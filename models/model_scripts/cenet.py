#!/usr/bin/env python3
"""
CENet: Context Encoding for Semantic Segmentation in 3D LiDAR
Reference: https://arxiv.org/abs/2105.08332
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import sys
from pathlib import Path
import yaml
import argparse

def load_params(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)

# ----------------------------------------------------------------------
# Context Encoding Module (CEM)
# ----------------------------------------------------------------------
class ContextEncodingModule(nn.Module):
    """
    Learnable context vectors to capture global scene context
    """
    def __init__(self, in_channels, num_codes=32):
        super().__init__()
        self.num_codes = num_codes
        
        # Learnable codebook (context vectors)
        self.codebook = nn.Parameter(torch.Tensor(num_codes, in_channels))
        self.codebook.data.normal_(0, 0.1)
        
        # Attention mechanism
        self.attention = nn.Sequential(
            nn.Conv2d(in_channels, num_codes, 1),
            nn.Softmax(dim=1)
        )
        
        # Projection after aggregation
        self.projection = nn.Sequential(
            nn.Conv2d(in_channels + in_channels, in_channels, 1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        b, c, h, w = x.shape
        
        # Reshape to [b, c, h*w]
        x_flat = x.view(b, c, -1)
        
        # Compute attention weights [b, num_codes, h*w]
        attn_weights = self.attention(x).view(b, self.num_codes, -1)
        attn_weights = F.softmax(attn_weights, dim=1)  # Normalize across codes
        
        # Weighted combination of codebook vectors
        # [b, num_codes, h*w] @ [num_codes, c] -> [b, c, h*w]
        context = torch.bmm(attn_weights.transpose(1, 2), 
                           self.codebook.unsqueeze(0).expand(b, -1, -1))
        context = context.transpose(1, 2).view(b, c, h, w)
        
        # Concatenate with original features
        out = torch.cat([x, context], dim=1)
        out = self.projection(out)
        
        return out

# ----------------------------------------------------------------------
# Dual Attention Module (DAM)
# ----------------------------------------------------------------------
class DualAttentionModule(nn.Module):
    """
    Combines spatial attention and channel attention
    """
    def __init__(self, in_channels, reduction=16):
        super().__init__()
        
        # Spatial attention
        self.spatial_conv = nn.Conv2d(in_channels, 1, kernel_size=1)
        
        # Channel attention
        self.channel_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.channel_fc1 = nn.Linear(in_channels, in_channels // reduction)
        self.channel_fc2 = nn.Linear(in_channels // reduction, in_channels)
        
        # Fusion
        self.fusion = nn.Sequential(
            nn.Conv2d(in_channels * 2, in_channels, 1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, x):
        # Spatial attention
        spatial_weights = torch.sigmoid(self.spatial_conv(x))
        spatial_out = x * spatial_weights
        
        # Channel attention (Squeeze-and-Excitation)
        b, c, h, w = x.shape
        channel_avg = self.channel_avg_pool(x).view(b, c)
        channel_weights = torch.sigmoid(self.channel_fc2(self.channel_fc1(channel_avg)))
        channel_weights = channel_weights.view(b, c, 1, 1)
        channel_out = x * channel_weights
        
        # Combine
        out = torch.cat([spatial_out, channel_out], dim=1)
        out = self.fusion(out)
        
        return out

# ----------------------------------------------------------------------
# CENet Encoder
# ----------------------------------------------------------------------
class CENetEncoder(nn.Module):
    def __init__(self, input_channels=5):
        super().__init__()
        self.relu = nn.ReLU(inplace=True)
        
        # Entry flow (like ERFNet)
        self.conv1 = nn.Conv2d(input_channels, 32, 3, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        
        self.conv2 = nn.Conv2d(32, 64, 3, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        
        # Downsampling blocks
        self.down1 = self._make_downsample(64, 128)
        self.down2 = self._make_downsample(128, 256)
        self.down3 = self._make_downsample(256, 512)
        
        # Context Encoding Module at bottleneck
        self.context_module = ContextEncodingModule(512, num_codes=64)
        self.dual_attention = DualAttentionModule(512)
        
        # Additional downsampling
        self.down4 = self._make_downsample(512, 1024)
        
        self._initialize_weights()
    
    def _make_downsample(self, in_ch, out_ch):
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, stride=2, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        skip_connections = []
        
        x = self.relu(self.bn1(self.conv1(x)))
        skip_connections.append(x)  # H/2 - 32 channels
        
        x = self.relu(self.bn2(self.conv2(x)))
        skip_connections.append(x)  # H/4 - 64 channels
        
        x = self.down1(x)
        skip_connections.append(x)  # H/8 - 128 channels
        
        x = self.down2(x)
        skip_connections.append(x)  # H/16 - 256 channels
        
        x = self.down3(x)  # H/32 - 512 channels
        
        # Context encoding and attention
        x = self.context_module(x)
        x = self.dual_attention(x)
        
        x = self.down4(x)  # H/64 - 1024 channels
        
        return x, skip_connections

# ----------------------------------------------------------------------
# CENet Decoder (FIXED)
# ----------------------------------------------------------------------
class CENetDecoder(nn.Module):
    def __init__(self, num_classes=7):
        super().__init__()
        self.relu = nn.ReLU(inplace=True)
        
        # Decoder blocks with skip connections
        self.deconv1 = nn.ConvTranspose2d(1024, 512, 3, stride=2, padding=1, output_padding=1)
        self.deconv1_bn = nn.BatchNorm2d(512)
        
        # FIXED: Match actual skip connection sizes
        # skip[3] = after down2 -> 256 channels
        self.decoder_conv1 = nn.Sequential(
            nn.Conv2d(512 + 256, 512, 3, padding=1),  # 512 + 256 = 768
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )
        
        self.deconv2 = nn.ConvTranspose2d(512, 256, 3, stride=2, padding=1, output_padding=1)
        self.deconv2_bn = nn.BatchNorm2d(256)
        
        # skip[2] = after down1 -> 128 channels
        self.decoder_conv2 = nn.Sequential(
            nn.Conv2d(256 + 128, 256, 3, padding=1),  # 256 + 128 = 384
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        
        self.deconv3 = nn.ConvTranspose2d(256, 128, 3, stride=2, padding=1, output_padding=1)
        self.deconv3_bn = nn.BatchNorm2d(128)
        
        # skip[1] = after conv2 -> 64 channels
        self.decoder_conv3 = nn.Sequential(
            nn.Conv2d(128 + 64, 128, 3, padding=1),  # 128 + 64 = 192
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        
        self.deconv4 = nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=1)
        self.deconv4_bn = nn.BatchNorm2d(64)
        
        # skip[0] = after conv1 -> 32 channels
        self.decoder_conv4 = nn.Sequential(
            nn.Conv2d(64 + 32, 64, 3, padding=1),  # 64 + 32 = 96
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        
        # Final upsampling and classification
        self.final_conv = nn.Sequential(
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, num_classes, 1)
        )
    
    def forward(self, x, skips):
        # skips[0] = after conv1 -> 32 channels (H/2)
        # skips[1] = after conv2 -> 64 channels (H/4)
        # skips[2] = after down1 -> 128 channels (H/8)
        # skips[3] = after down2 -> 256 channels (H/16)
        
        # Decoder stage 1: H/64 -> H/32
        x = self.relu(self.deconv1_bn(self.deconv1(x)))
        skip = skips[3]  # 256 channels
        if x.shape[2:] != skip.shape[2:]:
            x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=False)
        x = torch.cat([x, skip], dim=1)  # 512 + 256 = 768
        x = self.decoder_conv1(x)
        
        # Decoder stage 2: H/32 -> H/16
        x = self.relu(self.deconv2_bn(self.deconv2(x)))
        skip = skips[2]  # 128 channels
        if x.shape[2:] != skip.shape[2:]:
            x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=False)
        x = torch.cat([x, skip], dim=1)  # 256 + 128 = 384
        x = self.decoder_conv2(x)
        
        # Decoder stage 3: H/16 -> H/8
        x = self.relu(self.deconv3_bn(self.deconv3(x)))
        skip = skips[1]  # 64 channels
        if x.shape[2:] != skip.shape[2:]:
            x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=False)
        x = torch.cat([x, skip], dim=1)  # 128 + 64 = 192
        x = self.decoder_conv3(x)
        
        # Decoder stage 4: H/8 -> H/4
        x = self.relu(self.deconv4_bn(self.deconv4(x)))
        skip = skips[0]  # 32 channels
        if x.shape[2:] != skip.shape[2:]:
            x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=False)
        x = torch.cat([x, skip], dim=1)  # 64 + 32 = 96
        x = self.decoder_conv4(x)
        
        # Final classification
        x = self.final_conv(x)
        return x

# ----------------------------------------------------------------------
# Main CENet Model
# ----------------------------------------------------------------------
class CENet(nn.Module):
    def __init__(self,
                 num_classes=7,
                 input_channels=5,
                 height=128,
                 width=2048,
                 config_path=None):
        super().__init__()
        
        self.num_classes = num_classes
        self.height = height
        self.width = width
        self.input_channels = input_channels
        
        # Default class names
        self.class_names = [
            'unlabeled', 'car', 'person', 'road',
            'building', 'vegetation', 'parking'
        ]
        
        DEFAULT_COLOR_MAP = torch.tensor([
            [0, 0, 0],
            [255, 0, 0],
            [0, 255, 255],
            [128, 0, 128],
            [70, 70, 70],
            [0, 255, 0],
            [255, 255, 0]
        ], dtype=torch.uint8)
        self.color_map = DEFAULT_COLOR_MAP
        
        # Load config
        if config_path is not None and os.path.exists(config_path):
            try:
                params = load_params(config_path)
                mp = params.get('model', {})
                
                self.num_classes = mp.get('num_classes', self.num_classes)
                self.height = mp.get('height', self.height)
                self.width = mp.get('width', self.width)
                self.input_channels = mp.get('input_channels', self.input_channels)
                
                names_dict = mp.get('names', None)
                if names_dict is not None:
                    self.class_names = [names_dict[i] for i in range(len(names_dict))]
                    print(f"✅ Loaded class names from config: {self.class_names}")
                
                color_dict = mp.get('color_map', None)
                if color_dict is not None:
                    colors = []
                    for i in range(self.num_classes):
                        if i in color_dict:
                            colors.append(color_dict[i])
                        else:
                            colors.append([0, 0, 0])
                    self.color_map = torch.tensor(colors, dtype=torch.uint8)
                    print(f"✅ Loaded custom colour map from config.")
                
                print(f"✅ Loaded config from: {config_path}")
            except Exception as e:
                print(f"⚠️ Error loading config: {e}, keeping provided parameters")
        
        # Normalization stats
        self.register_buffer('input_mean', torch.tensor([12.12, 0.0, 0.0, -1.04, 0.21]).view(1, 5, 1, 1))
        self.register_buffer('input_std', torch.tensor([12.32, 11.7, 6.72, 0.86, 0.16]).view(1, 5, 1, 1))
        
        # Build network
        self.backbone = CENetEncoder(input_channels=self.input_channels)
        self.decoder = CENetDecoder(num_classes=self.num_classes)
        
        self.dropout = nn.Dropout2d(0.1)
        
        print(f"✅ CENet model initialized:")
        print(f"   - Input: {self.input_channels}x{self.height}x{self.width}")
        print(f"   - Classes: {self.num_classes}")
        print(f"   - Context Codes: 64")
        print(f"   - Total parameters: {sum(p.numel() for p in self.parameters()):,}")
    
    def forward(self, x, return_probabilities=False):
        x, skips = self.backbone(x)
        x = self.decoder(x, skips)
        x = self.dropout(x)
        
        # Upsample to original size
        x = F.interpolate(x, size=(self.height, self.width),
                         mode='bilinear', align_corners=False)
        
        if return_probabilities:
            return F.softmax(x, dim=1)
        return x
    
    @torch.no_grad()
    def get_prediction(self, x):
        probs = self.forward(x, return_probabilities=True)
        pred_classes = torch.argmax(probs, dim=1)
        colored = self.color_map[pred_classes].permute(0, 3, 1, 2).contiguous()
        return pred_classes, probs, colored

# ----------------------------------------------------------------------
# Utility function to create and save model
# ----------------------------------------------------------------------
def create_and_save_model(config_path=None, save_path=None):
    script_dir = Path(__file__).parent.absolute()
    package_root = script_dir.parent
    
    # Handle config path
    if config_path is None:
        # Try to find config in standard locations
        possible_paths = [
            os.path.join(package_root, 'config', 'params.yaml'),
            os.path.join(package_root, 'src', 'config', 'params.yaml'),
            os.path.join(script_dir, 'config', 'params.yaml'),
            os.path.join(script_dir, '../config/params.yaml'),
            os.path.join(script_dir, '../../config/params.yaml'),
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                config_path = path
                print(f"🔧 Auto-discovered config at: {config_path}")
                break
        
        if config_path is None:
            print(f"⚠️ No config found, using defaults")
    
    if save_path is None:
        model_dir = os.path.join(package_root, 'models', 'pretrained_models')
        os.makedirs(model_dir, exist_ok=True)
        save_path = os.path.join(model_dir, 'CENet.pth')
    
    print("🔄 Creating CENet model…")
    model = CENet(config_path=config_path)
    model.eval()
    
    # Test forward pass
    test_input = torch.randn(1, model.input_channels, model.height, model.width)
    output = model(test_input)
    print(f"   ✅ Forward pass OK – output shape: {output.shape}")
    
    # Save model
    torch.save({
        'model_state_dict': model.state_dict(),
        'model_architecture': 'CENet',
        'num_classes': model.num_classes,
        'class_names': model.class_names,
        'color_map': model.color_map,
        'input_channels': model.input_channels,
        'height': model.height,
        'width': model.width,
        'config_path': config_path,
        'model_info': {
            'description': 'CENet with Context Encoding and Dual Attention',
            'total_parameters': sum(p.numel() for p in model.parameters()),
            'trainable_parameters': sum(p.numel() for p in model.parameters() if p.requires_grad)
        }
    }, save_path)
    
    size_mb = os.path.getsize(save_path) / (1024 * 1024)
    print(f"✅ Model saved to {save_path} ({size_mb:.2f} MB)")
    return model

# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description='CENet: Context Encoding Network for 3D LiDAR Segmentation'
    )
    parser.add_argument('--config', '-c', type=str, default=None,
                       help='Path to configuration YAML file')
    parser.add_argument('--save', '-s', type=str, default=None,
                       help='Path to save the model')
    args = parser.parse_args()
    
    # Determine config path (priority: CLI > ENV > Auto-discover)
    config_path = args.config
    
    if config_path is None and os.environ.get('CONFIG_PATH'):
        config_path = os.environ.get('CONFIG_PATH')
        print(f"🔧 Using config from ENV: {config_path}")
    
    if config_path is not None and not os.path.exists(config_path):
        print(f"⚠️ Config not found at {config_path}, using defaults")
        config_path = None
    
    create_and_save_model(config_path=config_path, save_path=args.save)
    print("\n🎉 Done. You can now use CENet with your inference pipeline.")

if __name__ == "__main__":
    main()