#!/usr/bin/env python3
"""
SqueezeSeg: Real-time Semantic Segmentation for 3D LiDAR Point Clouds
Reference: https://arxiv.org/abs/1710.07368
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import sys
from pathlib import Path
import yaml

def load_params(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)

# ----------------------------------------------------------------------
# Fire Module (SqueezeNet building block)
# ----------------------------------------------------------------------
class FireModule(nn.Module):
    def __init__(self, in_channels, squeeze_channels, expand1x1, expand3x3):
        super().__init__()
        self.squeeze = nn.Conv2d(in_channels, squeeze_channels, 1)
        self.squeeze_bn = nn.BatchNorm2d(squeeze_channels)
        
        self.expand1x1 = nn.Conv2d(squeeze_channels, expand1x1, 1)
        self.expand1x1_bn = nn.BatchNorm2d(expand1x1)
        
        self.expand3x3 = nn.Conv2d(squeeze_channels, expand3x3, 3, padding=1)
        self.expand3x3_bn = nn.BatchNorm2d(expand3x3)
        
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x):
        x = self.relu(self.squeeze_bn(self.squeeze(x)))
        out1 = self.relu(self.expand1x1_bn(self.expand1x1(x)))
        out2 = self.relu(self.expand3x3_bn(self.expand3x3(x)))
        return torch.cat([out1, out2], dim=1)

# ----------------------------------------------------------------------
# SqueezeSeg Backbone (Modified SqueezeNet)
# ----------------------------------------------------------------------
class SqueezeSegBackbone(nn.Module):
    def __init__(self, input_channels=5):
        super().__init__()
        self.relu = nn.ReLU(inplace=True)
        
        # Initial convolution (no downsampling)
        self.conv1 = nn.Conv2d(input_channels, 64, 3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.pool1 = nn.MaxPool2d(2, stride=2)
        
        # Fire modules with downsampling at specific points
        self.fire2 = FireModule(64, 16, 64, 64)      # 64+64=128
        self.fire3 = FireModule(128, 16, 64, 64)     # 64+64=128
        self.fire4 = FireModule(128, 32, 128, 128)   # 128+128=256
        self.pool4 = nn.MaxPool2d(2, stride=2)
        
        self.fire5 = FireModule(256, 32, 128, 128)   # 128+128=256
        self.fire6 = FireModule(256, 48, 192, 192)   # 192+192=384
        self.fire7 = FireModule(384, 48, 192, 192)   # 192+192=384
        self.fire8 = FireModule(384, 64, 256, 256)   # 256+256=512
        self.pool8 = nn.MaxPool2d(2, stride=2)
        
        self.fire9 = FireModule(512, 64, 256, 256)   # 256+256=512
        
        self._initialize_weights()
        
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
        skip_connections.append(x)  # Skip 0: after conv1 (H, W) - 64 channels
        
        x = self.pool1(x)  # H/2, W/2
        x = self.fire2(x)  # 128 channels
        x = self.fire3(x)  # 128 channels
        x = self.fire4(x)  # 256 channels
        skip_connections.append(x)  # Skip 1: after fire4 (H/2, W/2) - 256 channels
        
        x = self.pool4(x)  # H/4, W/4
        x = self.fire5(x)  # 256 channels
        x = self.fire6(x)  # 384 channels
        x = self.fire7(x)  # 384 channels
        x = self.fire8(x)  # 512 channels
        skip_connections.append(x)  # Skip 2: after fire8 (H/4, W/4) - 512 channels
        
        x = self.pool8(x)  # H/8, W/8
        x = self.fire9(x)  # Deepest features: 512 channels
        
        return x, skip_connections

# ----------------------------------------------------------------------
# SqueezeSeg Decoder (with skip connections)
# ----------------------------------------------------------------------
class SqueezeSegDecoder(nn.Module):
    def __init__(self, num_classes=7):
        super().__init__()
        self.relu = nn.ReLU(inplace=True)
        
        # Decoder layers with upsampling
        self.deconv1 = nn.ConvTranspose2d(512, 512, 3, stride=2, padding=1, output_padding=1)
        self.deconv1_bn = nn.BatchNorm2d(512)
        
        # Skip 2: from fire8 (512 channels) -> match with deconv output (512 channels)
        # After concatenation: 512+512=1024 -> Fire10 output 512
        self.fire10 = FireModule(1024, 64, 256, 256)  # 256+256=512
        
        self.deconv2 = nn.ConvTranspose2d(512, 256, 3, stride=2, padding=1, output_padding=1)
        self.deconv2_bn = nn.BatchNorm2d(256)
        
        # Skip 1: from fire4 (256 channels) -> match with deconv output (256 channels)
        # After concatenation: 256+256=512 -> Fire11 output 384
        self.fire11 = FireModule(512, 48, 192, 192)  # 192+192=384
        
        self.deconv3 = nn.ConvTranspose2d(384, 128, 3, stride=2, padding=1, output_padding=1)
        self.deconv3_bn = nn.BatchNorm2d(128)
        
        # Skip 0: from conv1 (64 channels) -> match with deconv output (128 channels)
        # Need to project skip to 128 channels first
        self.skip_conv0 = nn.Conv2d(64, 128, 1)
        self.skip_bn0 = nn.BatchNorm2d(128)
        
        # After concatenation: 128+128=256 -> Fire12 output 128
        self.fire12 = FireModule(256, 32, 64, 64)  # 64+64=128
        
        # Final convolution
        self.final_conv = nn.Conv2d(128, num_classes, 1)
        
    def forward(self, x, skips):
        # skips[0] = after conv1 (H, W) - 64 channels
        # skips[1] = after fire4 (H/2, W/2) - 256 channels
        # skips[2] = after fire8 (H/4, W/4) - 512 channels
        
        # Decoder stage 1: H/8 -> H/4
        x = self.relu(self.deconv1_bn(self.deconv1(x)))  # 512 channels
        skip = skips[2]  # 512 channels
        if x.shape[2:] != skip.shape[2:]:
            x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=False)
        x = torch.cat([x, skip], dim=1)  # 1024 channels
        x = self.fire10(x)  # 512 channels
        
        # Decoder stage 2: H/4 -> H/2
        x = self.relu(self.deconv2_bn(self.deconv2(x)))  # 256 channels
        skip = skips[1]  # 256 channels
        if x.shape[2:] != skip.shape[2:]:
            x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=False)
        x = torch.cat([x, skip], dim=1)  # 512 channels
        x = self.fire11(x)  # 384 channels
        
        # Decoder stage 3: H/2 -> H
        x = self.relu(self.deconv3_bn(self.deconv3(x)))  # 128 channels
        skip = skips[0]  # 64 channels
        skip = self.relu(self.skip_bn0(self.skip_conv0(skip)))  # Project 64->128
        if x.shape[2:] != skip.shape[2:]:
            x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=False)
        x = torch.cat([x, skip], dim=1)  # 256 channels
        x = self.fire12(x)  # 128 channels
        
        x = self.final_conv(x)
        return x

# ----------------------------------------------------------------------
# Main SqueezeSeg Model
# ----------------------------------------------------------------------
class SqueezeSeg(nn.Module):
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
        
        # Default class names (adjust for your dataset)
        self.class_names = [
            'unlabeled', 'car', 'person', 'road',
            'building', 'vegetation', 'parking'
        ]
        
        DEFAULT_COLOR_MAP = torch.tensor([
            [0, 0, 0],        # unlabeled
            [255, 0, 0],      # car
            [0, 255, 255],    # person
            [128, 0, 128],    # road
            [70, 70, 70],     # building
            [0, 255, 0],      # vegetation
            [255, 255, 0]     # parking
        ], dtype=torch.uint8)
        self.color_map = DEFAULT_COLOR_MAP
        
        # Load config if provided
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
        
        # Normalization stats (typical for range images)
        self.register_buffer('input_mean', torch.tensor([12.12, 0.0, 0.0, -1.04, 0.21]).view(1, 5, 1, 1))
        self.register_buffer('input_std', torch.tensor([12.32, 11.7, 6.72, 0.86, 0.16]).view(1, 5, 1, 1))
        
        # Build network
        self.backbone = SqueezeSegBackbone(input_channels=self.input_channels)
        self.decoder = SqueezeSegDecoder(num_classes=self.num_classes)
        
        # Dropout for regularization
        self.dropout = nn.Dropout2d(0.1)
        
        print(f"✅ SqueezeSeg model initialized:")
        print(f"   - Input: {self.input_channels}x{self.height}x{self.width}")
        print(f"   - Classes: {self.num_classes}")
        print(f"   - Class names: {self.class_names}")
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
# Utility function to find config file
# ----------------------------------------------------------------------
def find_config_file(config_path=None):
    """
    Find config file by checking multiple possible locations
    """
    if config_path is not None and os.path.exists(config_path):
        return config_path
    
    # Possible config locations
    possible_paths = [
        "/home/ronak/ouster_perception_ws/config/params.yaml",
        "/home/ronak/ouster_perception_ws/src/config/params.yaml",
        os.path.expanduser("~/ouster_perception_ws/config/params.yaml"),
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "config", "params.yaml"),
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "config", "params.yaml"),
    ]
    
    for path in possible_paths:
        if os.path.exists(path):
            print(f"✅ Found config at: {path}")
            return path
    
    return None

# ----------------------------------------------------------------------
# Utility function to create and save model
# ----------------------------------------------------------------------
def create_and_save_model(config_path=None, save_path=None):
    script_dir = Path(__file__).parent.absolute()
    package_root = script_dir.parent
    
    # Find config file
    config_path = find_config_file(config_path)
    
    if config_path is None:
        print("⚠️ No config file found - using defaults")
    else:
        print(f"🔧 Using config from: {config_path}")
    
    if save_path is None:
        model_dir = os.path.join(package_root, 'models', 'pretrained_models')
        os.makedirs(model_dir, exist_ok=True)
        save_path = os.path.join(model_dir, 'SqueezeSeg.pth')
    
    print("🔄 Creating SqueezeSeg model…")
    model = SqueezeSeg(config_path=config_path)
    model.eval()
    
    # Test forward pass
    test_input = torch.randn(1, model.input_channels, model.height, model.width)
    output = model(test_input)
    print(f"   ✅ Forward pass OK – output shape: {output.shape}")
    
    # Save model
    torch.save({
        'model_state_dict': model.state_dict(),
        'model_architecture': 'SqueezeSeg',
        'num_classes': model.num_classes,
        'class_names': model.class_names,
        'color_map': model.color_map,
        'input_channels': model.input_channels,
        'height': model.height,
        'width': model.width,
        'model_info': {
            'description': 'SqueezeSeg with Fire modules and skip connections',
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
    # Try multiple ways to specify config path
    config_path = None
    
    # 1. Check environment variable
    if "CONFIG_PATH" in os.environ:
        config_path = os.environ["CONFIG_PATH"]
        print(f"📌 Using CONFIG_PATH from environment: {config_path}")
    
    # 2. Check if user passed as argument
    if len(sys.argv) > 1:
        config_path = sys.argv[1]
        print(f"📌 Using config from command line: {config_path}")
    
    # 3. Use default path
    if config_path is None:
        config_path = "/home/ronak/ouster_perception_ws/config/params.yaml"
        print(f"📌 Using default config path: {config_path}")
    
    create_and_save_model(config_path=config_path)
    print("\n🎉 Done. You can now use SqueezeSeg with your inference pipeline.")

if __name__ == "__main__":
    main()