#!/usr/bin/env python3
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
# Darknet53 backbone
# ----------------------------------------------------------------------
class Darknet53(nn.Module):
    def __init__(self, input_channels=5):
        super().__init__()
        self.relu = nn.LeakyReLU(0.1, inplace=False)

        self.conv1 = nn.Conv2d(input_channels, 32, 3, padding=1)
        self.bn1   = nn.BatchNorm2d(32)

        self.layer1 = self._make_layer(32, 64, 1)      
        self.layer2 = self._make_layer(64, 128, 2)     
        self.layer3 = self._make_layer(128, 256, 8)    
        self.layer4 = self._make_layer(256, 512, 8)   
        self.layer5 = self._make_layer(512, 1024, 4) 

        self._initialize_weights()

    def _make_layer(self, in_ch, out_ch, num_blocks):
        layers = [
            nn.Conv2d(in_ch, out_ch, 3, stride=2, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(0.1, inplace=False)
        ]
        for _ in range(num_blocks):
            layers.append(ResidualBlock(out_ch))
        return nn.Sequential(*layers)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        skip_connections = []
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x);  skip_connections.append(x)
        x = self.layer2(x);  skip_connections.append(x)
        x = self.layer3(x);  skip_connections.append(x)
        x = self.layer4(x);  skip_connections.append(x)
        x = self.layer5(x)   # deepest features (no skip here)
        return x, skip_connections

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels//2, 1)
        self.bn1   = nn.BatchNorm2d(channels//2)
        self.conv2 = nn.Conv2d(channels//2, channels, 3, padding=1)
        self.bn2   = nn.BatchNorm2d(channels)
        self.relu  = nn.LeakyReLU(0.1, inplace=False)

    def forward(self, x):
        residual = x
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        x = self.relu(x + residual)
        return x

# ----------------------------------------------------------------------
# Decoder block with skip connection
# ----------------------------------------------------------------------
class DecoderBlock(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)

        self.skip_conv = nn.Conv2d(skip_channels, out_channels, 1)
        self.skip_bn   = nn.BatchNorm2d(out_channels)
        self.relu      = nn.LeakyReLU(0.1, inplace=False)

        # after concatenation: in_channels + out_channels
        self.conv = nn.Conv2d(in_channels + out_channels, out_channels, 3, padding=1)
        self.bn   = nn.BatchNorm2d(out_channels)

        self.refine = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.1, inplace=False)
        )

    def forward(self, x, skip):
        x = self.up(x)
        skip = self.relu(self.skip_bn(self.skip_conv(skip)))
        if x.shape != skip.shape:
            x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=False)
        x = torch.cat([x, skip], dim=1)
        x = self.relu(self.bn(self.conv(x)))
        x = self.refine(x)
        return x

# ----------------------------------------------------------------------
# RangeNet++ model
# ----------------------------------------------------------------------
class RangeNetPlusPlus(nn.Module):
    def __init__(self,
                 num_classes=7,
                 input_channels=5,
                 height=128,
                 width=2048,
                 config_path=None):
        super().__init__()

        self.num_classes    = num_classes
        self.height         = height
        self.width          = width
        self.input_channels = input_channels

        self.class_names = [
            'unlabeled', 'car', 'person', 'road',
            'building', 'vegetation', 'parking'
        ]

        DEFAULT_COLOR_MAP = torch.tensor([
            [0,   0,   0  ],
            [255, 0,   0  ],
            [0,   255, 255],
            [128, 0,   128],
            [70,  70,  70 ],
            [0,   255, 0  ],
            [255, 255, 0  ]
        ], dtype=torch.uint8)
        self.color_map = DEFAULT_COLOR_MAP

        if config_path is not None and os.path.exists(config_path):
            try:
                params = load_params(config_path)
                mp = params.get('model', {})

                # Override only what's present in the config
                self.num_classes    = mp.get('num_classes', self.num_classes)
                self.height         = mp.get('height', self.height)
                self.width          = mp.get('width', self.width)
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

        self.register_buffer('input_mean', torch.tensor([12.12, 0.0, 0.0, -1.04, 0.21]).view(1,5,1,1))
        self.register_buffer('input_std',  torch.tensor([12.32, 11.7, 6.72, 0.86, 0.16]).view(1,5,1,1))

        # ---- Build network ----
        self.backbone = Darknet53(input_channels=self.input_channels)

        self.decoder4 = DecoderBlock(1024, 512, 256)
        self.decoder3 = DecoderBlock(256,  256, 128)
        self.decoder2 = DecoderBlock(128,  128, 64)
        self.decoder1 = DecoderBlock(64,   64,  32)

        self.final_conv = nn.Sequential(
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.LeakyReLU(0.1, inplace=False),
            nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.LeakyReLU(0.1, inplace=False),
            nn.Conv2d(64, self.num_classes, 1)
        )

        self.dropout = nn.Dropout2d(0.1)
        self.softmax = nn.Softmax(dim=1)

        print(f"✅ RangeNet++ model initialized:")
        print(f"   - Input: {self.input_channels}x{self.height}x{self.width}")
        print(f"   - Classes: {self.num_classes}")
        print(f"   - Class names: {self.class_names}")
        print(f"   - Total parameters: {sum(p.numel() for p in self.parameters()):,}")

    def forward(self, x, return_probabilities=False):
        x, skips = self.backbone(x)
        x = self.decoder4(x, skips[3])
        x = self.decoder3(x, skips[2])
        x = self.decoder2(x, skips[1])
        x = self.decoder1(x, skips[0])
        x = self.dropout(x)
        x = self.final_conv(x)
        x = F.interpolate(x, size=(self.height, self.width),
                          mode='bilinear', align_corners=False)
        if return_probabilities:
            return self.softmax(x)
        return x

    @torch.no_grad()
    def get_prediction(self, x):
        probs = self.forward(x, return_probabilities=True)
        pred_classes = torch.argmax(probs, dim=1)
        colored = self.color_map[pred_classes].permute(0,3,1,2).contiguous()
        return pred_classes, probs, colored

def create_and_save_model(config_path=None, save_path=None):
    script_dir  = Path(__file__).parent.absolute()
    package_root = script_dir.parent
    if config_path and not os.path.isabs(config_path):
        config_path = os.path.join(package_root, config_path)

    if save_path is None:
        model_dir = os.path.join(package_root, 'models', 'pretrained_models')
        os.makedirs(model_dir, exist_ok=True)
        save_path = os.path.join(model_dir, 'rangenet_os0.pth')

    print("🔄 Creating RangeNet++ model…")
    model = RangeNetPlusPlus(config_path=config_path)
    model.eval()

    test_input = torch.randn(1, model.input_channels, model.height, model.width)
    output = model(test_input)
    print(f"   ✅ Forward pass OK – output shape: {output.shape}")

    torch.save({
        'model_state_dict': model.state_dict(),
        'model_architecture': 'RangeNetPlusPlus',
        'num_classes': model.num_classes,
        'class_names': model.class_names,
        'color_map': model.color_map,
        'input_channels': model.input_channels,
        'height': model.height,
        'width': model.width,
        'model_info': {
            'description': 'RangeNet++ with Darknet53 backbone and skip connections',
            'total_parameters': sum(p.numel() for p in model.parameters()),
            'trainable_parameters': sum(p.numel() for p in model.parameters() if p.requires_grad)
        }
    }, save_path)

    size_mb = os.path.getsize(save_path) / (1024*1024)
    print(f"✅ Model saved to {save_path} ({size_mb:.2f} MB)")
    return model

def main():
    config_path = os.environ.get("CONFIG_PATH", "/home/ronak/3D/ouster_os0/ouster_os0/src/config/params.yaml")
    
    if not os.path.exists(config_path):
        print(f"⚠️ Config not found at {config_path} – using defaults")
        config_path = None
    else:
        print(f"🔧 Using config from: {config_path}")

    create_and_save_model(config_path=config_path)
    print("\n🎉 Done. You can now use this model with your inference pipeline.")

if __name__ == "__main__":
    main()