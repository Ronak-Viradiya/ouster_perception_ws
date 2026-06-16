#!/usr/bin/env python3
import os
import sys
import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

def load_params(path):
    import yaml
    with open(path, 'r') as f:
        return yaml.safe_load(f) or {}

# ============================================================================
# SALSAnext BUILDING BLOCKS
# ============================================================================

class ResidualBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, stride: int = 1, dilation: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=dilation,
                               dilation=dilation, bias=False)
        self.bn1   = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=dilation,
                               dilation=dilation, bias=False)
        self.bn2   = nn.BatchNorm2d(out_ch)
        self.relu  = nn.ReLU(inplace=False)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch)
            )

    def forward(self, x):
        identity = self.shortcut(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.relu(out + identity)


class ContextModule(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.res1 = ResidualBlock(in_ch, out_ch, dilation=1)
        self.res2 = ResidualBlock(in_ch, out_ch, dilation=2)
        self.res3 = ResidualBlock(in_ch, out_ch, dilation=4)
        self.res4 = ResidualBlock(in_ch, out_ch, dilation=8)

    def forward(self, x):
        return self.res1(x) + self.res2(x) + self.res3(x) + self.res4(x)


class DecoderBlock(nn.Module):
    def __init__(self, in_ch: int, skip_ch: int, out_ch: int):
        super().__init__()
        if in_ch % 4 != 0:
            proj_ch = ((in_ch + 3) // 4) * 4
            self.projection = nn.Sequential(
                nn.Conv2d(in_ch, proj_ch, 1, bias=False),
                nn.BatchNorm2d(proj_ch),
                nn.ReLU(inplace=False)
            )
            in_ch = proj_ch
        else:
            self.projection = nn.Identity()

        self.pixel_shuffle = nn.PixelShuffle(upscale_factor=2)
        self.conv = nn.Conv2d(in_ch // 4 + skip_ch, out_ch, 3, padding=1, bias=False)
        self.bn   = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=False)

    def forward(self, x, skip):
        x = self.projection(x)
        x = self.pixel_shuffle(x)
        if x.shape[2:] != skip.shape[2:]:
            x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=False)
        return self.relu(self.bn(self.conv(torch.cat([x, skip], dim=1))))

class SalsaNext(nn.Module):
    def __init__(self, num_classes: int = 7, input_channels: int = 5,
                 height: int = 128, width: int = 2048,
                 class_names: list = None, color_map=None):
        super().__init__()
        self.num_classes    = num_classes
        self.input_channels = input_channels
        self.height         = height
        self.width          = width

        # Store meta information for later use (inference, visualisation)
        self.class_names = class_names
        if color_map is not None:
            if isinstance(color_map, torch.Tensor):
                self.register_buffer('color_map', color_map)
            else:
                self.color_map = color_map  # list of [R,G,B]
        else:
            self.color_map = None

        self.encoder_conv1 = nn.Sequential(
            nn.Conv2d(self.input_channels, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32), nn.ReLU(inplace=False)
        )
        self.encoder_res1 = ResidualBlock(32,  64,  stride=2)
        self.encoder_res2 = ResidualBlock(64,  128, stride=2)
        self.encoder_res3 = ResidualBlock(128, 256, stride=2)
        self.encoder_res4 = ResidualBlock(256, 512, stride=2)

        self.context = ContextModule(512, 512)

        self.decoder4 = DecoderBlock(512, 256, 256)
        self.decoder3 = DecoderBlock(256, 128, 128)
        self.decoder2 = DecoderBlock(128,  64,  64)
        self.decoder1 = DecoderBlock( 64,  32,  32)

        self.final_conv = nn.Sequential(
            nn.Conv2d(32, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64), nn.ReLU(inplace=False),
            nn.Dropout2d(0.1),
            nn.Conv2d(64, self.num_classes, 1)
        )

        self._init_weights()
        print(f"✅ SalsaNext: {self.input_channels}×{self.height}×{self.width}  "
              f"{self.num_classes} classes  {sum(p.numel() for p in self.parameters()):,} params")

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None: nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1); nn.init.constant_(m.bias, 0)

    def forward(self, x):
        s1 = self.encoder_conv1(x)
        s2 = self.encoder_res1(s1)
        s3 = self.encoder_res2(s2)
        s4 = self.encoder_res3(s3)
        x  = self.encoder_res4(s4)
        x  = self.context(x)
        x  = self.decoder4(x, s4)
        x  = self.decoder3(x, s3)
        x  = self.decoder2(x, s2)
        x  = self.decoder1(x, s1)
        x  = self.final_conv(x)
        if x.shape[2:] != (self.height, self.width):
            x = F.interpolate(x, size=(self.height, self.width),
                              mode='bilinear', align_corners=False)
        return x

def create_and_save_model(config_path=None, save_path=None,
                          num_classes=None, height=None, width=None, input_channels=None):
    """
    Build SalsaNext, run sanity check, and save checkpoint.
    Reads class_names & color_map from config if provided.
    Saves into the fixed pretrained_models directory unless overridden.
    """
    class_names = None
    color_map_raw = None

    if config_path and os.path.exists(config_path):
        params = load_params(config_path)
        mp = params.get('model', {})
        if num_classes is None:   num_classes   = mp.get('num_classes', 7)
        if height is None:        height        = mp.get('height', 128)
        if width is None:         width         = mp.get('width', 2048)
        if input_channels is None:input_channels = mp.get('input_channels', 5)

        names_dict = mp.get('names', None)
        if names_dict is not None:
            class_names = [names_dict[i] for i in range(len(names_dict))]
            print(f"✅ Loaded class names: {class_names}")

        color_dict = mp.get('color_map', None)
        if color_dict is not None:
            color_map_raw = [color_dict[i] for i in range(num_classes)]
            print(f"✅ Loaded colour map from config")
    else:
        if num_classes is None:   num_classes   = 7
        if height is None:        height        = 128
        if width is None:         width         = 2048
        if input_channels is None:input_channels = 5

    print(f"🔄 Creating SalsaNext with {num_classes} classes, "
          f"{input_channels} ch, {height}×{width}")
    model = SalsaNext(num_classes=num_classes,
                      input_channels=input_channels,
                      height=height,
                      width=width,
                      class_names=class_names,
                      color_map=color_map_raw)
    model.eval()

    dummy = torch.randn(1, input_channels, height, width)
    with torch.no_grad():
        logits = model(dummy)
    expected = (1, num_classes, height, width)
    assert logits.shape == expected, f"Shape {logits.shape} ≠ {expected}"
    print(f"   ✅ logits shape: {logits.shape}")

    FIXED_DIR = "/home/ronak/3D/ouster_os0/ouster_os0/src/models/models/pretrained_models"
    if save_path is None:
        os.makedirs(FIXED_DIR, exist_ok=True)
        save_path = os.path.join(FIXED_DIR, "THAB_salsanext.pth")

    checkpoint = {
        'model_state_dict': model.state_dict(),
        'architecture':     'salsanext',
        'num_classes':      num_classes,
        'input_channels':   input_channels,
        'height':           height,
        'width':            width,
        'total_params':     sum(p.numel() for p in model.parameters()),
    }
    if class_names is not None:
        checkpoint['class_names'] = class_names
    if color_map_raw is not None:
        checkpoint['color_map'] = color_map_raw

    torch.save(checkpoint, save_path)
    size_mb = os.path.getsize(save_path) / 1024**2
    print(f"💾 Saved → {save_path}  ({size_mb:.1f} MB)")
    return model

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Create and save SalsaNext model')
    parser.add_argument('--config', type=str, default=None,
                        help='Path to params.yaml')
    parser.add_argument('--save_path', type=str, default=None,
                        help='Custom save path (default: fixed pretrained_models dir)')
    parser.add_argument('--num_classes', type=int, default=None)
    parser.add_argument('--height', type=int, default=None)
    parser.add_argument('--width', type=int, default=None)
    parser.add_argument('--input_channels', type=int, default=None)

    args = parser.parse_args()

    print("=" * 60)
    print("Creating SalsaNext model")
    print("=" * 60)

    create_and_save_model(
        config_path=args.config,
        save_path=args.save_path,
        num_classes=args.num_classes,
        height=args.height,
        width=args.width,
        input_channels=args.input_channels
    )
    print("\n🎉 Done!")
    print("=" * 60)