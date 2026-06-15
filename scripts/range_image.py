#!/usr/bin/env python3
import os, sys, argparse, importlib.util
from pathlib import Path
import numpy as np
import torch
import cv2
import yaml

# ----------------------------------------------------------------------
# Project root
# ----------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

# ----------------------------------------------------------------------
# Dynamic model loader
# ----------------------------------------------------------------------
def get_model_class(arch):
    MODEL_FILES = {
        "salsanext":  ("models/model_scripts/salsanext.py", "SalsaNext"),
        "rangenet++": ("models/model_scripts/rangenetpp.py", "RangeNetPlusPlus"),
    }

    if arch not in MODEL_FILES:
        raise ValueError(f"Unknown architecture: {arch}. Known: {list(MODEL_FILES.keys())}")

    rel_path, class_name = MODEL_FILES[arch]
    file_path = PROJECT_ROOT / rel_path
    if not file_path.exists():
        raise FileNotFoundError(f"Model file not found: {file_path}")

    spec = importlib.util.spec_from_file_location(arch, file_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[arch] = module
    spec.loader.exec_module(module)

    ModelClass = getattr(module, class_name, None)
    if ModelClass is None:
        raise AttributeError(f"Class {class_name} not found in {file_path}")
    return ModelClass

# ----------------------------------------------------------------------
# Spherical projection (Ouster OS0‑128)
# ----------------------------------------------------------------------
def point_cloud_to_range_image(points, height=128, width=2048,
                               fov_up=45.0, fov_down=-45.0, max_range=100.0, min_range=1.0):
    x, y, z = points[:, 0], points[:, 1], points[:, 2]
    intensity = points[:, 3] if points.shape[1] >= 4 else np.zeros_like(x)

    r = np.sqrt(x**2 + y**2 + z**2)
    mask = (r > min_range) & (r < max_range)
    r, x, y, z, intensity = r[mask], x[mask], y[mask], z[mask], intensity[mask]

    yaw = -np.arctan2(y, x)
    pitch = np.arcsin(z / r)

    fov_up_rad, fov_down_rad = np.deg2rad(fov_up), np.deg2rad(fov_down)
    proj_x = 0.5 * (yaw / np.pi + 1.0)
    proj_y = 1.0 - (pitch - fov_down_rad) / (fov_up_rad - fov_down_rad)

    col = np.floor(proj_x * (width - 1)).astype(np.int32).clip(0, width - 1)
    row = np.floor(proj_y * (height - 1)).astype(np.int32).clip(0, height - 1)

    range_img = np.zeros((5, height, width), dtype=np.float32)
    range_img[0, row, col] = r
    range_img[1, row, col] = x
    range_img[2, row, col] = y
    range_img[3, row, col] = z
    range_img[4, row, col] = intensity
    return range_img

# ----------------------------------------------------------------------
# Checkpoint resolver
# ----------------------------------------------------------------------
def find_checkpoint(arch, cfg):
    ckpt_map = cfg.get('checkpoints', {})
    if arch in ckpt_map:
        path = str(PROJECT_ROOT / ckpt_map[arch])
        if os.path.exists(path):
            return path

    legacy = cfg.get('paths', {}).get('model_weights', None)
    if legacy:
        path = str(PROJECT_ROOT / legacy)
        if os.path.exists(path):
            return path

    p1 = PROJECT_ROOT / 'models' / 'pretrained_models' / f'THAB_{arch}.pth'
    if p1.exists():
        return str(p1)

    p2 = PROJECT_ROOT / 'models' / 'models' / 'pretrained_models' / f'THAB_{arch}.pth'
    if p2.exists():
        return str(p2)

    searched = [f"YAML checkpoints.{arch}", "YAML paths.model_weights", str(p1), str(p2)]
    raise FileNotFoundError(
        f"No checkpoint found for '{arch}'.\nTried:\n" +
        "\n".join(f"  - {p}" for p in searched) +
        "\nPlease set a valid path in config/params.yaml or place the .pth file in one of the above locations."
    )

# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='config/params.yaml', help='YAML config file')
    parser.add_argument('--cloud', default=None, help='Path to .npy point cloud (optional)')
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        cfg = yaml.safe_load(f)

    arch        = cfg['model']['architecture']
    num_classes = cfg['model']['num_classes']
    height      = cfg['model']['height']
    width       = cfg['model']['width']
    input_ch    = cfg['model']['input_channels']
    color_dict  = cfg['model']['color_map']
    color_map   = np.array([color_dict[i] for i in range(num_classes)], dtype=np.uint8)

    ckpt_path = find_checkpoint(arch, cfg)
    print(f"📦 Using checkpoint: {ckpt_path}")

    ModelClass = get_model_class(arch)
    model = ModelClass(num_classes=num_classes, input_channels=input_ch,
                       height=height, width=width, color_map=color_dict)
    checkpoint = torch.load(ckpt_path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print(f"✅ Loaded {arch}")

    if args.cloud and os.path.exists(args.cloud):
        points = np.load(args.cloud)
        print(f"📂 Loaded point cloud: {args.cloud} ({points.shape[0]} points)")
    else:
        print("⚠️  No point cloud – using random test cloud")
        N = 50000
        angles = np.random.uniform(-np.pi, np.pi, N)
        pitches = np.random.uniform(-np.deg2rad(45), np.deg2rad(45), N)
        r = np.random.uniform(1.0, 100.0, N)
        x = r * np.cos(pitches) * np.cos(angles)
        y = r * np.cos(pitches) * np.sin(angles)
        z = r * np.sin(pitches)
        intensity = np.random.uniform(0, 1, N)
        points = np.stack([x, y, z, intensity], axis=1)

    range_img = point_cloud_to_range_image(points, height=height, width=width)
    tensor_in = torch.from_numpy(range_img).unsqueeze(0)

    for c in range(5):
        ch = tensor_in[:, c]
        min_val, max_val = ch.min(), ch.max()
        if max_val - min_val > 1e-6:
            tensor_in[:, c] = (ch - min_val) / (max_val - min_val)

    with torch.no_grad():
        logits = model(tensor_in)
        pred   = torch.argmax(logits, dim=1).squeeze(0).cpu().numpy()

    colored = color_map[pred]
    bgr = cv2.cvtColor(colored, cv2.COLOR_RGB2BGR)
    save_dir = Path(cfg['paths']['predictions_dir']).expanduser().resolve()
    save_dir.mkdir(parents=True, exist_ok=True)
    save_path = save_dir / f'colored_{arch}.png'
    cv2.imwrite(str(save_path), bgr)
    print(f"🖼️  Saved → {save_path}")

    unique, counts = np.unique(pred, return_counts=True)
    names = cfg['model']['names']
    print("📊 Class distribution:")
    for cls, cnt in zip(unique, counts):
        print(f"   {cls} ({names.get(cls, '?')}): {cnt} pixels")

if __name__ == '__main__':
    main()