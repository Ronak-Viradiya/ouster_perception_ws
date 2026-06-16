#!/usr/bin/env python3
import numpy as np
import os
import sys
import argparse
import random
from pathlib import Path
import yaml

class RangeImageProjector:
    def __init__(self, height, width, fov_up, fov_down, max_range, min_range):
        self.height = height
        self.width = width
        self.fov_up = np.deg2rad(fov_up)
        self.fov_down = np.deg2rad(fov_down)
        self.fov = abs(self.fov_up) + abs(self.fov_down)
        self.max_range = max_range
        self.min_range = min_range

    def project_with_labels(self, points, labels):
        x, y, z, intensity = points[:, 0], points[:, 1], points[:, 2], points[:, 3]
        r = np.sqrt(x*x + y*y + z*z)
        mask = (r > self.min_range) & (r < self.max_range)
        r, x, y, z, intensity = r[mask], x[mask], y[mask], z[mask], intensity[mask]
        labels = labels[mask]

        yaw = -np.arctan2(y, x)
        pitch = np.arcsin(z / np.clip(r, 1e-6, None))

        proj_x = 0.5 * (yaw / np.pi + 1.0)
        proj_y = 1.0 - (pitch - self.fov_down) / self.fov

        col = np.floor(proj_x * (self.width - 1)).astype(np.int32)
        row = np.floor(proj_y * (self.height - 1)).astype(np.int32)
        col = np.clip(col, 0, self.width - 1)
        row = np.clip(row, 0, self.height - 1)

        range_img = np.zeros((self.height, self.width), dtype=np.float32)
        x_img = np.zeros_like(range_img)
        y_img = np.zeros_like(range_img)
        z_img = np.zeros_like(range_img)
        intensity_img = np.zeros_like(range_img)
        label_img = np.full((self.height, self.width), 0, dtype=np.int64)
        mask_img = np.zeros_like(range_img, dtype=np.uint8)

        tmp_range = np.full((self.height, self.width), np.inf, dtype=np.float32)
        for i in range(len(r)):
            ri, ci = row[i], col[i]
            if r[i] < tmp_range[ri, ci]:
                tmp_range[ri, ci] = r[i]
                range_img[ri, ci] = r[i]
                x_img[ri, ci] = x[i]
                y_img[ri, ci] = y[i]
                z_img[ri, ci] = z[i]
                intensity_img[ri, ci] = intensity[i]
                label_img[ri, ci] = labels[i]
                mask_img[ri, ci] = 1

        return range_img, intensity_img, x_img, y_img, z_img, mask_img, label_img

def load_yaml(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f) or {}

def load_bin_file(bin_path):
    return np.fromfile(bin_path, dtype=np.float32).reshape(-1, 4)

def load_npz_labels(npz_path):
    data = np.load(npz_path)
    if 'labels' in data:
        return data['labels']
    if 'semantic_labels' in data:
        return data['semantic_labels']
    raise KeyError(f"No 'labels' or 'semantic_labels' in {npz_path}. Keys: {data.files}")

def prepare_training_data(data_root, sequence, scan_subdir='ouster',
                          random_seed=42, val_ratio=0.2, shuffle=True,
                          config_path='config/params.yaml'):
    cfg = load_yaml(config_path)
    sensor_cfg = cfg.get('sensor', {})
    model_cfg = cfg.get('model', {})

    num_classes = model_cfg.get('num_classes', 7)
    height = model_cfg.get('height', 128)
    width = model_cfg.get('width', 2048)
    fov_up = sensor_cfg.get('fov_up', 45.0)
    fov_down = sensor_cfg.get('fov_down', -45.0)
    max_range = sensor_cfg.get('max_range', 100.0)
    min_range = sensor_cfg.get('min_range', 1.0)

    names_dict = model_cfg.get('names', {})
    class_names = [names_dict.get(i, f"class_{i}") for i in range(num_classes)]

    projector = RangeImageProjector(height, width, fov_up, fov_down, max_range, min_range)

    bin_dir = os.path.join(data_root, sequence, scan_subdir)
    npz_dir = os.path.join(data_root, sequence, 'preprocessed_with_labels')
    output_dir = os.path.join(data_root, sequence, 'training')
    train_dir = os.path.join(output_dir, 'train')
    val_dir = os.path.join(output_dir, 'val')

    missing = []
    if not os.path.isdir(bin_dir):
        missing.append(bin_dir)
    if not os.path.isdir(npz_dir):
        missing.append(npz_dir)
    if missing:
        print("❌ Missing directories:")
        for m in missing:
            print(f"   {m}")
        print("Make sure you have run label.py first.")
        sys.exit(1)

    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)

    bin_map = {Path(f).stem: f for f in os.listdir(bin_dir) if f.endswith('.bin')}
    npz_map = {Path(f).stem: f for f in os.listdir(npz_dir) if f.endswith('.npz')}
    common_stems = sorted(set(bin_map) & set(npz_map))

    if not common_stems:
        print("❌ No matching .bin / .npz pairs found.")
        sys.exit(1)

    if shuffle:
        random.seed(random_seed)
        shuffled = common_stems.copy()
        random.shuffle(shuffled)
    else:
        shuffled = common_stems
    split_idx = int((1 - val_ratio) * len(shuffled))
    train_stems = set(shuffled[:split_idx])
    val_stems = set(shuffled[split_idx:])

    print(f"Sequence {sequence}: {len(common_stems)} pairs → {len(train_stems)} train / {len(val_stems)} val")
    print(f"Image size: {height}×{width}, classes: {num_classes}")

    successful = 0
    label_counts = np.zeros(num_classes, dtype=np.int64)

    for stem in common_stems:
        bin_path = os.path.join(bin_dir, bin_map[stem])
        npz_path = os.path.join(npz_dir, npz_map[stem])
        try:
            points = load_bin_file(bin_path)
            labels = load_npz_labels(npz_path).astype(np.int64).flatten()

            if len(labels) != len(points):
                min_len = min(len(labels), len(points))
                print(f"⚠️  {stem}: size mismatch, truncating to {min_len}")
                labels = labels[:min_len]
                points = points[:min_len]

            range_img, intensity_img, x_img, y_img, z_img, mask_img, label_img = \
                projector.project_with_labels(points, labels)

            for c in range(num_classes):
                label_counts[c] += np.sum(label_img == c)

            dst_dir = train_dir if stem in train_stems else val_dir
            np.savez_compressed(os.path.join(dst_dir, f'{stem}.npz'),
                                range=range_img,
                                x=x_img, y=y_img, z=z_img,
                                intensity=intensity_img,
                                label=label_img,
                                mask=mask_img)
            successful += 1
            if successful % 100 == 0 or successful <= 3:
                print(f"   [{successful:04d}] {stem}")

        except Exception as e:
            import traceback
            print(f"❌ Failed {stem}: {e}")
            traceback.print_exc()

    print(f"\n✅ Done. {successful}/{len(common_stems)} frames.")
    print("Label distribution:")
    total = label_counts.sum()
    for c in range(num_classes):
        pct = label_counts[c] / total * 100 if total > 0 else 0
        print(f"   {class_names[c]:12s}: {label_counts[c]:8d}  ({pct:5.1f}%)")

def main():
    parser = argparse.ArgumentParser(description='Prepare SalsaNext training data')
    default_data_root = str(Path(__file__).resolve().parent.parent / 'data' / 'sequences')
    parser.add_argument('--data_root', default=default_data_root,
                        help=f'Base directory containing sequence folders (default: {default_data_root})')
    parser.add_argument('--sequence', default='00', help='Sequence ID')
    parser.add_argument('--scan_subdir', default='ouster', help='Subfolder with .bin files')
    parser.add_argument('--val_ratio', type=float, default=0.2)
    parser.add_argument('--random_seed', type=int, default=42)
    parser.add_argument('--no_shuffle', action='store_true', help='Disable shuffling')
    parser.add_argument('--config', default='config/params.yaml',
                        help='Path to params.yaml')
    args = parser.parse_args()

    data_root = os.path.abspath(args.data_root)

    prepare_training_data(
        data_root=data_root,
        sequence=args.sequence,
        scan_subdir=args.scan_subdir,
        random_seed=args.random_seed,
        val_ratio=args.val_ratio,
        shuffle=not args.no_shuffle,
        config_path=args.config
    )

if __name__ == '__main__':
    main()