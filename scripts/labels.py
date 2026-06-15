#!/usr/bin/env python3
import os
import sys
import argparse
import numpy as np
from pathlib import Path

# ----------------------------------------------------------------------
# Load YAML
# ----------------------------------------------------------------------
try:
    import yaml
except ImportError:
    print("❌ PyYAML not installed. Run: pip install pyyaml")
    sys.exit(1)

def load_yaml(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)

# ----------------------------------------------------------------------
# Default paths relative to the project root 
# ----------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = PROJECT_ROOT / 'config' / 'params.yaml'
DEFAULT_POINTCLOUD_DIR = PROJECT_ROOT / 'data' / 'sequences' 

# ----------------------------------------------------------------------
# Read point cloud 
# ----------------------------------------------------------------------
def read_bin_file(path):
    return np.fromfile(path, dtype=np.float32).reshape(-1, 4)

# ----------------------------------------------------------------------
# Read label file
# ----------------------------------------------------------------------
def read_label_file(path, expected_points=None):
    file_size = os.path.getsize(path)
    if expected_points is not None and file_size == expected_points * 2:
        return np.fromfile(path, dtype=np.uint16), 'uint16'
    raw = np.fromfile(path, dtype=np.uint32)
    return (raw & 0xFFFF).astype(np.uint16), 'uint32'

# ----------------------------------------------------------------------
# Conversion with validation
# ----------------------------------------------------------------------
def apply_mapping(raw_labels, mapping, num_classes):
    out = np.zeros_like(raw_labels, dtype=np.uint8)
    for src, dst in mapping.items():
        if dst < 0 or dst >= num_classes:
            print(f"❌ Target class {dst} for source ID {src} is out of range (0–{num_classes-1}).")
            sys.exit(1)
        out[raw_labels == src] = dst
    return out

# ----------------------------------------------------------------------
# Diagnostic: show raw IDs present and which are unmapped
# ----------------------------------------------------------------------
def diagnose_frame(raw_labels, mapping, frame_num):
    uniq, cnt = np.unique(raw_labels, return_counts=True)
    print(f"\n🔍 Frame {frame_num} – {len(raw_labels)} points")
    print(f"   Raw IDs: {uniq.tolist()}")
    mapped_set = set(mapping.keys())
    unmapped = [u for u in uniq if u not in mapped_set]
    if unmapped:
        print(f"   ⚠️  Unmapped IDs (will become class 0): {unmapped}")
    else:
        print("   ✅ All IDs covered by label_map.")

# ----------------------------------------------------------------------
# Process one sequence
# ----------------------------------------------------------------------
def process_sequence(pointcloud_dir, label_base_dir, sequence, output_dir, mapping, num_classes):
    ouster_dir = pointcloud_dir / sequence / 'ouster'
    if not ouster_dir.is_dir():
        print(f"❌ Point cloud directory not found: {ouster_dir}")
        return False

    labels_dir = label_base_dir / 'sequences' / sequence / 'labels'
    if not labels_dir.is_dir():
        labels_dir = label_base_dir / sequence / 'labels'
    if not labels_dir.is_dir():
        print(f"❌ Label directory not found: {labels_dir}")
        return False

    output_dir = Path(output_dir) if output_dir else pointcloud_dir / sequence / 'preprocessed_with_labels'
    output_dir.mkdir(parents=True, exist_ok=True)

    pc_files = sorted(f for f in ouster_dir.iterdir() if f.suffix == '.bin')
    print(f"\n📂 Sequence {sequence}: {len(pc_files)} frames")
    print(f"   Point clouds: {ouster_dir}")
    print(f"   Labels:       {labels_dir}")
    print(f"   Output:       {output_dir}")

    processed = 0
    missing = 0
    target_counts = np.zeros(num_classes, dtype=np.int64)
    diag_done = False

    for i, pc_file in enumerate(pc_files):
        frame_num = int(pc_file.stem)
        points = read_bin_file(pc_file)
        num_points = points.shape[0]

        label_file = labels_dir / f"{frame_num:06d}.label"
        if label_file.is_file():
            raw, fmt = read_label_file(label_file, expected_points=num_points)
            if not diag_done and i < 2:
                diagnose_frame(raw, mapping, frame_num)
                if i == 1:
                    diag_done = True
            if len(raw) != num_points:
                if len(raw) > num_points:
                    raw = raw[:num_points]
                else:
                    raw = np.pad(raw, (0, num_points - len(raw)), constant_values=0)
            custom = apply_mapping(raw, mapping, num_classes)
        else:
            custom = np.zeros(num_points, dtype=np.uint8)
            missing += 1

        np.savez_compressed(output_dir / f"{frame_num:06d}.npz",
                            points=points, labels=custom)
        for c in range(num_classes):
            target_counts[c] += np.count_nonzero(custom == c)
        processed += 1

        if (i + 1) % 300 == 0:
            print(f"   ... {i+1}/{len(pc_files)}")

    print(f"\n✅ Done. Processed {processed} frames, {missing} missing labels.")
    total_labeled = target_counts[1:].sum()
    if total_labeled == 0:
        print("❌ No labeled points after mapping! Update label_map in params.yaml.")
    else:
        print("\n📊 Class distribution (custom):")
        for c in range(num_classes):
            if c == 0:
                print(f"   unlabeled : {target_counts[c]:>10,} points")
            else:
                pct = target_counts[c] / total_labeled * 100
                print(f"   class {c:<4}: {target_counts[c]:>10,} points ({pct:5.1f}%)")
    return True

# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description='Convert labels with minimal args')
    parser.add_argument('--config', default=str(DEFAULT_CONFIG),
                        help='Path to params.yaml')
    parser.add_argument('--sequence', default='00',
                        help='Sequence ID (e.g., 00)')
    parser.add_argument('--pointcloud_dir', default=None,
                        help='Override base directory for sequences (default: data/sequences/)')
    parser.add_argument('--label_base_dir', default=None,
                        help='Override label base directory (default: same as pointcloud_dir)')
    parser.add_argument('--output_dir', default=None,
                        help='Output directory for .npz files')
    args = parser.parse_args()

    config = load_yaml(args.config)

    if args.pointcloud_dir:
        pc_base = Path(args.pointcloud_dir)
    else:
        pc_base = DEFAULT_POINTCLOUD_DIR

    if args.label_base_dir:
        lbl_base = Path(args.label_base_dir)
    else:
        lbl_base = pc_base

    label_map = config.get('label_map', None)
    if not label_map:
        print("❌ Missing 'label_map' in config file.")
        return
    mapping = {int(k): int(v) for k, v in label_map.items()}

    num_classes = config.get('model', {}).get('num_classes', 7)

    invalid = [v for v in mapping.values() if v < 0 or v >= num_classes]
    if invalid:
        print(f"❌ Mapping targets out of range (0-{num_classes-1}): {set(invalid)}")
        return

    process_sequence(pc_base, lbl_base, args.sequence, args.output_dir,
                     mapping, num_classes)


if __name__ == '__main__':
    main()