#!/usr/bin/env python3
import os
import argparse
import struct
import numpy as np
import xml.etree.ElementTree as ET

# ==================== DEFAULT PATHS ====================
DEFAULT_INPUT_DIR = "/home/ronak/ouster_perception_ws/data/cvat_exports/sequence_00/"
DEFAULT_PCD_DIR = "/home/ronak/ouster_perception_ws/data/cvat_exports/sequence_00/velodyne_points/data/"
DEFAULT_OUTPUT_DIR = "/home/ronak/ouster_perception_ws/data/sequences/00/labels/"
# ========================================================

LABEL_MAPPING = {
    "Vegetation": 70,
    "Parking": 48,
    "Building": 50,
    "Road": 40,
    "Car": 10,
    "Person": 30,
}

def parse_boost_xml(xml_path):
    """Extract bounding boxes per frame from tracklet_labels.xml."""
    tree = ET.parse(xml_path)
    root = tree.getroot()

    objects = []
    for item in root.findall(".//item"):
        if item.find("objectType") is not None and item.find("poses") is not None:
            objects.append(item)

    print(f"Found {len(objects)} object entries in XML")
    frame_boxes = {}

    for obj in objects:
        obj_type = obj.find("objectType").text
        if obj_type not in LABEL_MAPPING:
            print(f"⚠️ Unknown objectType '{obj_type}' – skipping (add to LABEL_MAPPING if needed)")
            continue
        label_id = LABEL_MAPPING[obj_type]

        h = float(obj.find("h").text)
        w = float(obj.find("w").text)
        l = float(obj.find("l").text)
        first_frame = int(obj.find("first_frame").text)

        poses = obj.find("poses")
        pose_item = poses.find("item")
        if pose_item is None:
            continue
        tx = float(pose_item.find("tx").text)
        ty = float(pose_item.find("ty").text)
        tz = float(pose_item.find("tz").text)
        rz = float(pose_item.find("rz").text) if pose_item.find("rz") is not None else 0.0

        box = {
            'label': label_id,
            'cx': tx, 'cy': ty, 'cz': tz,
            'w': w, 'h': h, 'l': l,
            'rot': rz
        }
        frame_boxes.setdefault(first_frame, []).append(box)

    return frame_boxes

def read_pcd_binary(pcd_path):
    """
    Read a binary PCD file (little‑endian, fields: x y z intensity).
    Returns (N, 3) numpy array of (x, y, z).
    """
    with open(pcd_path, 'rb') as f:
        header = b''
        while True:
            line = f.readline()
            if not line:
                raise ValueError("PCD header incomplete")
            header += line
            if line.startswith(b'DATA'):
                data_type = line.decode().strip().split()[-1]
                if data_type not in ('binary', 'binary_compressed'):
                    raise ValueError(f"Unsupported PCD data format: {data_type}")
                break

        lines = header.decode().splitlines()
        for line in lines:
            if line.startswith('POINTS'):
                num_points = int(line.split()[-1])
                break
        else:
            raise ValueError("POINTS field not found in PCD header")

        point_size = 16  # 4 * 4 bytes (float)
        data = f.read(num_points * point_size)
        if len(data) != num_points * point_size:
            raise ValueError(f"PCD data size mismatch: expected {num_points*point_size} bytes, got {len(data)}")

        fmt = '<' + 'f' * (num_points * 4)  
        floats = struct.unpack(fmt, data)
        points = np.array(floats, dtype=np.float32).reshape(-1, 4)
        return points[:, :3]

def points_in_oriented_box(points, cx, cy, cz, w, h, l, rot):
    """Mask points inside oriented 3D box."""
    x = points[:, 0] - cx
    y = points[:, 1] - cy
    z = points[:, 2] - cz
    cos_r = np.cos(rot)
    sin_r = np.sin(rot)
    local_x =  x * cos_r + y * sin_r
    local_y = -x * sin_r + y * cos_r
    local_z = z
    return (np.abs(local_x) <= w/2) & (np.abs(local_y) <= l/2) & (np.abs(local_z) <= h/2)

def create_label_file(frame_num, boxes, pcd_path, output_dir):
    if not os.path.exists(pcd_path):
        print(f"❌ Missing: {pcd_path}")
        return False
    try:
        points = read_pcd_binary(pcd_path)
    except Exception as e:
        print(f"❌ Error reading {pcd_path}: {e}")
        return False

    if points.shape[0] == 0:
        print(f"⚠️ No points in {pcd_path}")
        return False

    labels = np.zeros(points.shape[0], dtype=np.uint32)
    for box in boxes:
        mask = points_in_oriented_box(points, box['cx'], box['cy'], box['cz'],
                                      box['w'], box['h'], box['l'], box['rot'])
        labels[mask] = box['label']

    out_file = os.path.join(output_dir, f"{frame_num:06d}.label")
    with open(out_file, 'wb') as f:
        f.write(labels.tobytes())
    print(f"✅ {out_file} – {np.sum(labels>0)} labeled points")
    return True

def main():
    parser = argparse.ArgumentParser(description="Convert CVAT XML + PCD to .label files")
    parser.add_argument("--input_dir", default=DEFAULT_INPUT_DIR,
                        help="Folder containing tracklet_labels.xml")
    parser.add_argument("--pcd_dir", default=DEFAULT_PCD_DIR,
                        help="Folder containing .pcd files (binary format)")
    parser.add_argument("--output_dir", default=DEFAULT_OUTPUT_DIR,
                        help="Folder where .label files will be saved (auto-created)")

    args = parser.parse_args()

    input_dir = args.input_dir
    pcd_dir = args.pcd_dir
    output_dir = args.output_dir

    xml_path = os.path.join(input_dir, "tracklet_labels.xml")
    if not os.path.exists(xml_path):
        print(f"❌ XML not found: {xml_path}")
        print("Check --input_dir or place tracklet_labels.xml there.")
        return

    if not os.path.isdir(pcd_dir):
        print(f"❌ PCD folder not found: {pcd_dir}")
        print("Check --pcd_dir or ensure it contains .pcd files.")
        return

    pcd_files = {}
    for f in os.listdir(pcd_dir):
        if f.endswith('.pcd'):
            try:
                frame_num = int(os.path.splitext(f)[0])
                pcd_files[frame_num] = os.path.join(pcd_dir, f)
            except ValueError:
                continue

    if not pcd_files:
        print(f"❌ No .pcd files in {pcd_dir}")
        return

    print(f"Found {len(pcd_files)} PCD files (frames {min(pcd_files.keys())}–{max(pcd_files.keys())})")

    frame_boxes = parse_boost_xml(xml_path)
    print(f"Parsed boxes for {len(frame_boxes)} frames")

    # Create output directory automatically
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory: {output_dir}")

    common_frames = set(frame_boxes.keys()) & set(pcd_files.keys())
    if not common_frames:
        print("⚠️ No matching frames between XML and PCD.")
        print(f"XML frames (first 10): {sorted(frame_boxes.keys())[:10]}")
        print(f"PCD frames (first 10): {sorted(pcd_files.keys())[:10]}")
        return

    print(f"Processing {len(common_frames)} common frames...")
    for frame in sorted(common_frames):
        create_label_file(frame, frame_boxes[frame], pcd_files[frame], output_dir)

    print("🎉 Done!")

if __name__ == "__main__":
    main()