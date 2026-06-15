#!/usr/bin/env python3
import numpy as np
import yaml
import struct
import os

# ─────────────────────────────────────────────────────────────────────────────
# YAML config loader
# ─────────────────────────────────────────────────────────────────────────────
def load_params(yaml_file):
    if not os.path.exists(yaml_file):
        print(f"⚠️  Parameter file not found: {yaml_file}")
        return {}

    with open(yaml_file, 'r') as f:
        params = yaml.safe_load(f)

    if params is None:
        print(f"⚠️  Parameter file is empty: {yaml_file}")
        return {}

    print(f"✅ Loaded parameters from {yaml_file}")
    return params
# ─────────────────────────────────────────────────────────────────────────────
# File I/O helpers  
# ─────────────────────────────────────────────────────────────────────────────

def save_bin_file(points, filename):
    points.astype(np.float32).tofile(filename)
    print(f"✅ Saved {points.shape[0]} points → {filename}")

def read_bin_file(filename):

    points = np.fromfile(filename, dtype=np.float32)
    return points.reshape(-1, 4)


def read_label_file(label_path):
    try:
        raw = np.fromfile(label_path, dtype=np.uint32).reshape(-1)
        if label_path.endswith('.label'):
            return (raw & 0xFFFF).astype(np.uint32)
        else:
            return raw
    except Exception as e:
        print(f"❌ Error reading label file {label_path}: {e}")
        return None


def pointcloud2_to_array(cloud_msg):
    try:
        n_pts = cloud_msg.width * cloud_msg.height
        if n_pts == 0:
            return np.zeros((0, 4), dtype=np.float32)

        step     = cloud_msg.point_step
        offsets  = {f.name: f.offset for f in cloud_msg.fields}
        x_off = offsets.get('x',         0)
        y_off = offsets.get('y',         4)
        z_off = offsets.get('z',         8)
        i_off = offsets.get('intensity', 16)
        raw = np.frombuffer(bytes(cloud_msg.data), dtype=np.uint8).reshape(n_pts, step)

        x         = raw[:, x_off : x_off + 4].copy().view(np.float32).reshape(-1)
        y         = raw[:, y_off : y_off + 4].copy().view(np.float32).reshape(-1)
        z         = raw[:, z_off : z_off + 4].copy().view(np.float32).reshape(-1)
        intensity = raw[:, i_off : i_off + 4].copy().view(np.float32).reshape(-1)

        valid = ~(np.isnan(x) | np.isnan(y) | np.isnan(z))
        result = np.stack(
            [x[valid], y[valid], z[valid], intensity[valid]], axis=1
        ).astype(np.float32)
        return result

    except Exception as e:
        print(f"⚠️  pointcloud2_to_array vectorised path failed: {e} — trying fallback")
        try:
            points = []
            for pt in _read_points_slow(
                    cloud_msg,
                    field_names=('x', 'y', 'z', 'intensity'),
                    skip_nans=True):
                points.append(list(pt))
            if not points:
                return np.zeros((0, 4), dtype=np.float32)
            return np.array(points, dtype=np.float32)
        except Exception as e2:
            print(f"❌ pointcloud2_to_array failed completely: {e2}")
            return np.zeros((0, 4), dtype=np.float32)


def _read_points_slow(cloud, field_names=None, skip_nans=False):
    
    point_step = cloud.point_step
    offsets    = {f.name: f.offset for f in cloud.fields}

    if field_names is None:
        field_names = [f.name for f in cloud.fields]

    data = cloud.data
    for i in range(cloud.width * cloud.height):
        point_data = data[i * point_step : (i + 1) * point_step]

        if skip_nans:
            # FIX BUG-5: safe .get() with default offsets 0,4,8
            x_val = struct.unpack_from('f', point_data, offsets.get('x', 0))[0]
            y_val = struct.unpack_from('f', point_data, offsets.get('y', 4))[0]
            z_val = struct.unpack_from('f', point_data, offsets.get('z', 8))[0]
            if np.isnan(x_val) or np.isnan(y_val) or np.isnan(z_val):
                continue

        values = []
        for name in field_names:
            if name in offsets:
                values.append(
                    struct.unpack_from('f', point_data, offsets[name])[0])
            else:
                values.append(0.0)
        yield tuple(values)


def array_to_pointcloud2(points, stamp, frame_id):
    from sensor_msgs.msg import PointCloud2, PointField

    n = points.shape[0]
    cloud_msg                  = PointCloud2()
    cloud_msg.header.stamp     = stamp
    cloud_msg.header.frame_id  = frame_id
    cloud_msg.height           = 1
    cloud_msg.width            = n
    cloud_msg.is_dense         = False
    cloud_msg.is_bigendian     = False

    pts = np.asarray(points, dtype=np.float32)

    if pts.shape[1] == 3:
        cloud_msg.fields = [
            PointField(name='x', offset=0,  datatype=PointField.FLOAT32, count=1),
            PointField(name='y', offset=4,  datatype=PointField.FLOAT32, count=1),
            PointField(name='z', offset=8,  datatype=PointField.FLOAT32, count=1),
        ]
        cloud_msg.point_step = 12
        cloud_msg.row_step   = 12 * n
        dtype = [('x', np.float32), ('y', np.float32), ('z', np.float32)]
        arr = np.zeros(n, dtype=dtype)
        arr['x'] = pts[:, 0]
        arr['y'] = pts[:, 1]
        arr['z'] = pts[:, 2]
        cloud_msg.data = arr.tobytes()

    elif pts.shape[1] == 4:
        cloud_msg.fields = [
            PointField(name='x',         offset=0,  datatype=PointField.FLOAT32, count=1),
            PointField(name='y',         offset=4,  datatype=PointField.FLOAT32, count=1),
            PointField(name='z',         offset=8,  datatype=PointField.FLOAT32, count=1),
            PointField(name='intensity', offset=12, datatype=PointField.FLOAT32, count=1),
        ]
        cloud_msg.point_step = 16
        cloud_msg.row_step   = 16 * n
        dtype = [('x', np.float32), ('y', np.float32),
                 ('z', np.float32), ('intensity', np.float32)]
        arr = np.zeros(n, dtype=dtype)
        arr['x']         = pts[:, 0]
        arr['y']         = pts[:, 1]
        arr['z']         = pts[:, 2]
        arr['intensity'] = pts[:, 3]
        cloud_msg.data = arr.tobytes()

    elif pts.shape[1] == 5:
        cloud_msg.fields = [
            PointField(name='x',         offset=0,  datatype=PointField.FLOAT32, count=1),
            PointField(name='y',         offset=4,  datatype=PointField.FLOAT32, count=1),
            PointField(name='z',         offset=8,  datatype=PointField.FLOAT32, count=1),
            PointField(name='intensity', offset=12, datatype=PointField.FLOAT32, count=1),
            PointField(name='rgb',       offset=16, datatype=PointField.UINT32,  count=1),
        ]
        cloud_msg.point_step = 20
        cloud_msg.row_step   = 20 * n
        dtype = [('x', np.float32), ('y', np.float32), ('z', np.float32),
                 ('intensity', np.float32), ('rgb', np.uint32)]
        arr = np.zeros(n, dtype=dtype)
        arr['x']         = pts[:, 0]
        arr['y']         = pts[:, 1]
        arr['z']         = pts[:, 2]
        arr['intensity'] = pts[:, 3]
        arr['rgb']       = pts[:, 4].astype(np.uint32)   # FIX BUG-7
        cloud_msg.data = arr.tobytes()

    elif pts.shape[1] == 7:
        # Column layout: [x, y, z, intensity, r, g, b]
        cloud_msg.fields = [
            PointField(name='x',         offset=0,  datatype=PointField.FLOAT32, count=1),
            PointField(name='y',         offset=4,  datatype=PointField.FLOAT32, count=1),
            PointField(name='z',         offset=8,  datatype=PointField.FLOAT32, count=1),
            PointField(name='intensity', offset=12, datatype=PointField.FLOAT32, count=1),
            PointField(name='rgb',       offset=16, datatype=PointField.UINT32,  count=1),
        ]
        cloud_msg.point_step = 20
        cloud_msg.row_step   = 20 * n
        r = pts[:, 4].astype(np.uint32) & 0xFF
        g = pts[:, 5].astype(np.uint32) & 0xFF
        b = pts[:, 6].astype(np.uint32) & 0xFF
        rgb_packed = (r << 16) | (g << 8) | b
        dtype = [('x', np.float32), ('y', np.float32), ('z', np.float32),
                 ('intensity', np.float32), ('rgb', np.uint32)]
        arr = np.zeros(n, dtype=dtype)
        arr['x']         = pts[:, 0]
        arr['y']         = pts[:, 1]
        arr['z']         = pts[:, 2]
        arr['intensity'] = pts[:, 3]
        arr['rgb']       = rgb_packed
        cloud_msg.data = arr.tobytes()

    else:
        raise ValueError(
            f"Unsupported point shape: {pts.shape[1]} columns "
            f"(supported: 3, 4, 5, 7)")

    return cloud_msg

def array_to_pointcloud2_rgb_packed(points, colors, stamp, frame_id='map'):
    from sensor_msgs.msg import PointCloud2, PointField
    msg = PointCloud2()
    msg.header.stamp = stamp
    msg.header.frame_id = frame_id
    msg.height = 1
    msg.width = points.shape[0]
    msg.is_dense = True
    msg.is_bigendian = False

    msg.fields = [
        PointField(name='x',   offset=0,  datatype=PointField.FLOAT32, count=1),
        PointField(name='y',   offset=4,  datatype=PointField.FLOAT32, count=1),
        PointField(name='z',   offset=8,  datatype=PointField.FLOAT32, count=1),
        PointField(name='rgb', offset=12, datatype=PointField.UINT32,  count=1),
    ]
    msg.point_step = 16
    msg.row_step   = 16 * points.shape[0]

    colors_safe = np.clip(np.asarray(colors), 0, 255).astype(np.uint32)
    r   = colors_safe[:, 0]
    g   = colors_safe[:, 1]
    b   = colors_safe[:, 2]
    rgb = (r << 16) | (g << 8) | b

    n = points.shape[0]
    packed = np.zeros(n, dtype=[('x',   np.float32),
                                 ('y',   np.float32),
                                 ('z',   np.float32),
                                 ('rgb', np.uint32)])
    packed['x']   = np.asarray(points[:, 0], dtype=np.float32)
    packed['y']   = np.asarray(points[:, 1], dtype=np.float32)
    packed['z']   = np.asarray(points[:, 2], dtype=np.float32)
    packed['rgb'] = rgb

    msg.data = packed.tobytes()
    return msg