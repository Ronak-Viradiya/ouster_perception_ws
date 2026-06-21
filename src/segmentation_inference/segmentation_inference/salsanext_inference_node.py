#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSDurabilityPolicy, QoSHistoryPolicy, QoSReliabilityPolicy
import numpy as np
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn

SRC_DIR = Path(__file__).resolve().parent.parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

WORKSPACE_ROOT = None
current = Path(__file__).resolve()
for parent in [current] + list(current.parents):
    if (parent / 'models' / 'model_scripts').exists() and (parent / 'src').exists():
        WORKSPACE_ROOT = parent
        break

if WORKSPACE_ROOT is None:
    for idx in (4, 5, 6):
        try:
            candidate = current.parents[idx]
            if (candidate / 'models' / 'model_scripts').exists():
                WORKSPACE_ROOT = candidate
                break
        except IndexError:
            pass

if WORKSPACE_ROOT and str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

MODEL_REGISTRY = {}

def register_model(name):
    def decorator(cls):
        MODEL_REGISTRY[name] = cls
        return cls
    return decorator

try:
    from models.model_scripts.salsanext import SalsaNext
    register_model('salsanext')(SalsaNext)
except ImportError:
    pass

try:
    from models.model_scripts.rangenetpp import RangeNetPlusPlus
    register_model('rangenetpp')(RangeNetPlusPlus)
except ImportError:
    pass


def create_model(model_name: str, **kwargs) -> nn.Module:
    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model '{model_name}'. Available: {list(MODEL_REGISTRY.keys())}")
    return MODEL_REGISTRY[model_name](**kwargs)

from lidar_utils.common import load_params, pointcloud2_to_array, array_to_pointcloud2_rgb_packed, point_cloud_to_range_image
from sensor_msgs.msg import PointCloud2, PointField
from visualization_msgs.msg import Marker, MarkerArray
import geometry_msgs.msg


class SalsaNextInference(Node):
    def __init__(self):
        super().__init__('salsanext_inference')

        self.declare_parameter('config_file', 'config/params.yaml')
        config_file = self.get_parameter('config_file').value
        self.params = load_params(config_file)

        sensor_cfg = self.params.get('sensor', {})
        model_cfg = self.params.get('model', {})
        topic_cfg = self.params.get('topics', {})
        paths_cfg = self.params.get('paths', {})

        self.num_classes = model_cfg.get('num_classes', 7)
        names_dict = model_cfg.get('names', {})
        if isinstance(names_dict, dict) and len(names_dict) == self.num_classes:
            self.class_names = [names_dict[i] for i in range(self.num_classes)]
        else:
            self.class_names = [f'class_{i}' for i in range(self.num_classes)]
            self.get_logger().warn('Class names missing/incomplete in config, using defaults')

        color_dict = model_cfg.get('color_map', {})
        if color_dict:
            self.colormap = [color_dict[i] for i in range(self.num_classes)]
        else:
            import random
            random.seed(42)
            self.colormap = [[random.randint(0, 255) for _ in range(3)] for _ in range(self.num_classes)]
        self.get_logger().info(f'Loaded {self.num_classes} classes: {self.class_names}')

        self.max_range = sensor_cfg.get('max_range', 100.0)
        self.min_range = sensor_cfg.get('min_range', 1.0)
        self.fov_up = sensor_cfg.get('fov_up', 45.0)
        self.fov_down = sensor_cfg.get('fov_down', -45.0)
        sensor_type = sensor_cfg.get('sensor_type', 'ouster').lower()
        if sensor_type == 'ouster':
            self.max_intensity = 65535.0
        elif sensor_type == 'velodyne':
            self.max_intensity = 1.0
        else:
            self.max_intensity = 65535.0
            self.get_logger().warn(f'Unknown sensor_type "{sensor_type}", using Ouster intensity norm.')

        self.proj_h = model_cfg.get('height', 128)
        self.proj_w = model_cfg.get('width', 2048)
        self.get_logger().info(f'Projection: {self.proj_h}×{self.proj_w}, FOV {self.fov_down}° to {self.fov_up}°')

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.get_logger().info(f'Device: {self.device}')

        self.model = None
        self.load_model()

        qos = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            durability=QoSDurabilityPolicy.VOLATILE,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=10
        )
        input_topic = topic_cfg.get('input_pointcloud', '/ouster/points')
        self.subscription = self.create_subscription(
            PointCloud2, input_topic, self.pointcloud_callback, qos)

        self.colored_pub = self.create_publisher(
            PointCloud2, topic_cfg.get('output_colored', '/rangenet/colored_cloud'), 10)
        self.label_pub = self.create_publisher(
            PointCloud2, topic_cfg.get('output_labels', '/rangenet/labels'), 10)
        self.marker_pub = self.create_publisher(
            MarkerArray, topic_cfg.get('output_markers', '/rangenet/semantic_markers'), 10)

        self.frame_count = 0
        self.total_time = 0.0
        self.processing = False
        self.last_marker_time = time.time()
        self.marker_interval = 1.0

        self.get_logger().info('✅ SalsaNext inference node ready.')

    def load_model(self):
        model_cfg = self.params.get('model', {})
        model_path = model_cfg.get('checkpoints', {}).get(
            'salsanext',
            'models/models/pretrained_models/THAB_salsanext.pth'
        )
        if not os.path.isabs(model_path):
            # Try to resolve relative to workspace root based on config file location
            config_file = self.get_parameter('config_file').value
            if config_file and os.path.exists(config_file):
                workspace_root = Path(config_file).resolve().parent.parent
                candidate = workspace_root / model_path
                if candidate.exists():
                    model_path = str(candidate)
                else:
                    # Fallback: try common workspace parent paths
                    candidates = [workspace_root, SRC_DIR, Path.cwd()]
                    resolved = Path(__file__).resolve()
                    for idx in (3, 6):
                        try:
                            candidates.append(resolved.parents[idx])
                        except IndexError:
                            pass

                    model_file = None
                    for base in candidates:
                        candidate = base / model_path
                        if candidate.exists():
                            model_file = candidate
                            break

                    if model_file is None:
                        model_file = workspace_root / model_path
                        self.get_logger().warn(
                            f'Model not found under workspace roots, using {model_file} and letting torch raise if missing.'
                        )
                    model_path = str(model_file)
            else:
                model_path = str(SRC_DIR / model_path)

        self.get_logger().info(f'Loading SalsaNext model from {model_path}')

        checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
        arch = checkpoint.get('architecture', 'salsanext')
        if arch != 'salsanext':
            self.get_logger().warn(f"Checkpoint architecture is '{arch}', but forcing to 'salsanext'")
            arch = 'salsanext'

        num_classes_ckpt = checkpoint.get('num_classes', self.num_classes)
        input_channels = checkpoint.get('input_channels', 5)
        height = checkpoint.get('height', 128)
        width = checkpoint.get('width', 2048)

        ckpt_names = checkpoint.get('class_names', None)
        if ckpt_names is not None and len(ckpt_names) == num_classes_ckpt:
            self.class_names = ckpt_names

        self.model = create_model(
            arch,
            num_classes=num_classes_ckpt,
            input_channels=input_channels,
            height=height,
            width=width
        )
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()

        self.num_classes = num_classes_ckpt
        self.get_logger().info(f'Loaded {arch} ({num_classes_ckpt} classes)')
        val_acc = checkpoint.get('val_acc')
        if val_acc is not None:
            self.get_logger().info(f'Validation accuracy: {val_acc:.2f}%')

    def pointcloud_callback(self, msg):
        if self.processing:
            return
        self.processing = True
        t_start = time.time()

        try:
            self.frame_count += 1

            points = pointcloud2_to_array(msg)
            if points.shape[0] == 0:
                return

            range_image_full = point_cloud_to_range_image(
                points,
                height=self.proj_h,
                width=self.proj_w,
                fov_up=self.fov_up,
                fov_down=self.fov_down,
                max_range=self.max_range,
                min_range=self.min_range
            )
            range_img = range_image_full[0]
            x_img = range_image_full[1]
            y_img = range_image_full[2]
            z_img = range_image_full[3]
            intensity_img = range_image_full[4]
            mask = range_img > self.min_range

            x_met = x_img.copy()
            y_met = y_img.copy()
            z_met = z_img.copy()
            i_met = intensity_img.copy()

            range_norm = np.clip(range_img / self.max_range, 0.0, 1.0)
            x_norm = x_img / self.max_range
            y_norm = y_img / self.max_range
            z_norm = z_img / self.max_range
            intensity_norm = np.clip(intensity_img / self.max_intensity, 0.0, 1.0)

            tensor = np.stack([range_norm, x_norm, y_norm, z_norm, intensity_norm], axis=0)
            tensor = torch.from_numpy(tensor).float().unsqueeze(0).to(self.device)

            with torch.no_grad():
                logits = self.model(tensor)
                preds = torch.argmax(logits, dim=1)
                probs = torch.softmax(logits, dim=1)
                conf, _ = torch.max(probs, dim=1)

            labels = preds.squeeze(0).cpu().numpy()
            conf_img = conf.squeeze(0).cpu().numpy()

            if self.frame_count % 10 == 0:
                avg_conf = conf_img[mask].mean() if mask.sum() else 0.0
                self.get_logger().info(f'Frame {self.frame_count}: avg conf {avg_conf:.3f}')

            valid = mask.reshape(-1)
            if valid.any():
                x_flat = x_met.reshape(-1)[valid]
                y_flat = y_met.reshape(-1)[valid]
                z_flat = z_met.reshape(-1)[valid]
                i_flat = i_met.reshape(-1)[valid]
                lbl_flat = labels.reshape(-1)[valid].astype(np.int32)
                lbl_flat = np.clip(lbl_flat, 0, self.num_classes - 1)

                colors = np.array(self.colormap, dtype=np.float32)[lbl_flat]

                colored_cloud = np.zeros((len(x_flat), 7), dtype=np.float32)
                colored_cloud[:, 0] = x_flat
                colored_cloud[:, 1] = y_flat
                colored_cloud[:, 2] = z_flat
                colored_cloud[:, 3] = i_flat
                colored_cloud[:, 4:7] = colors

                colored_msg = array_to_pointcloud2_rgb_packed(
                    points=colored_cloud[:, :3],
                    colors=colored_cloud[:, 4:7],
                    stamp=msg.header.stamp,
                    frame_id=msg.header.frame_id
                )
                self.colored_pub.publish(colored_msg)

                labeled_cloud = np.zeros((len(x_flat), 5), dtype=np.float32)
                labeled_cloud[:, 0] = x_flat
                labeled_cloud[:, 1] = y_flat
                labeled_cloud[:, 2] = z_flat
                labeled_cloud[:, 3] = lbl_flat.astype(np.float32) / max(1, self.num_classes - 1)
                labeled_cloud[:, 4] = lbl_flat.astype(np.float32)
                self.publish_labeled_cloud(labeled_cloud, msg.header.stamp, msg.header.frame_id)

                now = time.time()
                if now - self.last_marker_time >= self.marker_interval:
                    self.publish_markers(colored_cloud, lbl_flat, msg.header.frame_id)
                    self.last_marker_time = now

            dt = time.time() - t_start
            self.total_time += dt
            if self.frame_count % 10 == 0:
                avg = self.total_time / self.frame_count
                self.get_logger().info(f'Inference {dt*1000:.1f} ms, avg {avg*1000:.1f} ms')

        except Exception as e:
            self.get_logger().error(f'Error: {e}')
            import traceback
            traceback.print_exc()
        finally:
            self.processing = False

    def publish_labeled_cloud(self, labeled_cloud, stamp, frame_id):
        msg = PointCloud2()
        msg.header.stamp = stamp
        msg.header.frame_id = frame_id
        msg.height = 1
        msg.width = labeled_cloud.shape[0]
        msg.is_dense = True
        msg.is_bigendian = False
        msg.fields = [
            PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
            PointField(name='intensity', offset=12, datatype=PointField.FLOAT32, count=1),
            PointField(name='class_id', offset=16, datatype=PointField.FLOAT32, count=1),
        ]
        msg.point_step = 20
        msg.row_step = msg.point_step * msg.width
        msg.data = labeled_cloud.astype(np.float32).tobytes()
        self.label_pub.publish(msg)

    def publish_markers(self, colored_cloud, labels, frame_id):
        ma = MarkerArray()
        step = max(1, len(colored_cloud) // 1000)
        for cls_id in range(1, self.num_classes):
            idxs = np.where(labels == cls_id)[0]
            if len(idxs) == 0:
                continue
            sampled = colored_cloud[idxs[::step]]
            m = Marker()
            m.header.frame_id = frame_id
            m.header.stamp = self.get_clock().now().to_msg()
            m.ns = f"sem_{cls_id}"
            m.id = cls_id
            m.type = Marker.POINTS
            m.action = Marker.ADD
            color = self.colormap[cls_id]
            m.color.r = color[0] / 255.0
            m.color.g = color[1] / 255.0
            m.color.b = color[2] / 255.0
            m.color.a = 0.8
            m.scale.x = 0.1
            m.scale.y = 0.1
            m.scale.z = 0.1
            for pt in sampled:
                p = geometry_msgs.msg.Point()
                p.x = float(pt[0])
                p.y = float(pt[1])
                p.z = float(pt[2])
                m.points.append(p)
            ma.markers.append(m)
        self.marker_pub.publish(ma)


def main(args=None):
    rclpy.init(args=args)
    node = SalsaNextInference()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
