#!/usr/bin/env python3
import os
import sys
import argparse
from pathlib import Path
import datetime
from rclpy.serialization import deserialize_message

from lidar_utils.common import save_bin_file, pointcloud2_to_array

import rclpy
from rclpy.node import Node
from rosbag2_py import SequentialReader, StorageOptions, ConverterOptions
from sensor_msgs.msg import PointCloud2
import numpy as np

class BagExtractor(Node):
    def __init__(self, bag_file, output_dir, topic_name="/ouster/points",
                 min_range=1.0, max_range=50.0, intensity_max=65535.0):
        super().__init__('bag_extractor')
        self.bag_file = bag_file
        self.output_dir = output_dir
        self.topic_name = topic_name
        self.min_range = min_range
        self.max_range = max_range
        self.intensity_max = intensity_max

        os.makedirs(output_dir, exist_ok=True)

        self.count = 0  # counter for saved files

        print(f"📦 Bag file: {bag_file}")
        print(f"📁 Output directory: {output_dir}")
        print(f"📡 Topic: {topic_name}")
        print(f"🔧 Range filter: {min_range}m – {max_range}m")
        print(f"🔧 Intensity normalization: / {intensity_max}")

    def extract(self):
        storage_options = StorageOptions(
            uri=self.bag_file,
            storage_id='sqlite3'  # typical ROS2 bag storage
        )
        converter_options = ConverterOptions(
            input_serialization_format='cdr',
            output_serialization_format='cdr'
        )

        reader = SequentialReader()
        reader.open(storage_options, converter_options)

        topic_types = reader.get_all_topics_and_types()
        type_dict = {topic.name: topic.type for topic in topic_types}

        if self.topic_name not in type_dict:
            print(f"❌ Topic {self.topic_name} not found in bag!")
            print(f"Available topics: {list(type_dict.keys())}")
            return

        print(f"✅ Found topic: {self.topic_name} ({type_dict[self.topic_name]})")
        print("🔄 Extracting point clouds...")

        while reader.has_next():
            topic, data, timestamp = reader.read_next()
            if topic != self.topic_name:
                continue

            msg = deserialize_message(data, PointCloud2)

            points = pointcloud2_to_array(msg)

            if points is None or len(points) == 0:
                continue

            points = points[~np.isnan(points).any(axis=1)]
            if len(points) == 0:
                continue

            dist = np.linalg.norm(points[:, :3], axis=1)
            points = points[(dist > self.min_range) & (dist < self.max_range)]
            if len(points) == 0:
                continue

            points[:, 3] = points[:, 3] / self.intensity_max
            # Optional: clip to [0,1] after division
            points[:, 3] = np.clip(points[:, 3], 0.0, 1.0)

            filename = os.path.join(self.output_dir, f"{self.count:06d}.bin")
            save_bin_file(points, filename)
            self.count += 1

            if self.count % 100 == 0:
                print(f"  Extracted {self.count} point clouds...")

        print(f"\n✅ Extraction complete! Saved {self.count} point clouds to {self.output_dir}")

def main():
    parser = argparse.ArgumentParser(description='Extract point clouds from ROS2 bag')
    parser.add_argument('--bag', required=True, help='Path to ROS2 bag file')
    parser.add_argument('--output', required=True, help='Output directory for .bin files')
    parser.add_argument('--topic', default='/ouster/points', help='Point cloud topic name')
    parser.add_argument('--min_range', type=float, default=1.0, help='Minimum range in meters (filter out closer points)')
    parser.add_argument('--max_range', type=float, default=50.0, help='Maximum range in meters (filter out farther points)')
    parser.add_argument('--intensity_max', type=float, default=65535.0, help='Maximum intensity value for normalization (e.g., 65535 for 16-bit)')
    args = parser.parse_args()

    bag_name = Path(args.bag).stem
    timestamp_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(args.output, f"{bag_name}_{timestamp_str}")
    os.makedirs(output_dir, exist_ok=True)
    print(f"📂 Saving .bin files in: {output_dir}")

    rclpy.init()

    extractor = BagExtractor(
        bag_file=args.bag,
        output_dir=output_dir,
        topic_name=args.topic,
        min_range=args.min_range,
        max_range=args.max_range,
        intensity_max=args.intensity_max
    )

    extractor.extract()
    extractor.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()