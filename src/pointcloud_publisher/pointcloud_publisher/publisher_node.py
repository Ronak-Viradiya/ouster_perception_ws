#!/usr/bin/env python3
import os
import yaml
import time

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSDurabilityPolicy, QoSHistoryPolicy
from rosbag2_py import SequentialReader, StorageOptions, ConverterOptions
from rclpy.serialization import deserialize_message
from sensor_msgs.msg import PointCloud2


class BagPlaybackNode(Node):
    def __init__(self):
        super().__init__('bag_playback_node')

        self.declare_parameter('config_file', '')
        config_path = self.get_parameter('config_file').value

        if not config_path:
            config_path = os.path.join(
                os.getenv('HOME', '/home/ronak'),
                'ouster_perception_ws/config/config.yaml'
            )
        self.config_path = config_path

        self.load_config()

        qos = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            durability=QoSDurabilityPolicy.VOLATILE,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=10
        )

        self.publisher = self.create_publisher(PointCloud2, self.topic_name, qos)

        self.timer = self.create_timer(1.0 / self.publish_rate, self.timer_callback)
        self.reader = None
        self.setup_reader()

        self.get_logger().info(f'BagPlaybackNode started – bag: {self.bag_file}, topic: {self.topic_name}')

    def load_config(self):
        try:
            with open(self.config_path, 'r') as f:
                cfg = yaml.safe_load(f) or {}
        except Exception as e:
            self.get_logger().error(f'Failed to load config: {e}')
            raise

        self.bag_file = cfg.get('bag_file', '')
        self.topic_name = cfg.get('topic', '/ouster/points')
        self.publish_rate = float(cfg.get('publish_rate', 10.0))
        self.loop = cfg.get('loop', True)
        self.single_shot = cfg.get('single_shot', False)
        self.frame_id_override = cfg.get('frame_id', '')  

        if not self.bag_file:
            raise ValueError('bag_file not specified in config')

    def setup_reader(self):
        """Open the bag file and prepare reader."""
        storage_options = StorageOptions(uri=self.bag_file, storage_id='sqlite3')
        converter_options = ConverterOptions(
            input_serialization_format='cdr',
            output_serialization_format='cdr'
        )
        self.reader = SequentialReader()
        self.reader.open(storage_options, converter_options)

        type_dict = {t.name: t.type for t in self.reader.get_all_topics_and_types()}
        if self.topic_name not in type_dict:
            self.get_logger().error(f'Topic {self.topic_name} not found in bag. Available: {list(type_dict.keys())}')
            rclpy.shutdown()
        self.get_logger().info(f'Opened bag with topic {self.topic_name}')

    def timer_callback(self):
        if not self.reader.has_next():
            if not self.loop:
                self.get_logger().info('Bag finished – single shot complete')
                rclpy.shutdown()
                return
            self.setup_reader()

        topic, data, timestamp = self.reader.read_next()
        while topic != self.topic_name:
            if not self.reader.has_next():
                if not self.loop:
                    rclpy.shutdown()
                    return
                self.setup_reader()
            topic, data, timestamp = self.reader.read_next()

        msg = deserialize_message(data, PointCloud2)

        if self.frame_id_override:
            msg.header.frame_id = self.frame_id_override

        self.publisher.publish(msg)
        self.get_logger().debug(f'Published message on {self.topic_name}')


def main(args=None):
    rclpy.init(args=args)
    node = BagPlaybackNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()