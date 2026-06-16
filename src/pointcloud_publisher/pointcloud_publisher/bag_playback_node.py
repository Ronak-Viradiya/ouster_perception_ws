#!/usr/bin/env python3
"""
Bag playback node for PointCloud2 messages.
All settings are loaded from a ROS2 parameter file.
Usage:
    ros2 run pointcloud_publisher bag_playback_node --ros-args --params-file config/bag_playback_params.yaml
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSDurabilityPolicy, QoSHistoryPolicy
from rclpy.serialization import deserialize_message
from rosbag2_py import SequentialReader, StorageOptions, ConverterOptions
from sensor_msgs.msg import PointCloud2
import time


class BagPlaybackNode(Node):
    def __init__(self):
        super().__init__('bag_playback_node')

        # Declare parameters with defaults (overridden by YAML)
        self.declare_parameter('bag_file', '')
        self.declare_parameter('topic', '/ouster/points')
        self.declare_parameter('publish_rate', 10.0)   # Hz, 0 = as fast as possible
        self.declare_parameter('loop', False)
        self.declare_parameter('single_shot', False)
        self.declare_parameter('frame_id', 'ouster')

        # Get final values
        bag_file = self.get_parameter('bag_file').get_parameter_value().string_value
        self.topic_name = self.get_parameter('topic').get_parameter_value().string_value
        self.publish_rate = self.get_parameter('publish_rate').get_parameter_value().double_value
        self.loop = self.get_parameter('loop').get_parameter_value().bool_value
        self.single_shot = self.get_parameter('single_shot').get_parameter_value().bool_value
        self.frame_id = self.get_parameter('frame_id').get_parameter_value().string_value

        if not bag_file:
            self.get_logger().error('No bag_file parameter provided! Exiting.')
            raise ValueError('bag_file parameter is required')

        # QoS for point cloud
        qos = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            durability=QoSDurabilityPolicy.VOLATILE,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=10
        )
        self.publisher = self.create_publisher(PointCloud2, self.topic_name, qos)

        # Open bag
        self.reader = SequentialReader()
        storage_options = StorageOptions(uri=bag_file, storage_id='sqlite3')
        converter_options = ConverterOptions(
            input_serialization_format='cdr',
            output_serialization_format='cdr'
        )
        self.reader.open(storage_options, converter_options)

        # Check topic exists
        topic_types = self.reader.get_all_topics_and_types()
        type_dict = {t.name: t.type for t in topic_types}
        if self.topic_name not in type_dict:
            self.get_logger().error(f'Topic "{self.topic_name}" not in bag. Available: {list(type_dict.keys())}')
            raise RuntimeError('Topic missing')

        self.get_logger().info(f'Playing bag: {bag_file}')
        self.get_logger().info(f'  Topic: {self.topic_name}, Rate: {self.publish_rate} Hz, Loop: {self.loop}, SingleShot: {self.single_shot}, Frame: {self.frame_id}')

        # Timer to drive publishing
        timer_period = 1.0 / self.publish_rate if self.publish_rate > 0 else 0.01
        self.timer = self.create_timer(timer_period, self.playback_callback)

    def playback_callback(self):
        if self.single_shot and hasattr(self, '_single_shot_done'):
            return

        while self.reader.has_next():
            topic, data, timestamp = self.reader.read_next()
            if topic != self.topic_name:
                continue

            msg = deserialize_message(data, PointCloud2)
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.header.frame_id = self.frame_id
            self.publisher.publish(msg)

            if self.single_shot:
                self._single_shot_done = True
                self.get_logger().info('Single shot published. Holding cloud.')
                self.destroy_timer(self.timer)

            # Rate limiting (only if rate > 0, we already set timer period,
            # but for fast publishing we want to stay in the loop)
            # We'll just return to let the timer fire again later
            return

        # Bag finished
        if self.loop:
            self.get_logger().info('Looping bag...')
            bag_file = self.get_parameter('bag_file').get_parameter_value().string_value
            storage_options = StorageOptions(uri=bag_file, storage_id='sqlite3')
            self.reader.open(storage_options, ConverterOptions(
                input_serialization_format='cdr', output_serialization_format='cdr'))
        else:
            self.get_logger().info('Bag finished. Shutting down.')
            rclpy.shutdown()


def main(args=None):
    rclpy.init(args=args)
    try:
        node = BagPlaybackNode()
        rclpy.spin(node)
    except Exception as e:
        print(f'Error: {e}')
    finally:
        if rclpy.ok():
            rclpy.shutdown()