#!/usr/bin/env python3
"""
Launch RangeNet++ inference with bag playback and RViz.
Simple command: ros2 launch segmentation_inference rangenetpp.launch.py
"""
import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch_ros.actions import Node
from launch.substitutions import LaunchConfiguration


def generate_launch_description():

    ws_root = os.path.expanduser('~/ouster_perception_ws')
    bag_config = os.path.join(ws_root, 'config', 'bag_record.yaml')
    inference_config = os.path.join(ws_root, 'config', 'params.yaml')
    rviz_config = os.path.join(ws_root, 'src', 'pointcloud_publisher', 'launch', 'rviz.rviz')
    
    bag_config_arg = DeclareLaunchArgument(
        'bag_config', default_value=bag_config,
        description='Bag playback config'
    )
    inference_config_arg = DeclareLaunchArgument(
        'inference_config', default_value=inference_config,
        description='Inference config'
    )
    rviz_config_arg = DeclareLaunchArgument(
        'rviz_config', default_value=rviz_config,
        description='RViz config file'
    )

    # Bag playback node
    bag_node = Node(
        package='pointcloud_publisher',
        executable='publisher_node',           
        name='bag_playback_node',
        output='screen',
        parameters=[{'config_file': LaunchConfiguration('bag_config')}],
    )

    # RangeNet++ inference node
    rangenetpp_node = Node(
        package='segmentation_inference',
        executable='rangenetpp_inference_node',           
        name='rangenetpp_inference',
        output='screen',
        parameters=[{'config_file': LaunchConfiguration('inference_config')}],
    )

    # RViz for visualization
    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        output='screen',
        arguments=['-d', LaunchConfiguration('rviz_config')],
    )

    return LaunchDescription([
        bag_config_arg,
        inference_config_arg,
        rviz_config_arg,
        bag_node,
        rangenetpp_node,
        rviz_node,
    ])
