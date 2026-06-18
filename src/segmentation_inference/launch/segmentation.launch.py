#!/usr/bin/env python3
import os
from datetime import datetime
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess
from launch_ros.actions import Node
from launch.substitutions import LaunchConfiguration


def generate_launch_description():

    ws_root = os.path.expanduser('~/ouster_perception_ws')
    bag_config = os.path.join(ws_root, 'config', 'bag_record.yaml')
    inference_config = os.path.join(ws_root, 'config', 'params.yaml')
    rviz_config = os.path.join(ws_root, 'src', 'pointcloud_publisher', 'launch', 'rviz.rviz')
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    bag_record_path = os.path.join(ws_root, 'rosbags', f'recorded_session_{timestamp}')

    bag_config_arg = DeclareLaunchArgument(
        'bag_config', default_value=bag_config,
        description='config/bag_record.yaml'
    )
    inference_config_arg = DeclareLaunchArgument(
        'inference_config', default_value=inference_config,
        description='config/params.yaml'
    )
    rviz_config_arg = DeclareLaunchArgument(
        'rviz_config', default_value=rviz_config,
        description='RViz config file'
    )
    bag_record_arg = DeclareLaunchArgument(
        'bag_record_path', default_value=bag_record_path,
        description='Output directory for ros2 bag record'
    )

    bag_node = Node(
        package='pointcloud_publisher',
        executable='publisher_node',           
        name='bag_playback_node',
        output='screen',
        parameters=[{'config_file': LaunchConfiguration('bag_config')}],
    )

    inference_node = Node(
        package='segmentation_inference',
        executable='inference_node',           
        name='rangenet_inference',
        output='screen',
        parameters=[{'config_file': LaunchConfiguration('inference_config')}],
    )

    bag_record = ExecuteProcess(
        cmd=[
            'ros2', 'bag', 'record',
            '-o', LaunchConfiguration('bag_record_path'),
            '/ouster/points',
            '/rangenet/colored_cloud',
            '/rangenet/labels',
        ],
        output='screen',
    )

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
        bag_record_arg,
        bag_node,
        inference_node,
        bag_record,
        rviz_node,
    ])