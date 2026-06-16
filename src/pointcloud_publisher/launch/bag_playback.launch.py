from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
import os

def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument(
            'params_file',
            default_value=os.path.join(
                os.getenv('OUSTER_WS', '/home/ronak/ouster_perception_ws'),
                'config', 'bag_playback_params.yaml'
            ),
            description='Path to the bag playback parameters file'
        ),
        Node(
            package='pointcloud_publisher',
            executable='bag_playback_node',
            name='bag_playback_node',
            output='screen',
            parameters=[LaunchConfiguration('params_file')]
        )
    ])