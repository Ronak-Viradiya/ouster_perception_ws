from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, RegisterEventHandler
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch.event_handlers import OnProcessStart
import os
import yaml

def detect_frame_from_bag(params_path, topic_name):
    try:
        from rosbag2_py import SequentialReader, StorageOptions, ConverterOptions
        from rclpy.serialization import deserialize_message
        from sensor_msgs.msg import PointCloud2
    except Exception:
        return None

    try:
        with open(params_path, 'r') as f:
            cfg = yaml.safe_load(f) or {}
        if 'bag_playback_node' in cfg:
            params = cfg.get('bag_playback_node', {}).get('ros__parameters', {}) or {}
        else:
            params = cfg
        bag_file = params.get('bag_file', '')
        topic = params.get('topic', topic_name)
        if not bag_file:
            return None
    except Exception:
        return None

    try:
        reader = SequentialReader()
        storage_options = StorageOptions(uri=bag_file, storage_id='sqlite3')
        converter_options = ConverterOptions(input_serialization_format='cdr', output_serialization_format='cdr')
        reader.open(storage_options, converter_options)
        while reader.has_next():
            tpc, data, _ = reader.read_next()
            if tpc != topic:
                continue
            msg = deserialize_message(data, PointCloud2)
            frame = msg.header.frame_id if hasattr(msg, 'header') else None
            reader.open(storage_options, converter_options)
            return frame
    except Exception:
        return None

def generate_launch_description():
    # Default path to your single config file
    default_config = os.path.join(
        os.getenv('HOME', '/home/ronak'),
        'ouster_perception_ws/config/config.yaml'
    )
    
    rviz_template = os.path.join(
        os.path.dirname(__file__),
        'rviz.rviz'
    )

    detected_frame = detect_frame_from_bag(default_config, '/ouster/points')
    rviz_config = rviz_template
    if detected_frame:
        try:
            with open(rviz_template, 'r') as f:
                content = f.read()
            content = content.replace('Fixed Frame: base_link', f'Fixed Frame: {detected_frame}')
            tmp_path = os.path.join('/tmp', f'rviz_auto_{os.getpid()}.rviz')
            with open(tmp_path, 'w') as f:
                f.write(content)
            rviz_config = tmp_path
        except Exception:
            rviz_config = rviz_template

    try:
        with open(default_config, 'r') as f:
            cfg = yaml.safe_load(f) or {}
        if 'bag_playback_node' in cfg:
            params = cfg.get('bag_playback_node', {}).get('ros__parameters', {}) or {}
        else:
            params = cfg
        bag_file = params.get('bag_file', '')
    except Exception:
        bag_file = ''

    bag_cmd = ['ros2', 'bag', 'play']
    if bag_file:
        bag_cmd.append(bag_file)
    bag_cmd.append('-l')

    bag_proc = ExecuteProcess(
        cmd=bag_cmd,
        output='screen'
    )

    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        output='screen',
        arguments=['-d', rviz_config]
    )

    start_rviz_on_bag = RegisterEventHandler(
        event_handler=OnProcessStart(
            target_action=bag_proc,
            on_start=[rviz_node]
        )
    )

    return LaunchDescription([
        DeclareLaunchArgument(
            'config_file',
            default_value=default_config,
            description='ouster_perception_ws/config/config.yaml'
        ),
        bag_proc,
        start_rviz_on_bag,
    ])
