#!/usr/bin/env python
from ament_index_python import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource, AnyLaunchDescriptionSource
import os

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='gesture_detector',
            executable='leap',
            name='leap_publisher_node',
            output='screen',
        ),
        Node(
            package='gesture_detector',
            executable='static_detector',
            name='static_detector_node',
            output='screen',
        ),
        Node(
            package='gesture_detector',
            executable='dynamic_detector',
            name='dynamic_detector_node',
            output='screen',
        ),
        Node(
            package='gesture_detector',
            executable='gesture_detect',
            name='gesture_detector_node',
            output='screen',
            parameters=[{'l': 'static+dynamic'}]
        ),
        # IncludeLaunchDescription(
        #     PythonLaunchDescriptionSource([os.path.join(
        #         get_package_share_directory('rosbridge_server'),
        #         'launch',
        #         'rosbridge_websocket_launch.xml'
        #     )])
        # )
    ])




