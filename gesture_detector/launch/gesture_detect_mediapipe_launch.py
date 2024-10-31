#!/usr/bin/env python
from ament_index_python import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import ExecuteProcess
import gesture_detector

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='mediapipe_ros_pkg',
            executable='mediapipe_toolbox_node',
            name='mediapipe_publisher_node',
            output='screen',
        ),
        Node(
            package='gesture_detector',
            executable='custom_detector', # static detector
            name='static_detector_node',
            output='screen',
            parameters=[{'model': 'common_gestures'}]
        ),
        Node(
            package='gesture_detector',
            executable='custom_detector', # dynamic detector
            name='dynamic_detector_node',
            output='screen',
            parameters=[{'model': 'directional_swipes'}]
        ),
        Node(
            package='gesture_detector',
            executable='gesture_detect',
            name='gesture_detector_node',
            output='screen',
            parameters=[{'l': 'static+dynamic', 'r': 'static+dynamic'}]
        ),
        Node(
            package='rosbridge_server',
            executable='rosbridge_websocket',
            name='rosbridge_server_node',
            output='screen',
        ),
        ExecuteProcess(
            cmd=['python', '-m', 'http.server', '--directory', gesture_detector.path+"/live_display", '8000'],
            output='screen',
            shell=True
        ),
    ])




