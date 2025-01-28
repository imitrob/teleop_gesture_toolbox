#!/usr/bin/env python
from ament_index_python import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import ExecuteProcess

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='scene_getter',
            executable='scene_marker_pub',
            name='scene_marker_pub_node',
            output='screen',
        ),
        Node(
            package='pointing_object_selection',
            executable='selector_node',
            name='selector_node_node',
            output='screen',
        ),
        Node(
            package='gesture_sentence_maker',
            executable='deictic_processor',
            name='deictic_processor_node',
            output='screen',
        ),
    ])




