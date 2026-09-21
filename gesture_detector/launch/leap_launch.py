#!/usr/bin/env python
''' ros2 launch gesture_detector leap_launch.py

Same as `ros2 run gesture_detector leap`, but respawned on crash.
'''
from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        Node(
            package='gesture_detector',
            executable='leap',
            name='leap_publisher_node',
            output='screen',
            # bring the node back if fails (Leap SDK segfaults in its native
            # callback thread, uncatchable from Python)
            respawn=True,
            respawn_delay=1.0,
        ),
    ])
