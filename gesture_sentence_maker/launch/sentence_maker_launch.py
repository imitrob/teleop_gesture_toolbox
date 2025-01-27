#!/usr/bin/env python
from ament_index_python import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import ExecuteProcess

from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch.actions import DeclareLaunchArgument, SetLaunchConfiguration
import os 
import gesture_detector

def generate_launch_description():
    sensor_arg = DeclareLaunchArgument('sensor', default_value='leap', description='Sensor leap or realsense')
    set_sensor_config = SetLaunchConfiguration(
        'sensor', LaunchConfiguration('sensor')
    )
    user_name_arg = DeclareLaunchArgument('user_name', default_value='', description='User name.')


    return LaunchDescription([
        sensor_arg,
        set_sensor_config,
        # ros2 launch gesture_detector gesture_detect_launch.py
        IncludeLaunchDescription( 
            PythonLaunchDescriptionSource(
                os.path.join(
                    gesture_detector.package_path, 
                    'launch', 
                    'gesture_detect_launch.py'
                )
            ),
        ),
        # ros2 run pointing_object_selection selector_node
        Node( 
            package='pointing_object_selection',
            executable='selector_node',
            name='selector_node',
            output='screen',
        ),
        # ros2 run scene_getter scene_marker_pub # for rViz
        Node( 
            package='scene_getter',
            executable='scene_marker_pub',
            name='scene_marker_pub_node',
            output='screen',
        ),
        ## ONLY DEICTIC
        # ros2 run gesture_sentence_maker deictic_processor # Deictic Episode Processor
        # Node( 
        #     package='gesture_sentence_maker',
        #     executable='deictic_processor',
        #     name='deictic_processor_node',
        #     output='screen',
        # ),
        ## ALL
        # ros2 run gesture_sentence_maker sentence_maker # Gesture Sentence Processor
        Node( 
            package='gesture_sentence_maker',
            executable='sentence_maker',
            name='sentence_maker_node',
            output='screen',
        ),
        Node(
            package='scene_getter',
            executable='mocked_scene',
            name='mocked_scene_node',
            output='screen',
        ),
        user_name_arg,
        Node(
            package='gesture_meaning',
            executable='compound_gesture_user_meaning', #'gesture_meaning_service',
            name='gesture_meaning_service_node',
            output='screen',
            parameters=[{
                'user_name': LaunchConfiguration('user_name'),
            }]
        ),
    ])




