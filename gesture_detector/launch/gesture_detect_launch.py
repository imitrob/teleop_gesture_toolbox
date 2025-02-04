#!/usr/bin/env python
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch.actions import ExecuteProcess
import gesture_detector

def generate_nodes(context, *args, **kwargs):
    # Retrieve the value of 'sensor' argument at runtime
    sensor = LaunchConfiguration('sensor').perform(context)

    # Conditional logic for node selection
    if sensor == 'realsense':
        return [Node(
            package='gesture_detector',
            executable='realsense',
            name='realsense_publisher_node',
            output='screen',
        )]
    elif sensor == 'leap':
        return [Node(
            package='gesture_detector',
            executable='leap',
            name='leap_publisher_node',
            output='screen',
        )]
    else:
        raise ValueError(f"Invalid sensor argument: {sensor}. Use 'realsense' or 'leap'.")

def generate_launch_description():
    # Declare the 'sensor' argument
    sensor_arg = DeclareLaunchArgument(
        'sensor',
        default_value='leap',
        description='Choose which sensor node to launch: "realsense" or "leap"'
    )

    rviz_config_file_arg = DeclareLaunchArgument(
        'rviz_config_file',
        default_value=gesture_detector.path+"/live_display/hand_cfg.rviz",
        description='Path to the RViz2 configuration file'
    )

    # Define the Node action to launch RViz2
    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        output='screen',
        arguments=['-d', LaunchConfiguration('rviz_config_file')],
    )

    return LaunchDescription([
        sensor_arg,
        OpaqueFunction(function=generate_nodes),
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
            parameters=[{'model': 'directional_swipes2'}]
        ),
        Node(
            package='gesture_detector',
            executable='gesture_detect',
            name='gesture_detector_node',
            output='screen',
            parameters=[{'l_hand_mode': 'static+dynamic', 'r_hand_mode': 'static+dynamic'}]
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
        Node(
            package='gesture_detector',
            executable='hand_marker_pub',
            name='hand_marker_pub',
            output='screen',
        ),
        Node(
            package='pointing_object_selection',
            executable='tf_a404',
            name='static_tf_a404_node',
            output='screen',
        ),
        rviz_config_file_arg,
        rviz_node
    ])




