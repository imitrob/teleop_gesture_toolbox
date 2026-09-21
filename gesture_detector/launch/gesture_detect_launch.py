#!/usr/bin/env python
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration, PythonExpression
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch.actions import ExecuteProcess, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
import os
import gesture_detector

def generate_nodes(context, *args, **kwargs):
    # Retrieve the value of 'sensor' argument at runtime
    input_source = LaunchConfiguration('sensor').perform(context)

    # Conditional logic for node selection
    if input_source == 'realsense':
        return [Node(
            package='gesture_detector',
            executable='realsense',
            name='realsense_publisher_node',
            output='screen',
        )]
    elif input_source == 'leap':
        return [IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                os.path.join(gesture_detector.package_path, 'launch', 'leap_launch.py')
            ),
        )]
    elif input_source == 'bag':
        return []
    else:
        raise ValueError(
            f"Invalid sensor argument: {input_source}. "
            "Use 'realsense', 'leap', or 'bag'.")

def generate_launch_description():
    # Declare the 'sensor' argument
    sensor_arg = DeclareLaunchArgument(
        'sensor',
        default_value='leap',
        description='Choose an input source: "realsense", "leap", or "bag"'
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
            parameters=[{'model': 'directional_swipes'}]
        ),
        Node(
            package='gesture_detector',
            executable='gesture_detect',
            name='gesture_detector_node',
            output='screen',
            parameters=[{
                'l_hand_mode': 'static+dynamic',
                'r_hand_mode': 'static+dynamic',
                'replay_mode': ParameterValue(PythonExpression([
                    "'", LaunchConfiguration('sensor'), "' == 'bag'"
                ]), value_type=bool),
            }]
        ),
        Node(
            package='rosbridge_server',
            executable='rosbridge_websocket',
            name='rosbridge_server_node',
            output='screen',
        ),
        ExecuteProcess(
            cmd=['python', '-m', 'http.server', '--directory', gesture_detector.path+"/live_display", '6357'],
            output='screen',
            shell=True
        ),
        # Standalone full-screen hand scene viewer. The dashboard's embedded. "Hand Scene" card is served by 6357 above and does not need this.
        ExecuteProcess(
            cmd=['python3', gesture_detector.path+"/live_display/scene_viewer_test/server.py",
                 '--port', '6358'],
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
        # Note: rViz replaced by visualization in browser 
        # rviz_config_file_arg,
        # rviz_node
    ])


