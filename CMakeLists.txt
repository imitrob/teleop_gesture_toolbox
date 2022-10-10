cmake_minimum_required(VERSION 3.0.2)
project(teleop_gesture_toolbox)

find_package(catkin REQUIRED COMPONENTS
  roscpp
  rospy
  std_msgs
  moveit_visual_tools
  message_generation
  actionlib_msgs
  geometry_msgs
  sensor_msgs
)

## Generate messages in the 'msg' folder
 add_message_files(
   FILES
   Bone.msg
   Hand.msg
   Frame.msg
   LeapGestures.msg
   DetectionObservations.msg
   DetectionSolution.msg
   Object.msg
 )

## Generate services in the 'srv' folder
add_service_files (
  FILES
   ChangeNetwork.srv
	 SaveHandRecord.srv
   FindObjects.srv
   GetSyncedImages.srv
)

include($ENV{ROS_ROOT}/core/rosbuild/rosbuild.cmake)

## Generate actions in the 'action' folder
add_action_files(
 FILES
 ActuateGripper.action
 MoveToNamedPose.action
 MoveToPose.action
)

## Generate added messages and services with any dependencies listed here
generate_messages(
  DEPENDENCIES
  std_msgs
  geometry_msgs
  actionlib_msgs
  sensor_msgs
)

## C++ libs & executables

catkin_package(
  CATKIN_DEPENDS message_runtime
  )
