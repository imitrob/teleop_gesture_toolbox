cmake_minimum_required(VERSION 3.0.2)
project(mirracle_gestures)

#set(CMAKE_CXX_COMPILER "/home/pierro/anaconda3/envs/robostackenv/bin/x86_64-conda-linux-gnu-g++")

find_package(catkin REQUIRED COMPONENTS
  roscpp
  rospy
  std_msgs
  moveit_visual_tools
  message_generation
)

## Generate messages in the 'msg' folder
 add_message_files(
   FILES
   Bone.msg
   Hand.msg
   Frame.msg
   LeapGestures.msg
 )

## Generate services in the 'srv' folder
add_service_files (
  FILES
   ChangeNetwork.srv
#  AddOrEditObject.srv
#  RemoveObject.srv
#  GripperControl.srv
)

include($ENV{ROS_ROOT}/core/rosbuild/rosbuild.cmake)

## Generate actions in the 'action' folder
# add_action_files(
#   FILES
#   Action1.action
#   Action2.action
# )

## Generate added messages and services with any dependencies listed here
generate_messages(
  DEPENDENCIES
  std_msgs
  geometry_msgs
)

## C++ libs & executables

catkin_package()


# 1.
#add_subdirectory(src/leapmotion)
# 2. Main
#add_executable(sample src/leapmotion/Sample.cpp)
# 3. Add Leap
#target_include_directories(sample PUBLIC "/home/pierro/LeapSDK/include")
#target_link_directories(sample PUBLIC "/home/pierro/LeapSDK/lib/x64")
#target_link_libraries(sample PUBLIC hand_classes Leap)

# 4. Target leapmotion dir
#target_include_directories(sample PUBLIC
#                          "${PROJECT_BINARY_DIR}"
#                          "${PROJECT_SOURCE_DIR}/src/leapmotion"
#                          )



#
