cmake_minimum_required(VERSION 3.5)
project(gesture_msgs)

# Default to C99
if(NOT CMAKE_C_STANDARD)
  set(CMAKE_C_STANDARD 99)
endif()

# Default to C++14
if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 14)
endif()

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
# uncomment the following section in order to fill in
# further dependencies manually.
find_package(std_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(builtin_interfaces REQUIRED)
find_package(rosidl_default_generators REQUIRED)

rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/Bone.msg"
  "msg/Hand.msg"
  "msg/Frame.msg"
  "msg/Gestures.msg"
  "msg/DetectionObservations.msg"
  "msg/DetectionSolution.msg"
  "msg/DeicticSolution.msg"
  "msg/LeapGestures.msg"
  "msg/Intent.msg"
  "srv/SaveHandRecord.srv"
  "srv/GetModelConfig.srv"
  "srv/GestureToMeaning.srv"
  
  DEPENDENCIES builtin_interfaces std_msgs sensor_msgs geometry_msgs
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
