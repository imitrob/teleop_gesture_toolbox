cmake_minimum_required(VERSION 3.5)
project(teleop_gesture_toolbox)

# Find dependencies
find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(moveit_msgs REQUIRED)
find_package(control_msgs REQUIRED)
find_package(rosidl_default_generators REQUIRED)

rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/Bone.msg"
  "msg/DetectionObservations.msg"
  "msg/DetectionSolution.msg"
  "msg/Frame.msg"
  "msg/Hand.msg"
  "msg/LeapGestures.msg"
  "msg/Object.msg"
  "msg/EEPoseGoals.msg"
  "msg/JointAngles.msg"
  "srv/SaveHandRecord.srv"
  "srv/ChangeNetwork.srv"
  DEPENDENCIES geometry_msgs std_msgs sensor_msgs # Add packages that above messages depend on, in this case geometry_msgs for Sphere.msg
)

# Install Python package
ament_python_install_package(teleop_gesture_toolbox)

# Install examples
set(EXAMPLES_DIR teleop_gesture_toolbox)
install(PROGRAMS
    ${EXAMPLES_DIR}/leapmotion/leap.py
    ${EXAMPLES_DIR}/gesture_classification/main_sample_thread.py
    ${EXAMPLES_DIR}/gesture_classification/dynamic_sample_thread.py
    ${EXAMPLES_DIR}/main.py
    DESTINATION lib/${PROJECT_NAME}
)

ament_export_dependencies(rosidl_default_runtime)

# Setup the project
ament_package()
