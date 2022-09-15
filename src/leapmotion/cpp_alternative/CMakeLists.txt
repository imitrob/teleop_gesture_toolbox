add_library(hand_classes hand_classes.cpp)

## TODO: Change user name as parameter
target_include_directories(hand_classes PUBLIC "/home/<user>/LeapSDK/include")
target_link_directories(hand_classes PUBLIC "/home/<user>/LeapSDK/lib/x64")
target_link_libraries(hand_classes PUBLIC Leap)
