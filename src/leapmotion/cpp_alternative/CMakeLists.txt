add_library(hand_classes hand_classes.cpp)

target_include_directories(hand_classes PUBLIC "/home/pierro/LeapSDK/include")
target_link_directories(hand_classes PUBLIC "/home/pierro/LeapSDK/lib/x64")
target_link_libraries(hand_classes PUBLIC Leap)
