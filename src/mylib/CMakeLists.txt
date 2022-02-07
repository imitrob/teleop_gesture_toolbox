add_library(mylib mylib.cpp)

## TODO: Change user to parameter
target_include_directories(mylib PUBLIC "/home/<user>/LeapSDK/include")
target_link_directories(mylib PUBLIC "/home/<user>/LeapSDK/lib/x64")
target_link_libraries(mylib PUBLIC Leap)
