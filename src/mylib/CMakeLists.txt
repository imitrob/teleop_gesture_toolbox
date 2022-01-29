add_library(mylib mylib.cpp)

target_include_directories(mylib PUBLIC "/home/pierro/LeapSDK/include")
target_link_directories(mylib PUBLIC "/home/pierro/LeapSDK/lib/x64")
target_link_libraries(mylib PUBLIC Leap)
