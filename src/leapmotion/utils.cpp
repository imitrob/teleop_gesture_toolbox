
#include <iostream>


#include "ros/ros.h"
#include "std_msgs/String.h"

bool ROS_ENABLED(){

#ifdef _GLIBCXX_ROSCPP
  std::cout << "leap is here" << std::endl;
  return true;
#else
  std::cout << "leap is not here" << std::endl;
  return false;
#endif
}

int main(){

  ROS_ENABLED();
  return 0;
}
