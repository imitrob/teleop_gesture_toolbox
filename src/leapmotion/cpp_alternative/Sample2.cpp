
__asm__(".symver memcpy,memcpy@GLIBC_2.14");
#include <iostream>

#include "ros/ros.h"

#include "Leap.h"
#include "mylib.h"


int main(){
  MyClass v1;
  Leap::Vector v2;
  //v1.x = 1.0;
  std::cout << "SOMETHING!!!" << std::endl;
  return 0;
}
