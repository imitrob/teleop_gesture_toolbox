#include <iostream>
#include <cstring>
#include "Leap.h"

#include "ros/ros.h"
//#include "std_msgs/String.h"
//#include "teleop_gesture_toolbox.msg/Frame.h"

#include "hand_classes.h"

//using namespace Leap;

int main(){
//Leap::Vector v1;
Vector_ v2;
v2.x = 3.0;
//Vector_ v1;
/*Vector_ v;
std::cout << "v1 " << v.x << " " << v.y << " " << v.z << std::endl;
v.x = 1.0;
v.y = 2.0;
v.z = 3.0;
std::cout << "v2 " << v.x << " " << v.y << " " << v.z << std::endl;

Bone_ b1;
Finger_ f1;
Hand_ h1;
LeapGestures_ lg1;
Frame_ fr1;
*/
std::cout << "done" << v2.x << std::endl;

return 0;
}
