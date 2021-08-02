#!/usr/bin/env python

from __future__ import print_function
from moveit_msgs.srv import ApplyPlanningScene

import rospy

def handler(req):
    print("Success %b"%success)
    return success

def service_server():
    rospy.init_node('aaa')
    s = rospy.Service('/apply_planning_scene', ApplyPlanningScene, handler)
    rospy.spin()

if __name__ == "__main__":
    service_server()

def main():
    service_server()
