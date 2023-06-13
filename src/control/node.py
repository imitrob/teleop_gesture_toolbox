#!/usr/bin/env python3.8

import time
import argparse
from ruckig_controller import JointController

import numpy as np

import rospy

if __name__ == '__main__':
    rospy.init_node("/teleop_gesture_toolbox/controller")
    rospy.Subscriber("/hand_frame", Float64MultiArray, callback)
    rospy.spin()

def callback(msg):
    self.j.tac_control_add_new_goal(msg)
