#!/usr/bin/env python3.7
import sys
import os
sys.path.append(os.path.expanduser('~/PyRep'))

import settings
settings.init(minimal=True)

from os.path import dirname, join, abspath
from pyrep import PyRep
from pyrep.robots.arms.panda import Panda
from pyrep.objects.shape import Shape
from pyrep.const import PrimitiveShape
from pyrep.errors import ConfigurationPathError
import numpy as np
import math
import time

from threading import Thread

# ROS imports
from std_msgs.msg import Int8, Float64MultiArray, Header
from relaxed_ik.msg import EEPoseGoals, JointAngles
from sensor_msgs.msg import JointState
from geometry_msgs.msg import Pose, Point, Quaternion
import rospy
import time

joints = []

def callback(data):
    ''' When received joints, publish to coppeliaSim
    '''
    global joints
    j = []
    for ang in data.angles:
        j.append(ang.data)
    joints = j

def main():
    if settings.GRIPPER_NAME == 'none':
        SCENE_FILE = settings.COPPELIA_SCENE_PATH+'/'+'scene_panda.ttt'
    elif settings.GRIPPER_NAME == 'franka_hand':
        SCENE_FILE = settings.COPPELIA_SCENE_PATH+'/'+'scene_panda_franka_gripper.ttt'
    elif settings.GRIPPER_NAME == 'franka_hand_with_camera':
        SCENE_FILE = settings.COPPELIA_SCENE_PATH+'/'+'scene_panda_custom_gripper.ttt'
    else: raise Exception("Wrong selected gripper, (probably in demo.launch file)")

    global pr
    pr.launch(SCENE_FILE, headless=False)
    pr.start()
    agent = Panda()
    while joints == []:
        pass
    while True:
        # publish to coppelia
        agent.set_joint_target_positions(joints)
        pr.step()
        # delay
        time.sleep(0.1)
        # publish eef
        eef_msg = Pose()
        eef_msg.position = Point(*agent.get_tip().get_position())
        ## temporary correction -> coppelia environment table-> will be changed
        eef_msg.position.x -= 0.0
        eef_msg.position.y -= 0.0
        eef_msg.position.z -= 0.0
        eef_msg.orientation = Quaternion(*agent.get_tip().get_quaternion())
        eef_pub.publish(eef_msg)
        # publish joint_states
        joint_state_msg = JointState()
        joint_state_msg.position = agent.get_joint_positions()
        joint_state_msg.header = Header()
        joint_state_msg.header.stamp = rospy.Time.now()
        joint_states_pub.publish(joint_state_msg)

rospy.init_node('coppeliaPublisher', anonymous=True)
rospy.Subscriber('/relaxed_ik/joint_angle_solutions', JointAngles, callback)
joint_states_pub = rospy.Publisher('/joint_states_coppelia', JointState, queue_size=5)
eef_pub = rospy.Publisher('/pose_eef', Pose, queue_size=5)

pr = PyRep()
thread = Thread(target = main)
thread.daemon=True
thread.start()

input()
print("Ending in 5s")
pr.stop()
pr.shutdown()
