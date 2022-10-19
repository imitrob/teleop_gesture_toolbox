#!/usr/bin/env python3
import sys, os, threading

# for ros2 run, which has current working directory in ~/<colcon_ws>
sys.path.append("install/teleop_gesture_toolbox/lib/python3.9/site-packages/teleop_gesture_toolbox")
if os.getcwd()[-4:] == '.ros': # if running from roslaunch
    import rospkg; rospack = rospkg.RosPack(); rospack.list()
    sys.path.append(rospack.get_path('teleop_gesture_toolbox')+'/src/')
sys.path.append("..")
import os_and_utils.settings as settings; settings.init()


import numpy as np
import rclpy

from teleop_gesture_toolbox.msg import DetectionSolution, DetectionObservations
from teleop_gesture_toolbox.srv import ChangeNetwork
from std_msgs.msg import Int8, Float64MultiArray

from os_and_utils.nnwrapper import NNWrapper

from gesture_classification.main_sample_thread import ClassificationSampler
print(f"Launching dynamic sampling thread!")
rclpy.init()
dynamic_classification_node = ClassificationSampler(type='dynamic')
rclpy.spin(dynamic_classification_node)


#
