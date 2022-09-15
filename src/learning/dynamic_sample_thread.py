#!/usr/bin/env python3
import sys, os, threading
if os.getcwd()[-4:] == '.ros': # if running from roslaunch
    import rospkg; rospack = rospkg.RosPack(); rospack.list()
    sys.path.append(rospack.get_path('mirracle_gestures')+'/src/')
sys.path.append("..")
import settings; settings.init()

import numpy as np
import rospy

from mirracle_gestures.msg import DetectionSolution, DetectionObservations
from mirracle_gestures.srv import ChangeNetwork, ChangeNetworkResponse
from std_msgs.msg import Int8, Float64MultiArray

from os_and_utils.nnwrapper import NNWrapper

from learning.main_sample_thread import ClassificationSampler
print(f"Launching dynamic sampling thread!")
ClassificationSampler(type='dynamic')


#
