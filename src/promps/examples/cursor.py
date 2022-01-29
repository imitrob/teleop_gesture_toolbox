if True:
    import sys, os, argparse
    import numpy as np
    import matplotlib.pyplot as plt

    # append utils library relative to this file
    sys.path.append('../os_and_utils')
    # create GlobalPaths() which changes directory path to package src folder
    from utils import GlobalPaths
    GlobalPaths()

    sys.path.append('leapmotion') # .../src/leapmotion
    from loading import HandDataLoader, DatasetLoader

from os_and_utils.transformations import Transformations as tfm


## init Coppelia
from geometry_msgs.msg import Pose, Point, Quaternion
import rospy
import numpy as np

# Import utils.py from src folder
import time
sys.path.append(GlobalPaths().home+"/"+GlobalPaths().ws_folder+'/src/mirracle_sim/src')

from utils import *
from mirracle_sim.srv import AddOrEditObject, AddOrEditObjectResponse, RemoveObject, RemoveObjectResponse, GripperControl, GripperControlResponse
from mirracle_sim.msg import ObjectInfo
from sensor_msgs.msg import JointState, Image

from coppelia_sim_ros_lib import CoppeliaROSInterface

rospy.init_node("coppeliaSimPublisherTopic", anonymous=True)

'''
# Prepare scene
pose = Pose()
pose.position = Point(0., 0.3, 0.0)
pose.orientation = Quaternion(0.7071067811865476, 0.7071067811865476, 0.0, 0.0)
CoppeliaROSInterface.add_or_edit_object(name="object1",pose=pose, shape='sphere', color='r', dynamic='false', size=[0.01,0.01,0.01], collision='false')
#CoppeliaROSInterface.gripper_control(position=0.0)

time.sleep(2.0)

#promp_paths = approach.construct_promp_trajectories2(Xpalm, DXpalm, Ypalm, start='mean')
promp_paths.shape
from copy import deepcopy
promp_paths_scene = deepcopy(promp_paths)
promp_paths_scene = paths_to_scene(promp_paths_scene)


sim = CoppeliaROSInterface()
for path in promp_paths_scene:
    for point in path:

        pose.position = Point(*point)
        sim.add_or_edit_object(name="object1",pose=pose)
        sim.go_to_pose(pose, blocking=False)

        #time.sleep(0.005)
    time.sleep(2)

print("DONE")
'''
# Init functions create global placeholder objects for data
import settings
settings.init()
import os_and_utils.move_lib as ml
ml.init()
import os_and_utils.scenes as sl
sl.init()
import gestures_lib as gl
gl.init()
import ui_lib as ui

def main():
    delay = 1.0
    rate = rospy.Rate(1./delay)

    while not rospy.is_shutdown():
        ml.md.eef_pose_array.append(ml.md.eef_pose)
        ml.md.goal_pose_array.append(ml.md.goal_pose)

        if ml.md.r_present():
            ml.md.goal_pose = tfm.transformLeapToScene(ml.md.frames[-1].r.palm_pose())
            sim.add_or_edit_object(name="object1",pose=ml.md.goal_pose)
            print(f"ml.md.goal_pose {ml.md.goal_pose.position}")
        rate.sleep()

main()











#
