#!/usr/bin/env python2.7

import sys
import os
from threading import Thread
import settings
settings.init()
import leapmotionlistener as lml
import ui_lib as ui
from time import sleep
import moveit_lib
import time
from copy import deepcopy
#import modern_robotics as mr
import numpy as np
import trajectory_action_client
import rospy
import math
from numpy import pi
import random

import matplotlib.pyplot as plt
import visualizer_lib
from std_msgs.msg import Int8, Float64MultiArray, Int32
from geometry_msgs.msg import Pose, PoseStamped, Point, Quaternion
from moveit_msgs.msg import RobotTrajectory
from trajectory_msgs.msg import JointTrajectoryPoint
from control_msgs.msg import FollowJointTrajectoryGoal, JointTolerance
from relaxed_ik.msg import EEPoseGoals, JointAngles
from visualization_msgs.msg import MarkerArray, Marker
from sensor_msgs.msg import JointState


def callbackpymc(data):
    settings.pymcout = data.data

def main():
    settings.mo = mo = moveit_lib.MoveGroupPythonInteface()
    if rospy.get_param("/mirracle_config/launch_ui") == "true":
        thread2 = Thread(target = launch_ui)
        #thread2.daemon=True
        thread2.start()
    if rospy.get_param("/mirracle_config/launch_leap") == "true":
        thread = Thread(target = launch_lml)
        thread.daemon=True
        thread.start()
        while not settings.frames_adv: # wait until receive Leap data
            time.sleep(2)
            print("[WARN*] Leap data not received")


    if rospy.get_param("/mirracle_config/launch_gesture_detection") == "true":
        pymc_out_sub = rospy.Subscriber('/pymcout', Int8, callbackpymc)
        settings.pymc_in_pub = rospy.Publisher('/pymcin', Float64MultiArray, queue_size=5)

    ## Check if everything is running
    while not settings.mo:
        time.sleep(2)
        print("[WARN*] settings.mo not init!!")
    while not settings.goal_pose:
        time.sleep(2)
        print("[WARN*] settings.goal_pose not init!!")
    while not settings.goal_joints:
        time.sleep(2)
        settings.mo.relaxedIK_publish(pose_r = settings.mo.relaxik_t(settings.goal_pose))
        print("[WARN*] settings.goal_joints not init!!")
    while not settings.joints:
        time.sleep(2)
        print("[WARN*] settings.joints not init!!")

    thread3 = Thread(target = main_manager)
    thread3.daemon=True
    thread3.start()
    thread4 = Thread(target = mo.createMarker)
    thread4.daemon=True
    thread4.start()
    thread5 = Thread(target = updateValues)
    thread5.daemon=True
    thread5.start()
    print("[Info*] Main ready")

    settings.md.Mode = ''
    mo.testInit()
    print("[Info*] Tests Done")
    #settings.md.Mode = 'live'
    rospy.spin()

def updateValues():
    GEST_DET = rospy.get_param("/mirracle_config/launch_gesture_detection")
    while True:
        # 1. Publish to relaxedIK topic
        settings.mo.relaxedIK_publish(pose_r = settings.mo.relaxik_t(settings.goal_pose))
        # 2. Send hand values to PyMC topic
        if GEST_DET == "true":
            settings.mo.fillInputPyMC()
        # 3.
        ## When simulator is coppelia settings.eef_pose are updated with topic
        if settings.SIMULATOR_NAME != 'coppelia': # Coppelia updates eef_pose through callback
            settings.eef_pose = settings.mo.fk.getCurrentFK(settings.EEF_NAME).pose_stamped[0].pose
        settings.eef_robot.append(settings.eef_pose)
        settings.eef_goal.append(settings.goal_pose)
        # This is simple check if relaxedIK results and joint states are same
        #if np.sum(np.subtract(settings.goal_joints,settings.joints)) > 0.1:
        #    print("[WARN*] joints not same!")
        # Sleep
        time.sleep(0.1)

def launch_lml():
    lml.main()

def launch_ui():
    ui.main()

def main_manager():
    delay = 0.1
    seq = 0
    enable_plot =True
    time_on_one_pose = 0.0
    mo = settings.mo
    print("[INFO*] Main manager initialized")

    #plt.ion()
    sample_states = [
    #[ 0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0]#,
    mo.get_random_joints(),
    mo.get_random_joints(),
    mo.get_random_joints()
    #mo.get_random_joints()#,
    #[ 0.0, -pi/4, 0.0, -pi/2, 0.0,  pi/3, 0.0]
                    ]

    while True:
        settings.loopn += 1
        time.sleep(delay)
        first_time = (settings.loopn == 1)
        settings.mo.perform_path(joint_states = settings.goal_joints, first_time=first_time)

        '''
        if settings.loopn < 4:
            pass
            #settings.mo.perform_optimized_path(joint_states = sample_states[settings.loopn-1], first_time=first_time)
            mo.callViz()
        else:
            if enable_plot:
                mo.callViz(load_data=True)
                plt.ioff()
                plt.show()
                print("[INFO*] Plot complete")
                enable_plot = False
        '''
        if settings.md.Mode == 'live':
            o = settings.frames_adv[-1].r.pPose.pose.orientation
            if (np.sqrt(o.x**2 + o.y**2 + o.z**2 + o.w**2) - 1 > 0.000001):
                print("[WARN*] Not valid orientation!")
            if settings.frames_adv[-1].r.visible:
                ### MODE 1 default
                if settings.md.liveMode == 'default':
                    settings.goal_pose = settings.mo.transformLeapToScene(settings.frames_adv[-1].r.pPose.pose, normdir=settings.frames_adv[-1].r.pNormDir)

                ### MODE 2 interactive
                elif settings.md.liveMode == 'interactive':
                    goal_pose = settings.mo.transformLeapToScene(settings.frames_adv[-1].r.pPose.pose, normdir=settings.frames_adv[-1].r.pNormDir)
                    z = settings.mo.inSceneObj(goal_pose)
                    if 'drawer' in z: # drawer outside cannot be attached
                        z.remove('drawer')
                    if 'button_out' in z:
                        z.remove('button_out')
                    print("Z", z)
                    if z:
                        min_dist = np.inf
                        min_name = ''

                        for n, mesh_name in enumerate(settings.scene.mesh_names):
                            if mesh_name in z:
                                if settings.mo.distancePoses(settings.scene.mesh_poses[n], goal_pose) < min_dist:
                                    min_dist = settings.mo.distancePoses(settings.scene.mesh_poses[n], goal_pose)
                                    min_name = mesh_name

                        if settings.md.attached is False and settings.gd.r.poses[settings.gd.r.POSES['grab']].toggle:
                            if settings.scene.NAME == 'pushbutton' or settings.scene.NAME == 'pushbutton2':
                                print("Button name: ", min_name, " clicked")
                            else:
                                # TODO: maybe move closer
                                settings.mo.attach_item(item=min_name)
                                settings.md.STRICT_MODE = True

                        settings.md.ACTION = True
                    else:
                        settings.md.ACTION = False
                        if settings.md.STRICT_MODE:
                            if settings.scene.NAME == 'drawer' or settings.scene.NAME == 'drawer2':
                                settings.goal_pose.position.x = goal_pose.position.x
                            else:
                                settings.goal_pose = goal_pose

                            if not settings.gd.r.poses[settings.gd.r.POSES['grab']].toggle:
                                settings.md.STRICT_MODE = False
                                item = settings.md.attached[0]
                                settings.mo.detach_item(item=item)
                                n = 0
                                for i in range(0, len(settings.scene.mesh_poses)):
                                    if settings.scene.mesh_names[i] == item:
                                        n = i
                                settings.scene.mesh_poses[n].position.x =settings.goal_pose.position.x
                                print("set value of drawer to ", settings.goal_pose.position.x)
                        else:
                            settings.goal_pose = goal_pose


                ### MODE 3 gesture controlled
                elif settings.md.liveMode == 'gesture':
                    settings.goal_pose = settings.md.gestures_goal_pose

        if settings.md.Mode == 'play':
            # controls everything:
            #settings.HoldValue
            ## -> Path that will be performed
            pp = settings.md.PickedPath
            ## -> HoldValue (0-100) to targetPose number (0,len(path))
            targetPose = int(settings.HoldValue / (100/len(settings.sp[pp].poses)))
            if targetPose >= len(settings.sp[pp].poses):
                targetPose = len(settings.sp[pp].poses)-1
            #diff_pose_progress = 100/len(settings.sp[pp].poses)
            if targetPose == settings.currentPose:
                time_on_one_pose = 0.0
                continue
            ## 1 - Forward, -1 - Backward
            direction = 1 if targetPose - settings.currentPose > 0 else -1
            ## Attaching/Detaching when moving backwards
            if settings.leavingAction and time_on_one_pose <= 0.1 and direction == -1 and settings.sp[pp].actions[settings.currentPose] is not "":
                if settings.md.attached:
                    settings.mo.detach_item(item=settings.sp[pp].actions[settings.currentPose])
                    settings.leavingAction = False
                else:
                    settings.mo.attach_item(item=settings.sp[pp].actions[settings.currentPose])
                    settings.leavingAction = False
            ## Set goal_pose one pose in direction
            settings.goal_pose = deepcopy(settings.sp[pp].poses[settings.currentPose+direction])
            ## On new pose target or over time limit
            if time_on_one_pose > 0.1 or mo.samePoses(settings.eef_pose, settings.sp[pp].poses[settings.currentPose+direction]):
                 #(mo.fk.getCurrentFK(settings.EEF_NAME).pose_stamped[0].pose.position, settings.sp[pp].poses[settings.currentPose+direction].position)
                settings.leavingAction = True
                settings.currentPose = settings.currentPose+direction
                ## Attaching/Detaching when moving forward
                if settings.sp[pp].actions[settings.currentPose] is not "" and direction == 1:
                    if not settings.md.attached:
                        settings.mo.attach_item(item=settings.sp[pp].actions[settings.currentPose])
                    else:
                        settings.mo.detach_item(item=settings.sp[pp].actions[settings.currentPose])
                time_on_one_pose = 0.0
            time_on_one_pose += delay


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print('[Info*] Interrupted')
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)
