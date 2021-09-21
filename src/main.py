#!/usr/bin/env python2.7

import sys
import os
from threading import Thread
import time
from copy import deepcopy
import numpy as np
import rospy
from numpy import pi
import random

import settings
settings.init()

import leapmotionlistener as lml
import ui_lib as ui
import moveit_lib
from moveit_lib import IK_bridge
from markers_publisher import MarkersPublisher
import trajectory_action_client
#import modern_robotics as mr

import matplotlib.pyplot as plt
from visualizer_lib import VisualizerLib
from std_msgs.msg import Int8, Float64MultiArray, Int32, Bool
from geometry_msgs.msg import Pose, PoseStamped, Point, Quaternion, Vector3
from moveit_msgs.msg import RobotTrajectory
from trajectory_msgs.msg import JointTrajectoryPoint
from control_msgs.msg import FollowJointTrajectoryGoal, JointTolerance
from relaxed_ik.msg import EEPoseGoals, JointAngles
from visualization_msgs.msg import MarkerArray, Marker
from sensor_msgs.msg import JointState


# Temporary solution (reading from mirracle_sim pkg)
sys.path.append('/home/pierro/my_ws/src/mirracle_sim/src')
from coppelia_sim_ros_lib import CoppeliaROSInterface

def callbackik(data):
    joints = []
    for ang in data.angles:
        joints.append(ang.data)
    settings.goal_joints = joints

coppelia_eef_pose = Pose()
def coppelia_eef_callback(data):
    global coppelia_eef_pose
    coppelia_eef_pose = data

def camera_angle_callback(data):
    global camera_orientation
    camera_orientation = data

def save_joints(data):
    ''' Saves joint_states to * append array 'settings.joint_in_time' circle buffer.
                              * latest data 'settings.joints', 'settings.velocity', 'settings.effort'
        Topic used:
            - CoppeliaSim (PyRep) -> topic "/joint_states_coppelia"
            - Other (Gazebo sim / Real) -> topic "/joint_states"

    '''
    settings.joints_in_time.append(data)

    if settings.ROBOT_NAME == 'iiwa':
        if data.name[0][1] == '1': # r1 robot
            settings.joints = data.position # Position = Angles [rad]
            settings.velocity = data.velocity # [rad/s]
            settings.effort = data.effort # [Nm]
        ''' Enable second robot arm
        elif data.name[0][1] == '2': # r2 robot
            r2_joint_pos = data.position # Position = Angles [rad]
            r2_joint_vel = data.velocity # [rad/s]
            r2_joint_eff = data.effort # [Nm]
            float(data.header.stamp.to_sec())
        '''
    elif settings.ROBOT_NAME == 'panda':
        settings.joints = data.position[-7:] # Position = Angles [rad]
        settings.velocity = data.velocity[-7:] # [rad/s]
        settings.effort = data.effort[-7:] # [Nm]

    else: raise Exception("Wrong robot name!")


def sendInputPyMC():
    dd = settings.frames_adv[-1]
    if settings.observation_type == 'user_defined':
        f = [dd.r.OC[0], dd.r.OC[1], dd.r.OC[2], dd.r.OC[3], dd.r.OC[4], dd.r.TCH12, dd.r.TCH23, dd.r.TCH34, dd.r.TCH45, dd.r.TCH13, dd.r.TCH14, dd.r.TCH15]
    elif settings.observation_type == 'all_defined':
        f = []
        f.extend(dd.r.wrist_hand_angles_diff[1:3])
        f.extend(ext_fingers_angles_diff(dd.r.fingers_angles_diff))
        f.extend(ext_pos_diff_comb(dd.r.pos_diff_comb))
        if settings.position == 'absolute':
            f.extend(dd.r.pRaw)
            if settings.position == 'absolute+finger':
                f.extend(dd.r.index_position)
    settings.pymcin = Float64MultiArray()
    settings.pymcin.data = f
    settings.pymc_in_pub.publish(settings.pymcin)

def saveOutputPyMC(data):
    settings.pymcout = data.data

def publish_eef_goal_pose():
    ''' Publish goal_pose /relaxed_ik/ee_pose_goals to relaxedIK with its transform
        Publish goal_pose /ee_pose_goals the goal eef pose
    '''
    settings.ee_pose_goals_pub.publish(settings.goal_pose)
    settings.mo.ik_node_publish(pose_r = settings.mo.relaxik_t(settings.goal_pose))


def main():
    # Saving the joint_states
    if settings.SIMULATOR_NAME == 'coppelia':
        rospy.Subscriber("joint_states_coppelia", JointState, save_joints)
    else:
        rospy.Subscriber("joint_states", JointState, save_joints)
    # Saving relaxedIK output

    rospy.Subscriber('/relaxed_ik/joint_angle_solutions', JointAngles, callbackik)
    # Goal pose publisher
    settings.ee_pose_goals_pub = rospy.Publisher('/ee_pose_goals', Pose, queue_size=5)


    if settings.SIMULATOR_NAME == 'coppelia':
        rospy.Subscriber('/pose_eef', Pose, coppelia_eef_callback)

        rospy.Subscriber('/coppelia/camera_angle', Vector3, camera_angle_callback)

    settings.mo = mo = moveit_lib.MoveGroupPythonInteface()
    # Don't use any mode for start
    settings.md.Mode = ''
    if rospy.get_param("/mirracle_config/launch_ui") == "true":
        thread_ui = Thread(target = launch_ui)
        thread_ui.daemon=True
        thread_ui.start()
    if rospy.get_param("/mirracle_config/launch_leap") == "true":
        thread_leap = Thread(target = launch_lml)
        thread_leap.daemon=True
        thread_leap.start()
    if rospy.get_param("/mirracle_config/launch_gesture_detection") == "true":
        rospy.Subscriber('/mirracle_gestures/pymcout', Int8, saveOutputPyMC)
        settings.pymc_in_pub = rospy.Publisher('/mirracle_gestures/pymcin', Float64MultiArray, queue_size=5)

    ## Check if everything is running
    while not settings.mo:
        time.sleep(2)
        print("[WARN*] settings.mo not init!!")
    while not settings.goal_pose:
        time.sleep(2)
        print("[WARN*] settings.goal_pose not init!!")
    while not settings.goal_joints:
        time.sleep(2)
        # Putting additional relaxed_ik transform if this solver is used
        publish_eef_goal_pose()
        print("[WARN*] settings.goal_joints not init!!")
    while not settings.joints:
        time.sleep(2)
        print("[WARN*] settings.joints not init!!")

    thread_main_manager = Thread(target = main_manager)
    thread_main_manager.daemon=True
    thread_main_manager.start()
    thread_markers = Thread(target = MarkersPublisher.markersThread)
    thread_markers.daemon=True
    thread_markers.start()
    thread_updater = Thread(target = updateValues)
    thread_updater.daemon=True
    thread_updater.start()

    if settings.VIS_ON == 'true':
        settings.viz = VisualizerLib()
    print("[Info*] Main ready")

def updateValues():
    GEST_DET = rospy.get_param("/mirracle_config/launch_gesture_detection")
    while not rospy.is_shutdown():
        # 1. Publish to ik topic, putting additional relaxed_ik transform if this solver is used
        publish_eef_goal_pose()
        # 2. Send hand values to PyMC topic
        if settings.frames_adv and GEST_DET == "true":
            sendInputPyMC()
        # 3.
        if settings.SIMULATOR_NAME == 'coppelia': # Coppelia updates eef_pose through callback
            settings.eef_pose = coppelia_eef_pose
        else:
            settings.eef_pose = settings.mo.ik_bridge.getFKmoveitPose()
        settings.eef_pose_array.append(settings.eef_pose)
        settings.goal_pose_array.append(settings.goal_pose)
        # This is simple check if relaxedIK results and joint states are same
        #if np.sum(np.subtract(settings.goal_joints,settings.joints)) > 0.1:
        #    print("[WARN*] joints not same!")
        # 4. Update camera angle
        if settings.SIMULATOR_NAME == 'coppelia':
            settings.md.camera_orientation = camera_orientation
        # Sleep
        time.sleep(0.1)

def launch_lml():
    lml.main()

def launch_ui():
    ui.main()

def main_manager():
    delay = 0.1
    seq = 0
    enable_plot = True
    time_on_one_pose = 0.0
    mo = settings.mo
    print("[INFO*] Main manager initialized")

    while not rospy.is_shutdown():
        settings.loopn += 1
        time.sleep(delay)
        settings.mo.go_to_joint_state(joints = settings.goal_joints)

        if settings.VIS_ON == 'true':
            if settings.loopn < 5:
                mo.plotJointsCallViz()
            else:
                if enable_plot:
                    mo.plotJointsCallViz(load_data=True)
                    settings.viz.show()
                    print("[INFO*] Plot complete")
                    enable_plot = False

        if settings.md.Mode == 'live':
            if not settings.frames_adv:
                print("[WARN*] Leap Motion frames not published")
                continue
            o = settings.frames_adv[-1].r.pPose.pose.orientation
            if (np.sqrt(o.x**2 + o.y**2 + o.z**2 + o.w**2) - 1 > 0.000001):
                print("[WARN*] Not valid orientation!")
            if settings.frames_adv[-1].r.visible:
                ### MODE 1 default
                if settings.md.liveMode == 'default':
                    settings.goal_pose = settings.mo.transformLeapToScene(settings.frames_adv[-1].r.pPose.pose, normdir=settings.frames_adv[-1].r.pNormDir)

                ### MODE 2 interactive
                elif settings.md.liveMode == 'interactive':
                    settings.goal_pose = goal_pose = settings.mo.transformLeapToScene(settings.frames_adv[-1].r.pPose.pose, normdir=settings.frames_adv[-1].r.pNormDir)

                    # 1. Gesture output
                    gripper_position_ = settings.gd.r.poses[settings.gd.r.POSES["grab"]].prob
                    gripper_position_ = 1.-gripper_position_
                    # 2. the gripper control is running
                    if settings.loopn%5==0:
                        action = ''
                        if gripper_position_ < 0.1:
                            action = 'grasp'
                        if gripper_position_ > 0.9:
                            action = 'release'
                        CoppeliaROSInterface.gripper_control(gripper_position_, effort=0.04, action=action, object='box')

                    '''
                    z = settings.mo.inSceneObj(goal_pose)
                    if 'drawer' in z: # drawer outside cannot be attached
                        z.remove('drawer')
                    if 'button_out' in z:
                        z.remove('button_out')
                    print("Z", z)
                    if z:
                        min_dist = np.inf
                        min_name = ''

                        for n, mesh_name in enumerate(settings.scene.object_names):
                            if mesh_name in z:
                                if settings.mo.distancePoses(settings.scene.object_poses[n], goal_pose) < min_dist:
                                    min_dist = settings.mo.distancePoses(settings.scene.object_poses[n], goal_pose)
                                    min_name = mesh_name

                        if settings.md.attached is False and settings.gd.r.poses[settings.gd.r.POSES['grab']].toggle:
                            if settings.scene.NAME == 'pushbutton' or settings.scene.NAME == 'pushbutton2':
                                print("Button name: ", min_name, " clicked")
                            else:
                                # TODO: maybe move closer
                                settings.mo.pick_object(name=min_name)
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
                                name = settings.md.attached[0]
                                settings.mo.release_object(name=name)
                                n = 0
                                for i in range(0, len(settings.scene.object_poses)):
                                    if settings.scene.object_names[i] == item:
                                        n = i
                                settings.scene.object_poses[n].position.x =settings.goal_pose.position.x
                                print("set value of drawer to ", settings.goal_pose.position.x)
                        else:
                            settings.goal_pose = goal_pose
                    '''

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
                    settings.mo.release_object(name=settings.sp[pp].actions[settings.currentPose])
                    settings.leavingAction = False
                else:
                    settings.mo.pick_object(name=settings.sp[pp].actions[settings.currentPose])
                    settings.leavingAction = False
            ## Set goal_pose one pose in direction
            settings.goal_pose = deepcopy(settings.sp[pp].poses[settings.currentPose+direction])
            ## On new pose target or over time limit
            if time_on_one_pose > 0.1 or mo.samePoses(settings.eef_pose, settings.sp[pp].poses[settings.currentPose+direction]):
                settings.leavingAction = True
                settings.currentPose = settings.currentPose+direction
                ## Attaching/Detaching when moving forward
                if settings.sp[pp].actions[settings.currentPose] is not "" and direction == 1:
                    if not settings.md.attached:
                        settings.mo.pick_object(name=settings.sp[pp].actions[settings.currentPose])
                    else:
                        settings.mo.release_object(name=settings.sp[pp].actions[settings.currentPose])
                time_on_one_pose = 0.0
            time_on_one_pose += delay


if __name__ == '__main__':
    main()
    while not rospy.is_shutdown():
        time.sleep(1)
    print('[Info*] Interrupted')
    try:
        sys.exit(0)
    except SystemExit:
        os._exit(0)
