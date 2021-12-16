#!/usr/bin/env python3.8
'''
    |||
  \ ||||
   \AAA/
     ||
'''
import sys, os, time
from threading import Thread
from copy import deepcopy
import numpy as np
import rospy

from os_and_utils.utils import GlobalPaths

import settings
settings.init()
from os_and_utils.move_lib import MoveData
from gestures_lib import GestureDetection

#import ui_lib as ui
import moveit_lib
from inverse_kinematics.ik_lib import IK_bridge
from os_and_utils.markers_publisher import MarkersPublisher
import os_and_utils.trajectory_action_client
sys.path.append(os.path.join(sys.path[0],'leapmotion'))
from leapmotion.frame_lib import Frame

import matplotlib.pyplot as plt
from os_and_utils.visualizer_lib import VisualizerLib
from std_msgs.msg import Int8, Float64MultiArray, Int32, Bool
from geometry_msgs.msg import Pose, PoseStamped, Point, Quaternion, Vector3
from moveit_msgs.msg import RobotTrajectory
from trajectory_msgs.msg import JointTrajectoryPoint
from control_msgs.msg import FollowJointTrajectoryGoal, JointTolerance
from relaxed_ik.msg import EEPoseGoals, JointAngles
from visualization_msgs.msg import MarkerArray, Marker
from sensor_msgs.msg import JointState

import mirracle_gestures.msg as rosm

# Temporary solution (reading from mirracle_sim pkg)
sys.path.append(settings.paths.home+"/"+settings.paths.ws_folder+'/src/mirracle_sim/src')
from coppelia_sim_ros_lib import CoppeliaROSInterface

class Callbacks():
    @staticmethod
    def ik(data):
        joints = []
        for ang in data.angles:
            joints.append(ang.data)
        md.goal_joints = joints

    @staticmethod
    def coppelia_eef(data):
        global coppelia_eef_pose
        coppelia_eef_pose = data

    @staticmethod
    def camera_angle(data):
        global camera_orientation
        camera_orientation = data

    @staticmethod
    def hand_data(data):
        ''' Hand data received by ROS msg is saved
        '''
        md.frames.append(Frame().import_from_ros(data))

    @staticmethod
    def save_joints(data):
        ''' Saves joint_states to * append array 'settings.joint_in_time' circle buffer.
                                  * latest data 'md.joints', 'md.velocity', 'md.effort'
            Topic used:
                - CoppeliaSim (PyRep) -> topic "/joint_states_coppelia"
                - Other (Gazebo sim / Real) -> topic "/joint_states"

        '''
        md.joint_states.append(data)

        if settings.robot == 'iiwa':
            if data.name[0][1] == '1': # r1 robot
                md.joints = data.position # Position = Angles [rad]
                md.velocity = data.velocity # [rad/s]
                md.effort = data.effort # [Nm]
            ''' Enable second robot arm
            elif data.name[0][1] == '2': # r2 robot
                r2_joint_pos = data.position # Position = Angles [rad]
                r2_joint_vel = data.velocity # [rad/s]
                r2_joint_eff = data.effort # [Nm]
                float(data.header.stamp.to_sec())
            '''
        elif settings.robot == 'panda':
            md.joints = data.position[-7:] # Position = Angles [rad]
            md.velocity = data.velocity[-7:] # [rad/s]
            md.effort = data.effort[-7:] # [Nm]

        else: raise Exception("Wrong robot name!")

    @staticmethod
    def saveOutputPyMC(data):
        settings.gd.r.pymcout = data.data

def sendInputPyMC():
    msg = Float64MultiArray()
    msg.data = md.frames[-1].r.get_learning_data()

    ps.pymc_in_pub.publish(msg)



def main():
    md = MoveData(settings)
    ps = PubsSubs()
    gd = GestureDetection()
    ik_bridge = IK_bridge()

    #settings.mo = mo = moveit_lib.MoveGroupPythonInteface()
    # Don't use any mode for start
    md.Mode = ''
    if rospy.get_param("/mirracle_config/launch_ui", 'false') == "true":
        thread_ui = Thread(target = launch_ui)
        thread_ui.daemon=True
        thread_ui.start()


    ## Check if everything is running
    #while not settings.mo:
    #    time.sleep(2)
    #    print("[WARN*] settings.mo not init!!")
    while not md.goal_pose:
        time.sleep(2)
        print("[WARN*] md.goal_pose not init!!")
    while not md.goal_joints:
        time.sleep(2)
        # Putting additional relaxed_ik transform if this solver is used
        PubsSubs.publish_eef_goal_pose()
        print("[WARN*] md.goal_joints not init!!")
    while not md.joints:
        time.sleep(2)
        print("[WARN*] md.joints not init!!")

    thread_main_manager = Thread(target = main_manager)
    thread_main_manager.daemon=True
    thread_main_manager.start()
    thread_markers = Thread(target = MarkersPublisher.markersThread)
    thread_markers.daemon=True
    thread_markers.start()
    thread_updater = Thread(target = updateValues)
    thread_updater.daemon=True
    thread_updater.start()

    print("[Main] Ready")

def updateValues():
    GEST_DET = rospy.get_param("/mirracle_config/launch_gesture_detection")
    while not rospy.is_shutdown():
        # 1. Publish to ik topic, putting additional relaxed_ik transform if this solver is used
        PubsSubs.publish_eef_goal_pose()
        # 2. Send hand values to PyMC topic
        if md.frames and GEST_DET == "true":
            sendInputPyMC()
        # 3.
        if settings.simulator == 'coppelia': # Coppelia updates eef_pose through callback
            md.eef_pose = coppelia_eef_pose
        else:
            md.eef_pose = ik_bridge.getFKmoveitPose()
        md.goal_pose_array.append(md.eef_pose)
        md.goal_pose_array.append(md.goal_pose)
        # 4. Update camera angle
        if settings.simulator == 'coppelia':
            settings.md.camera_orientation = camera_orientation
        # Sleep
        time.sleep(0.1)


def launch_ui():
    pass
    #ui.main()

def main_manager():
    delay = 3.0
    rate = rospy.Rate(1./delay)
    time_on_one_pose = 0.0
    print("[INFO*] Main manager initialized")

    # first initialize goal
    md.ENV = md.ENV_DAT['above']
    # initialize pose
    pose = Pose()
    pose.orientation = md.ENV_DAT['above']['ori']
    pose.position = Point(0.4,0.,1.0)
    md.goal_pose = deepcopy(pose)

    self.ik_bridge = IK_bridge()

    while not rospy.is_shutdown():
        settings.mo.go_to_joint_state(joints = md.goal_joints)

        if settings.md.Mode == 'live':

            o = md.frames[-1].r.palm_pose().orientation
            if (np.sqrt(o.x**2 + o.y**2 + o.z**2 + o.w**2) - 1 > 0.000001):
                print("[WARN*] Not valid orientation!")
            if md.frames[-1].r.visible:
                ### MODE 1 default
                if settings.md.liveMode == 'default':
                    md.goal_pose = tfm.transformLeapToScene(md.frames[-1].r.palm_pose())

                ### MODE 2 interactive
                elif settings.md.liveMode == 'interactive':
                    md.goal_pose = goal_pose = tfm.transformLeapToScene(md.frames[-1].r.palm_pose())

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
                                if tfm.distancePoses(settings.scene.object_poses[n], goal_pose) < min_dist:
                                    min_dist = tfm.distancePoses(settings.scene.object_poses[n], goal_pose)
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
                                md.goal_pose.position.x = goal_pose.position.x
                            else:
                                md.goal_pose = goal_pose

                            if not settings.gd.r.poses[settings.gd.r.POSES['grab']].toggle:
                                settings.md.STRICT_MODE = False
                                name = settings.md.attached[0]
                                settings.mo.release_object(name=name)
                                n = 0
                                for i in range(0, len(settings.scene.object_poses)):
                                    if settings.scene.object_names[i] == item:
                                        n = i
                                settings.scene.object_poses[n].position.x =md.goal_pose.position.x
                                print("set value of drawer to ", md.goal_pose.position.x)
                        else:
                            md.goal_pose = goal_pose
                    '''

                ### MODE 3 gesture controlled
                elif settings.md.liveMode == 'gesture':
                    md.goal_pose = settings.md.gestures_goal_pose

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
            if settings.leavingAction and time_on_one_pose <= 0.1 and direction == -1 and settings.sp[pp].actions[settings.currentPose] != "":
                if settings.md.attached:
                    settings.mo.release_object(name=settings.sp[pp].actions[settings.currentPose])
                    settings.leavingAction = False
                else:
                    settings.mo.pick_object(name=settings.sp[pp].actions[settings.currentPose])
                    settings.leavingAction = False
            ## Set goal_pose one pose in direction
            md.goal_pose = deepcopy(settings.sp[pp].poses[settings.currentPose+direction])
            ## On new pose target or over time limit
            if time_on_one_pose > 0.1 or tfm.samePoses(md.eef_pose, settings.sp[pp].poses[settings.currentPose+direction]):
                settings.leavingAction = True
                settings.currentPose = settings.currentPose+direction
                ## Attaching/Detaching when moving forward
                if settings.sp[pp].actions[settings.currentPose] != "" and direction == 1:
                    if not settings.md.attached:
                        settings.mo.pick_object(name=settings.sp[pp].actions[settings.currentPose])
                    else:
                        settings.mo.release_object(name=settings.sp[pp].actions[settings.currentPose])
                time_on_one_pose = 0.0
            time_on_one_pose += delay
        rate.sleep()

class PubsSubs():
    def __init__(self):
        # Saving the joint_states
        if settings.simulator == 'coppelia':
            rospy.Subscriber("joint_states_coppelia", JointState, Callbacks.save_joints)
        else:
            rospy.Subscriber("joint_states", JointState, Callbacks.save_joints)
        # Saving relaxedIK output

        rospy.Subscriber('/relaxed_ik/joint_angle_solutions', JointAngles, Callbacks.ik)
        # Goal pose publisher
        self.ee_pose_goals_pub = rospy.Publisher('/ee_pose_goals', Pose, queue_size=5)

        rospy.Subscriber('/hand_frame', rosm.Frame, Callbacks.hand_data)

        if settings.simulator == 'coppelia':
            rospy.Subscriber('/pose_eef', Pose, Callbacks.coppelia_eef)

            rospy.Subscriber('/coppelia/camera_angle', Vector3, Callbacks.camera_angle)

        if rospy.get_param("/mirracle_config/launch_gesture_detection", 'false') == "true":
            rospy.Subscriber('/mirracle_gestures/pymcout', Int8, Callbacks.saveOutputPyMC)
            self.pymc_in_pub = rospy.Publisher('/mirracle_gestures/pymcin', Float64MultiArray, queue_size=5)


    @staticmethod
    def publish_eef_goal_pose():
        ''' Publish goal_pose /relaxed_ik/ee_pose_goals to relaxedIK with its transform
            Publish goal_pose /ee_pose_goals the goal eef pose
        '''
        ps.ee_pose_goals_pub.publish(md.goal_pose)
        settings.mo.ik_node_publish(pose_r = settings.mo.relaxik_t(md.goal_pose))



if __name__ == '__main__':
    main()
    while not rospy.is_shutdown():
        time.sleep(1)
    print('[Main] Interrupted')
    try:
        sys.exit(0)
    except SystemExit:
        os._exit(0)
