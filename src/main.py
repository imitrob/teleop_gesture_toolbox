#!/usr/bin/env python3.8
'''
    |||
  \ ||||
   \|||/
     ||
'''
import sys, os, time
from threading import Thread
from copy import deepcopy
import numpy as np
import rospy

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

from inverse_kinematics.ik_lib import IK_bridge
from leapmotion.frame_lib import Frame
from os_and_utils.visualizer_lib import VisualizerLib
from os_and_utils.transformations import Transformations as tfm
from os_and_utils.utils_ros import samePoses

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

class ROSComm():
    ''' ROS communication of main thread: Subscribers (init & callbacks) and Publishers
    '''
    def __init__(self, ik_bridge):
        # Saving the joint_states
        if settings.simulator == 'coppelia':
            rospy.Subscriber("joint_states_coppelia", JointState, self.joint_states)

            rospy.Subscriber('/pose_eef', Pose, self.coppelia_eef)
            rospy.Subscriber('/coppelia/camera_angle', Vector3, self.camera_angle)
        else:
            rospy.Subscriber("joint_states", JointState, self.joint_states)

        # Saving relaxedIK output
        rospy.Subscriber('/relaxed_ik/joint_angle_solutions', JointAngles, self.ik)
        # Goal pose publisher
        self.ee_pose_goals_pub = rospy.Publisher('/ee_pose_goals', Pose, queue_size=5)

        rospy.Subscriber('/hand_frame', rosm.Frame, self.hand_frame)

        if settings.launch_gesture_detection == "true":
            rospy.Subscriber('/mirracle_gestures/pymcout', Int8, self.saveOutputPyMC)
            self.pymc_in_pub = rospy.Publisher('/mirracle_gestures/pymcin', Float64MultiArray, queue_size=5)

        self.controller = rospy.Publisher('/mirracle_gestures/target', Float64MultiArray, queue_size=5)
        self.ik_bridge = ik_bridge

    def publish_eef_goal_pose(self, goal_pose):
        ''' Publish goal_pose /relaxed_ik/ee_pose_goals to relaxedIK with its transform
            Publish goal_pose /ee_pose_goals the goal eef pose
        '''
        self.ee_pose_goals_pub.publish(goal_pose)
        self.ik_bridge.relaxedik.ik_node_publish(pose_r = self.ik_bridge.relaxedik.relaxik_t(goal_pose))


    @staticmethod
    def ik(data):
        joints = []
        for ang in data.angles:
            joints.append(ang.data)
        ml.md.goal_joints = joints

    @staticmethod
    def coppelia_eef(data):
        ml.md.eef_pose = data

    @staticmethod
    def camera_angle(data):
        ml.md.camera_orientation = data

    @staticmethod
    def hand_frame(data):
        ''' Hand data received by ROS msg is saved
        '''
        f = Frame()
        f.import_from_ros(data)
        ml.md.frames.append(f)

    @staticmethod
    def joint_states(data):
        ''' Saves joint_states to * append array 'settings.joint_in_time' circle buffer.
                                  * latest data 'ml.md.joints', 'ml.md.velocity', 'ml.md.effort'
            Topic used:
                - CoppeliaSim (PyRep) -> topic "/joint_states_coppelia"
                - Other (Gazebo sim / Real) -> topic "/joint_states"

        '''
        ml.md.joint_states.append(data)

        if settings.robot == 'iiwa':
            if data.name[0][1] == '1': # r1 robot
                ml.md.joints = data.position # Position = Angles [rad]
                ml.md.velocity = data.velocity # [rad/s]
                ml.md.effort = data.effort # [Nm]
            ''' Enable second robot arm
            elif data.name[0][1] == '2': # r2 robot
                r2_joint_pos = data.position # Position = Angles [rad]
                r2_joint_vel = data.velocity # [rad/s]
                r2_joint_eff = data.effort # [Nm]
                float(data.header.stamp.to_sec())
            '''
        elif settings.robot == 'panda':
            ml.md.joints = data.position[-7:] # Position = Angles [rad]
            ml.md.velocity = data.velocity[-7:] # [rad/s]
            ml.md.effort = data.effort[-7:] # [Nm]

        else: raise Exception("Wrong robot name!")

    @staticmethod
    def saveOutputPyMC(data):
        settings.gd.r.pymcout = data.data

    def sendInputPyMC(self):
        if not ml.md.r_present():
            return

        msg = Float64MultiArray()
        msg.data = ml.md.frames[-1].r.get_learning_data()

        self.pymc_in_pub.publish(msg)




def main():
    rospy.init_node('main_manager', anonymous=True)

    thread_main = Thread(target = main_manager)
    thread_main.daemon=True
    thread_main.start()

    if settings.launch_ui == "true":
        app = ui.QApplication(sys.argv)
        ex = ui.Example()
        sys.exit(app.exec_())

def main_manager():
    ## Check if everything is running
    #while not ml.md.goal_pose:
    #    time.sleep(2)
    #    print("[WARN*] ml.md.goal_pose not init!!")
    #while not ml.md.goal_joints:
    #    time.sleep(2)
    #    # Putting additional relaxed_ik transform if this solver is used
    #    roscm.publish_eef_goal_pose(ml.md.goal_pose)
    #    print("[WARN*] ml.md.goal_joints not init!!")
    #while not ml.md.joints:
    #    time.sleep(2)
    #    print("[WARN*] ml.md.joints not init!!")

    ik_bridge = IK_bridge()
    roscm = ROSComm(ik_bridge)

    delay = 1.0
    rate = rospy.Rate(1./delay)
    time_on_one_pose = 0.0

    # first initialize goal
    ml.md.ENV = ml.md.ENV_DAT['above']
    # initialize pose
    pose = Pose()
    pose.orientation = ml.md.ENV_DAT['above']['ori']
    pose.position = Point(0.4,0.,1.0)
    ml.md.goal_pose = deepcopy(pose)

    print("[Main] Main manager initialized")
    while not rospy.is_shutdown():
        if ml.md.r_present():
            print(f"fps {ml.md.frames[-1].fps}, id {ml.md.frames[-1].seq}")

        # 1. Publish to ik topic, putting additional relaxed_ik transform if this solver is used
        roscm.publish_eef_goal_pose(ml.md.goal_pose)
        # 2. Send hand values to PyMC topic
        if ml.md.frames and settings.gesture_detection_on == "true":
            roscm.sendInputPyMC()
        # 3.
        if settings.simulator == 'coppelia': # Coppelia updates eef_pose through callback
            pass
        else:
            ml.md.eef_pose = ik_bridge.getFKmoveitPose()
        ml.md.goal_pose_array.append(ml.md.eef_pose)
        ml.md.goal_pose_array.append(ml.md.goal_pose)

        # ~200-400us

        if ml.md.mode == 'live':

            #o = ml.md.frames[-1].r.palm_pose().orientation
            #if (np.sqrt(o.x**2 + o.y**2 + o.z**2 + o.w**2) - 1 > 0.000001):
            #    print("[WARN*] Not valid orientation!")
            if ml.md.r_present():
                ### MODE 1 default
                if ml.md.live_mode == 'default':
                    ml.md.goal_pose = tfm.transformLeapToScene(ml.md.frames[-1].r.palm_pose())

                ### MODE 2 interactive
                elif ml.md.live_mode == 'interactive':
                    ml.md.goal_pose = goal_pose = tfm.transformLeapToScene(ml.md.frames[-1].r.palm_pose())

                    # 1. Gesture output
                    gripper_position_ = 0.0
                    if hasattr(gl.gd.r.static, 'grab'): gripper_position_ = gl.gd.r.static.grab.prob
                    gripper_position_ = 1.-gripper_position_
                    # 2. the gripper control is running

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

                        if ml.md.attached is False and settings.gd.r.static.grab.toggle:
                            if settings.scene.NAME == 'pushbutton' or settings.scene.NAME == 'pushbutton2':
                                print("Button name: ", min_name, " clicked")
                            else:
                                # TODO: maybe move closer
                                settings.mo.pick_object(name=min_name)
                                ml.md.strict_mode = True

                        ml.md.action = True
                    else:
                        ml.md.action = False
                        if ml.md.strict_mode:
                            if settings.scene.NAME == 'drawer' or settings.scene.NAME == 'drawer2':
                                ml.md.goal_pose.position.x = goal_pose.position.x
                            else:
                                ml.md.goal_pose = goal_pose

                            if not settings.gd.r.static.grab.toggle:
                                ml.md.strict_mode = False
                                name = ml.md.attached[0]
                                settings.mo.release_object(name=name)
                                n = 0
                                for i in range(0, len(settings.scene.object_poses)):
                                    if settings.scene.object_names[i] == item:
                                        n = i
                                settings.scene.object_poses[n].position.x =ml.md.goal_pose.position.x
                                print("set value of drawer to ", ml.md.goal_pose.position.x)
                        else:
                            ml.md.goal_pose = goal_pose
                    '''

                ### MODE 3 gesture controlled
                elif ml.md.liveMode == 'gesture':
                    ml.md.goal_pose = ml.md.gestures_goal_pose

        if ml.md.mode == 'play':
            # controls everything:
            #settings.HoldValue
            ## -> Path that will be performed
            pp = ml.md.picked_path
            ## -> HoldValue (0-100) to targetPose number (0,len(path))
            targetPose = int(settings.HoldValue / (100/len(sl.paths[pp].poses)))
            if targetPose >= len(sl.paths[pp].poses):
                targetPose = len(sl.paths[pp].poses)-1
            #diff_pose_progress = 100/len(sl.sp[pp].poses)
            if targetPose == settings.currentPose:
                time_on_one_pose = 0.0
                continue
            ## 1 - Forward, -1 - Backward
            direction = 1 if targetPose - settings.currentPose > 0 else -1
            ## Attaching/Detaching when moving backwards
            if settings.leavingAction and time_on_one_pose <= 0.1 and direction == -1 and sl.paths[pp].actions[settings.currentPose] != "":
                if ml.md.attached:
                    settings.mo.release_object(name=sl.paths[pp].actions[settings.currentPose])
                    settings.leavingAction = False
                else:
                    settings.mo.pick_object(name=sl.paths[pp].actions[settings.currentPose])
                    settings.leavingAction = False
            ## Set goal_pose one pose in direction
            ml.md.goal_pose = deepcopy(sl.paths[pp].poses[settings.currentPose+direction])
            ## On new pose target or over time limit
            if time_on_one_pose > 0.1 or samePoses(ml.md.eef_pose, sl.paths[pp].poses[settings.currentPose+direction]):
                settings.leavingAction = True
                settings.currentPose = settings.currentPose+direction
                ## Attaching/Detaching when moving forward
                if sl.paths[pp].actions[settings.currentPose] != "" and direction == 1:
                    if not ml.md.attached:
                        settings.mo.pick_object(name=sl.paths[pp].actions[settings.currentPose])
                    else:
                        settings.mo.release_object(name=sl.paths[pp].actions[settings.currentPose])
                time_on_one_pose = 0.0
            time_on_one_pose += delay

        rate.sleep()


if __name__ == '__main__':
    main()
    while not rospy.is_shutdown():
        time.sleep(1)
    print('[Main] Interrupted')
    try:
        sys.exit(0)
    except SystemExit:
        os._exit(0)
