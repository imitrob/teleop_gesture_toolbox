#!/usr/bin/env python2.7

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
    if rospy.get_param("/mirracle_config/launch_leap") == "true":
        thread = Thread(target = launch_lml)
        thread.daemon=True
        thread.start()
        while not settings.frames_adv: # wait until receive Leap data
            pass
    if rospy.get_param("/mirracle_config/launch_ui") == "true":
        thread2 = Thread(target = launch_ui)
        thread2.daemon=True
        thread2.start()

    settings.mo = mo = moveit_lib.MoveGroupPythonInteface()

    if rospy.get_param("/mirracle_config/launch_gesture_detection") == "true":
        pymc_out_sub = rospy.Subscriber('/pymcout', Int8, callbackpymc)
        settings.pymc_in_pub = rospy.Publisher('/pymcin', Float64MultiArray, queue_size=5)

    ## DEPRECATED will be deleted
    # Coppelia fakce FCI controller
    if settings.ROBOT_NAME == 'panda' and settings.SIMULATOR_NAME == 'coppelia':
        settings.coppeliaFakeFCIpub = rospy.Publisher('/fakeFCI/joint_state', JointState, queue_size=5)
        coppeliaFakeFCIpubState = rospy.Publisher('/fakeFCI/robot_state', Int32, queue_size=5)
        msg = Int32()
        msg.data = 1
        coppeliaFakeFCIpubState.publish(msg)
        print("Coppelia Fake publisher init")
    ## ########### ##

    ## Check if everything is running
    while not settings.mo:
        time.sleep(5)
        print("[WARN*] settings.mo not init!!")
    while not settings.goal_pose:
        time.sleep(5)
        print("[WARN*] settings.goal_pose not init!!")
    while not settings.goal_joints:
        time.sleep(5)
        settings.mo.relaxedIK_publish(pose_r = settings.mo.relaxik_t(settings.goal_pose))
        print("[WARN*] settings.goal_joints not init!!")
    while not settings.joints:
        time.sleep(5)
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
    print("Done")

    settings.md.Mode = ''
    mo.testInit()
    print("Tests Done")
    settings.md.Mode = 'live'
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
        '''
        DEPRECATED -> will be deteleted
        else:
            pose = Pose()
            pose.position = Point(*moveit_lib.panda_forward_kinematics(settings.joints))

            settings.eef_pose = pose
        '''
        settings.eef_robot.append(settings.eef_pose)
        settings.eef_goal.append(settings.goal_pose)

        # Sleep
        time.sleep(0.1)

def launch_lml():
    lml.main()

def launch_ui():
    ui.main()

def main_manager():
    delay = 0.1
    seq = 0
    time_on_one_pose = 0.0
    mo = settings.mo
    first_time = True
    loopn = 0

    plt.ion()
    settings.fig, settings.ax = visualizer_lib.visualize_new_fig(title="Path", dim=2)
    #visualizer_lib.visualize_3d(settings.eef_goal, storeObj=settings, color='b', label="leap", units='m')
    #data = [settings.mo.extv(pose.position) for pose in list(settings.eef_robot)]
    #visualizer_lib.visualize_3d(data=data, storeObj=settings, color='r', label="robot", units='m')
    #data = [settings.mo.extv(settings.mo.transformLeapToScene(settings.frames_adv[i].r.pPose.pose).position) for i in range(0, settings.BUFFER_LEN)]
    #visualizer_lib.visualize_3d(data=data, storeObj=settings, color='b', label="leap", units='m')
    #plt.ioff()
    #plt.show()
    print("[INFO*] Main manager initialized")
    hmm =True
    while True:
        loopn += 1
        ## This is simple check if relaxedIK results and joint states are same
        #if np.sum(np.subtract(settings.goal_joints,settings.joints)) > 0.1:
        #    print("[WARN*] joints not same!")

        time.sleep(delay)
        ## deprecated --> will be deleted
        ## If relaxed ik output exists, execute it
        if settings.SIMULATOR_NAME == 'coppelia':
            msg = JointState()
            msg.header.frame_id = 'panda_link0'
            msg.header.stamp = rospy.Time.now()
            msg.header.seq = seq = seq+1
            msg.name = settings.JOINT_NAMES
            msg.position = settings.goal_joints
            #settings.coppeliaFakeFCIpub.publish(msg)
        else:
            if loopn < 5:
                settings.mo.perform_optimized_path(joint_states = settings.goal_joints, first_time=first_time)
                if first_time:
                    first_time = False
            else:
                if hmm:
                    plt.ioff()
                    plt.show()
                    hmm = False
        if settings.md.Mode == 'live':
            o = settings.frames_adv[-1].r.pPose.pose.orientation
            if (np.sqrt(o.x**2 + o.y**2 + o.z**2 + o.w**2) - 1 > 0.000001):
                print("not valid orientation!")
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
        print('Interrupted')
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)
