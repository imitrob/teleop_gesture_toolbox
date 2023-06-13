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
from os_and_utils.visualizer_lib import VisualizerLib
from os_and_utils.transformations import Transformations as tfm
from os_and_utils.utils_ros import samePoses
from os_and_utils.ros_communication_main import ROSComm

# Temporary solution (reading from coppelia_sim_ros_interface pkg)
sys.path.append(settings.paths.home+"/"+settings.paths.ws_folder+'/src/coppelia_sim_ros_interface/src')
from coppelia_sim_ros_client import CoppeliaROSInterface

from geometry_msgs.msg import Pose, Point

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

    roscm = ROSComm()

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
        ml.md.eef_pose_array.append(ml.md.eef_pose)
        ml.md.goal_pose_array.append(ml.md.goal_pose)

        # ~200-400us

        if ml.md.mode == 'live':

            #o = ml.md.frames[-1].r.palm_pose().orientation
            #if (np.sqrt(o.x**2 + o.y**2 + o.z**2 + o.w**2) - 1 > 0.000001):
            #    print("[WARN*] Not valid orientation!")
            if ml.md.r_present():
                ### MODE 1 default
                if ml.md.live_mode == 'Default':
                    ml.md.goal_pose = tfm.transformLeapToScene(ml.md.frames[-1].r.palm_pose(), ml.md.ENV, ml.md.scale)

                ### MODE 2 interactive
                elif ml.md.live_mode == 'Interactive':
                    ml.md.goal_pose = goal_pose = tfm.transformLeapToScene(ml.md.frames[-1].r.palm_pose(), ml.md.ENV, ml.md.scale)

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
            targetPose = int(ml.md.HoldValue / (100/len(sl.paths[pp].poses)))
            if targetPose >= len(sl.paths[pp].poses):
                targetPose = len(sl.paths[pp].poses)-1
            #diff_pose_progress = 100/len(sl.sp[pp].poses)
            if targetPose == ml.md.currentPose:
                time_on_one_pose = 0.0
                continue
            ## 1 - Forward, -1 - Backward
            direction = 1 if targetPose - ml.md.currentPose > 0 else -1
            ## Attaching/Detaching when moving backwards
            if settings.leavingAction and time_on_one_pose <= 0.1 and direction == -1 and sl.paths[pp].actions[ml.md.currentPose] != "":
                if ml.md.attached:
                    ml.md.release_object(name=sl.paths[pp].actions[ml.md.currentPose])
                    ml.md.leavingAction = False
                else:
                    md.md.pick_object(name=sl.paths[pp].actions[ml.md.currentPose])
                    ml.md.leavingAction = False
            ## Set goal_pose one pose in direction
            ml.md.goal_pose = deepcopy(sl.paths[pp].poses[ml.md.currentPose+direction])
            ## On new pose target or over time limit
            if time_on_one_pose > 0.1 or samePoses(ml.md.eef_pose, sl.paths[pp].poses[ml.md.currentPose+direction]):
                ml.md.leavingAction = True
                ml.md.currentPose = ml.md.currentPose+direction
                ## Attaching/Detaching when moving forward
                if sl.paths[pp].actions[ml.md.currentPose] != "" and direction == 1:
                    if not ml.md.attached:
                        ml.md.pick_object(name=sl.paths[pp].actions[ml.md.currentPose])
                    else:
                        ml.md.mo.release_object(name=sl.paths[pp].actions[ml.md.currentPose])
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
