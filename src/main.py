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

from std_msgs.msg import Int8, Float64MultiArray
from geometry_msgs.msg import Pose, PoseStamped, Point, Quaternion
from moveit_msgs.msg import RobotTrajectory
from trajectory_msgs.msg import JointTrajectoryPoint
from control_msgs.msg import FollowJointTrajectoryGoal, JointTolerance
from relaxed_ik.msg import EEPoseGoals, JointAngles
from visualization_msgs.msg import MarkerArray, Marker

#import roslib; roslib.load_manifest('visualization_marker_tutorials')

def callbackik(data):
    #if settings.goal_joints and not settings.mo.sameJoints(settings.mo.move_group.get_current_joint_values(), data.angles.data):
    joints = []
    for ang in data.angles:
        joints.append(ang.data)
    settings.goal_joints = joints

def callbackpymc(data):
    settings.pymcout = data.data

def main():
    thread = Thread(target = launch_lml)
    thread.start()
    thread2 = Thread(target = launch_ui)
    thread2.start()
    settings.mo = mo = moveit_lib.MoveGroupPythonInteface()
    ## Saving solutions
    goal_pos_sub = rospy.Subscriber('/relaxed_ik/joint_angle_solutions', JointAngles, callbackik)
    pymc_out_sub = rospy.Subscriber('/pymcout', Int8, callbackpymc)
    settings.pymc_in_pub = rospy.Publisher('/pymcin', Float64MultiArray, queue_size=5)
    ## Giving input goals
    thread3 = Thread(target = main_manager)
    thread3.start()
    thread4 = Thread(target = mo.createMarker)
    thread4.start()
    print("Done")




def testing_comp_torq():
    thetalist = np.array([0.5, 0.5, 0.0, 0.0, 0.0, 0.0])
    dthetalist = np.array([0.7, 0.7, 0.0, 0.0, 0.0, 0.0])

    thetalistd = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    dthetalistd = np.array([0.2, 0.2, 0.0, 0.0, 0.0, 0.0])
    ddthetalistd = np.array([0.1, 0.1, 0.0, 0.0, 0.0, 0.0])

    #mr.ComputedTorque(thetalist, dthetalist, settings.eint, settings.g, settings.Mlist, settings.Glist, settings.Slist, thetalistd, \
    # dthetalistd, ddthetalistd, settings.Kp, settings.Ki, settings.Kd)


def testing_toppra_remap():
    ''' Temporary testing function
    '''
    traj = RobotTrajectory()
    traj.joint_trajectory.header.frame_id = "world"
    traj.joint_trajectory.joint_names = ['r1_joint_1', 'r1_joint_2', 'r1_joint_3', 'r1_joint_4', 'r1_joint_5', 'r1_joint_6', 'r1_joint_7']

    point = JointTrajectoryPoint()
    point.positions = [-0.08054438377481699, -0.8232029298738625, 0.2525762172959391, -1.6485245928061523, 0.24475797684814748, -0.8565435756854809, -0.255304308507704]
    point.velocities = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    point.accelerations = [-0.13352455972148886, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    point.time_from_start.nsecs = 196648417
    traj.joint_trajectory.points.append(deepcopy(point))
    point = JointTrajectoryPoint()
    point.positions = [-0.08312612117897764, -0.8046237528537469, 0.2547171135137392, -1.6481982145495813, 0.2390041623802832, -0.8741749544202149, -0.2505752735931462]
    point.velocities = [-0.023081741211931704, 0.16321707977384275, 0.01903472294690833, 0.006600325818079781, -0.050592072613806154, -0.15196519829749505, 0.04079158687804039]
    point.accelerations = [-0.14312864627969457, 0.9884779944513767, 0.1171685238410183, 0.07104805357967676, -0.3067721321144603, -0.8959815980363419, 0.24077681040969648]
    point.time_from_start.nsecs = 278156605
    traj.joint_trajectory.points.append(deepcopy(point))
    point = JointTrajectoryPoint()
    point.positions = [-0.08581872672559407, -0.7857175209529341, 0.25693271169288157, -1.6472575328531098, 0.2331417066023944, -0.8916397953223283, -0.24588569981336367]
    point.velocities = [-0.05130508998361209, 0.3318508133989117, 0.041751230415190965, 0.03269097130475298, -0.0958998983550059, -0.29134203380641954, 0.07428227791014161]
    point.accelerations = [-0.1777071876536269, 1.004080884998614, 0.1401162552359974, 0.18451998489668228, -0.252551834603743, -0.8011980155127468, 0.18378046564131745]
    point.time_from_start.nsecs = 341147205
    traj.joint_trajectory.points.append(deepcopy(point))
    #print("hmm", mo.move_group.execute(traj, wait=True))
    new_traj = mo.retime(traj)
    print("new traj", new_traj)



def launch_lml():
    lml.main()

def launch_ui():
    ui.main()

def main_manager():
    delay = 0.1
    time_on_one_pose = 0.0
    while not settings.frames_adv:
        pass
    while not settings.mo:
        pass
    mo = settings.mo

    # demo (for testing)
    while True:
        if settings.frames_adv[-1].r.visible:
            settings.goal_pose = settings.mo.transformLeapToScene(settings.frames_adv[-1].r.pPose.pose, normdir=settings.frames_adv[-1].r.pNormDir)
            mo.go_to_pose_goal(pose = settings.goal_pose)
        time.sleep(delay)


    while True:
        ## Send hand values to PyMC topic
        mo.fillInputPyMC()

        ## Update forward kinematics in settings
        #settings.rd.eef_pose = mo.fk.getCurrentFK(settings.EEF_NAME).pose_stamped[0].pose

        settings.md.LeapInRviz = settings.mo.transformLeapToScene(settings.frames_adv[-1].r.pPose.pose, normdir=settings.frames_adv[-1].r.pNormDir)
        time.sleep(delay)

        ## Send pose to relaxed ik topic
        if settings.goal_pose:
            settings.mo.relaxedIK_publish(pose_r = settings.goal_pose)
        ## If relaxed ik output exists, execute it
        if settings.goal_joints:
            settings.mo.perform_optimized_path(joint_states = settings.goal_joints)

        if settings.md.Mode == 'live':
            o = settings.frames_adv[-1].r.pPose.pose.orientation
            if (np.sqrt(o.x**2 + o.y**2 + o.z**2 + o.w**2) - 1 > 0.000001):
                print("not valid orientation!")
            if settings.frames_adv[-1].r.visible:
                ### MODE 1 default
                if settings.md.liveMode == 'default':
                    settings.goal_pose = settings.mo.relaxik_t(settings.mo.transformLeapToScene(settings.frames_adv[-1].r.pPose.pose, normdir=settings.frames_adv[-1].r.pNormDir))

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
                                settings.goal_pose.position.x = settings.mo.relaxik_t(goal_pose).position.x
                            else:
                                settings.goal_pose = settings.mo.relaxik_t(goal_pose)

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
                            settings.goal_pose = settings.mo.relaxik_t(goal_pose)


                ### MODE 3 gesture controlled
                elif settings.md.liveMode == 'gesture':
                    settings.goal_pose = settings.mo.relaxik_t(settings.md.gestures_goal_pose)

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
            if time_on_one_pose > 5 or mo.samePoses(mo.relaxik_t(settings.rd.eef_pose), settings.sp[pp].poses[settings.currentPose+direction]):
                 #(mo.relaxik_t(mo.fk.getCurrentFK(settings.EEF_NAME).pose_stamped[0].pose).position, settings.sp[pp].poses[settings.currentPose+direction].position)
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



        if settings.md.Mode == 'nothin':
            pass


if __name__ == '__main__':
  main()
