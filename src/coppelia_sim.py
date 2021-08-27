#!/usr/bin/env python3.7
'''
Coppelia publisher. It communicates with CoppeliaSim via PyRep.

Loads Scene from dir var 'COPPELIA_SCENE_PATH' specified in 'settings.py'.
Three scenes are created based on ROSparam 'mirracle_config/gripper':
    - 'none' gripper -> 'scene_panda.ttt' loaded
    - 'franka_hand' gripper -> 'scene_panda_franka_gripper2.ttt' loaded
    - 'franka_hand_with_camera' gripper -> 'scene_panda_custom_gripper.ttt'
Uses Joint states controller.
Inverse Kinematics based on ROSparam 'mirracle_config/ik_solver' as:
    - 'relaxed_ik' -> RelaxedIK, computed in separate node, here receiving '/relaxed_ik/joint_angle_solutions'
    - 'pyrep' -> Uses PyRep IK, computed here, receiving '/relaxed_ik/ee_pose_goals', publishing to '/relaxed_ik/joint_angle_solutions'
        * TODO: In this option change the topic names as it is not /relaxed_ik in this option to not be misleading

Note: Install PyRep dir is set (line below) to ~/PyRep as written in README.md
'''
import sys
import os
sys.path.append(os.path.expanduser('~/PyRep'))

import settings
settings.init(minimal=True)

from os.path import dirname, join, abspath
from pyrep import PyRep
from pyrep.robots.arms.panda import Panda
from pyrep.robots.end_effectors.panda_gripper import PandaGripper
from pyrep.objects.shape import Shape
from pyrep.const import PrimitiveShape
from pyrep.errors import ConfigurationPathError, IKError, ConfigurationError
import numpy as np
import math
import time

from threading import Thread

# ROS imports
from std_msgs.msg import Int8, Float64MultiArray, Header, Float32
from relaxed_ik.msg import EEPoseGoals, JointAngles
from sensor_msgs.msg import JointState
from geometry_msgs.msg import Pose, Point, Quaternion
from mirracle_gestures.srv import AddMesh, AddMeshResponse, RemoveMesh, RemoveMeshResponse, GripperControl, GripperControlResponse
import rospy


joints = False
pose = False

## TODO: Move to better place
SceneObjects = {}

def callback_output_relaxedIK(data):
    ''' RelaxedIK solver option. When received joints, publish to coppeliaSim
    '''
    global joints
    j = []
    for ang in data.angles:
        j.append(ang.data)
    joints = j

def publish_as_output_ik(joints):
    ''' PyRep solver option. Publish joints as JointAngles type msg
    Parameters:
        joints (Float[7]): joints
    '''
    msg = JointAngles()
    for n,j in enumerate(joints):
        msg.angles.append(Float32(j))
    ik_solution_pub.publish(msg)

def callback_goal_poses(data):
    ''' PyRep solver option. Receved goal poses will be applied InverseKinematics in this node
    '''
    global pose
    if len(data.ee_poses) > 1:
        raise Exception("Two or more arms goals are sended, please update this callback")
    pose = data.ee_poses[0]

def simpleController(panda, pose):

    p1, q1 = panda.get_tip().get_position(), panda.get_tip().get_quaternion()
    p2, q2 = list(settings.extv(pose.position)), list(settings.extq(pose.orientation))
    p_diff = np.subtract(p2,p1)
    q_diff = np.subtract(q2,q1)
    p_diff_cropped = np.clip(p_diff, -0.1, 0.1) # set max step
    q_diff_cropped = np.clip(q_diff, -0.1, 0.1) # set max step

    p_out = np.add(p1, p_diff_cropped)
    q_out = np.add(q1, q_diff_cropped)

    pose = Pose()
    pose.position = Point(*p_out)
    pose.orientation = Quaternion(*q_out)

    return pose

def add_mesh_callback(req):
    ''' Receives service callback of object creation
    '''
    file = req.file
    name = req.name
    pose = req.pose
    frame_id = req.frame_id
    if 'box' in req.name:
        object = Shape.create(type=PrimitiveShape.CUBOID,
                      color=[0.,1.,0.], size=[0.075, 0.075, 0.075],
                      position=[pose.position.x, pose.position.y, pose.position.z])
        object.set_color([0., 1., 0.])
        object.set_position([pose.position.x, pose.position.y, pose.position.z])
    else:
        object = Shape.import_mesh(filename = file)
        object.set_pose(settings.extp(pose))

    SceneObjects[name] = object
    return AddMeshResponse(True)

def remove_mesh_callback(req):
    ''' Receives service callback of object deletion
    '''
    name = req.name
    SceneObjects[name].remove()
    return RemoveMeshResponse(True)


def gripper_control_callback(req):
    ''' Control the gripper with values
        - position 0.0 -> closed, 1.0 -> open
        - (pseudo) effort <0.0-1.0>
    '''
    position = req.position
    effort = req.effort
    # TODO: Add applying force
    '''
    PandaGripper -> error is occuring (https://github.com/stepjam/PyRep/issues/280)
    panda_gripper.actuate(position, # open position
                          effort) # (pseudo) effort
    '''
    return GripperControlResponse(True)

def main():
    global pr, joints, pose
    if settings.GRIPPER_NAME == 'none':
        SCENE_FILE = settings.COPPELIA_SCENE_PATH+'/'+'scene_panda.ttt'
    elif settings.GRIPPER_NAME == 'franka_hand':
        SCENE_FILE = settings.COPPELIA_SCENE_PATH+'/'+'scene_panda_franka_gripper.ttt'
    elif settings.GRIPPER_NAME == 'franka_hand_with_camera':
        SCENE_FILE = settings.COPPELIA_SCENE_PATH+'/'+'scene_panda_custom_gripper.ttt'
    else: raise Exception("Wrong selected gripper, (probably in demo.launch file)")

    pr.launch(settings.COPPELIA_SCENE_PATH+'/'+'scene_panda_reach_target.ttt', headless=False)
    pr.get_simulation_timestep()
    #pr.set_simulation_timestep(dt=0.1)
    pr.start()

    # ROBOT loading
    panda = Panda()
    '''
    PandaGripper -> error is occuring (https://github.com/stepjam/PyRep/issues/280)
    panda_gripper = PandaGripper()
    '''

    while not joints and not pose:
        time.sleep(2)
        print("[Coppelia Pub] Not received data yet!")
    while True:
        # publish to coppelia
        if settings.IK_SOLVER == 'relaxed_ik':
            panda.set_joint_target_positions(joints)
        elif settings.IK_SOLVER == 'pyrep':
            pose_send = simpleController(panda, pose)


            '''  https://github.com/stepjam/PyRep/issues/272
                 https://github.com/stepjam/PyRep/issues/285
            '''
            joints = panda.solve_ik_via_sampling(list(settings.extv(pose.position)), quaternion=list(settings.extq(pose.orientation)))[0]

            try:
                joints = panda.solve_ik_via_jacobian(list(settings.extv(pose_send.position)), quaternion=list(settings.extq(pose_send.orientation)))
            except IKError:
                print("[Coppelia Pub] Solving via jacobian failed")

                try:
                    joints = panda.solve_ik_via_sampling(list(settings.extv(pose.position)), quaternion=list(settings.extq(pose.orientation)))[0]
                except ConfigurationError:
                    print("[Coppelia Pub] No configuration found for goal pose:", pose)

            publish_as_output_ik(joints)
            panda.set_joint_target_positions(joints)
        else: raise Exception("[ERROR*] Wrong 'ik_solver' used in demo.launch!")
        pr.step()
        # delay
        time.sleep(0.1)
        # publish eef
        eef_msg = Pose()
        eef_msg.position = Point(*panda.get_tip().get_position())
        eef_msg.orientation = Quaternion(*panda.get_tip().get_quaternion())
        eef_pub.publish(eef_msg)
        # publish joint_states
        joint_state_msg = JointState()
        joint_state_msg.position = panda.get_joint_positions()
        joint_state_msg.velocity = panda.get_joint_velocities()
        joint_state_msg.header = Header()
        joint_state_msg.header.stamp = rospy.Time.now()
        joint_states_pub.publish(joint_state_msg)

rospy.init_node('coppeliaPublisher', anonymous=True)
if settings.IK_SOLVER == 'relaxed_ik':
    rospy.Subscriber('/relaxed_ik/joint_angle_solutions', JointAngles, callback_output_relaxedIK)
elif settings.IK_SOLVER == 'pyrep':
    rospy.Subscriber('/relaxed_ik/ee_pose_goals', EEPoseGoals, callback_goal_poses)
    ik_solution_pub = rospy.Publisher('/relaxed_ik/joint_angle_solutions', JointAngles, queue_size=5)
else: raise Exception("[ERROR*] Wrong 'ik_solver' used in demo.launch!")
joint_states_pub = rospy.Publisher('/joint_states_coppelia', JointState, queue_size=5)
eef_pub = rospy.Publisher('/pose_eef', Pose, queue_size=5)

# Listen for service
rospy.Service('add_mesh', AddMesh, add_mesh_callback)
rospy.Service('remove_mesh', AddMesh, remove_mesh_callback)
rospy.Service('gripper_control', GripperControl, gripper_control_callback)

pr = PyRep()
thread = Thread(target = main)
thread.daemon=True
thread.start()

print("Ending in 5s")
pr.stop()
pr.shutdown()
