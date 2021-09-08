#!/usr/bin/env python3.7
'''
Coppelia publisher. It communicates with CoppeliaSim via PyRep.

Purpose: Compatibility communication with Python2.7 via services.
         After transfer to Python3, communication can be direct

Loads Scene from dir var 'COPPELIA_SCENE_PATH' specified in 'settings.py'.
Three scenes are created based on ROSparam 'mirracle_config/gripper':
    - 'none' gripper -> 'scene_panda.ttt' loaded
    - 'franka_hand' gripper -> 'scene_panda_franka_gripper2.ttt' loaded
    - 'franka_hand_with_camera' gripper -> 'scene_panda_custom_gripper.ttt'
Uses Joint states controller.
Inverse Kinematics based on ROSparam 'mirracle_config/ik_solver' as:
    - 'specify_topic' -> Output joints to be updated by settings.IK_TOPIC
    - 'relaxed_ik' -> RelaxedIK, computed in separate node, here receiving '/relaxed_ik/joint_angle_solutions'
    - 'pyrep' -> Uses PyRep IK, computed here, receiving '/relaxed_ik/ee_pose_goals', publishing to '/relaxed_ik/joint_angle_solutions'
        * TODO: In this option change the topic names as it is not /relaxed_ik in this option to not be misleading
    - IMPORTANT: 'relaxed_ik' updates self.joints, 'pyrep' updates self.eef_pose
Note: Install PyRep dir is set (line below) to ~/PyRep as written in README.md
'''
import sys
import os

import settings
settings.init(minimal=True)

from os.path import dirname, join, abspath
from pyrep import PyRep
from pyrep.robots.arms.panda import Panda
from pyrep.robots.end_effectors.panda_gripper import PandaGripper
from pyrep.objects.shape import Shape
from pyrep.objects.vision_sensor import VisionSensor
from pyrep.const import PrimitiveShape
from pyrep.errors import ConfigurationPathError, IKError, ConfigurationError
import numpy as np
import math
import time
#import cv2

from threading import Thread

# ROS imports
from std_msgs.msg import Int8, Float64MultiArray, Header, Float32, Bool
from relaxed_ik.msg import EEPoseGoals, JointAngles
from sensor_msgs.msg import JointState, Image
from geometry_msgs.msg import Pose, Point, Quaternion
from mirracle_gestures.srv import AddOrEditObject, AddOrEditObjectResponse, RemoveObject, RemoveObjectResponse, GripperControl, GripperControlResponse
import rospy
import py3_utils

# Real Sense publish frequency
IMAGE_PUBLISH_FREQ = float(rospy.get_param("/coppelia/image_publish_freq"))
global ALIVE; ALIVE = True

class CoppeliaSim():

    def __init__(self):
        rospy.init_node('coppeliaSim', anonymous=True)

        self.joints = [0.,0.,0.,0.,0.,0.,0.] # Updated by relaxed_ik or another topic
        # Received goal pose, the program will converge to this pose
        self.eef_pose = False
        # (inside controller) eef_pose -> eef_pose_controller_applied
        # This is next value published to Coppelia
        self.eef_pose_controller_applied = False
        # Markers to appropriate eef_pose
        self.eef_pose_marker = None
        self.eef_pose_controller_applied_marker = None
        # These are object types saved inside dictionary
        self.SceneObjects = {}

        # image publishers
        self.left_seq = 0
        self.right_seq = 0
        self.leftimagerpub = None
        self.rightimagerpub = None

        self.pr = PyRep()
        if settings.GRIPPER_NAME == 'none':
            SCENE_FILE = settings.COPPELIA_SCENE_PATH+'/'+'scene_panda.ttt'
        elif settings.GRIPPER_NAME == 'franka_hand':
            SCENE_FILE = settings.COPPELIA_SCENE_PATH+'/'+'scene_panda_franka_gripper_prox.ttt'
        elif settings.GRIPPER_NAME == 'franka_hand_with_camera':
            SCENE_FILE = settings.COPPELIA_SCENE_PATH+'/'+'scene_panda_custom_gripper.ttt'
        else: raise Exception("Wrong selected gripper, (probably in demo.launch file)")
        self.pr.launch(SCENE_FILE, headless=False) # Run coppelia
        self.pr.get_simulation_timestep()
        #self.pr.set_simulation_timestep(dt=0.1)
        self.pr.start()

        # ROBOT loading
        self.panda = Panda()
        if settings.GRIPPER_NAME == 'franka_hand':
            self.panda_gripper = PandaGripper()
        elif settings.GRIPPER_NAME == 'franka_hand_with_camera':
            self.panda_gripper = PandaGripper()
            # Publisher for Intel Realsense D435 image
            self.leftimagerpub = rospy.Publisher('/coppelia/left_camera', Image, queue_size=5)
            self.rightimagerpub = rospy.Publisher('/coppelia/right_camera', Image, queue_size=5)
            self.LeftImager = VisionSensor("LeftImager")
            self.RightImager = VisionSensor("RightImager")

        # For PyRep solver, this will publish joints solution to system
        self.ik_solution_pub = None
        if settings.IK_SOLVER == 'relaxed_ik':
            # Receives IK solutions (computed in relaxed ik node)
            rospy.Subscriber('/relaxed_ik/joint_angle_solutions', JointAngles, self.callback_output_nodeIK)
        elif settings.IK_SOLVER == 'pyrep':
            # Publishes IK solutions (computed here)
            self.ik_solution_pub = rospy.Publisher('/relaxed_ik/joint_angle_solutions', JointAngles, queue_size=5)
        elif settings.IK_SOLVER == 'specify_topic':
            # Received IK solutions (computed in specified topic)
            self.ik_solution_pub = rospy.Publisher(settings.IK_TOPIC, JointAngles, queue_size=5)
        else: raise Exception("[ERROR*] Wrong 'ik_solver' used in demo.launch!")
        # Receives the end-effector pose goal
        rospy.Subscriber('/ee_pose_goals', Pose, self.callback_goal_poses)
        # Publish joint states and end-effector
        self.joint_states_pub = rospy.Publisher('/joint_states_coppelia', JointState, queue_size=5)
        self.eef_pub = rospy.Publisher('/pose_eef', Pose, queue_size=5)

        # Listen for service
        rospy.Service('add_or_edit_object', AddOrEditObject, self.add_or_edit_object_callback)
        rospy.Service('remove_object', RemoveObject, self.remove_object_callback)
        rospy.Service('gripper_control', GripperControl, self.gripper_control_callback)


    def __enter__(self):
        thread = Thread(target = self.simulate)
        #thread.daemon=True
        thread.start()
        #if settings.IK_SOLVER == 'pyrep':
        #    thread2 = Thread(target = self.pyrep_ik_solver_thread)
        #    thread2.daemon=True
        #    thread2.start()

        # Publishes camera images
        if settings.GRIPPER_NAME == 'franka_hand_with_camera':
            thread3 = Thread(target = self.camera_publisher)
            #thread3.daemon=True
            thread3.start()

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        print("Ending in 5s")
        coppeliasim.pr.stop()
        coppeliasim.pr.shutdown()

    def callback_output_nodeIK(self, data):
        ''' RelaxedIK solver option. When received joints, publish to coppeliaSim
        '''
        j = []
        for ang in data.angles:
            j.append(ang.data)
        self.joints = j

    def publish_as_output_ik(self):
        ''' PyRep solver option. Publish joints as JointAngles type msg
        Parameters:
            joints (Float[7]): joints
        '''
        msg = JointAngles()
        for n,j in enumerate(self.joints):
            msg.angles.append(Float32(j))
        self.ik_solution_pub.publish(msg)

    def callback_goal_poses(self, data):
        ''' PyRep solver option. Receved goal poses will be applied InverseKinematics in this node
        '''
        #if len(data.ee_poses) > 1:
        #    raise Exception("Two or more arms goals are sended, please update this callback")
        #eef_pose = data.ee_poses[0]
        self.eef_pose = data
        self.eef_pose_controller_applied = self.simpleController(self.eef_pose)

    def simpleController(self, pose):
        ''' Proporcional controller, max p=0.1 meters per iteration in every axis
            - Probably not very good for this situation
        '''
        assert type(pose) == type(Pose()), "simpleController input not type Pose"
        p1, q1 = self.panda.get_tip().get_position(), self.panda.get_tip().get_quaternion()
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

    def add_or_edit_object_callback(self, msg):
        ''' Receives service callback of object creation
            - msg defined in AddOrEditObject.srv
        '''
        # Set specified color
        if   msg.color == 'b': color_rgb = [0.,0.,1.]
        elif msg.color == 'g': color_rgb = [0.,1.,0.]
        elif msg.color == 'b': color_rgb = [0.,0.,1.]
        elif msg.color == 'c': color_rgb = [0.,1.,1.]
        elif msg.color == 'm': color_rgb = [1.,0.,1.]
        elif msg.color == 'y': color_rgb = [1.,1.,0.]
        elif msg.color == 'k': color_rgb = [0.,0.,0.]
        else: raise Exception("Picked wrong color! color is not ('r','g','b',...) rgb is"+str(msg.color))
        if msg.shape == 'cube':
            shape = PrimitiveShape.CUBOID
        elif msg.shape == 'sphere':
            shape = PrimitiveShape.SPHERE
        elif msg.shape == 'cylinder':
            shape = PrimitiveShape.CYLINDER
        elif msg.shape == 'cone':
            shape = PrimitiveShape.CONE
        elif msg.shape != '':
            raise Exception("Picked wrong shape! Shape is not 'cube', 'sphere', 'cylinder', or 'cone', it is: "+str(msg.shape))

        object = None
        # 1. Edit existing object
        if msg.name in self.SceneObjects.keys(): # Check if object exists
            # Object won't be created, only edited
            object = self.SceneObjects[msg.name]
        else:
            # 2. Create Shape
            if msg.file:
                if not msg.size: msg.size = 1.
                object = Shape.import_mesh(filename = msg.file, scaling_factor=msg.size)
            # 3. Create Mesh
            elif msg.shape:
                if not msg.size: msg.size = 0.075
                object = Shape.create(type=shape, size=[msg.size, msg.size, msg.size])
            # 4. Create Box
            else:
                object = Shape.create(type=PrimitiveShape.CUBOID, size=[0.075, 0.075, 0.075])

        # Set parameters
        object.set_position(settings.extv(msg.pose.position))
        #q = py3_utils.quaternion_multiply(object.get_quaternion(), settings.extq(msg.pose.orientation))

        print("q1", np.round(object.get_quaternion(),2))
        object.set_quaternion(py3_utils.quaternion_multiply(object.get_quaternion(), settings.extq(msg.pose.orientation)))
        print("q2", np.round(py3_utils.quaternion_multiply(object.get_quaternion(), settings.extq(msg.pose.orientation)),2))


        object.set_color(color_rgb)
        if msg.friction: object.set_bullet_friction(msg.friction)

        self.SceneObjects[msg.name] = object
        return AddOrEditObjectResponse(True)

    def remove_object_callback(self, msg):
        ''' Receives service callback of object deletion
            - msg defined in Remove Object.srv
        '''
        if msg.name in self.SceneObjects.keys():
            self.SceneObjects.pop(msg.name).remove()
            return RemoveObjectResponse(True)
        else:
            print("[WARN*][Coppelia Pub] No object with name ", msg.name, " found!")
            return RemoveObjectResponse(False)


    def gripper_control_callback(self, msg):
        ''' Control the gripper with values
            - msg defined in GripperControl.srv
            - position 0.0 -> closed, 1.0 -> open
            - (pseudo) effort <0.0-1.0>
        '''
        # TODO: Add applying force
        '''
        PandaGripper -> error is occuring (https://github.com/stepjam/PyRep/issues/280)
        self.panda_gripper.actuate(msg.position, # open position
                              msg.effort) # (pseudo) effort
        '''
        if msg.action == 'grasp':
            #self.panda_gripper.grasp(msg.object)
            self.panda.grasp(msg.object)
        elif msg.action == 'release':
            self.panda.release()
            #self.panda_gripper.release()
        return GripperControlResponse(True)

    def simulate(self):
        global ALIVE
        while type(self.eef_pose) == type(False) or type(self.eef_pose) == type(None):
            time.sleep(1)
        while ALIVE:
            # viz the eef
            if self.eef_pose_marker: self.eef_pose_marker.remove()
            if self.eef_pose_controller_applied_marker: self.eef_pose_controller_applied_marker.remove()
            if self.eef_pose:
                self.eef_pose_marker = Shape.create(type=PrimitiveShape.SPHERE,
                              color=[0.,1.,0.], size=[0.04, 0.04, 0.04], respondable = False,
                              position=[self.eef_pose.position.x, self.eef_pose.position.y, self.eef_pose.position.z])
                self.eef_pose_marker.set_color([0., 0., 1.])
                self.eef_pose_marker.set_position([self.eef_pose.position.x, self.eef_pose.position.y, self.eef_pose.position.z])
                self.eef_pose_controller_applied_marker = Shape.create(type=PrimitiveShape.SPHERE,
                              color=[0.,1.,0.], size=[0.04, 0.04, 0.04], respondable = False,
                              position=[self.eef_pose_controller_applied.position.x, self.eef_pose_controller_applied.position.y, self.eef_pose_controller_applied.position.z])
                self.eef_pose_controller_applied_marker.set_color([1., 0., 0.])
                self.eef_pose_controller_applied_marker.set_position([self.eef_pose_controller_applied.position.x, self.eef_pose_controller_applied.position.y, self.eef_pose_controller_applied.position.z])
            # publish to coppelia
            if settings.IK_SOLVER == 'relaxed_ik':
                self.panda.set_joint_target_positions(self.joints)
            elif settings.IK_SOLVER == 'pyrep':
                PATH = False
                if PATH:
                    # Get a path to the target (rotate so z points down)
                    try:
                        path = self.panda.get_path(
                            position=list(settings.extv(self.eef_pose.position)),
                            quaternion=list(settings.extq(self.eef_pose.orientation)))
                    except ConfigurationPathError as e:
                        print('[Coppelia] Could not find path')
                        continue

                    done = False
                    while not done:
                        done = path.step()
                        pr.step()
                else:
                    '''  https://github.com/stepjam/PyRep/issues/272
                         https://github.com/stepjam/PyRep/issues/285
                    '''
                    self.pyrep_ik_solver() # self.eef_pose_controller_applied -> self.joints
                    self.publish_as_output_ik() # publishes ik output to other nodes
                    self.panda.set_joint_target_positions(self.joints)

            else: raise Exception("[ERROR*] Wrong 'ik_solver' used in demo.launch!")
            self.pr.step()
            # delay
            time.sleep(0.1)
            # publish eef
            eef_msg = Pose()
            eef_msg.position = Point(*self.panda.get_tip().get_position())
            eef_msg.orientation = Quaternion(*self.panda.get_tip().get_quaternion())
            self.eef_pub.publish(eef_msg)
            # publish joint_states
            joint_state_msg = JointState()
            joint_state_msg.position = self.panda.get_joint_positions()
            joint_state_msg.velocity = self.panda.get_joint_velocities()
            joint_state_msg.header = Header()
            joint_state_msg.header.stamp = rospy.Time.now()
            self.joint_states_pub.publish(joint_state_msg)


    def pyrep_ik_solver_thread(self):
        while not self.eef_pose_controller_applied:
            time.sleep(2)
            print("[Coppelia] Not received data yet!")
        while True:
            self.pyrep_ik_solver()


    def pyrep_ik_solver(self):
        time.sleep(0.1)
        try:
            self.joints = self.panda.solve_ik_via_jacobian(list(settings.extv(self.eef_pose_controller_applied.position)), quaternion=list(settings.extq(self.eef_pose_controller_applied.orientation)))
        except IKError:
            print("[Coppelia] Solving via jacobian failed")
            try:
                self.joints = self.panda.solve_ik_via_sampling(list(settings.extv(self.eef_pose_controller_applied.position)), quaternion=list(settings.extq(self.eef_pose_controller_applied.orientation)))[0]
            except ConfigurationError:
                print("[Coppelia] No configuration found for goal pose:", self.eef_pose_controller_applied)
        print("joints", self.joints)

    def camera_publisher(self):
        global ALIVE
        rate = rospy.Rate(IMAGE_PUBLISH_FREQ)
        while ALIVE:
            while ALIVE and CAMERA_ALIVE:
                leftdata = self.LeftImager.capture_rgb()
                img = Image()
                img.data = leftdata
                img.step = 720
                img.height = 720
                img.width = 1280
                img.header.seq = self.left_seq = self.left_seq + 1
                img.header.stamp = rospy.Time.now()
                img.header.frame_id = ""
                self.leftimagerpub.publish(img)

                rightdata = self.RightImager.capture_rgb()
                img = Image()
                img.data = rightdata
                img.step = 720
                img.height = 720
                img.width = 1280
                img.header.seq = self.right_seq = self.right_seq + 1
                img.header.stamp = rospy.Time.now()
                img.header.frame_id = ""
                self.rightimagerpub.publish(img)

                rate.sleep()


# __enter__ and __exit__ functions to be called
with CoppeliaSim() as coppeliasim:
    while True:
        msg = rospy.wait_for_message('/mirracle_gestures/quit', Bool)
        if msg.data == True:
            ALIVE = False
            break
