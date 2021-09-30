#!/usr/bin/env python2
'''
Library for manipulation with robot arm.

Source: https://ros-planning.github.io/moveit_tutorials/doc/move_group_python_interface/move_group_python_interface_tutorial.html
'''

from __future__ import print_function
from six.moves import input

import os
import tf
import sys
import copy
from copy import deepcopy
import time
import math
import rospy
import scipy
from scipy import interpolate
import random
import itertools
import time as t
import numpy as np
from math import pi
import moveit_commander
import moveit_msgs.msg
import geometry_msgs.msg
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.preprocessing import minmax_scale
from moveit_commander.conversions import pose_to_list

from visualizer_lib import VisualizerLib
from matplotlib import cm
from matplotlib.colors import ListedColormap, LinearSegmentedColormap

import trajectory_action_client

#from promp_lib import promp_
from std_msgs.msg import String, Bool, Int8, Float64MultiArray
from control_msgs.msg import FollowJointTrajectoryAction, FollowJointTrajectoryGoal
from geometry_msgs.msg import Pose, Point, Quaternion, PoseStamped, Vector3Stamped, QuaternionStamped, Vector3
from moveit_msgs.srv import ApplyPlanningScene
from trajectory_msgs.msg import JointTrajectoryPoint, JointTrajectory
from moveit_msgs.msg import RobotTrajectory
#import RelaxedIK.Utils.transformations as T
from sensor_msgs.msg import JointState
from relaxed_ik.msg import EEPoseGoals, JointAngles
from mirracle_sim.srv import AddOrEditObject, AddOrEditObjectResponse, RemoveObject, RemoveObjectResponse, GripperControl, GripperControlResponse

import kinematics_interface
import settings

from os.path import isfile

## just test, if qpOASES can be sourced by adding the given folder
# -> should be sourced by installation as in README.md
#sys.path.append("~/Documents/qpOASES-3.2.1/interfaces/python")
#sys.path.append("~/Documents/toppra-0.2.2a/examples")
import toppra as ta
from toppra import SplineInterpolator


def all_close(goal, actual, tolerance):
    """
    Convenience method for testing if a list of values are within a tolerance of their counterparts in another list
    @param: goal       A list of floats, a Pose or a PoseStamped
    @param: actual     A list of floats, a Pose or a PoseStamped
    @param: tolerance  A float
    @returns: bool
    """
    all_equal = True
    if type(goal) is list:
        for index in range(len(goal)):
            if abs(actual[index] - goal[index]) > tolerance:
                return False

    elif type(goal) is geometry_msgs.msg.PoseStamped:
        return all_close(goal.pose, actual.pose, tolerance)

    elif type(goal) is geometry_msgs.msg.Pose:
        return all_close(pose_to_list(goal), pose_to_list(actual), tolerance)
    return True


class MoveGroupPythonInteface(object):
    """MoveGroupPythonInteface, Not only MoveIt"""
    def __init__(self, env='above'):
        super(MoveGroupPythonInteface, self).__init__()

        moveit_commander.roscpp_initialize(sys.argv)
        rospy.init_node('move_group_python_interface', anonymous=True)

        print("[INFO*] GROUP NAME: ", settings.GROUP_NAME)
        robot = moveit_commander.RobotCommander()
        scene = moveit_commander.PlanningSceneInterface()
        move_group = moveit_commander.MoveGroupCommander(settings.GROUP_NAME)
        display_trajectory_publisher = rospy.Publisher('/move_group/display_planned_path',
                                                       moveit_msgs.msg.DisplayTrajectory,
                                                       queue_size=20)



        planning_frame = move_group.get_planning_frame()
        eef_link = move_group.get_end_effector_link()
        group_names = robot.get_group_names()
        # Misc variables
        self.box_name = ''
        self.robot = robot
        self.scene = scene
        self.move_group = move_group
        self.display_trajectory_publisher = display_trajectory_publisher
        self.planning_frame = planning_frame
        self.eef_link = eef_link
        self.group_names = group_names

        self.global_plan = None
        self.RECORDING_LENGTH = 2.0
        self.PLANNING_ATTEMPTS = 1
        self.ProMP_INPUT_NUM_PLANNERS = 1
        self.ProMP_INPUT_NUM_LEAP_RECORDS = 10

        # RelaxedIK publishers
        self.ik_goal_r_pub = rospy.Publisher('/ik_goal_r',PoseStamped,queue_size=5)
        self.ik_goal_l_pub = rospy.Publisher('/ik_goal_l',PoseStamped,queue_size=5)
        self.goal_pos_pub = rospy.Publisher('vive_position', Vector3Stamped,queue_size=5)
        self.goal_quat_pub = rospy.Publisher('vive_quaternion', QuaternionStamped,queue_size=5)
        self.ee_pose_goals_pub = rospy.Publisher('/relaxed_ik/ee_pose_goals', EEPoseGoals, queue_size=5)
        self.quit_pub = rospy.Publisher('/relaxed_ik/quit',Bool,queue_size=5)
        self.seq = 1

        # When directing control right to Coppelia Simulator, the output of RelaxedIk is sended directily to simulator
        if settings.SIMULATOR_NAME == 'gazebo' or settings.SIMULATOR_NAME == 'real':
            self.tac = trajectory_action_client.TrajectoryActionClient(arm=settings.GROUP_NAME, topic=settings.TAC_TOPIC, topic_joint_states = settings.JOINT_STATES_TOPIC)
        settings.md.ENV = settings.md.ENV_DAT[env]
        # initialize pose
        pose = Pose()
        pose.orientation = settings.md.ENV_DAT['above']['ori']
        pose.position = Point(0.4,0.,1.0)
        settings.goal_pose = deepcopy(pose)

        self.ik_bridge = IK_bridge()

        # For Path velocity
        self.savedVelocityVec = None
        self.timePassedPreviousTraj = None
        self.durationPreviousTraj = None

        self.MOVEIT = False


        # Will replace trajectory inside planning the trajectory
        self.trajectory_action_perform = True

    def getCurrentPose(self, stamped=False):
        ''' Returns pose of endeffector.
            - eef is changed based on gripper, in order to maintain place of object pickup
            - 'coppelia' -> updated by topic
            - other -> updated in main
        Parameters:
            (global) self.MOVEIT (bool): Use MoveIt! function
        Returns:
            pose (Pose())
        '''
        if self.MOVEIT:
            if stamped:
                return self.move_group.get_current_pose()
            return self.move_group.get_current_pose().pose
        return settings.eef_pose

    def getCurrentJoints(self):
        ''' Joint states across multiple systems. Latest joint values updated in function: 'save_joints'
            - MoveIt -> use function
        Parameters:
            (global) self.MOVEIT (bool): Use MoveIt! function
        Returns:
            joints (float[7])
        '''
        if self.MOVEIT:
            return self.move_group.get_current_joint_values()
        return settings.joints


    def go_to_joint_state(self, joints):
        ''' Planning to a Joint Goal
            Choosing how to perform the path between robots, simulators based on demo.launch/simulator.launch
            - MoveIt
            - TAC/Joint states controller
            !! TODO: Test MoveIt! fun: self.move_group.get_current_joint_values()
        '''
        assert isinstance(joints, list) and len(joints)==7, "Input not type list with len 7"

        if self.MOVEIT:
            self.move_group.go(joints, wait=True)
            self.move_group.stop()

            current_joints = self.move_group.get_current_joint_values()
            return all_close(joints, current_joints, 0.01)
        else:
            if settings.ROBOT_NAME == 'panda':
                if settings.SIMULATOR_NAME == 'gazebo' or settings.SIMULATOR_NAME == 'real':
                    self.plan_and_perform_trajectory_from_joints(joints)
                elif settings.SIMULATOR_NAME == 'coppelia':
                    pass # publsiher is enabled
                else: raise Exception("[ERROR*] Simulator name wrong")
            elif settings.ROBOT_NAME == 'iiwa':
                raise Exception("[ERROR*] Robot not setup, pick Panda.")
            else: raise Exception("[ERROR*] Robot name wrong")


    def go_to_pose_goal(self, pose=None):
        ''' Planning to a Pose Goal
            - Plan a motion for this group to a desired pose for the end-effector
            - TODO: Another option for execution than MoveIt!
        Parameters:
            (global) self.MOVEIT (Bool): Uses MoveIt! function
        '''
        assert isinstance(pose, Pose), "Input not type Pose"

        if self.MOVEIT:
            self.move_group.set_pose_target(pose)

            plan = self.move_group.go(wait=True)  # Planner compute the plan and execute it.
            self.move_group.stop()
            self.move_group.clear_pose_targets()

            current_pose = self.move_group.get_current_pose().pose
            return all_close(pose, current_pose, 0.01)
        raise Exception("TODO: Option without moveit")


    def plan_path(self, poses):
        ''' Cartesian Paths
        Plans Cartesian path directly by specifying a list of waypoints
        for the end-effector to go through.

        Parameters:
            poses (Array of Pose()): Waypoints
            (global) self.MOVEIT (Bool): Uses MoveIt! function
        Returns:
            plan (RobotTrajectory()): Output plan
            fraction (Float): Fraction how much path followed
        '''
        if self.MOVEIT:
            pose = self.getCurrentJoints()
            if self.samePoses(pose, poses[0]): # If first path point is the same as current one, discard it from path
                poses = poses[1:]
            (plan, fraction) = self.move_group.compute_cartesian_path(poses, 0.01, # eef_step
                                                                                0.0) # jump_threshold
            if fraction < 1.0:
                print("[Warn*] Plan path function did not found whole path, only: ", fraction*100., " %")
            return plan, fraction
        raise Exception("TODO: Option without moveit")


    def display_trajectory(self, plan):
        ''' Displaying a Trajectory
        You can ask RViz to visualize a plan (aka trajectory) for you. But the
        group.plan() method does this automatically so this is not that useful
        here (it just displays the same trajectory again):

        A `DisplayTrajectory`_ msg has two primary fields, trajectory_start and trajectory.
        We populate the trajectory_start with our current robot state to copy over
        any AttachedCollisionObjects and add our plan to the trajectory.
        '''
        if self.MOVEIT:
            display_trajectory = moveit_msgs.msg.DisplayTrajectory()
            display_trajectory.trajectory_start = self.robot.get_current_state()
            display_trajectory.trajectory.append(plan)
            print(display_trajectory)
            # Publish
            self.display_trajectory_publisher.publish(display_trajectory)
        raise Exception("TODO: Option without moveit")

    def execute_plan(self, plan):
        ''' Executing a Plan. Uses plan already computed.

        Parameters:
            plan (RobotTrajectory()): Input plan
            (global) self.MOVEIT (Bool): Execute with MoveIt! fun.
        '''
        if self.MOVEIT:
            return self.move_group.execute(plan, wait=True)
        raise Exception("TODO: Without Moveit!")


    def wait_for_state_update(self, box_is_known=False, box_is_attached=False, timeout=4):
        ''' MoveIt! fun. Ensuring Collision Updates Are Received
        If the Python node dies before publishing a collision object update message, the message
        could get lost and the box will not appear. To ensure that the updates are
        made, we wait until we see the changes reflected in the
        ``get_attached_objects()`` and ``get_known_object_names()`` lists.
        For the purpose of this tutorial, we call this function after adding,
        removing, attaching or detaching an object in the planning scene. We then wait
        until the updates have been made or ``timeout`` seconds have passed
        '''
        start = rospy.get_time()
        seconds = rospy.get_time()
        while (seconds - start < timeout) and not rospy.is_shutdown():
            # Test if the box is in attached objects
            attached_objects = self.scene.get_attached_objects([self.box_name])
            is_attached = len(attached_objects.keys()) > 0

            # Test if the box is in the scene.
            # Note that attaching the box will remove it from known_objects
            is_known = self.box_name in self.scene.get_known_object_names()

            # Test if we are in the expected state
            if (box_is_attached == is_attached) and (box_is_known == is_known):
                return True

            # Sleep so that we give other threads time on the processor
            rospy.sleep(0.1)
            # If we exited the while loop without returning then we timed out
            return False


    def add_or_edit_object(self, file='', name='', pose=Pose(), shape="", size=None, collision='true', color='', friction=-1, frame_id='panda_link7', mass=-1, inertia=np.zeros(9), inertiaTransformation=np.zeros(12), dynamic='true', pub_info='false', texture_file="", timeout=4):
        ''' Adds shape/mesh based on configuration
            - If no shape & no mesh specified -> box created
            - If both shape & mesh specified -> mesh is used
        Parameters:
            file (Str): Mesh file string (def. in dir include/models/) as .obj file
            name (Str): Name of object
            pose (Pose()): Pose of object to be spawned
            shape (Str): 'cube', 'sphere', 'cylinder', 'cone'
            color (Str): 'r', 'g', 'b', 'c', 'm', 'y', 'k'
            frame_id (Str): By respect to what frame_id to be spawned
            timeout (Int/Float): If fails, will end in given seconds
            (global) self.MOVEIT (Bool): Uses MoveIt! env.
        '''
        if self.MOVEIT:
            box_name = self.box_name
            scene = self.scene

            mesh_pose = geometry_msgs.msg.PoseStamped()
            mesh_pose.header.frame_id = frame_id
            mesh_pose.pose = pose
            mesh_name = name

            if file:
                scene.add_mesh(mesh_name, mesh_pose, file, size=(1.0, 1.0, 1.0))
            elif shape:
                raise Exception("TODO!")
                #self.scene.add_shape(mesh_name, mesh_pose, size=(0.075, 0.075, 0.075))
            else:
                self.scene.add_box(mesh_name, mesh_pose, size=(0.075, 0.075, 0.075))

            self.box_name=mesh_name # its the box
            return self.wait_for_state_update(box_is_known=True, timeout=timeout)
        # CoppeliaSim (PyRep)
        rospy.wait_for_service('add_or_edit_object')
        try:
            add_or_edit_object = rospy.ServiceProxy('add_or_edit_object', AddOrEditObject)
            resp1 = add_or_edit_object(name=name, init_file=file, init_shape=shape, init_size=size, init_collision=collision, pose=pose, color=color, friction=friction, frame_id=frame_id, mass=mass, inertia=inertia, inertiaTransformation=inertiaTransformation, dynamic=dynamic, pub_info=pub_info, texture_file=texture_file)
        except rospy.ServiceException as e:
            print("Service call failed: %s"%e)


    def remove_object(self, name=None, timeout=4):
        '''Removing Objects from the Planning Scene
        We can remove the box from the world.
        '''
        if self.MOVEIT:
            self.scene.remove_world_object(name)
            ## **Note:** The object must be detached before we can remove it from the world

            # We wait for the planning scene to update.
            return self.wait_for_state_update(box_is_attached=False, box_is_known=False, timeout=timeout)

        # CoppeliaSim (PyRep)
        rospy.wait_for_service('remove_object')
        try:
            remove_object = rospy.ServiceProxy('remove_object', RemoveObject)
            resp1 = remove_object(name)
            settings.md.attached = []
        except rospy.ServiceException as e:
            print("Service call failed: %s"%e)


    def pick_object(self, name=None, timeout=4):
        ''' Pick up item. Perform grasp while appling some force.
        '''
        # TODO: Position won't be 0.0, it will be more with pressure applied
        position = 0.0 # position of gripper
        effort = 0.4 # pseudo effort of gripper motion

        if self.MOVEIT:
            # Update planning scene
            self.attach_item_moveit(name=name, timeout=timeout, moveit=moveit)
        else:
            # CoppeliaSim (PyRep)
            rospy.wait_for_service('gripper_control')
            try:
                gripper_control = rospy.ServiceProxy('gripper_control', GripperControl)
                resp1 = gripper_control(position, effort, "grasp", name)
                settings.md.attached = [name]
            except rospy.ServiceException as e:
                print("Service call failed: %s"%e)




    def release_object(self, name=None, timeout=4):
        ''' Drops item. Perform gripper execution.

        '''
        position = 1.0 # position of gripper
        effort = 0.4 # pseudo effort of gripper motion
        if self.MOVEIT:
            # Update planning scene
            self.detach_item_moveit(name=name, timeout=timeout)
        else:
            # CoppeliaSim (PyRep)
            rospy.wait_for_service('gripper_control')
            try:
                gripper_control = rospy.ServiceProxy('gripper_control', GripperControl)
                resp1 = gripper_control(position, effort, "release", "")
            except rospy.ServiceException as e:
                print("Service call failed: %s"%e)


    def attach_item_moveit(self, name=None, timeout=4):
        ''' Attaching Mesh Objects to the Robot
            - Ignore collisions, adding link names to the ``touch_links`` array
            between those links and the box
            - Grasping group is based on robot 'settings.GRASPING_GROUP'
        Parameters:
            name (Str): Name id
        '''
        touch_links = self.robot.get_link_names(group=settings.GRASPING_GROUP)
        self.scene.attach_mesh(self.eef_link, name, touch_links=touch_links)

        ## Mark what item was attached
        settings.md.attached = [name]
        print("Attached item")

        # We wait for the planning scene to update.
        return self.wait_for_state_update(box_is_attached=True, box_is_known=False, timeout=4)


    def detach_item_moveit(self, name=None, timeout=4):
        '''Detaching Objects from the Robot
        We can also detach and remove the object from the planning scene:
        '''
        self.scene.remove_attached_object(self.eef_link, name=name)

        ## Mark that item was detached
        settings.md.attached = False
        print("Detached item")

        # We wait for the planning scene to update.
        return self.wait_for_state_update(box_is_known=True, box_is_attached=False, timeout=timeout)


    def print_info(self, timeout=4):
        # We can get the name of the reference frame for this robot:
        print("=== Planning frame: %s" % self.planning_frame)

        # We can also print the name of the end-effector link for this group:
        print("=== End effector link: %s" % self.eef_link)

        # We can get a list of all the groups in the robot:
        print("=== Available Planning Groups:", self.robot.get_group_names())

        # Sometimes for debugging it is useful to print the entire state of the
        # robot:
        print("=== Printing robot state")
        print(self.robot.get_current_state())
        print("")


    ''' Move functions
    '''
    def get_random_position(self):
        ''' Get random position (ret pose obj) within environment based on settings.md.ENV['max'|'min'] boundaries
            Orientation is set to default settings.md.ENV['ori']

        Returns:
            Pose (Pose()): Random pose
        '''
        x_len = settings.md.ENV['max'].x - settings.md.ENV['min'].x
        y_len = settings.md.ENV['max'].y - settings.md.ENV['min'].y
        z_len = settings.md.ENV['max'].z - settings.md.ENV['min'].z

        x = random.random()
        y = random.random()
        z = random.random()

        x_ = settings.md.ENV['min'].x + x_len * x
        y_ = settings.md.ENV['min'].y + y_len * y
        z_ = settings.md.ENV['min'].z + z_len * z

        pose = Pose()
        pose.position = Point(x_, y_, z_)
        pose.orientation = settings.md.ENV['ori']
        return pose

    def get_random_joints(self):
        ''' Returns random robot joints within bounds

        Returns:
            Joints (Float[7]): Robot joints float array based on configuration in settings
        '''
        joints_diff = np.array(settings.upper_lim) - np.array(settings.lower_lim)
        joints_diff_rand = [joints_diff[i] * random.random() for i in range(len(settings.upper_lim))]
        return np.add(settings.lower_lim, joints_diff_rand)


    '''
    Further custom functions
    '''
    def make_scene(self, scene=''):
        ''' Prepare scene, add objects for obstacle or manipulation.
        Parameters:
            scene (str): Scenes names are from settings.ss[:].NAME and are generated from settings.GenerateSomeScenes() function
            (you can get scene names settings.getSceneNames())
        '''
        scenes = settings.getSceneNames()
        if settings.scene: # When scene is initialized
            # get id of current scene
            id = scenes.index(settings.scene.NAME)
            # remove objects from current scene
            for i in range(0, len(settings.ss[id].object_names)):
                self.remove_object(name=settings.ss[id].object_names[i])
            if settings.md.attached:
                self.detach_item_moveit(name=settings.md.attached)
        # get id of new scene
        id = scenes.index(scene)

        for i in range(0, len(settings.ss[id].object_names)):
            obj_name = settings.ss[id].object_names[i] # object name
            size = settings.extv(settings.ss[id].object_sizes[i])
            color = settings.ss[id].object_colors[i]
            scale = settings.ss[id].object_scales[i]
            shape = settings.ss[id].object_shapes[i]
            mass = settings.ss[id].object_masses[i]
            friction = settings.ss[id].object_frictions[i]
            inertia = settings.ss[id].object_inertia[i]
            inertiaTransformation = settings.ss[id].object_inertiaTransform[i]
            dynamic = settings.ss[id].object_dynamic[i]
            pub_info = settings.ss[id].object_pub_info[i]
            texture_file = settings.ss[id].object_texture_file[i]
            file = settings.ss[id].object_file[i]

            if shape:
                self.add_or_edit_object(name=obj_name, frame_id=settings.BASE_LINK, size=size, color=color, pose=settings.ss[id].object_poses[i], shape=shape, mass=mass, friction=friction, inertia=inertia, inertiaTransformation=inertiaTransformation, dynamic=dynamic, pub_info=pub_info, texture_file=texture_file)
            elif file:
                if scale: size = [settings.ss[id].object_scales[i], 0, 0]
                else: size = [0,0,0]
                self.add_or_edit_object(file=settings.HOME+'/'+settings.WS_FOLDER+'/src/mirracle_gestures/include/models/'+file, size=size, color=color, mass=mass, friction=friction, inertia=inertia, inertiaTransformation=inertiaTransformation, dynamic=dynamic, pub_info=pub_info, texture_file=texture_file, name=obj_name, pose=settings.ss[id].object_poses[i], frame_id=settings.BASE_LINK)
            else:
                self.add_or_edit_object(name=obj_name, frame_id=settings.BASE_LINK, size=size, color=color, pose=settings.ss[id].object_poses[i], shape='cube', mass=mass, friction=friction, inertia=inertia, inertiaTransformation=inertiaTransformation, dynamic=dynamic, pub_info=pub_info, texture_file=texture_file)
        settings.scene = settings.ss[id]
        if id == 0:
            settings.scene = None

        print("[Make Scene] Scene "+scene+" ready!")


    def distancePoses(self, p1, p2):
        ''' Returns distance between two pose objects
        Parameters:
            pose1 (type Pose() from geometry_msgs.msg)
            pose2 (type Pose() from geometry_msgs.msg)
        Returns:
            distance (Float)
        '''
        p1 = p1.position
        p2 = p2.position
        return np.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2 + (p1.z - p2.z)**2)

    def pointToScene(self, point):
        x,y,z = point.x, point.y, point.z
        point_ = Point()
        point_.x = np.dot([x,y,z], settings.md.ENV['axes'][0])*settings.md.SCALE + settings.md.ENV['start'].x
        point_.y = np.dot([x,y,z], settings.md.ENV['axes'][1])*settings.md.SCALE + settings.md.ENV['start'].y
        point_.z = np.dot([x,y,z], settings.md.ENV['axes'][2])*settings.md.SCALE + settings.md.ENV['start'].z
        return point_

    def samePoses(self, pose1, pose2, accuracy=0.05):
        ''' Checks if two type poses are near each other
            (Only for cartesian (xyz), not orientation wise)
        Parameters:
            pose1 (type Pose(), Point(), list or tuple)
            pose2 (type Pose(), Point(), list or tuple)
            accuracy (Float): threshold of return value
        Returns:
            same poses (Bool)
        '''
        assert isinstance(pose1,(Pose,Point,np.ndarray,list,tuple)), "Not right datatype, pose1: "+str(pose1)
        assert isinstance(pose2,(Pose,Point,np.ndarray,list,tuple)), "Not right datatype, pose2: "+str(pose2)

        if isinstance(pose1,(list,tuple,np.ndarray)):
            pose1 = pose1[0:3]
        elif isinstance(pose1,Point):
            pose1 = [pose1.x, pose1.y, pose1.z]
        elif isinstance(pose1,Pose):
            pose1 = [pose1.position.x, pose1.position.y, pose1.position.z]
        if isinstance(pose2,(list,tuple,np.ndarray)):
            pose2 = pose2[0:3]
        elif isinstance(pose2,Point):
            pose2 = [pose2.x, pose2.y, pose2.z]
        elif isinstance(pose2,Pose):
            pose2 = [pose2.position.x, pose2.position.y, pose2.position.z]

        if np.sqrt((pose1[0] - pose2[0])**2 + (pose1[1] - pose2[1])**2 + (pose1[2] - pose2[2])**2) < accuracy:
            return True
        return False

    def sameJoints(self, joints1, joints2, accuracy=0.1):
        ''' Checks if two type joints are near each other
        Parameters:
            joints1 (type float[7])
            joints2 (type float[7])
            threshold (Float): sum of joint differences threshold
        '''
        assert isinstance(joints1[0],float) and len(joints1)==7, "Not datatype List w len 7, joints 1: "+str(joints1)
        assert isinstance(joints2[0],float) and len(joints2)==7, "Not datatype List w len 7, joints 2: "+str(joints2)

        if sum([abs(i[0]-i[1]) for i in zip(joints1, joints2)]) < accuracy:
            return True
        return False

    def transformLeapToScene(self, pose, normdir = None):
        ''' Leap -> rViz -> Scene
        '''
        assert isinstance(pose, Pose), "pose not right datatype"
        pose_ = Pose()
        pose_.orientation = deepcopy(pose.orientation)
        # Leap to rViz center point
        x = pose.position.x/1000
        y = -pose.position.z/1000
        z = pose.position.y/1000

        ## Camera rotation from CoppeliaSim
        ## TODO: Optimize for all environments
        if settings.POSITION_MODE == 'sim_camera':
            x = -pose.position.x/1000
            y = pose.position.y/1000
            z = -pose.position.z/1000
            camera = settings.md.camera_orientation
            camera_matrix = tf.transformations.euler_matrix(camera.x, camera.y, camera.z, 'rxyz')
            camera_matrix = np.array(camera_matrix)[0:3,0:3]
            x_cop = np.dot([x,y,z], camera_matrix[0])
            y_cop = np.dot([x,y,z], camera_matrix[1])
            z_cop = np.dot([x,y,z], camera_matrix[2])
            x,y,z = x_cop,y_cop,z_cop

        # Linear transformation to point with rotation
        # How the Leap position will affect system
        pose_.position.x = np.dot([x,y,z], settings.md.ENV['axes'][0])*settings.md.SCALE + settings.md.ENV['start'].x
        pose_.position.y = np.dot([x,y,z], settings.md.ENV['axes'][1])*settings.md.SCALE + settings.md.ENV['start'].y
        pose_.position.z = np.dot([x,y,z], settings.md.ENV['axes'][2])*settings.md.SCALE + settings.md.ENV['start'].z

        #hand.palm_normal.roll, hand.direction.pitch, hand.direction.yaw
        ## Orientation

        # apply rotation

        alpha, beta, gamma = tf.transformations.euler_from_quaternion([pose.orientation.x,pose.orientation.y,pose.orientation.z,pose.orientation.w])
        Rx = tf.transformations.rotation_matrix(alpha, settings.md.ENV['ori_live'][0])
        Ry = tf.transformations.rotation_matrix(beta,  settings.md.ENV['ori_live'][1])
        Rz = tf.transformations.rotation_matrix(gamma, settings.md.ENV['ori_live'][2])
        R = tf.transformations.concatenate_matrices(Rx, Ry, Rz)
        euler = tf.transformations.euler_from_matrix(R, 'rxyz')

        [alpha, beta, gamma] = euler
        Rx = tf.transformations.rotation_matrix(alpha, [1,0,0]) #settings.md.ENV['axes'][0])
        Ry = tf.transformations.rotation_matrix(beta,  [0,1,0]) #settings.md.ENV['axes'][1])
        Rz = tf.transformations.rotation_matrix(gamma, [0,0,1]) #settings.md.ENV['axes'][2])
        R = tf.transformations.concatenate_matrices(Rx, Ry, Rz)
        euler = tf.transformations.euler_from_matrix(R, 'rxyz')

        pose_.orientation = Quaternion(*tf.transformations.quaternion_multiply(tf.transformations.quaternion_from_euler(*euler), settings.extq(settings.md.ENV['ori'])))

        if settings.ORIENTATION_MODE == 'fixed':
            pose_.orientation = settings.md.ENV['ori']

        # only for this situtaiton
        return pose_

    def transformSceneToUI(self, pose, view='view'):
        ''' Scene -> rViz -> UI
        '''
        pose_ = Pose()
        pose_.orientation = pose.orientation
        p = Point(pose.position.x-settings.md.ENV['start'].x, pose.position.y-settings.md.ENV['start'].y, pose.position.z-settings.md.ENV['start'].z)
        # View transformation
        x = (np.dot([p.x,p.y,p.z], settings.md.ENV[view][0]) )*settings.ui_scale
        y = (np.dot([p.x,p.y,p.z], settings.md.ENV[view][1]) )*settings.ui_scale
        z = (np.dot([p.x,p.y,p.z], settings.md.ENV[view][2]) )*settings.ui_scale
        # Window to center, y is inverted
        pose_.position.x = x + settings.w/2
        pose_.position.y = -y + settings.h
        pose_.position.z = round(-(z-200)/10)
        return pose_

    def transformLeapToUIsimple(self, pose):
        x, y, z = pose.position.x, pose.position.y, pose.position.z
        x_ = 2*x + settings.w/2
        y_ = -2*y + settings.h
        z_ = round(-(z-200)/10)
        pose_ = Pose()
        pose_.orientation = pose.orientation
        pose_.position.x, pose_.position.y, pose_.position.z = x_, y_, z_
        return pose_

    def transformLeapToUI(self, pose):
        ''' Leap -> UI
        '''
        pose_ = self.transformLeapToScene(pose)
        pose__ = self.transformSceneToUI(pose_)
        return pose__
        x, y, z = pose.position.x, pose.position.y, pose.position.z
        x_ = 2*x + settings.w/2
        y_ = -2*y + settings.h
        z_ = round(-(z-200)/10)
        pose_ = Pose()
        pose_.orientation = pose.orientation
        pose_.position.x, pose_.position.y, pose_.position.z = x_, y_, z_
        return pose_

    def relaxik_t(self, pose1):
        ''' All position goals and orientation goals are specified with respect to specified initial configuration.
            -> This function relates, sets goal poses to origin [0.,0.,0.] with orientation pointing up [0.,0.,0.,1.]
        '''
        pose_ = deepcopy(pose1)
        # 1.
        if settings.ROBOT_NAME == 'panda':
            pose_.position.x -= 0.55442+0.04
            pose_.position.y -= 0.0
            pose_.position.z -= 0.62443
            pose_.orientation = Quaternion(*tf.transformations.quaternion_multiply([1.0, 0.0, 0.0, 0.0], settings.extq(pose_.orientation)))
            #pose_.position.z -= 0.926
            #pose_.position.x -= 0.107
        elif settings.ROBOT_NAME == 'iiwa':
            pose_.position.z -= 1.27
        else: raise Exception("Wrong robot name!")
        # 2.
        if settings.ROBOT_NAME == 'iiwa':
            pose_.position.y = -pose_.position.y
        return pose_

    def relaxik_t_inv(self, pose1):
        ''' Additional inverse transformation to relaxik_t()
        '''
        raise Exception("TODO!")
        pose_ = deepcopy(pose1)
        if settings.ROBOT_NAME == 'panda':
            pose_.position.z += 0.926
            pose_.position.x += 0.088
        #pose_.position.z += 1.27 # iiwa
        if settings.ROBOT_NAME == 'iiwa':
            pose_.position.y = -pose_.position.y
        return pose_

    def set_constraints(self):
        ''' Sets trajectory tolerances
        '''
        self.move_group.set_goal_position_tolerance(0.0001)
        self.move_group.set_goal_joint_tolerance(0.0001)
        self.move_group.set_goal_orientation_tolerance(0.001)
        print("the goal tolerance", self.move_group.get_goal_position_tolerance())
        print("the goal tolerance", self.move_group.get_goal_orientation_tolerance())
        print("the goal tolerance", self.move_group.get_goal_joint_tolerance())
        # construct a message
        joint_constraint = moveit_msgs.msg.JointConstraint()
        joint_constraint.joint_name = self.move_group.get_joints()[0]
        joint_constraint.position = 1.0
        joint_constraint.tolerance_above = 0.5
        joint_constraint.tolerance_below = 0.5
        joint_constraint.weight = 1.0

        joint_constraint_list = []
        joint_constraint_list.append(joint_constraint)

        constraint_list = moveit_msgs.msg.Constraints()
        constraint_list.name = 'start_of_travel'
        constraint_list.joint_constraints = joint_constraint_list

        self.move_group.set_path_constraints(constraint_list)
        ## other parts of trajectory
        constraint_list.name = 'middle_of_travel'
        self.move_group.set_path_constraints(constraint_list)
        #
        constraint_list.name = 'end_of_travel'
        self.move_group.set_path_constraints(constraint_list)

    def plan_to_points(self, data, addon='zzz'):
        ''' Plan ROS msg object to points in [[x1,y1,z1],[x2,y2,z2],...[xn,yn,zn]]
        '''
        try:
            data.joint_trajectory.points
        except NameError:
            print(data.joint_trajectory.points[0].positions,"\n","Name Error, Data is in unsupported type: ", type(data))
            return None
        Q = []
        time = []
        points = data.joint_trajectory.points
        if points == []:
            print("error, points empty")
        for point in points:
            p = Raw_Kinematics.forward_kinematics(point.positions)
            Q.append(p)
            time.append(point.time_from_start.to_sec())

        if addon=='normalize':
            Q0 = deepcopy(Q[0])
            for n in range(0, len(Q)):
                Q[n] = np.subtract(Q[n], Q0)
        return Q, time

    def point_in_env(self, point):
        if settings.md.ENV['min'].x <= point[0] <= settings.md.ENV['max'].x:
          if settings.md.ENV['min'].y <= point[1] <= settings.md.ENV['max'].y:
            if settings.md.ENV['min'].z <= point[2] <= settings.md.ENV['max'].z:
                return True
        return False

    def eulerToVector(self, euler):
        ''' Check if there are no exception
        '''
        roll, pitch, yaw = euler
        x = cos(yaw)*cos(pitch)
        y = sin(yaw)*cos(pitch)
        z = sin(pitch)
        return x,y,z


    def inSceneObj(self, point):
        ''' in the zone of a box with length l
            Compatible for: Pose, Point, [x,y,z]
            Cannot be in two zones at once, return id of first zone
        '''
        collisionObjs = []
        if not settings.scene:
            return False
        z = [False] * len(settings.scene.object_poses)
        assert settings.scene, "Scene not published yet"
        if isinstance(point, Pose):
            point = point.position
        if isinstance(point, Point):
            point = [point.x, point.y, point.z]
        for n, pose in enumerate(settings.scene.object_poses):
            zone_point = pose.position

            zone_point = self.PointAdd(zone_point, settings.scene.mesh_trans_origin[n])
            #print(n, ": \n",zone_point.z, "\n" ,settings.scene.object_sizes[n].z, "\n", point[2])
            if settings.scene.object_sizes[n].y > 0.0:
                if zone_point.x <= point[0] <= zone_point.x+settings.scene.object_sizes[n].x:
                  if zone_point.y <= point[1] <= zone_point.y+settings.scene.object_sizes[n].y:
                    if zone_point.z <= point[2] <= zone_point.z+settings.scene.object_sizes[n].z:
                        collisionObjs.append(settings.scene.object_names[n])

            else:
                if zone_point.x <= point[0] <= zone_point.x+settings.scene.object_sizes[n].x:
                  if zone_point.y >= point[1] >= zone_point.y+settings.scene.object_sizes[n].y:
                    if zone_point.z <= point[2] <= zone_point.z+settings.scene.object_sizes[n].z:
                        collisionObjs.append(settings.scene.object_names[n])
            '''
                    else:
                        print("z")
                  else:
                    print("y")
                else:
                  print("x")
            '''
        return collisionObjs


    def points_in_env(self, points):
        points_in_workspace = True
        for n in range(0, len(points)):
            if self.point_in_env(points[n]) == False:
                points_in_workspace = False
        return points_in_workspace

    def cropToLimits(self, joint_states, safety_distance = 0.01):
        ''' Crop the joints if they are near limits
        '''
        for n, joint in enumerate(joint_states):
            if joint_states[n] > settings.upper_lim[n] - safety_distance:
                joint_states[n] = settings.upper_lim[n] - safety_distance
            if joint_states[n] < settings.lower_lim[n] + safety_distance:
                joint_states[n] = settings.lower_lim[n] + safety_distance
        return joint_states


    def cutToNearestPoint(self, robot_traj):
        ''' Old -> new trajectory
            New trajectory will have first point nearest to actual/current joint positions
        '''
        goal_trajectory = JointTrajectory()
        goal_trajectory.header = robot_traj.joint_trajectory.header
        goal_trajectory.joint_names = robot_traj.joint_trajectory.joint_names
        diffPrev = np.inf
        indx = 0
        for n, point in enumerate(robot_traj.joint_trajectory.points):
            diff = np.linalg.norm(settings.joints - point.positions)
            if diff > diffPrev:
                indx = n
                break
            diffPrev = diff
        print("indx", indx)
        for n, point in enumerate(robot_traj.joint_trajectory.points):
            if n > np.ceil(indx):
                #point.time_from_start = point.time_from_start - rospy.Duration(time_from_start)
                goal_trajectory.points.append(deepcopy(point))

        return goal_trajectory

    def plan_and_perform_trajectory_from_joints(self, joint_states=None):
        ''' Performs/Updates/Replaces trajectory with actionlib.

        Parameters:
            joint_states (Float[7]): joint states of the robot
        Global parameters:
            settings.TOPPRA_ON (Bool): Enables toppra planning to trajectory
            self.trajectory_action_perform (Bool): If false, this function does only the planning
        '''
        ## Check two states
        if settings.md._goal:
            # 1. new joint states are near the desired one and the robot is not moving
            if np.sum(np.abs(np.array(joint_states) - np.array(settings.joints))) < 0.1 and np.sum(settings.joints_in_time[-1].velocity) < 0.1:
                print(joint_states)
                self.timePassedPreviousTraj = time.time()
                print("[Info*] Robot on place")
                return
            # 2. The goal does not changed to new one
            if settings.md._goal and (np.sum(np.abs(np.array(settings.md._goal.trajectory.points[-1].positions) - np.array(joint_states))) < 0.1):
                self.timePassedPreviousTraj = time.time()
                print("[Info*] Goal has not changed")
                return

        assert len(joint_states)==7, "Not right datatype"+str(type(joint_states))
        js2_joints = np.array(joint_states) # desired
        if not settings.md._goal:
            js1_joints = settings.joints
            goal = FollowJointTrajectoryGoal()
            goal.trajectory.joint_names = settings.JOINT_NAMES
            joints = JointTrajectoryPoint()
            joints.positions = js1_joints
            joints.time_from_start = rospy.Time(0.)
            goal.trajectory.points.append(deepcopy(joints))

            joints.positions = js2_joints
            joints.time_from_start = rospy.Time(3.)
            goal.trajectory.points.append(deepcopy(joints))


            point_before_toppra = [rospy.Time.now().to_sec(), settings.joints[settings.NJ]]
            if settings.TOPPRA_ON:
                goal.trajectory = self.retime_wrapper(goal.trajectory)
                print("n points: ", len(goal.trajectory.points), " joints diff: ", round(sum(js2_joints-js1_joints),2), " pose diff: ", round(self.distancePoses(settings.eef_pose, settings.goal_pose),2))
            goal.trajectory.header.stamp = rospy.Time.now()
            settings.md._goal = goal
            
            if self.trajectory_action_perform:
                self.tac.add_goal(goal)
                self.tac.replace()
            point_after_toppra = [rospy.Time.now().to_sec(), settings.joints[settings.NJ]]

        else:
            js0 = settings.md._goal.trajectory.points[0]
            # 1. choose t* > tc => find js1
            computation_time = settings.md.traj_update_horizon # time horizon of new trajectory
            t_s = settings.md._goal.trajectory.header.stamp
            t_now = rospy.Time.now()
            dt_now = t_now - t_s
            def findPointInTrajAfterThan(trajectory, computation_time):
                assert type(computation_time)==float, "Wrong type"
                assert type(trajectory)==type(JointTrajectory()), "Wrong type"
                for n, point in enumerate(trajectory.points):
                    if point.time_from_start.to_sec() > computation_time+dt_now.to_sec():
                        return n
                return len(trajectory.points)-1
            index = findPointInTrajAfterThan(settings.md._goal.trajectory, computation_time)
            js1 = settings.md._goal.trajectory.points[index]
            js1_joints = js1.positions

            # 2. make RobotTrajectory from js1 -> js2
            # with js2 velocity according to hand velocity (gesture) (acc is zero)
            goal = FollowJointTrajectoryGoal()
            goal.trajectory.joint_names = settings.JOINT_NAMES
            goal.trajectory.header.stamp = settings.md._goal.trajectory.header.stamp
            joints = JointTrajectoryPoint()
            joints.positions = js1_joints
            joints.time_from_start = rospy.Time(0.)
            goal.trajectory.points.append(deepcopy(joints))

            joints.positions = js2_joints
            joints.time_from_start = rospy.Time(1.)
            goal.trajectory.points.append(deepcopy(joints))
            point_before_toppra = [rospy.Time.now().to_sec(), settings.joints[settings.NJ]]
            if settings.TOPPRA_ON:
                goal.trajectory = self.retime_wrapper(goal.trajectory)
                print("n points: ", len(goal.trajectory.points), " joints diff: ", round(sum(js2_joints-js1_joints),2), " pose diff: ", round(self.distancePoses(settings.eef_pose, settings.goal_pose),2), "index ", index," max t ", goal.trajectory.points[-1].time_from_start.to_sec())

            # 3. combine with trajectory from js0 -> js1 -> js2
            settings.md._goal.trajectory.points = settings.md._goal.trajectory.points[0:index+1]
            tts = settings.md._goal.trajectory.points[index].time_from_start
            for n,point in enumerate(goal.trajectory.points):
                if n >= 1:
                    point.time_from_start = rospy.Time.from_sec(point.time_from_start.to_sec() + tts.to_sec())
                    settings.md._goal.trajectory.points.append(point)
            print("pos real ", settings.joints[0], " pose plan ", settings.md._goal.trajectory.points[index+1].positions[0])

            point_after_toppra = [rospy.Time.now().to_sec(), settings.joints[settings.NJ]]
            # 4. replace with action client
            print("traj point 1", settings.md._goal.trajectory.points[0].positions)
            ''' TEMPORARY (I will try reject everything before)
            '''
            def findPointInTrajAfterNow(trajectory):
                for n, point in enumerate(trajectory.points):
                    if point.time_from_start.to_sec()+trajectory.header.stamp.to_sec() > rospy.Time.now().to_sec():
                        return n
                return len(trajectory.points)-1
            index = findPointInTrajAfterNow(settings.md._goal.trajectory)
            def zeroTimeFromStart():
                time0 = deepcopy(settings.md._goal.trajectory.points[0].time_from_start)
                for n, pt in enumerate(settings.md._goal.trajectory.points):
                    pt.time_from_start = pt.time_from_start-time0
                    if n < 10:
                        pt.positions = np.add(np.multiply(((n) / 10.), np.array(pt.positions)), np.multiply(((10-n) / 10.), np.array(settings.joints)))

            settings.md._goal.trajectory.points = settings.md._goal.trajectory.points[index-2:]
            settings.md._goal.trajectory.header.stamp = rospy.Time.now()
            zeroTimeFromStart()

            if self.trajectory_action_perform:
                self.tac.add_goal(deepcopy(settings.md._goal))
                self.tac.replace()

        point_after_replace = [rospy.Time.now().to_sec(), settings.joints[settings.NJ]]
        # 5. (OPTIONAL) Save visualization data


        dataPlot = [pt.positions[settings.NJ] for pt in settings.md._goal.trajectory.points]
        timePlot = [pt.time_from_start.to_sec()+settings.md._goal.trajectory.header.stamp.to_sec() for pt in settings.md._goal.trajectory.points]
        settings.sendedPlot = zip(timePlot, dataPlot)

        dataJointPlot = [pt.position[settings.NJ] for pt in list(settings.joints_in_time)]
        timeJointPlot = [pt.header.stamp.to_sec() for pt in list(settings.joints_in_time)]
        settings.realPlot = zip(timeJointPlot, dataJointPlot)

        settings.point_before_toppra = point_before_toppra
        settings.point_after_toppra = point_after_toppra
        settings.point_after_replace = point_after_replace

        timePlotVel = [pt.time_from_start.to_sec()+settings.md._goal.trajectory.header.stamp.to_sec() for pt in settings.md._goal.trajectory.points]
        try:
            dataPlotVel = [pt.velocities[settings.NJ] for pt in settings.md._goal.trajectory.points]
        except IndexError:
            try:
                print("@@@@@@@@@@@@@@@@@@@",settings.md._goal.trajectory.points[:].velocities)
            except AttributeError:
                print("atrivbuete error")
                dataPlotVel = [0.] * len(timePlotVel)
        settings.sendedPlotVel = zip(timePlotVel, dataPlotVel)
        timeJointPlotVel = [pt.header.stamp.to_sec() for pt in list(settings.joints_in_time)]
        dataJointPlotVel = [pt.velocity[settings.NJ] for pt in list(settings.joints_in_time)]
        settings.realPlotVel = zip(timeJointPlotVel, dataJointPlotVel)


    def plotPosesCallViz(self, load_data=True):
        ''' Visualize data + show. Loading series from:
            - eef poses: settings.dataPosePlot
            - goal poses: settings.dataPoseGoalsPlot

        Parameters:
            load_data (bool): Loads the data from:
                - eef poses: settings.eef_pose_array
                - goal poses: settings.goal_pose_array
        '''

        if load_data:
            settings.dataPosePlot = [pt.position for pt in list(settings.eef_pose_array)]
            settings.dataPoseGoalsPlot = [pt.position for pt in list(settings.goal_pose_array)]

        if not settings.dataPosePlot:
            print("[ERROR*] No data when plotting poses were found, probably call with param: load_data=True")
            return

        # Plot positions
        settings.viz.visualize_new_fig(title="Trajectory executed - vis. poses of panda eef:", dim=3)
        settings.viz.visualize_3d(data=settings.dataPosePlot, color='b', label="Real trajectory poses")
        settings.viz.visualize_3d(data=settings.dataPoseGoalsPlot, color='r', label="Goal poses")


    def plotJointsCallViz(self, load_data=False, plotToppraPlan=False, plotVelocities=True, plotAccelerations=False, plotEfforts=False):
        ''' Visualize data + show. Loading series from:
                - Sended trajectory values: settings.sendedPlot, settings.sendedPlotVel
                - The joint states values: settings.realPlot, settings.realPlotVel
                - Section of toppra execution, start/end pts: [settings.point_before_toppra, settings.point_after_toppra]
                - Section of trajectory replacement, start/end pts: [settings.point_after_toppra, settings.point_after_replace]
            Note: Every plot visualization takes ~200ms

        Parameters:
            load_data (bool): Loads joint_states positions and velocities to get up-to-date trajectories
            plotVelocities (bool): Plots velocities
            plotToppraPlan (bool): Plots toppra RobotTrajectory plan
            plotAccelerations (bool): Plots accelerations
            plotEfforts (bool): Plots efforts
        '''
        # Load/Update Data
        if load_data or settings.SIMULATOR_NAME == 'coppelia':
            dataJointPlot = [pt.position[settings.NJ] for pt in list(settings.joints_in_time)]
            timeJointPlot = [pt.header.stamp.to_sec() for pt in list(settings.joints_in_time)]
            settings.realPlot = zip(timeJointPlot, dataJointPlot)
            timeJointPlotVel = [pt.header.stamp.to_sec() for pt in list(settings.joints_in_time)]
            dataJointPlotVel = [pt.velocity[settings.NJ] for pt in list(settings.joints_in_time)]
            settings.realPlotVel = zip(timeJointPlotVel, dataJointPlotVel)

        # Plot positions
        settings.viz.visualize_new_fig(title="Trajectory number "+str(settings.loopn)+" executed - vis. position of panda_joint"+str(settings.NJ+1), dim=2)
        if settings.ROBOT_NAME == 'panda' and (settings.SIMULATOR_NAME == 'gazebo' or settings.SIMULATOR_NAME == 'real'):
            settings.viz.visualize_2d(data=settings.sendedPlot, color='r', label="Replaced (sended) trajectory position", scatter_pts=True)
            settings.viz.visualize_2d(data=[settings.point_before_toppra, settings.point_after_toppra], color='y', label="Toppra executing")
            settings.viz.visualize_2d(data=[settings.point_after_toppra, settings.point_after_replace], color='k', label="Replace executing")
        else:
            pass
        settings.viz.visualize_2d(data=settings.realPlot, color='b', label="Real (joint states) trajectory position", xlabel='time (global rospy) [s]', ylabel='joint positons [rad]')

        # Plot velocities
        if plotVelocities:
            settings.viz.visualize_new_fig(title="Trajectory number "+str(settings.loopn)+" executed - vis. velocity of panda_joint"+str(settings.NJ+1), dim=2)

            if settings.ROBOT_NAME == 'panda' and (settings.SIMULATOR_NAME == 'gazebo' or settings.SIMULATOR_NAME == 'real'):
                settings.viz.visualize_2d(data=settings.sendedPlotVel, color='r', label="Replaced (sended) trajectory velocity", scatter_pts=True)
            else:
                pass
            settings.viz.visualize_2d(data=settings.realPlotVel, color='b', label="Real (states) velocity", xlabel='time (global rospy) [s]', ylabel='joint velocities [rad/s]')

        # Plot accelerations
        if plotAccelerations:
            dataPlot = [pt.accelerations[settings.NJ] for pt in settings.md._goal.trajectory.points]
            timePlot = [pt.time_from_start.to_sec()+settings.md._goal.trajectory.header.stamp.to_sec() for pt in settings.md._goal.trajectory.points]
            timeJointPlot = [pt.header.stamp.to_sec() for pt in list(settings.joints_in_time)]
            dataJointPlot = [pt.effort[settings.NJ] for pt in list(settings.joints_in_time)]
            settings.figdata = visualizer_lib.visualize_new_fig(title="Loop"+str(settings.loopn)+" ACC", dim=2)
            settings.viz.visualize_2d(data=zip(timePlot, dataPlot), color='r', label="sended trajectory accelerations", transform='front')
            settings.viz.visualize_2d(data=zip(timeJointPlot, dataJointPlot), color='b', label="real efforts")

        # Plot efforts
        if plotEfforts:
            #dataPlot = [pt.effort[settings.NJ] for pt in settings.md._goal.trajectory.points]
            timePlot = [pt.time_from_start.to_sec()+settings.md._goal.trajectory.header.stamp.to_sec() for pt in settings.md._goal.trajectory.points]
            timeJointPlot = [pt.header.stamp.to_sec() for pt in list(settings.joints_in_time)]
            dataJointPlot = [pt.effort[settings.NJ] for pt in list(settings.joints_in_time)]
            settings.viz.visualize_new_fig(title="Path", dim=2)
            #settings.viz.visualize_2d(data=zip(timePlot, dataPlot), color='r', label="sended trajectory effort")
            settings.viz.visualizer_lib.visualize_2d(data=zip(timeJointPlot, dataJointPlot), color='b', label="real effort")

        if plotToppraPlan:
            self.plot_plan(plan=settings.toppraPlan)

    def retime_wrapper(self, trajectory):
        ''' Calls retime function which takes RobotTrajectory and returns RobotTrajectory.
            This function wraps it to take JointTrajectory nad return JointTrajectory
        Parameters:
            trajectory (JointTrajectory())
        Returns:
            robot_traj_new (JointTrajectory())
        '''
        robot_traj = RobotTrajectory()
        robot_traj.joint_trajectory = deepcopy(trajectory)
        robot_traj_new = self.retime(plan=robot_traj)
        settings.toppraPlan = deepcopy(robot_traj_new)
        return robot_traj_new.joint_trajectory

    def plot_plan(self, plan, title='', save='', show=True, ret=False):
        ''' Plots toppra plan.
        '''
        ss = np.array([pt.time_from_start.to_sec() for pt in plan.joint_trajectory.points])
        way_pts = np.array([list(pt.positions) for pt in plan.joint_trajectory.points])
        vel = np.array([list(pt.velocities) for pt in plan.joint_trajectory.points])
        acc = np.array([list(pt.accelerations) for pt in plan.joint_trajectory.points])

        #print("ss: ", ss)
        #print("way_pts", way_pts)
        #print("vel: ", vel)
        #print("acc: ", acc)
        #eef_vel = np.array([np.linalg.norm(np.dot(panda_jacobian(pt),v)) for pt, v in zip(way_pts, vel)])


        fig, axs = plt.subplots(4, 1, sharex=True)
        fig.suptitle(title, fontsize=16)
        for i in range(way_pts.shape[1]):
            # plot the i-th joint trajectory
            axs[0].plot(ss, way_pts[:, i], c="C{:d}".format(i))
            axs[1].plot(ss, vel[:, i], c="C{:d}".format(i))
            axs[2].plot(ss, acc[:, i], c="C{:d}".format(i))
        #axs[3].plot(ss, eef_vel, c="r")
        axs[2].set_xlabel("Time (s)")
        axs[0].set_ylabel("Position (rad)")
        axs[1].set_ylabel("Velocity (rad/s)")
        axs[2].set_ylabel("Acceleration (rad/s2)")
        #axs[3].set_ylabel("EEF Speed (m/s)")
        if len(save) > 0:
            import os
            dir = os.path.split(save)[0]
            os.path.isdir(dir)
            fig.savefig(fname=save)
        if show:
            plt.show()
        if ret:
            return fig, axs

    ###
    ### From Jan Kristof Behrens
    ###
    def retime(self, plan=None, cart_vel_limit=-1.0, secondorder=False, pt_per_s=20, curve_len=None, start_delay=0.0, traj_duration=3):
        #ta.setup_logging("INFO")
        assert isinstance(plan, RobotTrajectory)

        if not curve_len is None and cart_vel_limit > 0:
            n_grid = np.ceil(pt_per_s * curve_len / cart_vel_limit)
        else:
            n_grid = np.inf

        active_joints = plan.joint_trajectory.joint_names
        # ALERT: This function is not found
        #lower, upper, vel, effort = self.robot.get_joint_limits(active_joints)
        lower, upper, vel, effort = settings.lower_lim, settings.upper_lim, settings.vel_lim, settings.effort_lim
        # prepare numpy arrays with limits for acceleration
        alims = np.zeros((len(active_joints), 2))
        alims[:, 1] = np.array(len(lower) * [3.0]) # 0.5
        alims[:, 0] = np.array(len(lower) * [-3.0])

        # ... and velocity
        vlims = np.zeros((len(active_joints), 2))
        vlims[:, 1] = np.array(vel)
        vlims[:, 0] = (-1.0) * np.array(vel)

        use_cart_vel_limit = False
        if cart_vel_limit > 0:
            use_cart_vel_limit = True

        ss = [pt.time_from_start.to_sec() for pt in plan.joint_trajectory.points]
        way_pts = [list(pt.positions) for pt in plan.joint_trajectory.points]

        path = SplineInterpolator(ss, way_pts)
        pc_vel = ta.constraint.JointVelocityConstraint(vlim=vlims)

        #path()

        def vlims_func(val):
            eps = 0.001
            limit = cart_vel_limit

            J = Raw_Kinematics.jacobian(path(val))
            direction = (path(val + eps) - path(val - eps)) / (2* eps)
            dir_norm = direction / np.linalg.norm(direction)
            x = limit / np.linalg.norm(np.dot(J, dir_norm))
            x = x * dir_norm

            x = np.abs(x)
            #print("{}: {}".format(val, np.max(x)))
            lim = np.zeros((7, 2))
            lim[:, 1] = np.max(x)
            # if val <= 2.5:
            #     lim = np.zeros((7,2))
            #     lim[:,1] = np.max(x)
            # else:
            #     lim = np.zeros((7, 2))
            #     lim[:, 1] = np.array(7 * [1.0])

            lim[:, 0] = -lim[:,1]
            return lim

        pc_vel2 = ta.constraint.JointVelocityConstraintVarying(vlim_func=vlims_func)
        # pc_vel2.discretization_type = DiscretizationType.Interpolation

        pc_acc = ta.constraint.JointAccelerationConstraint(alim=alims)

        # def inv_dyn(q, qd, qgg):
        #     # use forward kinematic formula and autodiff to get jacobian, then calc velocities from jacobian and joint
        #     # velocities
        #     J = panda_jacobian(q)
        #     cart_vel = np.dot(J, qd)
        #     return np.linalg.norm(cart_vel)
        #
        # def g(q):
        #     return ([-0.2, 0.2])
        #
        # def F(q):
        #     return np.eye(1)
        #

        if secondorder:
            def my_inv_dyn(q, qd, qgg):
                # use forward kinematic formula and autodiff to get jacobian, then calc velocities from jacobian and joint
                # velocities
                J = panda_jacobian(q)
                cart_vel_sq = np.dot(np.dot(qd.T, J.T), np.dot(J, qd))

                print(cart_vel_sq)
                return np.array(len(qd) * [10000 * cart_vel_sq])
            def my_g(q):
                return np.array(len(q) * [10000 * cart_vel_limit**2])
            def my_F(q):
                return 1 * np.eye(len(q))

            eef_vel = ta.constraint.SecondOrderConstraint(inv_dyn=my_inv_dyn, constraint_F=my_F, constraint_g=my_g, dof=7,
                                                          discretization_scheme=DiscretizationType.Interpolation)
            instance = ta.algorithm.TOPPRA([pc_vel, pc_acc, eef_vel], path, solver_wrapper='seidel', gridpoints=np.linspace(0.0, ss[-1], np.min([int(n_grid), np.ceil(pt_per_s*ss[-1])])))
            # instance = ta.algorithm.TOPPRA([eef_vel], path, solver_wrapper='seidel')
        elif False:
            def my_inv_dyn(q, qd, qgg):
                # use forward kinematic formula and autodiff to get jacobian, then calc velocities from jacobian and joint
                # velocities
                J = panda_jacobian(q)
                cart_vel = np.dot(J, qd)

                print(np.linalg.norm(cart_vel))
                return np.array(len(qd) * [100 * np.linalg.norm(cart_vel)])

            def my_g(q):
                return np.array(len(q) * [100 * cart_vel_limit])
            def my_F(q):
                return 1 * np.eye(len(q))

            eef_vel = ta.constraint.SecondOrderConstraint(inv_dyn=my_inv_dyn, constraint_F=my_F, constraint_g=my_g, dof=7,
                                                          discretization_scheme=DiscretizationType.Collocation)
            instance = ta.algorithm.TOPPRA([pc_vel, pc_acc, eef_vel], path, solver_wrapper='seidel')
            # instance = ta.algorithm.TOPPRA([eef_vel], path, solver_wrapper='seidel')
        else:
            #print("gg: ", n_grid)
            #print("N : ", np.ceil(pt_per_s*ss[-1]))
            #print("A : ", np.linspace(0.0, ss[-1], np.min([n_grid, int(np.ceil(pt_per_s*ss[-1]))])))
            instance = ta.algorithm.TOPPRAsd([pc_vel, pc_vel2, pc_acc], path, solver_wrapper='seidel', gridpoints=np.linspace(0.0, ss[-1], int(np.min([n_grid, int(np.ceil(pt_per_s*ss[-1]))]))))


        ''' Original function edited primarily HERE
        '''
        '''
        instance.set_desired_duration(traj_duration)
        pathVelocityNow = self.getPathVelocityNow()
        jnt_traj = self.extractToppraTraj(instance.compute_trajectory(pathVelocityNow, 0.))
        if not jnt_traj: return plan
        _, sd_vec, _ = instance.compute_parameterization(pathVelocityNow, 0.)
        print("sd_vec ", sd_vec)
        self.savedVelocityVec = sd_vec # update vals
        self.durationPreviousTraj = jnt_traj.duration
        '''
        ''' END Edit
        '''
        instance.set_desired_duration(traj_duration)
        jnt_traj = self.extractToppraTraj(instance.compute_trajectory(0., 0.))
        '''
        '''

        feas_set = instance.compute_feasible_sets()

        # ts_sample = np.linspace(0, jnt_traj.duration, 10*len(plan.joint_trajectory.points))
        ts_sample = np.linspace(0, jnt_traj.duration, int(np.ceil(100 * jnt_traj.duration)))
        qs_sample = jnt_traj(ts_sample)
        qds_sample = jnt_traj(ts_sample, 1)
        qdds_sample = jnt_traj(ts_sample, 2)


        new_plan = deepcopy(plan)
        new_plan.joint_trajectory.points = []
        for t, q, qd, qdd in zip(ts_sample, qs_sample, qds_sample, qdds_sample):
            pt = JointTrajectoryPoint()
            pt.time_from_start = rospy.Duration.from_sec(t + start_delay)
            pt.positions = q
            pt.velocities = qd
            pt.accelerations = qdd
            new_plan.joint_trajectory.points.append(pt)

        if rospy.get_param('plot_joint_trajectory', default=False):


            fig, axs = plt.subplots(3, 1, sharex=True)
            for i in range(path.dof):
                # plot the i-th joint trajectory
                axs[0].plot(ts_sample, qs_sample[:, i], c="C{:d}".format(i))
                axs[1].plot(ts_sample, qds_sample[:, i], c="C{:d}".format(i))
                axs[2].plot(ts_sample, qdds_sample[:, i], c="C{:d}".format(i))
            axs[2].set_xlabel("Time (s)")
            axs[0].set_ylabel("Position (rad)")
            axs[1].set_ylabel("Velocity (rad/s)")
            axs[2].set_ylabel("Acceleration (rad/s2)")
            plt.show()
        return new_plan

    def extractToppraTraj(self, jnt_traj_):
        ''' Toppra function compute_trajectory() sometimes returns the tuple,
            which usually contains e.g. [0, traj], this function will just return the traj itself
        '''
        jnt_traj = deepcopy(jnt_traj_)
        succ = False
        if type(jnt_traj_) == tuple:
            for traj in jnt_traj_:
                if traj is not None:
                    jnt_traj = traj
                    succ = True
                    break
        if succ:
            return jnt_traj
        if not succ:
            print("[WARN*] Toppra FAILED! No trajectory found")
            return None # will return old plan, if no plan as result

    def getPathVelocityNow(self):
        ''' Returns the path velocity now
        '''
        if self.timePassedPreviousTraj: #
            # Load previous plan path velocity series
            pathVelocitySeriesPrevous = self.savedVelocityVec
            # Create interpolation object from the series
            f = interpolate.interp1d(np.linspace(0, 1, len(pathVelocitySeriesPrevous)), pathVelocitySeriesPrevous)
            # time which passed from before
            previousPathTrajectoryDuration = time.time()-self.timePassedPreviousTraj
            # In previous planned path, get (normalized) distance (to previous path duration), how much time passed from before
            normDistFromTrajStart = previousPathTrajectoryDuration/self.durationPreviousTraj
            print("normDistFromTrajStart", normDistFromTrajStart)
            # Get the final path velocity from now
            if normDistFromTrajStart > 1.0: normDistFromTrajStart = 1.0
            pathVelocityNow = f(normDistFromTrajStart)
        else:
            pathVelocityNow = 0.0

        # update the values
        self.timePassedPreviousTraj = time.time()
        return pathVelocityNow


    def scale_speed(self, scaling_factor=1.0):
        '''
        This function scales the execution speed of the remaining trajectory according to the scaling factor. This works only before or during an execution,
        :param scaling_factor: A relative scaling factor for the remaining trajectory. The remaining time to each trajectory point is divided by it.
        :return: None
        '''
        #if self._client.get_state() not in [0, 1]:
        #    rospy.logerr("No trajectory execution in process nor pending.")
        #    return

        if scaling_factor < 0.01 or scaling_factor > 100:
            rospy.logerr("scaling_factor out of range [0.01 .. 10]: {}".format(scaling_factor))
            rospy.logerr("I don't scale the trajectory.")
            return
        t_s = settings.md._goal.trajectory.header.stamp
        t_now = rospy.Time.now()
        dt_now = t_now - t_s
        dt_now_ = rospy.Time(secs=dt_now.secs, nsecs=dt_now.nsecs)
        # create new goal with all non-passed TrajectoryPoints
        new_traj_goal = deepcopy(settings.md._goal)
        # type: FollowJointTrajectoryGoal
        for index, tp in enumerate(new_traj_goal.trajectory.points):
            # type: int, JointTrajectoryPoint
            print(type(tp.time_from_start), type(dt_now_))
            if tp.time_from_start >= dt_now_:
                new_traj_goal.trajectory.points = new_traj_goal.trajectory.points[index:]
                break
        USE_MOVEIT = True
        if USE_MOVEIT:
            plan = RobotTrajectory()
            plan.joint_trajectory = new_traj_goal.trajectory
            state = self.robot.get_current_state()
            new_plan = self.move_group.retime_trajectory(state, plan, scaling_factor)
            new_traj_goal.trajectory = new_plan.joint_trajectory
        else:
            # scale the time from now to each tp according to scaling_factor
            for index, tp in enumerate(new_traj_goal.trajectory.points):
                # type: int, JointTrajectoryPoint
                #     t_new = t_now + (tp.time_from_start - dt_now) / scaling_factor
                t_new_from_now = (tp.time_from_start - dt_now) / scaling_factor
                tp.time_from_start = t_new_from_now
                velocity_list = list(tp.velocities)
                for index, vel in enumerate(velocity_list):
                    velocity_list[index] = vel * scaling_factor
                tp.velocities = tuple(velocity_list)

        new_traj_goal.trajectory.header.stamp = rospy.Time.now()
        self._goal = new_traj_goal
        self._client.send_goal(new_traj_goal)

    def curve_shift():
        curve_shifted = curve - curve.mean(axis=0) + np.array([0.0, 0.0, 0.0, 1.0])
        #                       shift curve mean to origin, and then every point by this amount in x,y,z

    def get_ik(self, pose=None):
        ''' Uses kinematics interface.
            If pose not specified, returns from current pose.

        Parameters:
            pose (Pose()): From ROS msg
        Returns:
            joints (Float[7]): Robot joint values
        '''
        if not pose:
            pose = self.move_group.get_current_pose()
        IK_resp = ik_bridge.getIKmoveit(pose)
        if IK_resp.error_code.val != 1:
            print("[Kinematics Interface] IKT not found!",IK_resp.solution.joint_state.position)
        return IK_resp.solution.joint_state.position[0:7] # robot1


    def ik_node_publish(self, pose_r=None, pose_l=None):
        ''' Sends goal poses to topic '/relaxed_ik/ee_pose_goals'.
            Inverse kinematics is solved based on 'ik_soler' ROSparam.
                - 'relaxed_ik' -> Picked up by relaxedIK node
                - 'pyrep' -> Picked up by "coppelia_sim.py"
        Parameters:
            pose_r (Pose()): Object from ROS msg. Primary arm.
            pose_l (Pose()): Object from ROS msg. For future, secord arm.
        '''
        position_r, rotation_r, position_l, rotation_l = None, None, None, None
        if pose_r is not None:
            position_r = [pose_r.position.x, pose_r.position.y, pose_r.position.z]
            rotation_r = [pose_r.orientation.w, pose_r.orientation.x, pose_r.orientation.y, pose_r.orientation.z]
        if pose_l is not None:
            position_l = [pose_l.position.x, pose_l.position.y, pose_l.position.z]
            rotation_l = [pose_l.orientation.w, pose_l.orientation.x, pose_l.orientation.y, pose_l.orientation.z]

        if position_r:
            pose = PoseStamped()
            pose.pose.position.x = position_r[0]
            pose.pose.position.y = position_r[1]
            pose.pose.position.z = position_r[2]

            pose.pose.orientation.w = rotation_r[0]
            pose.pose.orientation.x = rotation_r[1]
            pose.pose.orientation.y = rotation_r[2]
            pose.pose.orientation.z = rotation_r[3]
            self.ik_goal_r_pub.publish(pose)

        if position_l:
            pose = PoseStamped()
            pose.pose.position.x = position_l[0]
            pose.pose.position.y = position_l[1]
            pose.pose.position.z = position_l[2]

            pose.pose.orientation.w = rotation_l[0]
            pose.pose.orientation.x = rotation_l[1]
            pose.pose.orientation.y = rotation_l[2]
            pose.pose.orientation.z = rotation_l[3]
            self.ik_goal_l_pub.publish(pose)

        if position_r:
            pos_goal = Vector3Stamped()
            pos_goal.vector.x = position_r[0]
            pos_goal.vector.y = position_r[1]
            pos_goal.vector.z = position_r[2]
            self.goal_pos_pub.publish(pos_goal)

            quat_goal = QuaternionStamped()
            quat_goal.quaternion.w = rotation_r[0]
            quat_goal.quaternion.x = rotation_r[1]
            quat_goal.quaternion.y = rotation_r[2]
            quat_goal.quaternion.z = rotation_r[3]
            self.goal_quat_pub.publish(quat_goal)

        ee_pose_goals = EEPoseGoals()
        if position_r:
            pose_r = Pose()
            pose_r.position.x = position_r[0]
            pose_r.position.y = position_r[1]
            pose_r.position.z = position_r[2]

            pose_r.orientation.w = rotation_r[0]
            pose_r.orientation.x = rotation_r[1]
            pose_r.orientation.y = rotation_r[2]
            pose_r.orientation.z = rotation_r[3]
            ee_pose_goals.ee_poses.append(pose_r)

        if position_l:
            pose_l = Pose()
            pose_l.position.x = position_l[0]
            pose_l.position.y = position_l[1]
            pose_l.position.z = position_l[2]

            pose_l.orientation.w = rotation_l[0]
            pose_l.orientation.x = rotation_l[1]
            pose_l.orientation.y = rotation_l[2]
            pose_l.orientation.z = rotation_l[3]
            ee_pose_goals.ee_poses.append(pose_l)

        if position_r or position_l:
            ee_pose_goals.header.seq = self.seq
            self.seq += 1

            self.ee_pose_goals_pub.publish(ee_pose_goals)

        q = Bool()
        q.data = False
        self.quit_pub.publish(q)

    def changePlayPath(self, path_=None):
        for n, path in enumerate(settings.sp):
            if not path_ or path.NAME == path_: # pick first path if path_ not given
                self.make_scene(scene=path.scene)
                settings.md.PickedPath = n
                settings.md.ENV = settings.md.ENV_DAT[path.ENV]
                settings.HoldValue = 0
                settings.currentPose = 0
                settings.goal_pose = deepcopy(settings.sp[1].poses[1])
                break

    def changeLiveMode(self, text):

        # Reset Gestures
        settings.gestures_goal_pose = Pose()
        settings.gestures_goal_pose.position = deepcopy(settings.md.ENV['start'])
        settings.gestures_goal_pose.orientation.w = 1.0
        if text == "Default":
            settings.md.liveMode = 'default'
        elif text == "Gesture based":
            settings.md.liveMode = 'gesture'
        elif text == "Interactive":
            settings.md.liveMode = 'interactive'

    def gestureGoalPoseUpdate(self, toggle, move):
        if abs(settings.frames[-1].timestamp - settings.frames_adv[-1].r.time_last_stop) < 200000:
            pass
        else:
            #print("blocked", settings.frames[-1].timestamp, settings.frames_adv[-1].r.time_last_stop)
            return

        move_ = 1 if move else -1
        move_ = move_ * settings.md.gestures_goal_stride

        pt = [0.] * 3
        pt[toggle] = move_
        # switch axes a) Leap
        Pt_a = [0.] * 3
        Pt_a[0] = np.dot(pt, settings.md.LEAP_AXES[0])
        Pt_a[1] = np.dot(pt, settings.md.LEAP_AXES[1])
        Pt_a[2] = np.dot(pt, settings.md.LEAP_AXES[2])
        # b) Scene
        Pt_b = [0.] * 3
        Pt_b[0] = np.dot(Pt_a, settings.md.ENV['axes'][0])
        Pt_b[1] = np.dot(Pt_a, settings.md.ENV['axes'][1])
        Pt_b[2] = np.dot(Pt_a, settings.md.ENV['axes'][2])

        # axis swtich
        settings.md.gestures_goal_pose.position = self.PointAdd(settings.md.gestures_goal_pose.position, Point(*Pt_b))

    def gestureGoalPoseRotUpdate(self, toggle, move):
        if abs(settings.frames[-1].timestamp - settings.frames_adv[-1].r.time_last_stop) < 200000:
            pass
        else:
            #print("blocked", settings.frames[-1].timestamp, settings.frames_adv[-1].r.time_last_stop)
            return
        move_ = 1 if move else -1
        move_ = move_ * settings.md.gestures_goal_rot_stride

        pt = [0.] * 3
        pt[toggle] = move_

        Pt_b = [0.] * 3
        Pt_b[0] = np.dot(pt, settings.md.ENV['ori_axes'][0])
        Pt_b[1] = np.dot(pt, settings.md.ENV['ori_axes'][1])
        Pt_b[2] = np.dot(pt, settings.md.ENV['ori_axes'][2])

        # axis swtich
        o = settings.md.gestures_goal_pose.orientation
        euler = list(tf.transformations.euler_from_quaternion([o.x, o.y, o.z, o.w]))
        euler[0] += Pt_b[0]
        euler[1] += Pt_b[1]
        euler[2] += Pt_b[2]
        settings.md.gestures_goal_pose.orientation = Quaternion(*tf.transformations.quaternion_from_euler(euler[0],euler[1],euler[2]))




    def PointAdd(self, p1, p2):
        assert (isinstance(p1, Point) or isinstance(p1, Vector3)) and (isinstance(p2, Point) or isinstance(p2, Vector3)), "Datatype assert, not Point or Vector3"
        return Point(p1.x+p2.x, p1.y+p2.y, p1.z+p2.z)




        def plot_examples(dists, xs, ys, name):
            """
            helper function to plot two colormaps
            """
            viridis = cm.get_cmap('viridis', 256)
            newcolors = viridis(np.linspace(0, 1, 256))
            pink = np.array([248/256, 24/256, 148/256, 1])
            newcolors[:25, :] = pink
            newcmp = ListedColormap(newcolors)
            cms = [viridis, newcmp]

            fig, axs = plt.subplots(1, 2, figsize=(6, 3), constrained_layout=True)
            for [ax, cmap] in zip(axs, cms):
                psm = ax.pcolormesh(dists, cmap=cmap, rasterized=True, vmin=0, vmax=0.1)
                fig.colorbar(psm, ax=ax)

            plt.sca(axs[0])
            plt.xticks(range(len(xs)), xs)
            plt.yticks(range(len(ys)), ys)
            plt.sca(axs[1])
            plt.xticks(range(len(xs)), xs)
            plt.yticks(range(len(ys)), ys)

            plt.savefig(settings.PLOTS_PATH+name+'.eps', format='eps')
            plt.show()

        plot_examples(dists, xs, ys, name)

    def advancedWait(self):
        ''' Desired settings.goal_pose waiting for real settings.eef_pose

        '''
        time.sleep(2)
        if not self.samePoses(settings.eef_pose, settings.goal_pose, accuracy=0.02):
            print("Pose diff: ", round(self.distancePoses(settings.eef_pose, settings.goal_pose),2))
        else: return
        time.sleep(2)
        if not self.samePoses(settings.eef_pose, settings.goal_pose, accuracy=0.06):
            print("Pose diff: ", round(self.distancePoses(settings.eef_pose, settings.goal_pose),2))
        else: return
        time.sleep(2)
        if not self.samePoses(settings.eef_pose, settings.goal_pose, accuracy=0.15):
            print("Pose diff: ", round(self.distancePoses(settings.eef_pose, settings.goal_pose),2))
            print("[WARN*] Position not accurate")
        else: return
        time.sleep(5)
        print("[WARN*] Position failed")

    def testInit(self):
        print("[MoveIt*] Init test")

        pose = Pose()
        pose.orientation = settings.md.ENV_DAT['above']['ori']
        pose.position = Point(0.4,0.,1.0)
        settings.goal_pose = deepcopy(pose)
        self.advancedWait()
        print("[MoveIt*] Init test 1, error: ", round(self.distancePoses(settings.eef_pose, pose), 2), " [x,y,z] diff: ", np.subtract(settings.extv(settings.eef_pose.position), settings.extv(settings.goal_pose.position)))

        pose.orientation = settings.md.ENV_DAT['wall']['ori']
        pose.position = Point(0.7,0.1,0.5)
        settings.goal_pose = deepcopy(pose)
        self.advancedWait()
        print("[MoveIt*] Init test 2 1/5, error: ", round(self.distancePoses(settings.eef_pose, pose), 2), " [x,y,z] diff: ", np.subtract(settings.extv(settings.eef_pose.position), settings.extv(settings.goal_pose.position)))
        pose.position = Point(0.7,-0.1,0.5)
        settings.goal_pose = deepcopy(pose)
        self.advancedWait()
        print("[MoveIt*] Init test 2 2/5, error: ", round(self.distancePoses(settings.eef_pose, pose), 2), " [x,y,z] diff: ", np.subtract(settings.extv(settings.eef_pose.position), settings.extv(settings.goal_pose.position)))
        pose.position = Point(0.7,-0.1,0.4)
        settings.goal_pose = deepcopy(pose)
        self.advancedWait()
        print("[MoveIt*] Init test 2 3/5, error: ", round(self.distancePoses(settings.eef_pose, pose), 2), " [x,y,z] diff: ", np.subtract(settings.extv(settings.eef_pose.position), settings.extv(settings.goal_pose.position)))
        pose.position = Point(0.7,0.1,0.4)
        settings.goal_pose = deepcopy(pose)
        self.advancedWait()
        print("[MoveIt*] Init test 2 4/5, error: ", round(self.distancePoses(settings.eef_pose, pose), 2), " [x,y,z] diff: ", np.subtract(settings.extv(settings.eef_pose.position), settings.extv(settings.goal_pose.position)))
        pose.position = Point(0.7,0.1,0.5)
        settings.goal_pose = deepcopy(pose)
        self.advancedWait()
        print("[MoveIt*] Init test 2 5/5, error: ", round(self.distancePoses(settings.eef_pose, pose), 2), " [x,y,z] diff: ", np.subtract(settings.extv(settings.eef_pose.position), settings.extv(settings.goal_pose.position)))

        pose.orientation = settings.md.ENV_DAT['table']['ori']
        pose.position = Point(0.4,-0.1,0.2)
        settings.goal_pose = deepcopy(pose)
        self.advancedWait()
        print("[MoveIt*] Init test 3 1/5, error: ", round(self.distancePoses(settings.eef_pose, pose), 2), " [x,y,z] diff: ", np.subtract(settings.extv(settings.eef_pose.position), settings.extv(settings.goal_pose.position)))
        pose.position = Point(0.6,-0.1,0.2)
        settings.goal_pose = deepcopy(pose)
        self.advancedWait()
        print("[MoveIt*] Init test 3 2/5, error: ", round(self.distancePoses(settings.eef_pose, pose), 2), " [x,y,z] diff: ", np.subtract(settings.extv(settings.eef_pose.position), settings.extv(settings.goal_pose.position)))
        pose.position = Point(0.6,0.1,0.2)
        settings.goal_pose = deepcopy(pose)
        self.advancedWait()
        print("[MoveIt*] Init test 3 3/5, error: ", round(self.distancePoses(settings.eef_pose, pose), 2), " [x,y,z] diff: ", np.subtract(settings.extv(settings.eef_pose.position), settings.extv(settings.goal_pose.position)))
        pose.position = Point(0.4,0.1,0.2)
        settings.goal_pose = deepcopy(pose)
        self.advancedWait()
        print("[MoveIt*] Init test 3 4/5, error: ", round(self.distancePoses(settings.eef_pose, pose), 2), " [x,y,z] diff: ", np.subtract(settings.extv(settings.eef_pose.position), settings.extv(settings.goal_pose.position)))
        pose.position = Point(0.4,-0.1,0.2)
        settings.goal_pose = deepcopy(pose)
        self.advancedWait()
        print("[MoveIt*] Init test 3 5/5, error: ", round(self.distancePoses(settings.eef_pose, pose), 2), " [x,y,z] diff: ", np.subtract(settings.extv(settings.eef_pose.position), settings.extv(settings.goal_pose.position)))

        print("[MoveIt*] Init test ended")


    def testMovementsInput(self):
        ''' User enters Cartesian coordinates x,y,z
        distance between given and real coordinates is displayed
        '''
        pose = Pose()
        while True:
            print("[MoveIt*] Input test started (enter any string to quit)")
            try:
                print("Go to pose, enter x position:")
                pose.position.x = float(raw_input())
                print("Enter y position:")
                pose.position.y = float(raw_input())
                print("Enter z position:")
                pose.position.z = float(raw_input())
                print("Enter x orientation:")
                pose.orientation.x = float(raw_input())
                print("Enter y orientation:")
                pose.orientation.y = float(raw_input())
                print("Enter z orientation:")
                pose.orientation.z = float(raw_input())
                print("Enter w orientation:")
                pose.orientation.w = float(raw_input())
                settings.goal_pose = deepcopy(pose)
                time.sleep(8)
                print("Distance between given and real coords.: ", round(self.distancePoses(settings.eef_pose, pose), 2), " [x,y,z] diff: ", np.subtract(settings.extv(settings.eef_pose.position), settings.extv(settings.goal_pose.position)))
            except ValueError:
                print("[MoveIt*] Test ended")
                break


    def testMovements(self):
        xs=list(range(4,9))
        ys=list(range(-2,3))
        name="table"

        poses = []
        pose = Pose()
        pose.orientation = settings.md.ENV_DAT['table']['ori']
        dists = []
        for i in ys:
            row = []
            for j in xs:
                pose.position = Point(j*0.1,i*0.1,0.1)#settings.md.ENV_DAT['table']['min']
                p = deepcopy(pose)

                settings.goal_pose = deepcopy(p)
                t=time.time()
                while not(abs(time.time()-t) > 10 or self.samePoses(settings.eef_pose, p, accuracy=0.01)):
                    pass
                row.append(self.distancePoses(settings.eef_pose, p))
            dists.append(row)


    def testTrajectoryActionClient(self):
        settings.md.Mode = ''
        self.trajectory_action_perform = False
        Js = [ [-1.72, 0.42, -1.61, -1.98, -1.36, 0.49, -1.48],
        [1.87, -1.49, -2.60, -1.13, -0.34, 3.19, -1.83],
        [-0.32, 1.53, -1.60, -1.45, 0.10, 3.21, 0.63],
        [2.43, -0.22, -0.49, -0.40, -1.88, 2.41, -0.99],
        [0.87, -0.10, 1.84, -2.86, -1.07, 2.28, 1.89],
        [2.01, 0.06, -0.40, -2.16, -1.54, 3.02, -1.59],
        [0.93, 0.24, -0.97, -2.86, 0.22, 3.19, -0.41],
        [1.09, 0.19, 1.35, -2.67, 0.12, 1.32, -0.44]
        ]
        while True:
            settings.mo.go_to_joint_state(joints = Js[random.choice(Js)])

            settings.mo.plotJointsCallViz(load_data=True)
            raw_input()

            settings.md._goal.trajectory.header.stamp = rospy.Time.now()
            settings.mo.tac.add_goal(deepcopy(settings.md._goal))
            settings.mo.tac.replace()

            time.sleep(3)
            settings.mo.plotJointsCallViz(load_data=True)
            raw_input()


    def inputPlotJointsAction(self):
        settings.viz = VisualizerLib()
        self.plotJointsCallViz(load_data=True, plotVelocities=False)
        settings.viz.show()

    def inputPlotPosesAction(self):
        settings.viz = VisualizerLib()
        self.plotPosesCallViz(load_data=True)
        settings.viz.show()

class Raw_Kinematics():
    ''' Includes forward kinematics and jacobian computation for Panda and KUKA iiwa
        TODO: Check DH parameters on some example
    '''

    def __init__():
        pass

    @staticmethod
    def forward_kinematics(joints, robot='panda', out='xyz'):
        ''' Direct Kinematics from iiwa/panda structure. Using its dimensions and angles.
            TODO: Panda has link8 which is is not here of lenght ~0.106m
        '''
        if robot == 'iiwa':
            #             /theta   , d   , a,     alpha
            DH=np.array([[joints[0], 0.34, 0, -90],
                         [joints[1], 0.0, 0, 90],
                         [joints[2], 0.4, 0, 90],
                         [joints[3], 0.0, 0, -90],
                         [joints[4], 0.4, 0, -90],
                         [joints[5], 0.0, 0, 90],
                         [joints[6], 0.126, 0, 0]])
        elif robot == 'panda':
            #             /theta   , d   , a,     alpha
            DH=np.array([[joints[0], 0.333, 0,      0],
                         [joints[1], 0.0,   0,      -90],
                         [joints[2], 0.316, 0,      90],
                         [joints[3], 0.0,   0.0825, 90],
                         [joints[4], 0.384, -0.0825,-90],
                         [joints[5], 0.0,   0,      90],
                         [joints[6], 0.0, 0.088,  90]])
        else: raise Exception("Wrong robot name chosen!")

        Tr = np.eye(4)
        for i in range(0, len(DH)):
            t = DH[i, 0]
            d = DH[i, 1]
            a = DH[i, 2]
            al= DH[i, 3]
            T = np.array([[math.cos(t), -math.sin(t)*math.cos(math.radians(al)), math.sin(t)*math.sin(math.radians(al)), a*math.cos(t)],
                  [math.sin(t), math.cos(t)*math.cos(math.radians(al)), -math.cos(t)*math.sin(math.radians(al)), a*math.sin(t)],
                  [0, math.sin(math.radians(al)), math.cos(math.radians(al)), d],
                  [0, 0, 0, 1]])
            Tr = np.matmul(Tr, T)
        if out=='xyz':
            return [Tr[0,3], Tr[1,3], Tr[2,3]]
        if out=='matrix':
            return Tr
        return None

    @staticmethod
    def jacobian(state, robot='panda'):
        fun = Raw_Kinematics.forward_kinematics
        eps = 0.001
        jacobian = np.zeros((3,7))

        inp = np.array(state)
        selector = np.array([0,1,2,3,4,5,6])

        for i in selector:
            jacobian[:,i] = (np.array(fun(inp + eps* (selector == i), robot=robot)) - np.array(fun(inp - eps* (selector == i), robot=robot))) / (2*eps)
        # print(jacobian)
        return jacobian

class IK_bridge():
    ''' PURPOSE: Compare different inverse kinematics solvers
        - MoveIt IK
        -

        What are joints configuration and what is the latest link?
        Because the last link in urdf file should be linked with the tip_point of
    '''
    def __init__(self):
        self.fk = kinematics_interface.ForwardKinematics(frame_id=settings.BASE_LINK)
        self.ikt = kinematics_interface.InverseKinematics()

    def getFKmoveit(self):
        return self.fk.getCurrentFK(settings.EEF_NAME).pose_stamped[0]

    def getFKmoveitPose(self):
        return self.fk.getCurrentFK(settings.EEF_NAME).pose_stamped[0].pose

    def getIKmoveit(self, pose):
        return self.ikt.getIK(self.move_group.get_name(), self.move_group.get_end_effector_link(), pose)


### Useful to remember
'''
# Set number of planning attempts
move_group.set_num_planning_attempts(self.PLANNING_ATTEMPTS)
# Display trajectory
self.display_trajectory(plan)

'''
