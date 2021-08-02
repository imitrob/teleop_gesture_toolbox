#!/usr/bin/env python
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

import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import ListedColormap, LinearSegmentedColormap

import trajectory_action_client

#from promp_lib import promp_
from std_msgs.msg import String, Bool, Int8, Float64MultiArray
from control_msgs.msg import FollowJointTrajectoryAction, FollowJointTrajectoryGoal
from geometry_msgs.msg import Pose, Point, Quaternion, PoseStamped, Vector3Stamped, QuaternionStamped, Vector3
from moveit_msgs.srv import ApplyPlanningScene
from trajectory_msgs.msg import JointTrajectoryPoint
from moveit_msgs.msg import RobotTrajectory
from relaxed_ik.msg import EEPoseGoals
#import RelaxedIK.Utils.transformations as T
from visualization_msgs.msg import MarkerArray, Marker
from sensor_msgs.msg import JointState

import kinematics_interface
import settings

from os.path import isfile

#import toppra as ta

#from toppra import SplineInterpolator


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

        print("GROUP NAME: ", settings.GROUP_NAME)
        robot = moveit_commander.RobotCommander()
        scene = moveit_commander.PlanningSceneInterface()
        move_group = moveit_commander.MoveGroupCommander(settings.GROUP_NAME)
        display_trajectory_publisher = rospy.Publisher('/move_group/display_planned_path',
                                                       moveit_msgs.msg.DisplayTrajectory,
                                                       queue_size=20)


        rospy.Subscriber("joint_states", JointState, save_joints)

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
        # direct kinematics
        self.fk = kinematics_interface.ForwardKinematics()
        self.ikt = kinematics_interface.InverseKinematics()

        # RelaxedIK publishers
        self.ik_goal_r_pub = rospy.Publisher('/ik_goal_r',PoseStamped,queue_size=5)
        self.ik_goal_l_pub = rospy.Publisher('/ik_goal_l',PoseStamped,queue_size=5)
        self.goal_pos_pub = rospy.Publisher('vive_position', Vector3Stamped,queue_size=5)
        self.goal_quat_pub = rospy.Publisher('vive_quaternion', QuaternionStamped,queue_size=5)
        self.ee_pose_goals_pub = rospy.Publisher('/relaxed_ik/ee_pose_goals', EEPoseGoals, queue_size=5)
        self.quit_pub = rospy.Publisher('/relaxed_ik/quit',Bool,queue_size=5)
        self.seq = 1

        self.tac = trajectory_action_client.TrajectoryActionClient(arm=settings.GROUP_NAME, topic=settings.TAC_TOPIC, topic_joint_states = settings.JOINT_STATES_TOPIC)
        settings.md.ENV = settings.md.ENV_DAT[env]


    def extq(self, q):
        return q.x, q.y, q.z, q.w

    def extv(self, v):
        return v.x, v.y, v.z

    def createMarker(self):
        publisher = rospy.Publisher('visualization_marker_array', MarkerArray, queue_size=5)
        markerArray = MarkerArray()
        while not rospy.is_shutdown():
            markerArray.markers = self.GenerateMarkers()
            publisher.publish(markerArray)
            rospy.sleep(0.25)

    def go_to_joint_state(self, joint_goal_ = [0, 0, 0, 0, 0, 0, 0]):
        ''' Planning to a Joint Goal
        '''
        assert isinstance(joint_goal_, list) and len(joint_goal_)==7, "Input not type list with len 7"
        self.move_group.go(joint_goal_, wait=True)
        self.move_group.stop()

        current_joints = self.move_group.get_current_joint_values()
        return all_close(joint_goal_, current_joints, 0.01)


    def go_to_pose_goal(self, pose=None):
        '''Planning to a Pose Goal
        Plan a motion for this group to a desired pose for the end-effector
        '''
        assert isinstance(pose, Pose), "Input not type Pose"
        self.move_group.set_pose_target(pose)

        plan = self.move_group.go(wait=True)  # Planner compute the plan and execute it.
        self.move_group.stop()
        self.move_group.clear_pose_targets()

        current_pose = self.move_group.get_current_pose().pose
        return all_close(pose, current_pose, 0.01)


    def plan_cartesian_path(self, scale=1):
        '''Cartesian Paths
        You can plan a Cartesian path directly by specifying a list of waypoints
        for the end-effector to go through. If executing  interactively in a
        Python shell, set scale = 1.0.
        '''
        waypoints = []

        wpose = self.move_group.get_current_pose().pose
        wpose.position.x += scale * 0.1  # Second move forward/backwards in (x)
        waypoints.append(deepcopy(wpose))

        (plan, fraction) = self.move_group.compute_cartesian_path(
                                           waypoints,   # waypoints to follow
                                           0.01,        # eef_step
                                           0.0)         # jump_threshold

        return plan, fraction

    def plan_cartesian_path_points(self, points):
        ''' Makes the plan with given waypoints
        '''
        waypoints = []
        pose = self.move_group.get_current_pose().pose
        pose.orientation = Quaternion(0.,0.,0.,1.)
        for point in points:
            pose.position.x = point[0]
            pose.position.y = point[1]
            pose.position.z = point[2]
            waypoints.append(deepcopy(pose))
            (plan, fraction) = self.move_group.compute_cartesian_path(waypoints, 0.01, 0.0)
        return plan, fraction


    def display_trajectory(self, plan):
        '''Displaying a Trajectory
        You can ask RViz to visualize a plan (aka trajectory) for you. But the
        group.plan() method does this automatically so this is not that useful
        here (it just displays the same trajectory again):

        A `DisplayTrajectory`_ msg has two primary fields, trajectory_start and trajectory.
        We populate the trajectory_start with our current robot state to copy over
        any AttachedCollisionObjects and add our plan to the trajectory.
        '''
        display_trajectory = moveit_msgs.msg.DisplayTrajectory()
        display_trajectory.trajectory_start = self.robot.get_current_state()
        display_trajectory.trajectory.append(plan)
        print(display_trajectory)
        # Publish
        self.display_trajectory_publisher.publish(display_trajectory);



    def execute_plan(self, plan):
        '''Executing a Plan
        Use execute if you would like the robot to follow
        the plan that has already been computed:
        '''

        return self.move_group.execute(plan, wait=True)
        ## **Note:** The robot's current joint state must be within some tolerance of the
        ## first waypoint in the `RobotTrajectory`_ or ``execute()`` will fail



    def wait_for_state_update(self, box_is_known=False, box_is_attached=False, timeout=4):
        '''Ensuring Collision Updates Are Received
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


    def add_box(self, timeout=4):
        '''Adding Objects to the Planning Scene
        First, we will create a box in the planning scene between the fingers:
        '''
        box_pose = PoseStamped()
        box_pose.header.frame_id = settings.BASE_LINK # "r1_gripper"
        box_pose.pose.orientation.w = 1.0
        box_pose.pose.position.z = 0.0 # above the panda_hand frame
        box_pose.pose.position.x = 0.3
        self.box_name = "box"

        self.wait_for_state_update(box_is_known=True, timeout=timeout)


    def add_mesh(self, file=None, name='box', pose=None, frame_id='r1_gripper', timeout=4):
        box_name = self.box_name
        scene = self.scene

        mesh_pose = geometry_msgs.msg.PoseStamped()
        mesh_pose.header.frame_id = frame_id
        mesh_pose.pose = pose
        mesh_name = name
        if name == "box" or name == "box2":
            self.scene.add_box(mesh_name, mesh_pose, size=(0.075, 0.075, 0.075))
        else:
            scene.add_mesh(mesh_name, mesh_pose, file, size=(1., 1., 1.))

        self.box_name=mesh_name # its the box
        return self.wait_for_state_update(box_is_known=True, timeout=timeout)

    def attach_item(self, item=None, timeout=4):
        '''Attaching Mesh Objects to the Robot
        Manipulating objects requires the
        robot be able to touch them without the planning scene reporting the contact as a
        collision. By adding link names to the ``touch_links`` array, we are telling the
        planning scene to ignore collisions between those links and the box. For the Panda
        robot, we set ``grasping_group = 'hand'``. If you are using a different robot,
        you should change this value to the name of your end effector group name.
        '''
        grasping_group = 'r1_gripper'
        touch_links = self.robot.get_link_names(group=grasping_group)
        self.scene.attach_mesh(self.eef_link, item, touch_links=touch_links)

        ## Mark what item was attached
        settings.md.attached = [item]
        print("attached item")

        # We wait for the planning scene to update.
        return self.wait_for_state_update(box_is_attached=True, box_is_known=False, timeout=timeout)


    def detach_item(self, item=None, timeout=4):
        '''Detaching Objects from the Robot
        We can also detach and remove the object from the planning scene:
        '''
        self.scene.remove_attached_object(self.eef_link, name=item)

        ## Mark that item was detached
        settings.md.attached = False
        print("detach item")

        # We wait for the planning scene to update.
        return self.wait_for_state_update(box_is_known=True, box_is_attached=False, timeout=timeout)


    def remove_item(self, item=None, timeout=4):
        '''Removing Objects from the Planning Scene
        We can remove the box from the world.
        '''
        self.scene.remove_world_object(item)
        ## **Note:** The object must be detached before we can remove it from the world

        # We wait for the planning scene to update.
        return self.wait_for_state_update(box_is_attached=False, box_is_known=False, timeout=timeout)

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
        pose.orientation = Quaternion(settings.md.ENV['x_ori'], settings.md.ENV['y_ori'], settings.md.ENV['z_ori'], settings.md.ENV['w_ori'])
        return pose

    '''
    Further custom functions
    '''
    def make_scene(self, scene=''):
        ''' Prepare scene, add objects for obstacle or manipulation.
        '''
        scenes = ['', 'drawer', 'pickplace', 'pushbutton', 'drawer2', 'pickplace2', 'pushbutton2']
        id = 0
        # get id of current scene
        if settings.scene:
            for n, i in enumerate(scenes):
                if i == settings.scene.NAME:
                    id = n
                    break
        # remove objects from current scene
        for i in range(0, len(settings.ss[id].mesh_names)):
            self.remove_item(item=settings.ss[id].mesh_names[i])
        if settings.md.attached:
            self.detach_item(item=settings.md.attached)

        id = 0
        for n, i in enumerate(scenes):
            if i == scene:
                id = n
                break

        for i in range(0, len(settings.ss[id].mesh_names)):
            self.add_mesh(file=settings.HOME+'/'+settings.WS_FOLDER+'/src/mirracle_gestures/include/models/'+settings.ss[id].mesh_names[i]+'.obj',
                name=settings.ss[id].mesh_names[i], pose=settings.ss[id].mesh_poses[i], frame_id=settings.BASE_LINK)
        settings.scene = settings.ss[id]
        if id == 0:
            settings.scene = None

        #settings.goal_pose = deepcopy(settings.sp[1].poses[0])
        #time.sleep(2)
        #settings.goal_pose = deepcopy(settings.sp[1].poses[1])
        #time.sleep(3)
        print("scene ready")



    def save_plan(self):
        ''' Saves plan to file, for ProMP.
        '''
        points = self.global_plan.joint_trajectory.points
        time_ = []
        Q_ = []
        for i in points:
            time_.append(i.time_from_start.to_sec())
            Q_.append(i.positions)

        f = open(settings.HOME+'/promp/examples/python_promp/strike_mov.npz', 'rb')
        data = np.load(f, allow_pickle=True)
        time = list(data['time'])
        Q = list(data['Q'])

        time_ = time_[:-1] # Issue, different sizes of data and time arr
        time.append(time_)
        Q.append(Q_)
        np.savez(settings.HOME+"/promp/examples/python_promp/strike_mov.npz", Q=Q, time=time)

    def load_LeapRecord(self, filen=""):
        ''' Loads last Leap Record
        '''
        if filen == "":
            for i in range(99,-1,-1):
                if isfile(settings.HOME+"/"+settings.WS_FOLDER+"/src/mirracle_gestures/include/data/leap_record_"+str(i)+".npz") == True:
                    filen = str(i)
                    print("Leap_record_"+filen+" loaded")
                    break
            if filen=="":
                sys.exit("FILE NOT FOUND!")
        f = open(settings.HOME+"/"+settings.WS_FOLDER+"/src/mirracle_gestures/include/data/leap_record_"+filen+".npz", 'rb')
        data = np.load(f, allow_pickle=True)
        time = list(data['time'])
        Q = list(data['Q'])

        return Q


    def samePoses(self, pose1, pose2, accuracy=0.05):
        ''' xyz, not orientation
        '''
        assert isinstance(pose1,Pose), "Not datatype Pose, pose1: "+str(pose1)
        assert isinstance(pose2,Pose), "Not datatype Pose, pose2: "+str(pose2)

        if np.sqrt((pose1.position.x - pose2.position.x)**2 + (pose1.position.y - pose2.position.y)**2 + (pose1.position.z - pose2.position.z)**2) < accuracy:
            return True
        return False

    def sameJoints(self, joints1, joints2):
        assert isinstance(joints1[0],float) and len(joints1)==7, "Not datatype List w len 7, joints 1: "+str(joints1)
        assert isinstance(joints2[0],float) and len(joints2)==7, "Not datatype List w len 7, joints 2: "+str(joints2)

        if sum([abs(i[0]-i[1]) for i in zip(joints1, joints2)]) < 0.1:
            return True
        return False

    def transformLeapToSceneRecord(self, p, filter='diff', targetPoints=20, scale='real',
    normalize=False, switchAxes=True, round_decp=3, two_dimensional=True):
        ''' p -> points, shape(len mov, 3)
            1. Reduce data size with filter: 'diff', 'modulo'
            2. Starting point is zero
            3. SWITCHES AXES
            4. Scale the data: 'real' - copying exact position, 'scaled' - trajectory scaled into workspace box
            *. Check if points reachable
            5. discard z axe
            6. rounding
            7. if first element zeros, delete it, as it fails when finding solution
        '''
        print("filter: ", filter, " targetPoints: ", targetPoints, " scale: ", scale,
        " normalize: ", normalize, " switchAxes: ", switchAxes, " round_decp: ", round_decp, " two_dimensional: ", two_dimensional)

        # get fewer points
        pred = []
        if filter == 'diff':
            pred = diffFilter(p, targetPoints)
        if filter == 'modulo':
            pred = moduloFilter(p, targetPoints)
        p = np.array(pred)

        # first point is zero
        if normalize:
            p0 = deepcopy(p[0])
            for n in range(0, len(p)):
                p[n] = np.subtract(p[n], p0)

        ## SWITCH AXES ##
        if switchAxes:
            x = deepcopy(p[:,0]) # x -> from left to right
            y = deepcopy(p[:,1]) # y -> from down to up
            z = deepcopy(p[:,2]) # z -> from front to back
            p[:,0] = x
            p[:,1] = np.multiply(z, [-1])
            p[:,2] = y

        # scale
        if scale=='real':
            for n in range(0, len(p)):
                p[n] = np.true_divide(p[n],1000) # milimeters->meters
            p[:,0] = p[:,0] + settings.md.ENV['start'].x * np.ones(len(p)) # transition into workspace
            p[:,1] = p[:,1] + settings.md.ENV['start'].y * np.ones(len(p))
            p[:,2] = p[:,2] + settings.md.ENV['start'].z * np.ones(len(p))
        if scale == 'scaled':
            p[:,0] = minmax_scale(p[:,0], feature_range=(settings.md.ENV['min'].x, settings.md.ENV['max'].x))
            p[:,1] = minmax_scale(p[:,1], feature_range=(settings.md.ENV['min'].y, settings.md.ENV['max'].y))
            p[:,2] = minmax_scale(p[:,2], feature_range=(settings.md.ENV['min'].z, settings.md.ENV['max'].z))

        # round 3 decimal points
        for n, i in enumerate(p):
            for m, j in enumerate(i):
                p[n,m] = np.round(j,round_decp)
        # keep only x,y for now
        if two_dimensional:
            p[:,2] = np.ones(len(p))
        # delete first element
        #if np.sum(p[0,:]) < 0.05:
        p = np.delete(p,0,0)
        # Check if reachable
        if self.points_in_env(p) == False:
            print("POINTS ARE NOT IN THE WORKSPACE")
        if self.points_reachable(p) == False:
            print("SOME POINT NOT REACHABLE")

        return p

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

        pose_.orientation = Quaternion(*tf.transformations.quaternion_multiply(tf.transformations.quaternion_from_euler(*euler), self.extq(settings.md.ENV['ori'])))
        '''
        if settings.md.ENV['ori_type'] == 'normal':
            dir = normdir[0:3]
        elif settings.md.ENV['ori_type'] == 'neg_normal':
            dir = np.dot(normdir[0:3], -1)
        elif settings.md.ENV['ori_type'] == 'direction':
            dir = normdir[3:6]
        elif settings.md.ENV['ori_type'] == 'neg_direction':
            dir = np.dot(normdir[3:6], -1)
        rx = dir[0]
        ry = -dir[2]
        rz = dir[1]
        '''
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
        pose_ = deepcopy(pose1)
        if settings.FIXED_ORI_TOGGLE:
            pose_.orientation = settings.md.ENV['ori']
        pose_.position.z -= 1.27
        pose_.position.y = -pose_.position.y
        return pose_

    def relaxik_t_inv(self, pose1):
        pose_ = deepcopy(pose1)
        pose_.position.z += 1.27
        pose_.position.y = -pose_.position.y
        return pose_

    def points_reachable(self, p):
        for point in p:
            # check the inverse kinematic
            pose = self.move_group.get_current_pose()
            pose.pose.position.x = point[0]
            pose.pose.position.y = point[1]
            pose.pose.position.z = point[2]
            pose.pose.orientation.x = 0.0
            pose.pose.orientation.y = 0.0
            pose.pose.orientation.z = 0.0
            pose.pose.orientation.w = 1.0
            IK_resp = self.ikt.getIK(self.move_group.get_name(),
                                        self.move_group.get_end_effector_link(),
                                        pose)
            if IK_resp.error_code.val != 1:
                print("this is it",IK_resp.solution.joint_state)
                return False
            #showcase.go_to_joint_state(IK_resp.solution.joint_state.position[0:7])
        return True

    def ProMPout_to_plan(self, Q, time):
        # ProMPJoints

        plan_ = self.move_group.plan()
        points_ = plan_.joint_trajectory.points
        point_ = JointTrajectoryPoint()
        for q, t in itertools.izip(Q, time):
            point_.positions = q
            point_.time_from_start = rospy.Duration(t * self.RECORDING_LENGTH)
            points_.append(deepcopy(point_))

        #self.display_trajectory(plan_)
        return plan_

    def process_ProMPRecord(self, q, time, filter='modulo', targetPoints=10):
        ''' q -> joints, shape(len mov, 7)
            1. Reduce data size with filter: 'modulo'
            2. if first element zeros, delete it, as it fails when finding solution
            3. Check if points are inside ENV
        '''
        # get fewer Qs
        qred = []
        timered = []
        if filter == 'modulo':
            qred = moduloFilter(q, targetPoints)
            timered = moduloFilter(time, targetPoints)

        # delete first element
        q = np.delete(q,0,0)
        time = np.delete(time,0)
        # Check if inside env
        if self.points_in_env([iiwa_forward_kinematics(q_) for q_ in q]) == False:
            print("POINTS ARE NOT IN WORKSPACE")

        return q, time

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
            p = iiwa_forward_kinematics(point.positions)
            Q.append(p)
            time.append(self.timeObj_to_double(point.time_from_start))

        if addon=='normalize':
            Q0 = deepcopy(Q[0])
            for n in range(0, len(Q)):
                Q[n] = np.subtract(Q[n], Q0)
        return Q, time

    def plan_to_joints(self, data):
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
            Q.append(point.positions)
            time.append(self.timeObj_to_double(point.time_from_start))

        return Q, time

    def timeObj_to_double(self, obj):
        return (obj.secs + obj.nsecs/1000000000.0)

    def promp_config_run(self, q_cond_init=None, q_cond_end=None, dim_basis=7, T=2.0, numInputTrajectories=5,
    n_samples = 1, max_iter=10, Sigma_rate=1e-5, mu_cartesian=[0, 0, 0], Sigma_cartesian=0.02**2,
    cond_time=0):
        '''
        q_cond_init -> starting condition of movement
        dim_basis -> dimension of all parameters of created basis
        T -> time of the movement, recording length
        numInputTrajectories -> number of input trajectories
        n_samples -> number of most probable samples to be displayed
        max_iter -> number of training iterations
        '''
        ## Basis configuration, what coefficient will shape through training
        full_basis = {
        'conf': [ #                                     num of RBF
                {"type": "sqexp", "nparams": 4, "conf": {"dim": 3}},
                  #                                     poly degree
                {"type": "poly", "nparams": 0, "conf": {"order": 3}}
            ],
        #                   scale 0.25,   centers, 0.5, 0.75
        'params': np.array([np.log(0.25), 0.0, 0.5, 1.0, 1.0, 1.0])
        }
        v_rate = 0.1*1
        # all in one
        ic = {'basis': full_basis, 'dim_basis': dim_basis, 'v_rate': v_rate, 'Sigma_rate': Sigma_rate, \
        'T': T, 'mu_cartesian': mu_cartesian, 'Sigma_cartesian': Sigma_cartesian, \
        'leap_records': numInputTrajectories, 'n_samples': n_samples, 'max_iter': max_iter, \
        'q_cond_init': q_cond_init, 'q_cond_end': q_cond_end, 'cond_time': cond_time,
        }
        result = promp_(self, input_config=ic)
        return result

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

    def applyTransformWithRotation():
        ''' TODO:
        '''
        pass


    def inSceneObj(self, point):
        ''' in the zone of a box with length l
            Compatible for: Pose, Point, [x,y,z]
            Cannot be in two zones at once, return id of first zone
        '''
        collisionObjs = []
        if not settings.scene:
            return False
        z = [False] * len(settings.scene.mesh_poses)
        assert settings.scene, "Scene not published yet"
        if isinstance(point, Pose):
            point = point.position
        if isinstance(point, Point):
            point = [point.x, point.y, point.z]
        for n, pose in enumerate(settings.scene.mesh_poses):
            zone_point = pose.position

            zone_point = self.PointAdd(zone_point, settings.scene.mesh_trans_origin[n])
            #print(n, ": \n",zone_point.z, "\n" ,settings.scene.mesh_sizes[n].z, "\n", point[2])
            if settings.scene.mesh_sizes[n].y > 0.0:
                if zone_point.x <= point[0] <= zone_point.x+settings.scene.mesh_sizes[n].x:
                  if zone_point.y <= point[1] <= zone_point.y+settings.scene.mesh_sizes[n].y:
                    if zone_point.z <= point[2] <= zone_point.z+settings.scene.mesh_sizes[n].z:
                        collisionObjs.append(settings.scene.mesh_names[n])

            else:
                if zone_point.x <= point[0] <= zone_point.x+settings.scene.mesh_sizes[n].x:
                  if zone_point.y >= point[1] >= zone_point.y+settings.scene.mesh_sizes[n].y:
                    if zone_point.z <= point[2] <= zone_point.z+settings.scene.mesh_sizes[n].z:
                        collisionObjs.append(settings.scene.mesh_names[n])
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







    '''
    def retime2(self, plan):

        assert isinstance(plan, RobotTrajectory)

        vel = [ 1.71, 1.71, 1.75, 2.27, 2.44, 3.14, 3.14 ]
        effort = [ 140, 140, 120, 100, 70, 70, 70 ]
        lower = [-2.97, -2.09, -2.97, -2.09, -2.97, -2.09, -3.05]
        upper = [2.97, 2.09, 2.97, 2.09, 2.97, 2.09, 3.05]  #self.get_joint_limits(self.mg.get_active_joints())
        alims = len(lower)* [[-1.0, 1.0]] # better use real acceleration limits

        print("plan", plan)
        ss = [pt.time_from_start.to_sec() *1.0 for pt in plan.joint_trajectory.points]
        way_pts = [list(pt.positions) for pt in plan.joint_trajectory.points]
        path = ta.SplineInterpolator(ss, way_pts)
        pc_vel = ta.constraint.JointVelocityConstraint(np.array([lower, upper]).transpose())
        pc_acc = ta.constraint.JointAccelerationConstraint(np.array(alims))
        print(path)
        instance = ta.algorithm.TOPPRA([pc_vel, pc_acc], path)
        # print(instance)
        # instance2 = ta.algorithm.TOPPRAsd([pc_vel, pc_acc], path)
        # instance2.set_desired_duration(60)
        jnt_traj = instance.compute_trajectory()
        # ts_sample = np.linspace(0, jnt_traj.duration, 10*len(plan.joint_trajectory.points))
        ts_sample = np.linspace(0, jnt_traj.duration, np.ceil(100*jnt_traj.duration))
        qs_sample = jnt_traj(ts_sample)
        qds_sample = jnt_traj(ts_sample, 1)
        qdds_sample = jnt_traj(ts_sample, 2)
        new_plan = deepcopy(plan)
        new_plan.joint_trajectory.points = []
        for t, q, qd, qdd in zip(ts_sample, qs_sample, qds_sample, qdds_sample):
            pt = JointTrajectoryPoint()
            pt.time_from_start = rospy.Duration.from_sec(t)
            pt.positions = q
            pt.velocities = qd
            pt.accelerations = qdd
            new_plan.joint_trajectory.points.append(pt)
        if rospy.get_param('plot_joint_trajectory', default=False):
            import matplotlib.pyplot as plt
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
    '''

    def perform_optimized_path(self, joint_states=None, toppra=False):
        '''
        '''
        assert len(joint_states)==7, "Not right datatype"+str(type(joint_states))
        goal = FollowJointTrajectoryGoal()
        goal.trajectory.joint_names = settings.JOINT_NAMES
        joints = JointTrajectoryPoint()
        joints.velocities = [0.] * 7
        joints.accelerations = [0.] * 7

        joints2 = np.array(joint_states)
        joints1 = np.array(settings.joints)

        joints_diff = joints2 - joints1
        for i in range(9,10):
            joints.positions = joints1 + joints_diff*i/10
            joints.time_from_start = rospy.Time((1./settings.md.speed)*i)
            goal.trajectory.points.append(deepcopy(joints))

        goal.trajectory.header.stamp = rospy.Time.now()
        #print("GOAL:", goal.trajectory)

        if toppra:
            pass
            '''
            robot_traj = RobotTrajectory()
            robot_traj.joint_trajectory = deepcopy(goal.trajectory)
            robot_traj_new = self.retime(plan=robot_traj)
            goal.trajectory = robot_traj_new.joint_trajectory
            '''

        # TODO: add tolerance: goal.path_tolerance = JointTolerance()
        settings.md._goal = deepcopy(goal)

        self.tac.add_goal(goal)
        #self.scale_speed()
        self.tac.replace()

        #instance = ta.algorithm.TOPPRA([pc_vel, pc_acc, eef_vel], path, solver_wrapper='seidel', gridpoints=np.linspace(0.0, ss[-1], np.min([int(n_grid), np.ceil(pt_per_s*ss[-1])])))

    '''
    ###
    ### From Jan Kristof Behrens
    ###
    def retime(self, plan=None, cart_vel_limit=-1.0, secondorder=False, pt_per_s=100, curve_len=None, start_delay=0.0):
        ta.setup_logging("INFO")
        assert isinstance(plan, RobotTrajectory)

        if not curve_len is None and cart_vel_limit > 0:
            n_grid = np.ceil(pt_per_s * curve_len / cart_vel_limit)
        else:
            n_grid = np.inf

        active_joints = plan.joint_trajectory.joint_names
        # ALERT: This function is not found
        #lower, upper, vel, effort = self.robot.get_joint_limits(active_joints)
        lower, upper, vel, effort = settings.md.lower_lim, settings.md.upper_lim, settings.md.vel_lim, settings.md.effort_lim
        # prepare numpy arrays with limits for acceleration
        alims = np.zeros((len(active_joints), 2))
        alims[:, 1] = np.array(len(lower) * [3.0])
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

        def vlims_func(val):
            eps = 0.001
            limit = cart_vel_limit

            J = iiwa_jacobian(path(val))
            direction = (path(val + eps) - path(val - eps)) / (2* eps)
            dir_norm = direction / np.linalg.norm(direction)
            x = limit / np.linalg.norm(np.dot(J, dir_norm))
            x = x * dir_norm

            x = np.abs(x)
            print("{}: {}".format(val, np.max(x)))
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
        #     J = iiwa_jacobian(q)
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
                J = iiwa_jacobian(q)
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
                J = iiwa_jacobian(q)
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
            instance = ta.algorithm.TOPPRA([pc_vel, pc_vel2, pc_acc], path, gridpoints=np.linspace(0.0, ss[-1], np.min(n_grid, np.ceil(pt_per_s*ss[-1]))))

            # instance = ta.algorithm.TOPPRA([pc_vel, pc_vel2, pc_acc], path, gridpoints=np.linspace(0.0, ss[-1], 10000))


        # print(instance)
        # instance2 = ta.algorithm.TOPPRAsd([pc_vel, pc_acc], path)
        # instance2.set_desired_duration(60)
        jnt_traj = instance.compute_trajectory()
        feas_set = instance.compute_feasible_sets()


        # ts_sample = np.linspace(0, jnt_traj.duration, 10*len(plan.joint_trajectory.points))
        ts_sample = np.linspace(0, jnt_traj.duration, np.ceil(100 * jnt_traj.duration))
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
            import matplotlib.pyplot as plt

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
    '''


    def scale_speed(self, scaling_factor=1.0):
        '''
        This function scales the execution speed of the remaining trajectory according to the scaling factor. This works onjy before or during an execution,
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

    def point_to_joints(self, x, y, z=1.0):

        pose = self.move_group.get_current_pose()
        joints = self.move_group.get_current_joint_values()
        pose.pose.position.x = x
        pose.pose.position.y = y
        pose.pose.position.z = z
        pose.pose.orientation.x = 0.0
        pose.pose.orientation.y = 0.0
        pose.pose.orientation.z = 0.0
        pose.pose.orientation.w = 1.0
        IK_resp = self.ikt.getIK(self.move_group.get_name(), self.move_group.get_end_effector_link(), pose)
        if IK_resp.error_code.val != 1:
            print("IKT NOT FOUND!!!",IK_resp.solution.joint_state.position)
        return IK_resp.solution.joint_state.position[0:7] # robot1



    def relaxedIK_publish(self, pose_r=None, pose_l=None):
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

    def fillInputPyMC(self):
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

    def testMovements(self, xs=list(range(4,9)), ys=list(range(-2,3)), name="table"):
        poses = []
        pose = Pose()
        pose.orientation = Quaternion(np.sqrt(2)/2, np.sqrt(2)/2., 0.0, 0.0)

        dists = []
        for i in ys:
            row = []
            for j in xs:
                pose.position = Point(j*0.1,i*0.1,0.1)#settings.md.ENV_DAT['table']['min']
                p = deepcopy(pose)

                settings.goal_pose = deepcopy(self.relaxik_t(p))
                t=time.time()
                while not(abs(time.time()-t) > 10 or self.samePoses(settings.rd.eef_pose, p, accuracy=0.01)):
                    pass
                row.append(self.distancePoses(settings.rd.eef_pose, p))
            dists.append(row)


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


    def distancePoses(self, p1,p2):
        p1 = p1.position
        p2 = p2.position
        #print(p1, "\n",p2)
        dist = np.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2 + (p1.z - p2.z)**2)
        #print("vector: ", str(p1.x - p2.x)," ", str(p1.y - p2.y)," ", str(p1.z - p2.z))
        #print("Distance: ", dist)
        #print("##")
        return dist

    def pointToScene(self, point):
        x,y,z = point.x, point.y, point.z
        point_ = Point()
        point_.x = np.dot([x,y,z], settings.md.ENV['axes'][0])*settings.md.SCALE + settings.md.ENV['start'].x
        point_.y = np.dot([x,y,z], settings.md.ENV['axes'][1])*settings.md.SCALE + settings.md.ENV['start'].y
        point_.z = np.dot([x,y,z], settings.md.ENV['axes'][2])*settings.md.SCALE + settings.md.ENV['start'].z
        return point_



    ### Long Data

    def GenerateMarkers(self):
        q_norm_to_dir = Quaternion(0.0, -math.sqrt(2)/2, 0.0, math.sqrt(2)/2)
        markers_array = []
        ## marker_interaction_box
        sx,sy,sz = self.extv(settings.md.ENV['start'])
        m = Marker()
        m.header.frame_id = "/"+settings.BASE_LINK
        m.type = m.SPHERE
        m.action = m.ADD
        m.scale.x = 0.01
        m.scale.y = 0.01
        m.scale.z = 0.01
        m.color.a = 1.0
        m.color.r = 1.0
        m.color.g = 1.0
        m.color.b = 0.0
        m.pose.orientation = settings.md.ENV['ori']
        m.pose.position = self.pointToScene(Point(+0.1175, +0.0735, +0.0825))
        m.id = 100
        markers_array.append(deepcopy(m))
        m.pose.position = self.pointToScene(Point(-0.1175, +0.0735, +0.0825))
        m.id += 1
        markers_array.append(deepcopy(m))
        m.pose.position = self.pointToScene(Point(-0.1175, -0.0735, +0.0825))
        m.id += 1
        markers_array.append(deepcopy(m))
        m.pose.position = self.pointToScene(Point(+0.1175, -0.0735, +0.0825))
        m.id += 1
        markers_array.append(deepcopy(m))
        m.pose.position = self.pointToScene(Point(+0.1175, +0.0735, +0.0825+0.3175))
        m.id += 1
        markers_array.append(deepcopy(m))
        m.pose.position = self.pointToScene(Point(-0.1175, +0.0735, +0.0825+0.3175))
        m.id += 1
        markers_array.append(deepcopy(m))
        m.pose.position = self.pointToScene(Point(-0.1175, -0.0735, +0.0825+0.3175))
        m.id += 1
        markers_array.append(deepcopy(m))
        m.pose.position = self.pointToScene(Point(+0.1175, -0.0735, +0.0825+0.3175))
        m.id += 1
        markers_array.append(deepcopy(m))

        ## Marker env + normal
        m = Marker()
        m.header.frame_id = "/"+settings.BASE_LINK
        m.type = m.ARROW
        m.action = m.ADD
        m.scale.x = 0.1
        m.scale.y = 0.01
        m.scale.z = 0.01
        m.color.a = 1.0
        m.color.r = 1.0
        m.color.g = 1.0
        m.color.b = 0.0
        m.pose.position = settings.md.ENV['start']
        m.pose.orientation = settings.md.ENV['ori']
        m.id += 1
        markers_array.append(deepcopy(m))
        m.color.g = 0.0
        m.pose.orientation = Quaternion(*tf.transformations.quaternion_multiply(self.extq(m.pose.orientation), self.extq(q_norm_to_dir)))
        m.id += 1
        markers_array.append(deepcopy(m))

        ## Marker endeffector + normal + action to attach
        m = Marker()
        m.header.frame_id = "/"+settings.EEF_NAME
        m.type = m.ARROW
        m.action = m.ADD
        m.scale.x = 0.2
        m.scale.y = 0.01
        m.scale.z = 0.01
        m.color.a = 1.0
        m.color.r = 1.0
        m.color.g = 1.0
        m.color.b = 0.0
        m.pose.orientation.x = 0.0
        m.pose.orientation.y = -math.sqrt(2)/2
        m.pose.orientation.z = 0.0
        m.pose.orientation.w = math.sqrt(2)/2
        m.pose.position.x = 0
        m.pose.position.y = 0
        m.pose.position.z = 0
        m.id += 1
        markers_array.append(deepcopy(m))
        m.color.g = 0.0
        m.pose.orientation = Quaternion(*tf.transformations.quaternion_multiply(self.extq(m.pose.orientation), self.extq(q_norm_to_dir)))
        m.id += 1
        markers_array.append(deepcopy(m))
        if settings.md.ACTION:
            m.type = m.SPHERE
            m.scale.x = 0.2
            m.scale.y = 0.2
            m.scale.z = 0.2
            m.id += 1
            markers_array.append(deepcopy(m))
        else:
            m.type = m.SPHERE
            m.scale.x = 0.01
            m.scale.y = 0.01
            m.scale.z = 0.01
            m.id += 1
            markers_array.append(deepcopy(m))

        ## Marker relaxit input
        m = Marker()
        m.header.frame_id = "/"+settings.BASE_LINK
        m.type = m.ARROW
        m.action = m.ADD
        m.scale.x = 0.2
        m.scale.y = 0.01
        m.scale.z = 0.01
        m.color.a = 1.0
        m.color.r = 1.0
        m.color.g = 1.0
        m.color.b = 0.0
        m.id += 1
        m.pose = Pose()
        if isinstance(settings.goal_pose, Pose):
            m.pose = self.relaxik_t_inv(pose1=settings.goal_pose)
        markers_array.append(deepcopy(m))
        m.color.g = 0.0
        m.pose.orientation = Quaternion(*tf.transformations.quaternion_multiply(self.extq(m.pose.orientation), self.extq(q_norm_to_dir)))
        m.id += 1
        markers_array.append(deepcopy(m))


        ## Writing trajectory
        m = Marker()
        m.header.frame_id = "/"+settings.BASE_LINK
        m.type = m.SPHERE
        m.action = m.ADD
        m.scale.x = 0.01
        m.scale.y = 0.01
        m.scale.z = 0.01
        m.color.a = 1.0
        m.color.r = 0.0
        m.color.g = 0.0
        m.color.b = 1.0
        m.id = 1000
        if settings.print_path_trace:
            for frame in list(settings.forward_kinematics):
                m.pose = frame
                m.id += 1
                markers_array.append(deepcopy(m))
        else:
            for frame in list(settings.forward_kinematics):
                m.action = m.DELETE
                m.id += 1
                markers_array.append(deepcopy(m))

        #goal_target_pose = Pose()
        #goal_target_pose.orientation.w = 1.0
        #goal_target_pose.position.x = 0.7
        #goal_target_pose.position.y = -0.2
        #goal_target_pose.position.z = 0.0

        #m.pose = goal_target_pose
        #m.action = m.ADD
        #m.id += 1
        #m.type = m.CUBE
        #m.color.r = 0.0
        #m.color.g = 1.0
        #m.color.b = 0.0
        #m.scale.x = 0.2
        #m.scale.y = 0.2
        #m.scale.z = 0.001
        #markers_array.append(deepcopy(m))

        return markers_array

def save_joints(data):
    if settings.ROBOT_NAME == 'iiwa':
        # On iiwa testbed, take data only from r1 robot
        if data.name[0][1] == '1':
            settings.joints = data.position # Position = Angles [rad]
    else: # 'panda'
        settings.joints = data.position[0:7] # Position = Angles [rad]
        '''
        r1_joint_vel = data.velocity # [rad/s]
        r1_joint_eff = data.effort # [Nm]
        r1_time = float(data.header.stamp.to_sec())
        global save_joint_pos
        global save_time
        save_joint_pos.append(r1_joint_pos)
        save_time.append(r1_time)
    elif data.name[0][1] == '2':
        global r2_joint_pos, r2_joint_vel, r2_joint_eff
        r2_joint_pos = data.position # Position = Angles [rad]
        r2_joint_vel = data.velocity # [rad/s]
        r2_joint_eff = data.effort # [Nm]
        '''

'''
class TrajectoryActionClient_Interface():
    def __init__():
        tac = trajectory_action_client.TrajectoryActionClient(arm="r1")

    def trajectory_action_client(self):

        goal = FollowJointTrajectoryGoal()
        goal.trajectory.joint_names = settings.JOINT_NAMES
        point = JointTrajectoryPoint()
        point.positions = [-0.08581872672559407, -0.7857175209529341, 0.25693271169288157, -1.6472575328531098, 0.2331417066023944, -0.8916397953223283, -0.24588569981336367]
        point.time_from_start = type('obj', (object,), {'secs' : 0, 'nsecs': 0})
        goal.trajectory.points.append(deepcopy(point))
        #jointtolearance = JointTolerance()
        #goal.path_tolerance = jointtolerance
        #goal.goal_tolerance =

        tac.add_goal(goal)
        tac.start()
'''


def diffFilter(p, targetNumPoints):
    ''' Reduces the trajectory points by difference
        Basic controller, optimizing fs by finding target number of points
    '''
    pred = []
    assert len(p) > targetNumPoints, "Points filter has ("+str(len(p))+") input lower than wanted output number ("+str(targetNumPoints)+")"

    fs = 0.0
    lastNumPoints = len(p)
    rateIncr = 1
    round = 0
    while True:
        round += 1
        fs += rateIncr
        pred = []
        threshold = 0.0
        prev = 0.0
        for n, i in enumerate(p):
            threshold += abs(i[0]) + abs(i[1]) + abs(i[2]) - prev
            if abs(threshold) > fs:
                pred.append(i)
                threshold = 0.0
            prev = abs(i[0]) + abs(i[1]) + abs(i[2])
        if abs(targetNumPoints - len(pred)) > abs(targetNumPoints - lastNumPoints):
            fs -= rateIncr
            rateIncr = -0.1 * rateIncr
        lastNumPoints = len(pred)
        if len(pred) > 0.9*targetNumPoints and len(pred) < 1.1*targetNumPoints:
            break
        if round > 100:
            break
    return pred

def moduloFilter(p, targetNumPoints):
    mod = round(len(p) / targetNumPoints)
    assert mod > 0, "Cannot devide zero, Points: "+str(len(p)) + " targetNumPoints: "+str(targetNumPoints)
    pred = []
    for n, i in enumerate(p):
        if n%mod == 0:
            pred.append(i)
    return pred



def iiwa_forward_kinematics(joints, out='xyz'):
    ''' Direct Kinematics from iiwa structure. Using its dimensions and angles.
    '''
    #joints = [0,0,0,0,0,0,0]
    DH=np.array([[joints[0], 0.34, 0, -90],
                 [joints[1], 0.0, 0, 90],
                 [joints[2], 0.4, 0, 90],
                 [joints[3], 0.0, 0, -90],
                 [joints[4], 0.4, 0, -90],
                 [joints[5], 0.0, 0, 90],
                 [joints[6], 0.126, 0, 0]])
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
    #pp = [Tr[0,3], Tr[1,3], T[2,3]]
    #pp
    if out=='xyz':
        return [Tr[0,3], Tr[1,3], Tr[2,3]]
    if out=='matrix':
        return Tr
    return None

def iiwa_jacobian(state):
    fun = iiwa_forward_kinematics
    eps = 0.001
    jacobian = np.zeros((3,7))

    inp = np.array(state)
    selector = np.array([0,1,2,3,4,5,6])

    for i in selector:
        jacobian[:,i] = (np.array(fun(inp + eps* (selector == i))) - np.array(fun(inp - eps* (selector == i)))) / (2*eps)
    # print(jacobian)
    return jacobian


### Useful to remember
'''
# Set number of planning attempts
move_group.set_num_planning_attempts(self.PLANNING_ATTEMPTS)
# Get Joint Values
self.move_group.get_current_joint_values()
# Display trajectory
self.display_trajectory(plan)

'''
