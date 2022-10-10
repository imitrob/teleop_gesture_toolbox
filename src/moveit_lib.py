#!/usr/bin/env python3.8
'''
Library for manipulation with robot arm.

Source: https://ros-planning.github.io/moveit_tutorials/doc/move_group_python_interface/move_group_python_interface_tutorial.html
'''

from __future__ import print_function
from six.moves import input

import os
import tf
import sys
import csv
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

from os_and_utils.visualizer_lib import VisualizerLib
from matplotlib import cm
from matplotlib.colors import ListedColormap, LinearSegmentedColormap

import os_and_utils.trajectory_action_client

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
from coppelia_sim_ros_interface.srv import AddOrEditObject, AddOrEditObjectResponse, RemoveObject, RemoveObjectResponse, GripperControl, GripperControlResponse

import inverse_kinematics.kinematics_interface
import settings

from os.path import isfile

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
    def __init__(self):
        super(MoveGroupPythonInteface, self).__init__()

        moveit_commander.roscpp_initialize(sys.argv)
        rospy.init_node('move_group_python_interface', anonymous=True)

        print("[INFO*] GROUP NAME: ", settings.group_name)
        robot = moveit_commander.RobotCommander()
        scene = moveit_commander.PlanningSceneInterface()
        move_group = moveit_commander.MoveGroupCommander(settings.group_name)
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

        # When directing control right to Coppelia Simulator, the output of RelaxedIk is sended directily to simulator
        if settings.simulator == 'gazebo' or settings.simulator == 'real':
            self.tac = trajectory_action_client.TrajectoryActionClient(arm=settings.group_name, topic=settings.tac_topic, topic_joint_states = settings.joint_states_topic)

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
        return md.eef_pose

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
        return md.joints


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
            if settings.robot == 'panda':
                if settings.simulator == 'gazebo' or settings.simulator == 'real':
                    self.plan_and_perform_trajectory_from_joints(joints)
                elif settings.simulator == 'coppelia':
                    pass # publsiher is enabled
                else: raise Exception("[ERROR*] Simulator name wrong")
            elif settings.robot == 'iiwa':
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


    def add_or_edit_object(self, file='', name='', pose=Pose(), shape="", size=None, collision='true', color='', friction=-1, frame_id='panda_link7', mass=-1, inertia=np.zeros(9), inertiaTransformation=np.zeros(12), dynamic='true', pub_info='false', texture_file="", object_state="", timeout=4):
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
            resp1 = add_or_edit_object(name=name, init_file=file, init_shape=shape, init_size=size, init_collision=collision, pose=pose, color=color, friction=friction, frame_id=frame_id, mass=mass, inertia=inertia, inertiaTransformation=inertiaTransformation, dynamic=dynamic, pub_info=pub_info, texture_file=texture_file, object_state=object_state)
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
            md.attached = []
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
                md.attached = [name]
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
            - Grasping group is based on robot 'settings.grasping_group'
        Parameters:
            name (Str): Name id
        '''
        touch_links = self.robot.get_link_names(group=settings.grasping_group)
        self.scene.attach_mesh(self.eef_link, name, touch_links=touch_links)

        ## Mark what item was attached
        md.attached = [name]
        print("Attached item")

        # We wait for the planning scene to update.
        return self.wait_for_state_update(box_is_attached=True, box_is_known=False, timeout=4)


    def detach_item_moveit(self, name=None, timeout=4):
        '''Detaching Objects from the Robot
        We can also detach and remove the object from the planning scene:
        '''
        self.scene.remove_attached_object(self.eef_link, name=name)

        ## Mark that item was detached
        md.attached = False
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





    def points_in_env(self, points):
        points_in_workspace = True
        for n in range(0, len(points)):
            if self.point_in_env(points[n]) == False:
                points_in_workspace = False
        return points_in_workspace


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
        t_s = md._goal.trajectory.header.stamp
        t_now = rospy.Time.now()
        dt_now = t_now - t_s
        dt_now_ = rospy.Time(secs=dt_now.secs, nsecs=dt_now.nsecs)
        # create new goal with all non-passed TrajectoryPoints
        new_traj_goal = deepcopy(md._goal)
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




    def gestureGoalPoseUpdate(self, toggle, move):
        if abs(md.frames[-1].timestamp - md.frames[-1].r.time_last_stop) < 200000:
            pass
        else:
            #print("blocked", md.frames[-1].timestamp, md.frames[-1].r.time_last_stop)
            return

        move_ = 1 if move else -1
        move_ = move_ * md.gestures_goal_stride

        pt = [0.] * 3
        pt[toggle] = move_
        # switch axes a) Leap
        Pt_a = [0.] * 3
        Pt_a[0] = np.dot(pt, md.LEAP_AXES[0])
        Pt_a[1] = np.dot(pt, md.LEAP_AXES[1])
        Pt_a[2] = np.dot(pt, md.LEAP_AXES[2])
        # b) Scene
        Pt_b = [0.] * 3
        Pt_b[0] = np.dot(Pt_a, md.ENV['axes'][0])
        Pt_b[1] = np.dot(Pt_a, md.ENV['axes'][1])
        Pt_b[2] = np.dot(Pt_a, md.ENV['axes'][2])

        # axis swtich
        md.gestures_goal_pose.position = self.PointAdd(md.gestures_goal_pose.position, Point(*Pt_b))

    def gestureGoalPoseRotUpdate(self, toggle, move):
        if abs(md.frames[-1].timestamp - settings.frames[-1].r.time_last_stop) < 200000:
            pass
        else:
            #print("blocked", md.frames[-1].timestamp, settings.frames[-1].r.time_last_stop)
            return
        move_ = 1 if move else -1
        move_ = move_ * md.gestures_goal_rot_stride

        pt = [0.] * 3
        pt[toggle] = move_

        Pt_b = [0.] * 3
        Pt_b[0] = np.dot(pt, md.ENV['ori_axes'][0])
        Pt_b[1] = np.dot(pt, md.ENV['ori_axes'][1])
        Pt_b[2] = np.dot(pt, md.ENV['ori_axes'][2])

        # axis swtich
        o = md.gestures_goal_pose.orientation
        euler = list(tf.transformations.euler_from_quaternion([o.x, o.y, o.z, o.w]))
        euler[0] += Pt_b[0]
        euler[1] += Pt_b[1]
        euler[2] += Pt_b[2]
        md.gestures_goal_pose.orientation = Quaternion(*tf.transformations.quaternion_from_euler(euler[0],euler[1],euler[2]))




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

            plt.savefig(settings.paths.plots_path+name+'.eps', format='eps')
            plt.show()

        plot_examples(dists, xs, ys, name)

    def advancedWait(self):
        ''' Desired md.goal_pose waiting for real md.eef_pose

        '''
        time.sleep(2)
        if not self.samePoses(md.eef_pose, md.goal_pose, accuracy=0.02):
            print("Pose diff: ", round(self.distancePoses(md.eef_pose, md.goal_pose),2))
        else: return
        time.sleep(2)
        if not self.samePoses(md.eef_pose, md.goal_pose, accuracy=0.06):
            print("Pose diff: ", round(self.distancePoses(md.eef_pose, md.goal_pose),2))
        else: return
        time.sleep(2)
        if not self.samePoses(md.eef_pose, md.goal_pose, accuracy=0.15):
            print("Pose diff: ", round(self.distancePoses(md.eef_pose, md.goal_pose),2))
            print("[WARN*] Position not accurate")
        else: return
        time.sleep(5)
        print("[WARN*] Position failed")

    def testInit(self):
        print("[MoveIt*] Init test")

        pose = Pose()
        pose.orientation = md.ENV_DAT['above']['ori']
        pose.position = Point(0.4,0.,1.0)
        md.goal_pose = deepcopy(pose)
        self.advancedWait()
        print("[MoveIt*] Init test 1, error: ", round(self.distancePoses(md.eef_pose, pose), 2), " [x,y,z] diff: ", np.subtract(extv(md.eef_pose.position), extv(md.goal_pose.position)))

        pose.orientation = md.ENV_DAT['wall']['ori']
        pose.position = Point(0.7,0.1,0.5)
        md.goal_pose = deepcopy(pose)
        self.advancedWait()
        print("[MoveIt*] Init test 2 1/5, error: ", round(self.distancePoses(md.eef_pose, pose), 2), " [x,y,z] diff: ", np.subtract(extv(md.eef_pose.position), extv(md.goal_pose.position)))
        pose.position = Point(0.7,-0.1,0.5)
        md.goal_pose = deepcopy(pose)
        self.advancedWait()
        print("[MoveIt*] Init test 2 2/5, error: ", round(self.distancePoses(md.eef_pose, pose), 2), " [x,y,z] diff: ", np.subtract(extv(md.eef_pose.position), extv(md.goal_pose.position)))
        pose.position = Point(0.7,-0.1,0.4)
        md.goal_pose = deepcopy(pose)
        self.advancedWait()
        print("[MoveIt*] Init test 2 3/5, error: ", round(self.distancePoses(md.eef_pose, pose), 2), " [x,y,z] diff: ", np.subtract(extv(md.eef_pose.position), extv(md.goal_pose.position)))
        pose.position = Point(0.7,0.1,0.4)
        md.goal_pose = deepcopy(pose)
        self.advancedWait()
        print("[MoveIt*] Init test 2 4/5, error: ", round(self.distancePoses(md.eef_pose, pose), 2), " [x,y,z] diff: ", np.subtract(extv(md.eef_pose.position), extv(md.goal_pose.position)))
        pose.position = Point(0.7,0.1,0.5)
        md.goal_pose = deepcopy(pose)
        self.advancedWait()
        print("[MoveIt*] Init test 2 5/5, error: ", round(self.distancePoses(md.eef_pose, pose), 2), " [x,y,z] diff: ", np.subtract(extv(md.eef_pose.position), extv(md.goal_pose.position)))

        pose.orientation = md.ENV_DAT['table']['ori']
        pose.position = Point(0.4,-0.1,0.2)
        md.goal_pose = deepcopy(pose)
        self.advancedWait()
        print("[MoveIt*] Init test 3 1/5, error: ", round(self.distancePoses(md.eef_pose, pose), 2), " [x,y,z] diff: ", np.subtract(extv(md.eef_pose.position), extv(md.goal_pose.position)))
        pose.position = Point(0.6,-0.1,0.2)
        md.goal_pose = deepcopy(pose)
        self.advancedWait()
        print("[MoveIt*] Init test 3 2/5, error: ", round(self.distancePoses(md.eef_pose, pose), 2), " [x,y,z] diff: ", np.subtract(extv(md.eef_pose.position), extv(md.goal_pose.position)))
        pose.position = Point(0.6,0.1,0.2)
        md.goal_pose = deepcopy(pose)
        self.advancedWait()
        print("[MoveIt*] Init test 3 3/5, error: ", round(self.distancePoses(md.eef_pose, pose), 2), " [x,y,z] diff: ", np.subtract(extv(md.eef_pose.position), extv(md.goal_pose.position)))
        pose.position = Point(0.4,0.1,0.2)
        md.goal_pose = deepcopy(pose)
        self.advancedWait()
        print("[MoveIt*] Init test 3 4/5, error: ", round(self.distancePoses(md.eef_pose, pose), 2), " [x,y,z] diff: ", np.subtract(extv(md.eef_pose.position), extv(md.goal_pose.position)))
        pose.position = Point(0.4,-0.1,0.2)
        md.goal_pose = deepcopy(pose)
        self.advancedWait()
        print("[MoveIt*] Init test 3 5/5, error: ", round(self.distancePoses(md.eef_pose, pose), 2), " [x,y,z] diff: ", np.subtract(extv(md.eef_pose.position), extv(md.goal_pose.position)))

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
                md.goal_pose = deepcopy(pose)
                time.sleep(8)
                print("Distance between given and real coords.: ", round(self.distancePoses(md.eef_pose, pose), 2), " [x,y,z] diff: ", np.subtract(extv(md.eef_pose.position), extv(md.goal_pose.position)))
            except ValueError:
                print("[MoveIt*] Test ended")
                break


    def testMovements(self):
        xs=list(range(4,9))
        ys=list(range(-2,3))
        name="table"

        poses = []
        pose = Pose()
        pose.orientation = md.ENV_DAT['table']['ori']
        dists = []
        for i in ys:
            row = []
            for j in xs:
                pose.position = Point(j*0.1,i*0.1,0.1)
                p = deepcopy(pose)

                md.goal_pose = deepcopy(p)
                t=time.time()
                while not(abs(time.time()-t) > 10 or self.samePoses(md.eef_pose, p, accuracy=0.01)):
                    pass
                row.append(self.distancePoses(md.eef_pose, p))
            dists.append(row)


    def testTrajectoryActionClient(self):
        md.mode = ''
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

            md._goal.trajectory.header.stamp = rospy.Time.now()
            settings.mo.tac.add_goal(deepcopy(md._goal))
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



### Useful to remember
'''
# Set number of planning attempts
move_group.set_num_planning_attempts(self.PLANNING_ATTEMPTS)
# Display trajectory
self.display_trajectory(plan)

'''
