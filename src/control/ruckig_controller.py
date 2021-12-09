#!/usr/bin/env python3.8
'''
**Tests around Trajectory replacement (ruckig version)**
Standalone script:
Launch:
    1. term: roslaunch panda_launch start_robot.launch
    2. term: rosrun mirracle_gestures joint_controller.py
        - For launch test
rosrun mirracle_gestures joint_controller_ruckig.py home

'''
from sys import path
path.append('..')
import trajectory_action_client

import rospy
from sensor_msgs.msg import JointState
from moveit_msgs.msg import RobotTrajectory
from control_msgs.msg import FollowJointTrajectoryGoal, JointTolerance
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

import numpy as np
from copy import deepcopy, copy
import collections

from ruckig import InputParameter, OutputParameter, Result, Ruckig, Trajectory
import time

class JointController():

    def __init__(self, args):
        self.args = args
        self.trajectory_points = args.trajectory_points
        rospy.init_node("joint_controller")

        self.waypoints_queue_joints = []

        self.joint_state = JointState()
        self.joint_state.position = [0.,0.,0.,0.,0.,0.,0.]
        self.joint_state.velocity = [0.,0.,0.,0.,0.,0.,0.]
        self.joint_state.effort =   [0.,0.,0.,0.,0.,0.,0.]
        self.joint_states = collections.deque(maxlen=20)

        rospy.Subscriber("/joint_states", JointState, self.save_joints)

        self.tac = trajectory_action_client.TrajectoryActionClient(arm='panda_arm', topic='/position_joint_trajectory_controller/follow_joint_trajectory', topic_joint_states = '/franka_state_controller/joint_states')
        self.rate = rospy.Rate(self.args.rate)

        self._goal = FollowJointTrajectoryGoal()
        self._goal.trajectory.joint_names = ['panda_joint1', 'panda_joint2', 'panda_joint3', 'panda_joint4', 'panda_joint5', 'panda_joint6', 'panda_joint7']
        self._goal.trajectory.header.seq = 0
        self._goal.trajectory.header.frame_id = 'panda_link0'
        for i in range(self.trajectory_points):
            jtp = JointTrajectoryPoint()
            jtp.positions = [0.,0.,0.,0.,0.,0.,0.]
            jtp.velocities = [0.,0.,0.,0.,0.,0.,0.]
            jtp.accelerations = [0.,0.,0.,0.,0.,0.,0.]
            jtp.time_from_start = 0.01*i
            self._goal.trajectory.points.append(deepcopy(jtp))

        self._tmpgoal = FollowJointTrajectoryGoal()
        for i in range(self.trajectory_points):
            jtp = JointTrajectoryPoint()
            jtp.positions = [0.,0.,0.,0.,0.,0.,0.]
            jtp.velocities = [0.,0.,0.,0.,0.,0.,0.]
            jtp.accelerations = [0.,0.,0.,0.,0.,0.,0.]
            jtp.time_from_start = 0.0
            self._tmpgoal.trajectory.points.append(deepcopy(jtp))

        self._tmpgoal2 = FollowJointTrajectoryGoal()
        for i in range(self.trajectory_points):
            jtp = JointTrajectoryPoint()
            jtp.positions = [0.,0.,0.,0.,0.,0.,0.]
            jtp.velocities = [0.,0.,0.,0.,0.,0.,0.]
            jtp.accelerations = [0.,0.,0.,0.,0.,0.,0.]
            jtp.time_from_start = 0.0
            self._tmpgoal2.trajectory.points.append(deepcopy(jtp))

        self.r = ruckig_wrapper(self.trajectory_points)

        # time duration of replacement function
        self.time_of_replace = 0.01

        # TEMPORARY: Additional topic for publishing into bag files
        self.pub_traj = rospy.Publisher('/followjointtrajectorygoalforplot',FollowJointTrajectoryGoal,queue_size=5)
        print("initializing publisher")
        time.sleep(5) ### It won't publish a first message otherwise, is it a bug?
        print("done init")

    def save_joints(self, data):
        self.joint_state = data
        self.joint_states.append(data)
        # Delete waypoint from queue, if close
        # %timeit sameJoints ~1.5us
        if self.waypoints_queue_joints and self.sameJoints(data.position, self.waypoints_queue_joints[0]):
            del self.waypoints_queue_joints[0]
            print("deleted")

    def sameJoints(self, joints1, joints2, accuracy=0.02):
        ''' Checks if two type joints are near each other
            Copied from moveit_lib.py, edited
        Parameters:
            joints1 (type float[7])
            joints2 (type float[7])
            threshold (Float): sum of joint differences threshold
        '''
        if sum([abs(i[0]-i[1]) for i in zip(joints1, joints2)]) < accuracy:
            return True
        return False

    def tac_control_single(self, target_joints):
        ''' Basic trajectory action client control (NOT replacement of trajectory)
        '''
        self._goal.trajectory.header.seq += 1

        # ADDON: interpolate first?
        accelerations = np.gradient([joint_state.velocity for joint_state in self.joint_states], [joint_state.header.stamp.to_sec() for joint_state in self.joint_states], axis=0)
        acceleration = accelerations[-1,:]

        # this function will update self._goal trajectory
        self.r.compute_multiple(current_position=self.joint_state.position, current_velocity=self.joint_state.velocity, current_acceleration=acceleration, target_position=target_joints, target_velocity=np.zeros(7), target_acceleration=np.zeros(7), _goal=self._goal, _tmpgoal=self._tmpgoal)


        self._goal.trajectory.header.stamp = rospy.Time.now() + rospy.Duration(0.1)
        for n,pt in enumerate(self._goal.trajectory.points):
            self._goal.trajectory.points[n].time_from_start += rospy.Duration(0.05)
        self.tac.add_goal(self._goal)
        self.tac.replace()
        # TMP: publish Trajectory
        self.pub_traj.publish(self._goal)
        # TMP: return duration of trajectory (time_from_start of first point is zero)
        return self._goal.trajectory.points[-1].time_from_start.to_sec()

    def findPointInTrajAfterTime(self, time_horizon):
        ''' Compares two absolute times
        Parameters:
            time_horizon (rospy Duration()): Absolute time [s]
        Returns:
            Point (Int or None if no result): Point in trajectory after threshold
        '''
        for n, point in enumerate(self._goal.trajectory.points):
            if self._goal.trajectory.header.stamp.to_sec()+point.time_from_start.to_sec() > time_horizon.to_sec():
                return n
        return None

    def tac_control_rewrite_new_goal(self, target_joints):
        '''
        Parameters:
            target_joints (Float[waypoints x 7])
        '''
        # Handle single goal input
        if is_it_single_goal(target_joints): target_joints = [target_joints]
        # Only for validation purposes
        self.waypoints_queue_joints = []
        self.waypoints_queue_joints.extend(deepcopy(target_joints))

        self.tac_control_auto(target_joints)

    def tac_control_add_new_goal(self, target_joints):
        '''
        Parameters:
            target_joints (Float[waypoints x 7])
        '''
        # Handle single goal input
        if is_it_single_goal(target_joints): target_joints = [target_joints]
        # Only for validation purposes
        self.waypoints_queue_joints.extend(deepcopy(target_joints))

        self.tac_control_auto(target_joints)

    def tac_control_auto(self, target_joints):
        ''' Trajectory action client control with trajectory replacement
        Parameters:
            target_joints (Float[waypoints x 7]): Target robot positions
        '''
        # First time - call standard tac_control_single
        if self._goal.trajectory.header.seq == 0:
            self.tac_control_single(target_joints)
            return
        # 1. from time_hotizon (t*) -> find js1 - time where old trajectory changes to new one
        time_horizon = rospy.Time.now() + rospy.Duration(self.args.time_horizon)

        # js1 is the point where original trajectory is changed to new one
        index_js1 = self.findPointInTrajAfterTime(time_horizon)
        if not index_js1:
            print("INFO: Occured that previous trajectory ending earlier than new horizon, creating new trajectory!")
            time.sleep(self.args.time_horizon)
            self.tac_control_single(target_joints)
            return
        js1 = self._goal.trajectory.points[index_js1]

        # Drop points after js1
        # I don't want to drop point and append them again, shuffle the pointers to trajectory
        last_tfs = self._goal.trajectory.points[index_js1].time_from_start.to_sec()
        self._goal.trajectory.points = self._goal.trajectory.points[0:index_js1]

        # 2. make new trajectory (from js1 -> js2)
        self._goal.trajectory.header.seq += 1
        # ADDITIONAL: incrstartease trajectory velocity based on gesture
        self.r.compute_multiple(current_position=js1.positions, current_velocity=js1.velocities, current_acceleration=js1.accelerations, target_position=target_joints, target_velocity=np.zeros(7), target_acceleration=np.zeros(7), _goal=self._tmpgoal, _tmpgoal=self._tmpgoal2)

        # 3. combine with trajectory from js0 -> js1 -> js2 -> js3 -> js_n
        for n,point in enumerate(self._tmpgoal.trajectory.points):
            self._tmpgoal.trajectory.points[n].time_from_start = rospy.Time.from_sec(point.time_from_start.to_sec() + last_tfs)
        last_tfs = self._tmpgoal.trajectory.points[-1].time_from_start.to_sec() # every new trajectory aligns to old one
        self._goal.trajectory.points.extend(self._tmpgoal.trajectory.points)

        # 4. Choose the point which will be the first point executed by sent trajectory
        #     - There will be chosen time in the future, for example 10ms in the future
        t0= time.perf_counter()
        #     - Point of stamp is based on previous time_of_replace + 40ms additional offset
        time_stamp = rospy.Time.now() + rospy.Duration(self.time_of_replace) + rospy.Duration(0.04)
        #     - From this time, the index in computed trajectory is chosen
        index_js0_stamp = self.findPointInTrajAfterTime(time_stamp)
        #     - And all points before are discarded
        self._goal.trajectory.points = self._goal.trajectory.points[index_js0_stamp:]
        #     - The stamp should be also the time
        self._goal.trajectory.header.stamp = time_stamp
        #     - Zero time_from_start to start at zero
        tfs = self._goal.trajectory.points[0].time_from_start
        for n,point in enumerate(self._goal.trajectory.points):
            self._goal.trajectory.points[n].time_from_start -= tfs
        #     - Below is fixed Panda Issue, discarding 40ms of points in trajectory, because of inner Panda SplineInterpolator
        ind = 0
        for n,point in enumerate(self._goal.trajectory.points):
            if point.time_from_start.to_sec() > 0.04:
                ind = n
                break
        self._goal.trajectory.points = self._goal.trajectory.points[ind:]

        self.tac.add_goal(self._goal)
        self.tac.replace()
        print(f"Number: {self._goal.trajectory.header.seq}, points {len(self._goal.trajectory.points)}, index_js0_stamp: {index_js0_stamp}, time_from_start {self._goal.trajectory.points[0].time_from_start.to_sec()}, time now: {rospy.Time.now()}, time of stamp: {self._goal.trajectory.header.stamp.to_sec()}, self.time_of_replace {self.time_of_replace}")
        self.time_of_replace = time.perf_counter()-t0

        # TMP: Publish traj
        self.pub_traj.publish(self._goal)

def is_it_single_goal(goal):
    ''' Get input target_joints
    Parameters:
        goal (ndarray): 1. 1Darray with len 7
                        2. 2Darray with shape(waypoints x 7)
    Returns:
        is_single (Boolean): True if goal is single
    '''
    if isinstance(goal[0], (float,int)):
        return True
    return False

class ruckig_wrapper():
    def __init__(self, trajectory_points):
        ''' Initialized for Panda
        r = ruckig_wrapper()
        trajecotry = r.compute(current_position, current_velocity, current_acceleration, target_position, target_velocity, target_acceleration)
        '''
        self.trajectory_points = trajectory_points

        self.otg = Ruckig(7)  # DoFs
        self.inp = InputParameter(7)
        self.out = OutputParameter(7)
        self.trajectory = Trajectory(7)

        self.inp.max_velocity = [2.1750, 2.1750, 2.1750, 2.1750, 2.6100, 2.6100, 2.6100]
        self.inp.max_acceleration = [15, 7.5, 10, 12.5, 15, 20, 20]
        self.inp.max_acceleration = [i*0.1 for i in self.inp.max_acceleration]
        self.inp.max_jerk = [7500, 3750, 5000, 6250, 7500, 10000, 10000]
        self.inp.max_jerk = [i*0.07 for i in self.inp.max_jerk]
        #self.inp.minimum_duration = xxx

        # additional
        self.min_positions = [-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973]
        self.max_positions = [2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973]
        self.max_effort = [87, 87, 87, 87, 12, 12, 12]
        self.max_deffort = [1000, 1000, 1000, 1000, 1000, 1000, 1000]

    def compute(self, current_position, current_velocity, current_acceleration, target_position, target_velocity, target_acceleration, _goal):
        ''' Ruckig wrapper function to compute trajectory (single trajectory)
        '''
        self.inp.current_position = current_position
        self.inp.current_velocity = current_velocity
        self.inp.current_acceleration = current_acceleration
        self.inp.target_position = target_position
        self.inp.target_velocity = target_velocity
        self.inp.target_acceleration = target_acceleration

        # ERROR? Init Ruckig() again?
        result = self.otg.calculate(self.inp, self.trajectory)
        if result == Result.ErrorInvalidInput:
            print(self.inp)
            raise Exception('Invalid input!')

        trajectory_points = len(_goal.trajectory.points)
        ss = np.linspace(0,self.trajectory.duration,trajectory_points)
        # I didn't found better way to sample trajectory (https://github.com/pantor/ruckig/blob/master/include/ruckig/trajectory.hpp)
        # ~1ms for 1000 points
        #return [self.trajectory.at_time(ss[i]) for i in range(trajectory_points)]
        # ~1.5ms for 1000 points, but already in _goal variable
        for i in range(0, trajectory_points):
            _goal.trajectory.points[i].time_from_start = rospy.Duration(ss[i])
            _goal.trajectory.points[i].positions, _goal.trajectory.points[i].velocities, _goal.trajectory.points[i].accelerations = self.trajectory.at_time(ss[i])

    def compute_multiple(self, current_position, current_velocity, current_acceleration, target_position, target_velocity, target_acceleration, _goal, _tmpgoal):
        ''' Calls compute multiple times, rewrites _goal trajectory -> multiple trajectories are added to each other
            - uses _tmpgoal as memory for each waypoint trajectory
        '''
        last_tfs = 0.
        _goal.trajectory.points = []
        lastpositions, lastvelocities, lastacceleraions = current_position, current_velocity, current_acceleration
        for n,target_position_single in enumerate(target_position):
            # this function will update _goal trajectory
            current_p_target_positions = [current_position]
            current_p_target_positions.extend(target_position)
            target_velocity = self.target_velocity_based_on_future_waypoints(current_p_target_positions[n:])
            print(f" target velocity {target_velocity}")
            self.compute(current_position=lastpositions, current_velocity=lastvelocities, current_acceleration=lastacceleraions, target_position=target_position_single, target_velocity=target_velocity, target_acceleration=target_acceleration, _goal=_tmpgoal)

            # 3. combine with trajectory from js0 -> js1 -> js2 -> js3 -> js_n
            for n,point in enumerate(_tmpgoal.trajectory.points):
                _tmpgoal.trajectory.points[n].time_from_start = rospy.Time.from_sec(point.time_from_start.to_sec() + last_tfs)

            lastpositions = _tmpgoal.trajectory.points[-1].positions
            lastvelocities = _tmpgoal.trajectory.points[-1].velocities
            lastacceleraions = _tmpgoal.trajectory.points[-1].accelerations

            _goal.trajectory.points.extend(deepcopy(_tmpgoal.trajectory.points))
            last_tfs = _goal.trajectory.points[-1].time_from_start.to_sec() # every new trajectory aligns to old one

    def target_velocity_based_on_future_waypoints(self, target_position):
        '''
        1. (target_position[0] - current_position) < 0 defines zero return target velocity
        2. target_position[1][j]-target_position[0][j] defines how big the velocity will be
        Parameters:
            target_positions (Float[waypoints x 7]): All future waypoints
        '''
        if len(target_position) < 2:
            raise Exception("Wrong number of arguments")
        elif len(target_position) < 3:
            return np.zeros(7)

        p_diff_1 = (np.array(target_position[1]) - np.array(target_position[0]))
        p_diff_2 = (np.array(target_position[2]) - np.array(target_position[1]))

        def single_joint(p_diff, v_max, k=0.4):
            v_end = p_diff * v_max * k
            if v_end > v_max: v_end = v_max
            return v_end

        v_end = np.zeros(7)
        print(f"p_diff_1[1] {p_diff_1[1]}")
        for j in range(0,7):
            if (p_diff_1[j] > 0) == (p_diff_2[j] > 0):
                v_end[j] = single_joint(min(p_diff_1[j], p_diff_2[j], key=abs), self.inp.max_velocity[j])
            else:
                v_end[j] = 0.
        return v_end

    def online_compute(self, current_position, current_velocity, current_acceleration, target_position, target_velocity, target_acceleration):
        ''' ERROR: otg needs to be initialized to timediff as second parameter
        '''
        self.inp.current_position = current_position
        self.inp.current_velocity = current_velocity
        self.inp.current_acceleration = current_acceleration

        self.inp.target_position = target_position
        self.inp.target_velocity = target_velocity
        self.inp.target_acceleration = target_acceleration

        result = otg.update(self.inp, self.out)
        self.out.pass_to_input(self.inp)
        if result == Result.Working:
            raise Exception('Invalid input!')
        return self.out






###
