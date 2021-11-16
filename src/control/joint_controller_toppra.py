#!/usr/bin/env python3
'''
**Tests around Trajectory replacement**
Standalone script:
Launch:
    1. term: roslaunch panda_launch start_robot.launch
    2. term: rosrun mirracle_gestures joint_controller.py
        - For launch test
rosrun mirracle_gestures joint_controller_toppra.py home

'''
import sys
import rospy
import argparse

from sensor_msgs.msg import JointState
from moveit_msgs.msg import RobotTrajectory
from control_msgs.msg import FollowJointTrajectoryGoal, JointTolerance
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

import trajectory_action_client

import numpy as np
from copy import deepcopy

import toppra as ta
from toppra import SplineInterpolator

import matplotlib.pyplot as plt
import random
import time
import collections

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
            T = np.array([[np.cos(t), -np.sin(t)*np.cos(np.radians(al)), np.sin(t)*np.sin(np.radians(al)), a*np.cos(t)],
                  [np.sin(t), np.cos(t)*np.cos(np.radians(al)), -np.cos(t)*np.sin(np.radians(al)), a*np.sin(t)],
                  [0, np.sin(np.radians(al)), np.cos(np.radians(al)), d],
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

class JointController():

    def __init__(self, args):
        self.args = args
        rospy.init_node("joint_controller")

        self.joint_state = JointState()
        self.joint_state.position = [0.,0.,0.,0.,0.,0.,0.]
        self.joint_state.velocity = [0.,0.,0.,0.,0.,0.,0.]
        self.joint_state.effort = [0.,0.,0.,0.,0.,0.,0.]
        self.joint_states = collections.deque(maxlen=300)

        self.pub = rospy.Publisher("/franka_state_controller/joint_states", JointState, queue_size=5)
        rospy.Subscriber("/joint_states", JointState, self.save_joints)
        self.seq = 0

        self.tac = trajectory_action_client.TrajectoryActionClient(arm='panda_arm', topic='/position_joint_trajectory_controller/follow_joint_trajectory', topic_joint_states = '/franka_state_controller/joint_states')

        self.plotgoalpub1 = rospy.Publisher("/plot_goal", JointState, queue_size=5)
        self.plotgoalpub2 = rospy.Publisher("/plot_goal2", JointState, queue_size=5)
        self.plotgoalpub3 = rospy.Publisher("/plot_goal3", JointState, queue_size=5)
        #raw_input("/plot_goal topic publishers initialized, press enter to conitnue.")
        self.plotgoalpub = [self.plotgoalpub1, self.plotgoalpub2, self.plotgoalpub3]

        self.SAMPLE_JOINTS = [ [] ]
        self.ALIMS = 3.0
        self.rate = rospy.Rate(self.args.rate)

        self._goal = None

        self.trajn = 0
        self.savedPrevVel = None

    def __enter__(self):
        self.start_mode()

    def __exit__(self):
        pass

    def save_joints(self, data):
        self.joint_state = data
        self.joint_states.append(data)

    def start_mode(self):
        if self.args.experiment == 'home':
            self.goHome()
            return

        if self.args.experiment == 'random_joints':
            self.testRandomJoints()
        elif self.args.experiment == 'shortest_distance':
            self.testShortestDistance()
        elif self.args.experiment == 'shortest_distance_0':
            self.testShortestDistance(joint=0)
        elif self.args.experiment == 'shortest_distance_1':
            self.testShortestDistance(joint=1)
        elif self.args.experiment == 'repeat_same':
            self.testPubSamePlace()
        elif self.args.experiment == 'acceleration':
            self.testAccelerations()
        elif self.args.experiment == 'durations':
            self.testDurations()
        elif self.args.experiment == 'trajectory_replacement':
            self.testTwoTrajectories()

        print("Done")
        time.sleep(200)

    def testTwoTrajectories(self):
        self.SAMPLE_JOINTS = []
        # Init. waypoints, where every waypoint is newly created trajectory
        for i in range(0,10):
            self.SAMPLE_JOINTS.append([0.8, -0.5, -0.8, -1.9, 0.25, 2.3, 0.63])

        for loop in range(0,len(self.SAMPLE_JOINTS)):
            if rospy.is_shutdown():
                break
            self.tac_control_replacement(loop)
            self.rate.sleep()


    def plotGoalPubFun(self, _goal):
        ''' Publish Goal Trajectory to plot
        '''
        stamp = _goal.trajectory.header.stamp
        for i, pt in enumerate(_goal.trajectory.points):
            msg = JointState()
            msg.name = ['joint']
            msg.header.seq = i
            msg.header.stamp = rospy.Time(stamp.to_sec() + pt.time_from_start.to_sec())
            msg.position = pt.positions[0:1]
            msg.velocity = pt.velocities[0:1]
            ff=self.plotgoalpub[self.trajn]
            ff.publish(msg)

        self.trajn +=1

    def robotStopped(self, waitStart=3, waitEnd=0.):
        # first condition (robot got into move), maybe need to be altered later
        time.sleep(waitStart)
        # second condition (robot stopped)
        while not rospy.is_shutdown() and sum(abs(np.array(self.joint_state.velocity))) > 0.005:
            print("velocity ", sum(abs(np.array(self.joint_state.velocity))))
            time.sleep(0.5)
        time.sleep(waitEnd)

    def goHome(self):
        self.SAMPLE_JOINTS = [ [0.8, 0.2, -0.8, -1.9, 0.25, 2.3, 0.63] ]
        self.tac_control(0)
        self.robotStopped(waitEnd=3.)


    def testRandomJoints(self):
        self.SAMPLE_JOINTS = [ [-0.32, 1.53, -1.60, -1.45, 0.10, 3.21, 0.63],
        [-1.72, 0.42, -1.61, -1.98, -1.36, 0.49, -1.48],
        [2.43, -0.22, -0.49, -0.40, -1.88, 2.41, -0.99],
        [2.01, 0.06, -0.40, -2.16, -1.54, 3.02, -1.59],
        [0.93, 0.24, -0.97, -2.86, 0.22, 3.19, -0.41],
        [1.09, 0.19, 1.35, -2.67, 0.12, 1.32, -0.44] ]

        for loop in range(0,len(self.SAMPLE_JOINTS)):
            if rospy.is_shutdown():
                break
            self.tac_control(loop)

            print("velocity ", sum(abs(np.array(self.joint_state.velocity))))
            self.robotStopped()
            self.rate.sleep()

    def testAccelerations(self):
        self.SAMPLE_JOINTS = [ [-1.72, 0.42, -1.61, -1.98, -1.36, 0.49, -1.48],
        [-0.32, 1.53, -1.60, -1.45, 0.10, 3.21, 0.63],
        [2.43, -0.22, -0.49, -0.40, -1.88, 2.41, -0.99],
        [2.01, 0.06, -0.40, -2.16, -1.54, 3.02, -1.59],
        [0.93, 0.24, -0.97, -2.86, 0.22, 3.19, -0.41],
        [1.09, 0.19, 1.35, -2.67, 0.12, 1.32, -0.44] ]
        ALIMS_arr = [0.1, 0.2, 0.5, 1.0, 2.0, 3.0]

        for loop in range(0,len(self.SAMPLE_JOINTS)):
            if rospy.is_shutdown():
                break
            self.ALIMS = ALIMS_arr[loop]
            self.tac_control(loop)

            print("velocity ", sum(abs(np.array(self.joint_state.velocity))))
            self.robotStopped()
            self.rate.sleep()

    def testDurations(self):
        self.SAMPLE_JOINTS = [ [-1.72, 0.42, -1.61, -1.98, -1.36, 0.49, -1.48],
        [-0.32, 1.53, -1.60, -1.45, 0.10, 3.21, 0.63],
        [2.43, -0.22, -0.49, -0.40, -1.88, 2.41, -0.99],
        [2.01, 0.06, -0.40, -2.16, -1.54, 3.02, -1.59],
        [0.93, 0.24, -0.97, -2.86, 0.22, 3.19, -0.41],
        [1.09, 0.19, 1.35, -2.67, 0.12, 1.32, -0.44] ]
        DUR_arr = [10.0, 5.0, 3.0, 2.0, 1.0, 0.5]

        for loop in range(0,len(self.SAMPLE_JOINTS)):
            if rospy.is_shutdown():
                break

            self.args.trajectory_duration = DUR_arr[loop]
            self.tac_control(loop)

            print("velocity ", sum(abs(np.array(self.joint_state.velocity))))
            self.robotStopped()
            self.rate.sleep()

    def testShortestDistance(self, joint=6):
        self.SAMPLE_JOINTS = [ [-0.32, 1.53, -1.60, -1.45, 0.10, 3.21, 0.63] ]

        arr = np.exp([0.,1.,2.,3.]) * 0.01
        for i in arr:
            js = self.SAMPLE_JOINTS[0]
            js[joint] += i
            self.SAMPLE_JOINTS.append(js)

        for loop in range(0,len(self.SAMPLE_JOINTS)):
            if rospy.is_shutdown():
                break

            self.tac_control(loop)

            print("velocity ", sum(abs(np.array(self.joint_state.velocity))))
            self.robotStopped()
            self.rate.sleep()

    def testPubSamePlace(self):
        self.SAMPLE_JOINTS = [ [-0.32, 1.53, -1.60, -1.45, 0.10, 3.21, 0.63] ]

        arr = np.arange(0,20)
        for i in arr:
            js = self.SAMPLE_JOINTS[0]
            self.SAMPLE_JOINTS.append(js)

        for loop in range(0,len(self.SAMPLE_JOINTS)):
            if rospy.is_shutdown():
                break

            self.tac_control(loop)

            print("velocity ", sum(abs(np.array(self.joint_state.velocity))))
            self.robotStopped()
            self.rate.sleep()

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

    def retime_wrapper(self, trajectory):
        robot_traj = RobotTrajectory()
        robot_traj.joint_trajectory = deepcopy(trajectory)
        robot_traj_new = self.retime(plan=robot_traj, traj_duration=self.args.trajectory_duration)
        return robot_traj_new.joint_trajectory

    def tac_control(self, loop):
        ''' Basic trajectory action client control (NOT replacement of trajectory)
        '''
        goal = FollowJointTrajectoryGoal()
        self.seq += 1
        goal.trajectory.joint_names = ['panda_joint1', 'panda_joint2', 'panda_joint3', 'panda_joint4', 'panda_joint5', 'panda_joint6', 'panda_joint7']
        goal.trajectory.header.seq = self.seq
        goal.trajectory.header.frame_id = 'panda_link0'
        point = JointTrajectoryPoint()
        point.positions =     self.joint_state.position
        point.velocities =    self.joint_state.velocity
        point.accelerations = self.joint_state.effort
        point.time_from_start = rospy.Duration(0.0)
        goal.trajectory.points.append(deepcopy(point))

        point = JointTrajectoryPoint()
        point.positions =     self.SAMPLE_JOINTS[loop]
        print(point.positions)
        point.velocities =    [0.,  0.,  0.,  0.,  0.,  0.,  0.]
        point.accelerations = [0.,  0.,  0.,  0.,  0.,  0.,  0.]
        point.time_from_start = rospy.Duration(self.args.trajectory_duration)
        goal.trajectory.points.append(deepcopy(point))

        goal.trajectory = self.retime_wrapper(goal.trajectory)

        print("last time_from_start: ", goal.trajectory.points[-1].time_from_start.to_sec())
        for point in goal.trajectory.points:
            point.time_from_start = point.time_from_start + rospy.Duration(0.1)
        goal.trajectory.header.stamp = rospy.Time.now() + rospy.Duration(1.0)

        self.tac.add_goal(goal)
        self.tac.replace()

    def tac_control_replacement(self, loop):
        ''' Trajectory action client control with trajectory replacement
        '''
        # First time, new trajectory is generated
        if not self._goal:
            # new _goal trajectory
            self._goal = FollowJointTrajectoryGoal()
            self._goal.trajectory.joint_names = ['panda_joint1', 'panda_joint2', 'panda_joint3', 'panda_joint4', 'panda_joint5', 'panda_joint6', 'panda_joint7']
            # Current joints (now) (Robot is not moving) -> as start position
            js1_joints = self.joint_state.position
            joints = JointTrajectoryPoint()
            joints.positions = js1_joints
            joints.time_from_start = rospy.Time(0.)
            self._goal.trajectory.points.append(deepcopy(joints)) # 1. start point

            # Desired joint values, updated every loop
            js2_joints = np.array(self.SAMPLE_JOINTS[loop])
            print("sample joints", js2_joints)
            joints.positions = js2_joints
            joints.time_from_start = rospy.Time(self.args.trajectory_duration)
            self._goal.trajectory.points.append(deepcopy(joints)) # 2. goal point

            # Retime Trajectory Optimize Parametrization (Spline Interpolator 2 -> 1000 trajectory points)
            self._goal.trajectory = self.retime_wrapper(self._goal.trajectory)

            # Postpone start of trajectory by 0.1s set as default
            for point in self._goal.trajectory.points:
                point.time_from_start = point.time_from_start + rospy.Duration(0.1)

            self._goal.trajectory.header.stamp = rospy.Time.now()
            self.tac.add_goal(self._goal)
            self.tac.replace()
            #self.plotGoalPubFun(self._goal)
        else:
            # Starting point (joints) of the previous trajectory
            js0 = self._goal.trajectory.points[0]
            # 1. choose t* > tc => find js1
            computation_time = self.args.computation_time # time horizon of new trajectory

            def findPointInTrajAfterTime(trajectory, computation_time):
                assert type(computation_time)==float, "Wrong type, it is:"+str(type(computation_time))
                assert type(trajectory)==type(JointTrajectory()), "Wrong type"
                for n, point in enumerate(trajectory.points):
                    if point.time_from_start.to_sec() > computation_time:
                        return n
                return len(trajectory.points)-1

            # Point where original trajectory is changed to new one
            index_js1 = findPointInTrajAfterTime(self._goal.trajectory, computation_time)
            js1 = self._goal.trajectory.points[index_js1]
            js1_joints = js1.positions

            # 2. make RobotTrajectory from js1 -> js2
            # with js2 velocity according to hand velocity (gesture) (acc is zero)
            goal = FollowJointTrajectoryGoal()
            goal.trajectory.joint_names = ['panda_joint1', 'panda_joint2', 'panda_joint3', 'panda_joint4', 'panda_joint5', 'panda_joint6', 'panda_joint7']
            goal.trajectory.header.stamp = self._goal.trajectory.header.stamp
            joints = JointTrajectoryPoint()
            joints.positions = js1_joints
            joints.time_from_start = rospy.Time(0.)
            goal.trajectory.points.append(deepcopy(joints)) # 1. start point

            joints.positions = js2_joints
            joints.time_from_start = rospy.Time(self.args.trajectory_duration)
            goal.trajectory.points.append(deepcopy(joints)) # 2. goal point

            goal.trajectory = self.retime_wrapper(goal.trajectory)

            # 3. combine with trajectory from js0 -> js1 -> js2
            self._goal.trajectory.points = self._goal.trajectory.points[0:index_js1+1]

            tfs = self._goal.trajectory.points[index_js1].time_from_start
            # VARIABLE: Offset of new distance, must be bigger
            #   - Temporary, should be zero if second trajectory setup properly
            offset_new_distance = 0.5
            for n,point in enumerate(goal.trajectory.points):
                # VARIABLE: Discarding n points of new trajectory
                #   - Temporary, should be zero fi second trajectory setup properly
                if n >= 50:
                    point.time_from_start = rospy.Time.from_sec(point.time_from_start.to_sec() + tfs.to_sec() + offset_new_distance)
                    self._goal.trajectory.points.append(point)

            # 3.1 compute time between two trajectories
            def zeroTimeFromStart(offset):
                time0 = deepcopy(self._goal.trajectory.points[0].time_from_start)
                for n, pt in enumerate(self._goal.trajectory.points):
                    pt.time_from_start = pt.time_from_start-time0+rospy.Duration(offset)

            zeroTimeFromStart(0.)
            # VARIABLE: discard the first 0.1s
            DISCARD_FIRST = 0.1
            index = findPointInTrajAfterNow(self._goal.trajectory, DISCARD_FIRST)
            self._goal.trajectory.points = self._goal.trajectory.points[index:]

            self._goal.trajectory.header.stamp = rospy.Time.now()
            self.tac.replace()
            #self.plotGoalPubFun(self._goal)

    def retime(self, plan=None, cart_vel_limit=-1.0, secondorder=False, pt_per_s=20, curve_len=None, start_delay=0.0, traj_duration=5):
        assert isinstance(plan, RobotTrajectory)


        if not curve_len is None and cart_vel_limit > 0:
            n_grid = np.ceil(pt_per_s * curve_len / cart_vel_limit)
        else:
            n_grid = np.inf

        active_joints = plan.joint_trajectory.joint_names
        # ALERT: This function is not found
        #lower, upper, vel, effort = self.robot.get_joint_limits(active_joints)
        vel_lim = [2.1750, 2.1750, 2.1750, 2.1750, 2.6100, 2.6100, 2.6100]
        effort_lim = [ 87, 87, 87, 87, 12, 12, 12 ]
        lower_lim = [-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973]
        upper_lim = [2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973]

        lower, upper, vel, effort = lower_lim, upper_lim, vel_lim, effort_lim
        # prepare numpy arrays with limits for acceleration
        alims = np.zeros((len(active_joints), 2))
        alims[:, 1] = np.array(len(lower) * [self.ALIMS]) # 0.5
        alims[:, 0] = np.array(len(lower) * [-self.ALIMS])

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

        instance.set_desired_duration(traj_duration)
        sd_start = 0.0
        if not isinstance(self.savedPrevVel, type(None)):
            v1, lenx = self.savedPrevVel
            sd_start = v1 * len(instance.compute_feasible_sets()) / lenx



        jnt_traj = self.extractToppraTraj(instance.compute_trajectory(sd_start, 0.))

        # ts_sample = np.linspace(0, jnt_traj.duration, 10*len(plan.joint_trajectory.points))
        ts_sample = np.linspace(0, jnt_traj.duration, int(np.ceil(100 * jnt_traj.duration)))
        qs_sample = jnt_traj(ts_sample)
        qds_sample = jnt_traj(ts_sample, 1)
        qdds_sample = jnt_traj(ts_sample, 2)

        _, sd_vec, _ = instance.compute_parameterization(sd_start, 0.)
        v1 = sd_vec[int(self.args.computation_time*len(sd_vec)/jnt_traj.duration)]
        self.savedPrevVel = [v1, len(sd_vec)]


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


    def control(self):
        msg = JointState()
        self.seq += 1
        msg.header.seq = self.seq
        msg.header.stamp = rospy.Time.now()
        msg.header.frame_id = 'panda_link0'

        msg.name = ['panda_joint1', 'panda_joint2', 'panda_joint3', 'panda_joint4', 'panda_joint5', 'panda_joint6', 'panda_joint7']
        msg.position = [0.,0.,0.,0.,0.,0.,0.,0.]
        msg.velocity = [0.04,0.04,0.04,0.04,0.04,0.04,0.04]
        msg.effort = [0.,0.,0.,0.,0.,0.,0.,0.]

        self.pub.publish(msg)

    def savitzky_golay(self, y, window_size, order, deriv=0, rate=1):
        r"""Smooth (and optionally differentiate) data with a Savitzky-Golay filter.
        The Savitzky-Golay filter removes high frequency noise from data.
        It has the advantage of preserving the original shape and
        features of the signal better than other types of filtering
        approaches, such as moving averages techniques.
        Parameters
        ----------
        y : array_like, shape (N,)
            the values of the time history of the signal.
        window_size : int
            the length of the window. Must be an odd integer number.
        order : int
            the order of the polynomial used in the filtering.
            Must be less then `window_size` - 1.
        deriv: int
            the order of the derivative to compute (default = 0 means only smoothing)
        Returns
        -------
        ys : ndarray, shape (N)
            the smoothed signal (or it's n-th derivative).
        Notes
        -----
        The Savitzky-Golay is a type of low-pass filter, particularly
        suited for smoothing noisy data. The main idea behind this
        approach is to make for each point a least-square fit with a
        polynomial of high order over a odd-sized window centered at
        the point.
        Examples
        --------
        t = np.linspace(-4, 4, 500)
        y = np.exp( -t**2 ) + np.random.normal(0, 0.05, t.shape)
        ysg = savitzky_golay(y, window_size=31, order=4)
        import matplotlib.pyplot as plt
        plt.plot(t, y, label='Noisy signal')
        plt.plot(t, np.exp(-t**2), 'k', lw=1.5, label='Original signal')
        plt.plot(t, ysg, 'r', label='Filtered signal')
        plt.legend()
        plt.show()
        References
        ----------
        .. [1] A. Savitzky, M. J. E. Golay, Smoothing and Differentiation of
           Data by Simplified Least Squares Procedures. Analytical
           Chemistry, 1964, 36 (8), pp 1627-1639.
        .. [2] Numerical Recipes 3rd Edition: The Art of Scientific Computing
           W.H. Press, S.A. Teukolsky, W.T. Vetterling, B.P. Flannery
           Cambridge University Press ISBN-13: 9780521880688
        """
        import numpy as np
        from math import factorial

        try:
            window_size = np.abs(np.int(window_size))
            order = np.abs(np.int(order))
        except ValueError:
            raise ValueError("window_size and order have to be of type int")
        if window_size % 2 != 1 or window_size < 1:
            raise TypeError("window_size size must be a positive odd number")
        if window_size < order + 2:
            raise TypeError("window_size is too small for the polynomials order")
        order_range = range(order+1)
        half_window = (window_size -1) // 2
        # precompute coefficients
        b = np.mat([[k**i for i in order_range] for k in range(-half_window, half_window+1)])
        m = np.linalg.pinv(b).A[deriv] * rate**deriv * factorial(deriv)
        # pad the signal at the extremes with
        # values taken from the signal itself
        firstvals = y[0] - np.abs( np.subtract(y[1:half_window+1][::-1], y[0]) )
        lastvals = y[-1] + np.abs( np.subtract(y[-half_window-1:-1][::-1], y[-1]))
        y = np.concatenate((firstvals, y, lastvals))
        return np.convolve( m[::-1], y, mode='valid')

parser=argparse.ArgumentParser(description='')

parser.add_argument('--experiment', default="trajectory_replacement", type=str, help='(default=%(default)s)', required=True, choices=['home', 'random_joints', 'shortest_distance', 'shortest_distance_0', 'shortest_distance_1', 'repeat_same', 'acceleration', 'durations', 'trajectory_replacement'])
parser.add_argument('--trajectory_duration', default=10, type=float, help='(default=%(default)s)')
parser.add_argument('--rate', default=1, type=float, help='(default=%(default)s)')
parser.add_argument('--computation_time', default=3., type=float, help='(default=%(default)s)')

args=parser.parse_args()

with JointController(args):
    input()



###
'''
We need acceleration in the middle to be zero or setting it to the actual one

save the trajectory in rosbag


'''
