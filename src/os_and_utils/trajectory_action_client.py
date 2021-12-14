#!/usr/bin/env python2
"""
Copyright (c) 2019 Jan Behrens
All rights reserved.

This source code is licensed under the BSD-3-Clause license found in the
LICENSE file in the root directory of this source tree.

@author: Jan Behrens
"""
from __future__ import print_function

import time
# from scipy.interpolate import bsplines

from actionlib import ClientGoalHandle
from math import pi
from copy import deepcopy

import rospy
import trajectory_msgs
import moveit_commander
from rospy import Duration

from sensor_msgs.msg import JointState
import cProfile, pstats
try:
    from StringIO import StringIO ## for Python 2
except ImportError:
    from io import StringIO ## for Python 3

from copy import copy
import sys

import actionlib

from control_msgs.msg import FollowJointTrajectoryAction, FollowJointTrajectoryGoal
from control_msgs.srv import QueryTrajectoryState, QueryTrajectoryStateRequest, QueryTrajectoryStateResponse

from rospy import ROSInitException
from trajectory_msgs.msg import JointTrajectoryPoint
from moveit_commander.move_group import RobotState, RobotTrajectory

import numpy as np
from scipy import interpolate
import cmath

class TrajectoryActionClient:
    def __init__(self, arm=None, moveGroupCommander=None, topic='/position_joint_trajectory_controller/follow_joint_trajectory', robotCommander=None, topic_joint_states='/franka_state_controller/joint_states'):
        if moveGroupCommander is None:
            assert isinstance(arm, str)
            if arm == 'r1' or arm == 'r2':
                # TODO: this is a workaround to keep the code working for the kukas and for other robots like the kawada nextage
                moveGroupCommander = moveit_commander.MoveGroupCommander(arm + "_arm")
            else:
                moveGroupCommander = moveit_commander.MoveGroupCommander(arm)
            rospy.sleep(2.0)
        else:
            assert isinstance(moveGroupCommander, moveit_commander.MoveGroupCommander)

        if robotCommander is None:
            robotCommander = moveit_commander.RobotCommander()
            rospy.sleep(2.0)
        else:
            assert isinstance(robotCommander, moveit_commander.RobotCommander)

        self._mg = moveGroupCommander
        self._rc = robotCommander  # type: moveit_commander.RobotCommander

        try:
            rospy.get_time()
        except ROSInitException:
            print("ros node not initialized.")
            rospy.init_node("traj_action_client")

        self._activeJoints = moveGroupCommander.get_active_joints()

        self._client = actionlib.SimpleActionClient(topic, FollowJointTrajectoryAction)

        # self._client.feedback_cb

        self._goal = FollowJointTrajectoryGoal()

        self._current_state = None

        self._state_sub = rospy.Subscriber(topic_joint_states, JointState, self.robotstate_cb)

        # TODO: make parameter for topic
        self._querystate_srv = rospy.ServiceProxy('/r2/trajectory_controller/query_state', QueryTrajectoryState)

        server_up = self._client.wait_for_server(timeout=rospy.Duration(15.0))

        if not server_up:
            rospy.logerr("Timed out waiting for Joint Trajectory"
                         " Action Server to connect. Start the action server"
                         " and check the procided topic.")
            rospy.signal_shutdown("Timed out waiting for Action Server")
            sys.exit(1)

        self.clear()

    def robotstate_cb(self, msg):
        if self._current_state is not None:
            self._current_accel = np.true_divide(np.array(msg.velocity) - np.array(self._current_state.velocity), (msg.header.stamp - self._current_state.header.stamp).to_sec())
        self._current_state = msg

        #print(self._current_accel)
        #print(self._current_state)

    def add_plan(self, plan):
        #  type: (plan) -> None
        self.clear()
        for point in plan.joint_trajectory.points:
            self.add_joint_trajectory_point(point)

    def add_goal(self, goal):
        assert isinstance(goal, FollowJointTrajectoryGoal)
        self._goal = goal

    def add_joint_trajectory_point(self, point):
        point = deepcopy(point)
        self._goal.trajectory.points.append(point)

    def add_point(self, positions, time):
        point = JointTrajectoryPoint()
        point.positions = copy(positions)
        point.time_from_start = rospy.Duration(time)
        self._goal.trajectory.points.append(point)

    def replace(self, start_time=0.0):
        #self._goal.trajectory.header.stamp = rospy.Time.now() + rospy.Duration.from_sec(start_time)
        self._client.send_goal(self._goal)

    def start(self, delay=0.0, done_cb=None, active_cb=None, feedback_cb=None):
        self._goal.trajectory.header.stamp = rospy.Time.now() + rospy.Duration(delay)
        self._client.send_goal(self._goal, done_cb=done_cb, active_cb=active_cb, feedback_cb=feedback_cb)

    def stop(self):
        self._client.cancel_all_goals()

    def wait(self, timeout=15.0):
        self._client.wait_for_result(rospy.Duration().from_sec(timeout))
        return self.result()

    def result(self):
        # return self._client.get_result()
        return self._client.get_result()

    def state(self):
        # Documentation of states: http://docs.ros.org/kinetic/api/actionlib_msgs/html/msg/GoalStatus.html
        if len(self._goal.trajectory.points) == 0:
            return 3
        gh = self._client.gh  # type: ClientGoalHandle
        try:
            return gh.comm_state_machine.latest_goal_status.status
        except AttributeError:
            # returning success if gh is None
            return 9

    def clear(self):
        self._goal = FollowJointTrajectoryGoal()
        self._goal.trajectory.joint_names = self._activeJoints

    def get_current_tp(self):
        t_s = self._goal.trajectory.header.stamp
        t_now = rospy.Time.now()
        gh = self._client.gh  # type: ClientGoalHandle
        t_gh = gh.comm_state_machine.latest_goal_status.goal_id.stamp
        # goal_status = gh.get_goal_status()

        dt_now = t_now - t_s
        dt_gh = t_gh - t_s

        for index, tp in enumerate(self._goal.trajectory.points):  # type: int, JointTrajectoryPoint
            if tp.time_from_start >= dt_now:
                return index, tp



    def status(self):
        print("get_state: {}".format(self._client.get_state()))
        print("get_result: {}".format(self._client.get_result()))
        print("simple_state: {}".format(self._client.simple_state))

        gh = self._client.gh  # type: ClientGoalHandle

        if gh is None:
            return
        t_s = self._goal.trajectory.header.stamp
        t_now = rospy.Time.now()
        t_gh = gh.comm_state_machine.latest_goal_status.goal_id.stamp

        dt_now = t_now - t_s
        dt_gh = t_gh - t_s
        print("dt_now: {}".format(dt_now.to_sec()))
        print("dt_gh: {}".format(dt_gh.to_sec()))

    @staticmethod
    def plot_plan(plans=[], start_times=[], joints=[], name=''):
        import numpy as np
        import matplotlib.pyplot as plt
        import matplotlib
        matplotlib.rcParams['pdf.fonttype'] = 42

        if len(start_times) < len(plans):
            start_times += (len(plans) - len(start_times)) * [0.0]

        color_list = plt.cm.Set3(np.linspace(0, 1, len(joints)*len(plans)))
        fig = plt.Figure()
        ax = fig.add_subplot(111)  # type: plt.Axes

        idx = 0

        for plan_idx, plan in enumerate(plans):
            assert isinstance(plan, RobotTrajectory)
            for joint in joints:
                j = plan.joint_trajectory.joint_names.index(joint)
                t = []
                data = []
                pos = []
                acc = []
                for tp in plan.joint_trajectory.points:
                    t.append(tp.time_from_start.to_sec() + start_times[plan_idx])
                    data.append(tp.velocities[j])
                    pos.append(tp.positions[j])
                    acc.append(tp.accelerations[j])

                # ax.scatter(t, data, color=color_list[idx], marker="^", label='plan {}, velocity joint {}'.format(plan_idx, joint))
                # ax.scatter(t, pos, color=color_list[idx], label='plan {}, position joint {}'.format(plan_idx, joint))
                # ax.scatter(t, acc, color=color_list[idx], marker='x', label='plan {}, acceleration joint {}'.format(plan_idx, joint))

                ax.scatter(t, data, color=color_list[idx], marker="^",
                           label='plan {}, vel'.format(plan_idx))
                ax.scatter(t, pos, color=color_list[idx], label='plan {}, pos'.format(plan_idx))
                ax.scatter(t, acc, color=color_list[idx], marker='x',
                           label='plan {}, acc'.format(plan_idx))

                idx += 1

                vs = interpolate.CubicSpline(t, data)
                BSpline
                # ss = vs.integrate(0, t[-1])
                aspline = vs.derivative()
                ss = vs.antiderivative()

                t_fine = np.arange(t[0], t[-1], 0.01)
                ax.plot(t_fine, vs(t_fine), color='r')

        ax.legend(loc=3)
        ax.set_xlabel('time [s]')
        ax.set_ylabel('position $[rad]$, velocity $[rad/s]$, and acceleration $[rad/s^2]$')
        fig.tight_layout()
        fig.savefig("plan_{}.pdf".format(name))
        return fig

    @staticmethod
    def pos(k, t):
        return np.polyval(k[::-1], t)

    @staticmethod
    def vel(k, t):
        der = np.polyder(np.poly1d(k[::-1]))
        return der(t)

    @staticmethod
    def acc(k, t):
        der = np.polyder(np.poly1d(k[::-1]), 2)
        return der(t)

    @staticmethod
    def pos3(k, t, pos0):
        return np.polyint(np.poly1d(k[::-1]), k=pos0)(t)


    @staticmethod
    def vel3(k, t):
        p = np.poly1d(k[::-1])
        return p(t)

    @staticmethod
    def acc3(k, t):
        der = np.polyder(np.poly1d(k[::-1]), 1)
        return der(t)

    def soft_stop_lin(self, te=0.8, a_max=3.0, a_min=-3.0):

        DEBUG_PLOT = True
        if DEBUG_PLOT:
            data = {}
        dt_calc = 0.09

        if self._client.get_state() not in [0, 1]:
            rospy.logerr("No trajectory execution in process nor pending.")
            return

        plan_orig = RobotTrajectory()
        plan_orig.joint_trajectory = deepcopy(self._goal.trajectory)

        if DEBUG_PLOT:
            data['orig_plan'] = plan_orig

        t_s = self._goal.trajectory.header.stamp
        t_now = rospy.Time.now()
        dt_now = t_now - t_s

        if DEBUG_PLOT:
            data['t_s'] = t_s
            data['t_now'] = t_now
            data['dt_now'] = dt_now

        # current_tp = JointTrajectoryPoint()
        # current_tp.positions = self._current_state.position
        # current_tp.velocities = self._current_state.velocity
        # current_tp.accelerations = self._current_accel
        # current_tp.time_from_start = self._current_state.header.stamp - t_now

        # print('current tp: ')
        # print(current_tp)

        current_tp_traj = None
        time_in = time.time()
        # create new goal with all non-passed TrajectoryPoints
        new_traj_goal = deepcopy(self._goal)  # type: FollowJointTrajectoryGoal
        for index, tp in enumerate(new_traj_goal.trajectory.points):  # type: int, JointTrajectoryPoint
            if tp.time_from_start >= dt_now + rospy.Duration.from_sec(dt_calc):
                current_tp_traj = new_traj_goal.trajectory.points[index]  # -1]
                new_traj_goal.trajectory.points = new_traj_goal.trajectory.points[index:]
                break

        if current_tp_traj is None:
            print(
                "No future points in trajectory. That means we are nearly stopped and can just wait for the trajectory to end.")
            return

        if DEBUG_PLOT:
            data['current_tp_traj'] = current_tp_traj

        # TODO: make the scaling funtions more general: at the moment they expect to be for a 7 DoF robot.
        if len(current_tp_traj.velocities) != 7:
            print('Soft_stop_lin')
            print('Current goal: \n')
            print(self._goal)

            print('Traj start: {}'.format(t_s))
            print('T_now: {}'.format(t_now))
            print('dt_now: {}'.format(dt_now))

            print(
                'The normal traj goal cannot help us. We just cancel the goal. This might activate the breaks of the robot.')
            # self.stop()
            return

        # # get start and final values for each joint
        # v0 = list(current_tp.velocities)
        # s0 = list(current_tp.positions)
        # a0 = list(current_tp.accelerations)
        #
        # # time of joint state + processing time
        # dt_simulate = dt_calc - current_tp.time_from_start.to_sec()
        # print("dt_simulate: {}".format(dt_simulate))
        #
        # # account for change of position during reaction time
        # s0 = np.array(s0) + np.array(v0) * dt_simulate
        # v0 = np.array(v0) + np.array(a0) * dt_simulate

        v0 = list(current_tp_traj.velocities)
        s0 = list(current_tp_traj.positions)
        a0 = list(current_tp_traj.accelerations)

        ve = 7 * [0.0]
        ae = 7 * [0.0]

        # find te by checking which joint has to break longest
        # dv = np.array(v0) - np.array(ve)
        dv = np.array(ve) - np.array(v0)

        # max_accel = np.max(np.array(self._current_state.velocity) * self._current_accel)
        # print('max_accel: {}'.format(max_accel))
        #
        # # if the robot is accelerating, give more time to stop
        # if max_accel > 0.2:
        #     dt_adapt = 0.0
        # else:
        #     dt_adapt = 0.0
        # te = np.max(np.abs(np.true_divide(dv, 4.0)))  # + dt_adapt
        te_decel = np.max(np.true_divide(dv, a_min))
        te_accel = np.max(np.true_divide(dv, a_max))
        te = max([te_decel, te_accel])

        # print("abs test: ")
        # print(np.abs(np.true_divide(dv, 3.5)))

        # s1 = np.array(s0) + np.array(v0) *dt_adapt

        if te < 0.05:
            te = 0.5

        if DEBUG_PLOT:
            data['te'] = te

        # calculate powers of te for the lin equation system
        t = [1, te]

        x = {}
        x3 = {}
        # for every joint, we create a linear equation system
        # v_0 = a_0
        # v_e = a_0 + t * a_1
        for joint in range(0, len(v0)):
            M3 = np.array([[1, 0],
                           [1, t[1]]])
            b3 = np.array([v0[joint], \
                           ve[joint]])

            # solve the thing
            x3[joint] = np.linalg.solve(M3, b3)

        # create a list of JointTrajectoryPoints and populate them with the data from the polynomial
        discr = 10

        tps = [JointTrajectoryPoint() for _ in range(0, discr)]

        t_start_dec = (current_tp_traj.time_from_start + new_traj_goal.trajectory.header.stamp - t_now).to_sec()
        # lin = np.linspace(dt_calc, dt_calc + te, discr)
        #lin = np.linspace(t_start_dec, t_start_dec + te, discr)
        lin = np.linspace(0.0, te, discr)
        print(lin)
        for tp, t in zip(tps, lin):
            tp.time_from_start = rospy.Duration.from_sec(current_tp_traj.time_from_start.to_sec() + t)
            tp.accelerations = 7 * [0.0]
            tp.velocities = 7 * [0.0]
            tp.positions = 7 * [0.0]

        # TODO: are we using the polynomial out of the range?
        for joint, pol in x3.items():
            st = TrajectoryActionClient.pos3(pol, lin, s0[joint])
            # st = TrajectoryActionClient.pos3(pol, lin, s1[joint])
            vt = TrajectoryActionClient.vel3(pol, lin)
            at = TrajectoryActionClient.acc3(pol, lin)
            for pos1, vel1, acc1, tp in zip(st, vt, at, tps):
                tp.positions[joint] = pos1
                tp.velocities[joint] = vel1
                tp.accelerations[joint] = acc1

        if DEBUG_PLOT:
            data['v0'] = v0
            data['s0'] = s0
            data['a0'] = a0
            data['ve'] = tps[-1].velocities
            data['se'] = tps[-1].positions
            data['ae'] = ae
            data['t_start_dec'] = t_start_dec
            data['x3'] = x3
            data['tps'] = tps

        # set the new trajectory to be the last point of our deceleration. This leaves the maximum freedom to the controller.
        # new_traj_goal.trajectory.points = [tps[-1]]
        # current_tp.time_from_start = rospy.Duration.from_sec(0.0)
        # first_tp = JointTrajectoryPoint()
        # first_tp.time_from_start = rospy.Duration.from_sec(dt_calc)
        # first_tp.positions = list(s0)
        # first_tp.velocities = list(v0)
        # first_tp.accelerations = 7 * [0.0]
        #
        # new_traj_goal.trajectory.points = [first_tp, tps[-1]]
        # new_traj_goal.trajectory.points = tps[3:]

        # tp = JointTrajectoryPoint()
        tps[-1].time_from_start = rospy.Duration.from_sec(current_tp_traj.time_from_start.to_sec() + te)
        #tps[-1].accelerations = 7 * [0.0]
        #tps[-1].velocities = 7 * [0.0]

        new_traj_goal.trajectory.points = tps# + [tps[-1]]

        time_out = time.time()
        tdiff = time_out - time_in

        print("te: {}".format(te))
        stopping_new = (plan_orig.joint_trajectory.header.stamp + current_tp_traj.time_from_start - t_now).to_sec() + te
        stopping_old = (plan_orig.joint_trajectory.header.stamp + plan_orig.joint_trajectory.points[
            -1].time_from_start - t_now).to_sec()
        print('stopping new: {}'.format(stopping_new))
        print('End of trajectory in {} s.'.format(stopping_old))

        if DEBUG_PLOT:
            data['new_traj_goal'] = new_traj_goal
            data['tdiff'] = tdiff
            data['stopping_new'] = stopping_new
            data['stopping_old'] = stopping_old

        if stopping_new >= stopping_old:
            print("Stopping time of motion would be later than normal stop. We do nothing.")
            return

        # new_traj_goal.trajectory.header.stamp = rospy.Time.now() - rospy.Duration.from_sec(tdiff)
        # new_traj_goal.trajectory.header.stamp = rospy.Time.now() - rospy.Duration.from_sec(tdiff)

        # new_traj_goal.trajectory.points = tps + plan.joint_trajectory.points[i+1:]
        self._goal = new_traj_goal
        self._client.send_goal(new_traj_goal)

        if DEBUG_PLOT:
            import cPickle as pickle
            path_to_file = "soft_stop_lin_log.pkl"
            with open(path_to_file, 'wb') as output:
                pickle.dump(data, output, pickle.HIGHEST_PROTOCOL)

        # print("trajectory: ")
        # print(new_traj_goal.trajectory.points)
        # print(new_traj_goal.trajectory.points[-1].time_from_start.to_sec())
        print("runtime of soft_stop_lin: {} s".format(tdiff))


    def get_joint_state(self, t_ref, t):
        req = QueryTrajectoryStateRequest()
        req.time = t_ref + t
        res = self._querystate_srv.call(req)
        assert isinstance(res, QueryTrajectoryStateResponse)
        print(res)

        tp = JointTrajectoryPoint()
        tp.accelerations = res.acceleration
        tp.positions = res.position
        tp.velocities = res.velocity

        tp.time_from_start = t

        return tp


    def scale_speed_on_path(self, scaling_factor=1.0, PROFILE=False, a_max=3.0, a_min=-3.0):

        if PROFILE:
            pr = cProfile.Profile()
            pr.enable()

        dt_calc = 0.4

        t_s = self._goal.trajectory.header.stamp
        t_now = rospy.Time.now()
        dt_now = t_now - t_s

        #goal_old = deepcopy(self._goal)
        plan_old = RobotTrajectory()
        #plan_old.joint_trajectory = goal_old.trajectory
        plan_old.joint_trajectory = self._goal.trajectory


        current_tp_traj = None
        time_in = time.time()
        # create new goal with all non-passed TrajectoryPoints
        #new_traj_goal = deepcopy(self._goal)  # type: FollowJointTrajectoryGoal

        test_time = dt_now + rospy.Duration.from_sec(dt_calc)

        # x_ref = self.get_joint_state(t_s,test_time)

        for index, tp in enumerate(plan_old.joint_trajectory.points):  # type: int, JointTrajectoryPoint
            if tp.time_from_start >= test_time:
                current_tp_traj = tp #new_traj_goal.trajectory.points[index]  # -1]
                x = index
                # new_traj_goal.trajectory.points = new_traj_goal.trajectory.points[index:]
                break

        if current_tp_traj is None:
            print(
                "No future points in trajectory. That means we are nearly stopped and can just wait for the trajectory to end.")
            return

        state = RobotState()
        state.joint_state.name = plan_old.joint_trajectory.joint_names
        state.joint_state.position = plan_old.joint_trajectory.points[0].positions
        state.joint_state.velocity = plan_old.joint_trajectory.points[0].velocities
        state.joint_state.header.stamp = plan_old.joint_trajectory.header.stamp
        state.joint_state.header.frame_id = plan_old.joint_trajectory.header.frame_id

        traj_in = RobotTrajectory()
        traj_in.joint_trajectory = plan_old.joint_trajectory


        plan_new = self._mg.retime_trajectory(ref_state_in=state, traj_in=traj_in,
                                           velocity_scaling_factor=scaling_factor)


        plan_merged = deepcopy(plan_old)
        # plan_merged = RobotTrajectory()
        # plan_merged.joint_trajectory.header = plan_old.joint_trajectory.header
        # plan_merged.joint_trajectory.joint_names = plan_old.joint_trajectory.joint_names

        dt = np.zeros(len(plan_new.joint_trajectory.points[x:-1]))

        idx = x - 1
        pt_old_start = plan_old.joint_trajectory.points[x]
        vel = np.array(pt_old_start.velocities)
        # plan_new.joint_trajectory.points[x].velocities = pt_old_start.velocities
        # plan_new.joint_trajectory.points[x].accelerations = pt_old_start.accelerations
        for pt1, pt2, pt_old, pt1_merge, pt2_merge in zip(plan_new.joint_trajectory.points[x:-1],
                                                          plan_new.joint_trajectory.points[x + 1:],
                                                          plan_old.joint_trajectory.points[x:-1],
                                                          plan_merged.joint_trajectory.points[x:-1],
                                                          plan_merged.joint_trajectory.points[x + 1:]):
            idx += 1
            print('IDX: {}'.format(idx))
            dx = np.array(pt2.positions) - np.array(pt1.positions)
            vel = np.array(pt1_merge.velocities)

            vel_new = np.array(pt1.velocities)
            if np.linalg.norm(vel_new - vel) < 0.0001:
                print('taking new point')
                pt2_merge.velocities = pt2.velocities
                pt1_merge.accelerations = pt1.accelerations
                dt[idx - x] = (pt2.time_from_start - pt1.time_from_start).to_sec()
                continue

            # vel = np.array(pt_old.velocities)
            #print(dx)
            #print(np.linalg.norm(vel))
            # acc = np.array([a_max if v_merged < v_n else a_min for v_merged, v_n in zip(pt1_merge.velocities, pt1.velocities)])
            acc = np.array(
                [a_max if v_merged < v_n else a_min for v_merged, v_n in zip(pt_old.velocities, pt1.velocities)])
            #print('acc: {}'.format(acc))

            # acc_max = np.true_divide(-(vel + vel ** 2), 2 * dx)
            # acc_min = np.true_divide(- (vel ** 2), 2 * dx)

            avg_vel = 0.5 * (vel + vel_new)
            #print('avg_vel: {}'.format(avg_vel))
            te_goal = dx / avg_vel
            a_tegoal = np.true_divide(vel_new - vel, te_goal)
            #print('te_goal: {}'.format(te_goal))
            #print('a te_goal: {}'.format(a_tegoal))

            a_limit = np.where(np.abs(avg_vel) < 0.01 * np.ones(a_tegoal.shape),
                               np.sign(a_tegoal) * 0.01 * np.ones(a_tegoal.shape), a_tegoal)

            # acc_max = np.true_divide(-(vel**2), 2*dx)
            # acc = np.where(np.abs(acc_max) < np.abs(acc), acc_max, acc)
            # acc = np.where(acc_min < acc, acc_min, acc)
            acc = np.where(np.abs(a_limit) < np.abs(acc), a_limit, acc)
            #print('acc: {}'.format(acc))

            te1, te2 = TrajectoryActionClient.solve_quad(0.5 * acc, vel, -dx)

            te1 = np.where(np.abs(dx) < 0.00001, -0.02 * np.ones(te1.shape), te1)
            te2 = np.where(np.abs(dx) < 0.00001, -0.02 * np.ones(te2.shape), te2)

            te1 = np.fmax(te1, -0.01 * np.ones(te1.shape))
            te2 = np.fmax(te2, -0.01 * np.ones(te2.shape))

            te_matrix = np.vstack((te1, te2))
            vmax = np.max(te_matrix, axis=0)
            vmin = np.min(te_matrix, axis=0)

            te = np.max(np.where((vmin >= 0), vmin, vmax))

            # te = np.max(np.fmin(te1[np.nonzero(te1)], te2[np.nonzero(te2)]))

            # te = np.max(np.fmin(te1[np.nonzero(te1)], te2[np.nonzero(te2)]))
            dt[idx - x] = te

            # if te < 0.0001:
            #     print('replace time with new travel time.')
            #     # acc_new = - np.true_divide(vel**2, 2.0 * dx)
            #     #
            #     # te1, te2 = solve_quad(0.5 *acc_new, vel, -dx)
            #     #
            #     # te1 = np.fmax(te1, np.zeros(te1.shape))
            #     # te2 = np.fmax(te2, np.zeros(te2.shape))
            #     #
            #     # te_matrix = np.vstack((te1, te2))
            #     # vmax = np.max(te_matrix,axis=0)
            #     # vmin = np.min(te_matrix,axis=0)
            #     #
            #     # te = np.max(np.where((vmin > 0), vmin, vmax))
            #     #
            #     # dt[idx-x] = te
            #     dt[idx - x] = np.max(np.true_divide(dx, np.true_divide(vel + np.array(pt1.velocities, 2))))
            #     te = dt[idx - x]

            #print('te: {}'.format(te))

            a = 2 * np.true_divide(dx - (vel * te), te ** 2)
            #print('a: ')
            #print(a)

            vel_e = vel + a * te

            #print(vel_e)

            #print("vel: old vs new")
            #print(np.abs(pt_old.velocities))
            #print(np.abs(pt1.velocities))

            #print(np.min(np.abs(pt1.velocities) - np.abs(pt_old.velocities)))
            #print(np.max(np.abs(pt1.velocities) - np.abs(pt_old.velocities)))

            if np.min(np.abs(pt1.velocities) - np.abs(pt_old.velocities)) < -0.0001:
                print("deceleration")
                vel_e_max = np.sign(pt2.velocities) * np.fmax(np.abs(vel_e), np.abs(pt2.velocities))
                a_final = np.where(np.abs(vel_e) > np.abs(pt2.velocities), a, pt1.accelerations)
            elif np.max(np.abs(pt1.velocities) - np.abs(pt_old.velocities)) > 0.0001:
                print("acceleration")
                vel_e_max = np.sign(pt2.velocities) * np.fmin(np.abs(vel_e), np.abs(pt2.velocities))
                a_final = np.where(np.abs(vel_e) < np.abs(pt2.velocities), a, pt1.accelerations)
            else:
                # vel_e_max = np.sign(pt2.velocities) * np.fmax(np.abs(vel_e), np.abs(pt2.velocities))
                vel_e_max = np.sign(pt2.velocities) * np.fmin(np.abs(vel_e), np.abs(pt2.velocities))

                print("desaster: no clear direction...")

            #print(vel_e_max)

            pt2_merge.velocities = tuple(vel_e_max)


            # a_final = np.array(pt1.accelerations)[np.isnan(vel_e)] +  a[~np.isnan(vel_e)]
            # a_final = np.where(np.isnan(vel_e), pt1.accelerations, a) #np.array(pt1.accelerations)[np.isnan(vel_e)] +  a[~np.isnan(vel_e)]

            #print(a_final)

            pt1_merge.accelerations = tuple(a_final)
            # print('----------------------------')

        dt_cumul = np.cumsum(dt)
        # t0 = plan_brand_new.joint_trajectory.points[x].time_from_start
        t0 = plan_old.joint_trajectory.points[x].time_from_start
        # plan_brand_new.joint_trajectory.points[x].time_from_start = t0
        for pt, delta_t in zip(plan_merged.joint_trajectory.points[x + 1:], dt_cumul):
            pt.time_from_start = t0 + rospy.Duration.from_sec(delta_t)

        self.plot_plan([plan_merged], joints=self._mg.get_active_joints())

        new_traj_goal = self._goal
        new_traj_goal.trajectory = plan_merged.joint_trajectory

        time_out = time.time()
        tdiff = time_out - time_in

        print('tdiff: {}'.format(tdiff))

        if PROFILE:
            pr.disable()

        self._goal = new_traj_goal
        self._client.send_goal(new_traj_goal)

        s = StringIO()
        sortby = 'cumtime'
        if PROFILE:
            ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
            ps.print_stats()
            print(s.getvalue())


    def scale_speed_on_path_plan(self, plan_old, dt_now, scaling_factor=1.0, PROFILE=False, a_max=3.0, a_min=-3.0):

        if PROFILE:
            pr = cProfile.Profile()
            pr.enable()

        dt_calc = 0.1

        # t_s = self._goal.trajectory.header.stamp
        # t_now = rospy.Time.now()
        # dt_now = t_now - t_s
        assert isinstance(dt_now, rospy.Duration)

        #goal_old = deepcopy(self._goal)
        # plan_old = RobotTrajectory()
        assert isinstance(plan_old, RobotTrajectory)
        #plan_old.joint_trajectory = goal_old.trajectory
        # plan_old.joint_trajectory = self._goal.trajectory


        current_tp_traj = None
        time_in = time.time()
        # create new goal with all non-passed TrajectoryPoints
        #new_traj_goal = deepcopy(self._goal)  # type: FollowJointTrajectoryGoal

        test_time = dt_now + rospy.Duration.from_sec(dt_calc)

        # x_ref = self.get_joint_state(t_s,test_time)

        for index, tp in enumerate(plan_old.joint_trajectory.points):  # type: int, JointTrajectoryPoint
            if tp.time_from_start >= test_time:
                current_tp_traj = tp #new_traj_goal.trajectory.points[index]  # -1]
                x = index
                # new_traj_goal.trajectory.points = new_traj_goal.trajectory.points[index:]
                break

        if current_tp_traj is None:
            print(
                "No future points in trajectory. That means we are nearly stopped and can just wait for the trajectory to end.")
            return

        state = RobotState()
        state.joint_state.name = plan_old.joint_trajectory.joint_names
        state.joint_state.position = plan_old.joint_trajectory.points[0].positions
        state.joint_state.velocity = plan_old.joint_trajectory.points[0].velocities
        state.joint_state.header.stamp = plan_old.joint_trajectory.header.stamp
        state.joint_state.header.frame_id = plan_old.joint_trajectory.header.frame_id

        traj_in = RobotTrajectory()
        traj_in.joint_trajectory = plan_old.joint_trajectory


        plan_new = self._mg.retime_trajectory(ref_state_in=state, traj_in=traj_in,
                                           velocity_scaling_factor=scaling_factor)


        plan_merged = deepcopy(plan_old)
        # plan_merged = RobotTrajectory()
        # plan_merged.joint_trajectory.header = plan_old.joint_trajectory.header
        # plan_merged.joint_trajectory.joint_names = plan_old.joint_trajectory.joint_names

        dt = np.zeros(len(plan_new.joint_trajectory.points[x:-1]))

        idx = x - 1
        pt_old_start = plan_old.joint_trajectory.points[x]
        vel = np.array(pt_old_start.velocities)
        # a_max_jerk = pt_old_start.accelerations + te_goal * 10
        # a_min_jerk = plan_old.joint_trajectory.points[x:-1] - te_goal * 10

        # plan_new.joint_trajectory.points[x].velocities = pt_old_start.velocities
        # plan_new.joint_trajectory.points[x].accelerations = pt_old_start.accelerations
        for pt1, pt2, pt_old, pt0_merge, pt1_merge, pt2_merge in zip(plan_new.joint_trajectory.points[x:-1],
                                                          plan_new.joint_trajectory.points[x + 1:],
                                                          plan_old.joint_trajectory.points[x:-1],
                                                                     plan_merged.joint_trajectory.points[x-1:-2],
                                                          plan_merged.joint_trajectory.points[x:-1],
                                                          plan_merged.joint_trajectory.points[x + 1:]):
            idx += 1
            print('IDX: {}'.format(idx))
            dx = np.array(pt2.positions) - np.array(pt1.positions)
            vel = np.array(pt1_merge.velocities)
            current_acc = np.array(pt0_merge.accelerations)

            vel_new = np.array(pt1.velocities)
            vel_next = np.array(pt2.velocities)
            if np.linalg.norm(vel_new - vel) < 0.0001:
                print('taking new point')
                pt2_merge.velocities = pt2.velocities
                pt1_merge.accelerations = pt1.accelerations
                dt[idx - x] = (pt2.time_from_start - pt1.time_from_start).to_sec()
                continue

            # vel = np.array(pt_old.velocities)
            #print(dx)
            #print(np.linalg.norm(vel))
            # acc = np.array([a_max if v_merged < v_n else a_min for v_merged, v_n in zip(pt1_merge.velocities, pt1.velocities)])
            acc = np.array(
                [a_max if v_merged < v_n else a_min for v_merged, v_n in zip(pt_old.velocities, pt1.velocities)])
            #print('acc: {}'.format(acc))

            # acc_max = np.true_divide(-(vel + vel ** 2), 2 * dx)
            # acc_min = np.true_divide(- (vel ** 2), 2 * dx)

            avg_vel = 0.5 * (vel + vel_next)
            #print('avg_vel: {}'.format(avg_vel))
            te_goal = dx / avg_vel
            a_tegoal = np.true_divide(vel_next - vel, te_goal)
            #print('te_goal: {}'.format(te_goal))
            #print('a te_goal: {}'.format(a_tegoal))

            a_limit = np.where(np.abs(avg_vel) < 0.01 * np.ones(a_tegoal.shape),
                               np.sign(a_tegoal) * 0.01 * np.ones(a_tegoal.shape), a_tegoal)

            # acc_max = np.true_divide(-(vel**2), 2*dx)
            # acc = np.where(np.abs(acc_max) < np.abs(acc), acc_max, acc)
            # acc = np.where(acc_min < acc, acc_min, acc)

            # te_goal is dependent on the acceleration
            a_max_jerk = current_acc + te_goal * 10
            a_min_jerk = current_acc - te_goal * 10

            a_dir_bound = np.where(a_max_jerk * a_min_jerk > 0, 1, 0)



            # find the factor to scale down the accelerations to not violate the limits
            acc_rel_max = np.true_divide(a_tegoal, np.fmin(np.array(7*[a_max]), a_max_jerk))
            acc_rel_min = np.true_divide(a_tegoal, np.fmax(np.array(7*[a_min]), a_min_jerk))
            acc_rel_mat = np.vstack((acc_rel_min, acc_rel_max))

            # acc_rel_rel = np.where(check_these==0, np.max(acc_rel_max, acc_rel_min), 0) + np.
            scale = 1.0
            opposite_bound = np.logical_and(np.where(acc_rel_min < 0, 1, 0), np.where(acc_rel_max < 0, 1, 0))
            if np.any(opposite_bound):
                # we want to decelerate, but we are still accelerating in the opposite direction
                # scale will be negative. We want the largest scale (closest to what we actually want)
                acc_rel_mat_max = np.max(acc_rel_mat, axis=0)
                acc_rel_mat_min = np.min(acc_rel_mat, axis=0)
                scale = np.min(np.where(opposite_bound==1, acc_rel_mat_min, 1))

                # raise NotImplementedError
            elif np.any(a_dir_bound):
                # we move in the right direction, but we have upper and lower relative acceleration both positive
                # we have to select the highest lower rel acc
                acc_rel_mat_min = np.min(acc_rel_mat, axis=0)
                scale1 = np.max(acc_rel_mat_min)
                scale2 = np.where(a_dir_bound==0, np.fmax(acc_rel_max, acc_rel_min), 7*[0])
                scale = max([scale1, np.max(scale2)])
            else:
                scale = np.max([np.max(acc_rel_max), np.max(acc_rel_min)])

            if scale > 1.0 or scale < 0:
                acc = a_tegoal / scale
            else:
                acc = a_tegoal
            # acc = np.where(np.abs(a_limit) < np.abs(acc), a_limit, acc)
            #print('acc: {}'.format(acc))

            te1, te2 = TrajectoryActionClient.solve_quad(0.5 * acc, vel, -dx)

            # replace the the times for joints which have small or no motion with a small negtive number
            te1 = np.where(np.abs(dx) < 0.00001, -0.02 * np.ones(te1.shape), te1)
            te2 = np.where(np.abs(dx) < 0.00001, -0.02 * np.ones(te2.shape), te2)

            #
            te1 = np.fmax(te1, -0.01 * np.ones(te1.shape))
            te2 = np.fmax(te2, -0.01 * np.ones(te2.shape))

            #
            te_matrix = np.vstack((te1, te2))
            vmax = np.max(te_matrix, axis=0)
            vmin = np.min(te_matrix, axis=0)

            # take vmin if it is positive, otherwise vmax. the endtime is the max of all of these
            te = np.max(np.where((vmin >= 0), vmin, vmax))

            # te = np.max(np.fmin(te1[np.nonzero(te1)], te2[np.nonzero(te2)]))

            # te = np.max(np.fmin(te1[np.nonzero(te1)], te2[np.nonzero(te2)]))
            dt[idx - x] = te

            # if te < 0.0001:
            #     print('replace time with new travel time.')
            #     # acc_new = - np.true_divide(vel**2, 2.0 * dx)
            #     #
            #     # te1, te2 = solve_quad(0.5 *acc_new, vel, -dx)
            #     #
            #     # te1 = np.fmax(te1, np.zeros(te1.shape))
            #     # te2 = np.fmax(te2, np.zeros(te2.shape))
            #     #
            #     # te_matrix = np.vstack((te1, te2))
            #     # vmax = np.max(te_matrix,axis=0)
            #     # vmin = np.min(te_matrix,axis=0)
            #     #
            #     # te = np.max(np.where((vmin > 0), vmin, vmax))
            #     #
            #     # dt[idx-x] = te
            #     dt[idx - x] = np.max(np.true_divide(dx, np.true_divide(vel + np.array(pt1.velocities, 2))))
            #     te = dt[idx - x]

            #print('te: {}'.format(te))

            a = 2 * np.true_divide(dx - (vel * te), te ** 2)
            #print('a: ')
            #print(a)

            vel_e = vel + a * te

            #print(vel_e)

            #print("vel: old vs new")
            #print(np.abs(pt_old.velocities))
            #print(np.abs(pt1.velocities))

            #print(np.min(np.abs(pt1.velocities) - np.abs(pt_old.velocities)))
            #print(np.max(np.abs(pt1.velocities) - np.abs(pt_old.velocities)))

            if np.min(np.abs(pt1.velocities) - np.abs(pt_old.velocities)) < -0.0001:
                print("deceleration")
                vel_e_max = np.sign(pt2.velocities) * np.fmax(np.abs(vel_e), np.abs(pt2.velocities))
                # a_final = np.where(np.abs(vel_e) > np.abs(pt2.velocities), a, pt1.accelerations)
                a_final = a
            elif np.max(np.abs(pt1.velocities) - np.abs(pt_old.velocities)) > 0.0001:
                print("acceleration")
                vel_e_max = np.sign(pt2.velocities) * np.fmin(np.abs(vel_e), np.abs(pt2.velocities))
                a_final = a # np.where(np.abs(vel_e) < np.abs(pt2.velocities), a, pt1.accelerations)
            else:
                # vel_e_max = np.sign(pt2.velocities) * np.fmax(np.abs(vel_e), np.abs(pt2.velocities))
                vel_e_max = np.sign(pt2.velocities) * np.fmin(np.abs(vel_e), np.abs(pt2.velocities))

                print("desaster: no clear direction...")

            #print(vel_e_max)

            pt2_merge.velocities = tuple(vel_e_max)


            # a_final = np.array(pt1.accelerations)[np.isnan(vel_e)] +  a[~np.isnan(vel_e)]
            # a_final = np.where(np.isnan(vel_e), pt1.accelerations, a) #np.array(pt1.accelerations)[np.isnan(vel_e)] +  a[~np.isnan(vel_e)]

            #print(a_final)

            pt1_merge.accelerations = tuple(a_final)
            # print('----------------------------')

        dt_cumul = np.cumsum(dt)
        # t0 = plan_brand_new.joint_trajectory.points[x].time_from_start
        t0 = plan_old.joint_trajectory.points[x].time_from_start
        # plan_brand_new.joint_trajectory.points[x].time_from_start = t0
        for pt, delta_t in zip(plan_merged.joint_trajectory.points[x + 1:], dt_cumul):
            pt.time_from_start = t0 + rospy.Duration.from_sec(delta_t)

        self.plot_plan([plan_merged], joints=self._mg.get_active_joints())

        # new_traj_goal = self._goal
        new_traj_goal = FollowJointTrajectoryGoal()
        new_traj_goal.trajectory = plan_merged.joint_trajectory
        # new_traj_goal.trajectory = plan_merged.joint_trajectory

        time_out = time.time()
        tdiff = time_out - time_in

        print('tdiff: {}'.format(tdiff))

        if PROFILE:
            pr.disable()

        # self._goal = new_traj_goal
        # self._client.send_goal(new_traj_goal)

        s = StringIO()
        sortby = 'cumtime'
        if PROFILE:
            ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
            ps.print_stats()
            print(s.getvalue())

        return new_traj_goal


    @staticmethod
    def solve_quad(a, b, c):
        # Solve the quadratic equation ax**2 + bx + c = 0

        # calculate the discriminant
        d = (b ** 2) - (4 * a * c)

        # find two solutions
        sol1 = (-b - np.sqrt(d)) / (2 * a)
        sol2 = (-b + np.sqrt(d)) / (2 * a)

        # print('The solution are {0} and {1}'.format(sol1, sol2))

        return sol1, sol2
