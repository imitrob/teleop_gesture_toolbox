#!/usr/bin/env python2
"""
ROS action server example
"""
import copy

import numpy as np
from relaxed_ik.msg import EEPoseGoals, JointAngles
from sensor_msgs.msg import JointState
from std_msgs.msg import ColorRGBA

import rospy
import actionlib
from visualization_msgs.msg import Marker
from geometry_msgs.msg import PoseStamped, Point, Pose, Quaternion, Transform, Vector3
from moveit_commander.move_group import MoveGroupCommander
from moveit_msgs.msg import RobotTrajectory, RobotState
from scipy import interpolate
from scipy.spatial.distance import euclidean
from tf.transformations import translation_matrix, quaternion_from_matrix, translation_from_matrix, quaternion_matrix
from traject_msgs.msg import CurveExecutionAction, CurveExecutionGoal, CurveExecutionResult, CurveExecutionFeedback, \
    ExecutionOptions, LoggingTask
from traject_msgs.srv import PlanTolerancedTrajecoryRequest, PlanTolerancedTrajecory, PlanTolerancedTrajecoryResponse
from trajectory_msgs.msg import JointTrajectoryPoint, JointTrajectory

from traj_complete_ros.toppra_eef_vel_ct import retime, plot_plan

from RelaxedIK.relaxedIK import RelaxedIK
# from start_here import config_file_name
from traj_complete_ros.LoggerProxy import LoggerProxy
from traj_complete_ros.geometry import R_axis_angle
from traj_complete_ros.trajectory_action_client import TrajectoryActionClient
from control_msgs.msg import FollowJointTrajectoryActionFeedback, FollowJointTrajectoryActionGoal, \
    FollowJointTrajectoryGoal, FollowJointTrajectoryFeedback

from pykdl_utils.kdl_kinematics import KDLKinematics
from urdf_parser_py.urdf import URDF, Collision, Mesh, Cylinder, Box, Sphere

from traj_complete_ros.iiwa_fk import iiwa_jacobian

def get_current_robot_state(mg):
    joints = mg.get_active_joints()
    js = mg.get_current_joint_values()
    if np.allclose(js, [0.0 for _ in range(7)]):
        js = mg.get_current_joint_values()

    rs1 = RobotState()
    rs1.joint_state.name = joints
    rs1.joint_state.position = js

    return rs1


def state_from_plan_point(plan, point):
    rs = RobotState()
    rs.joint_state.name = plan.joint_trajectory.joint_names
    pt = plan.joint_trajectory.points[point]
    assert isinstance(pt, JointTrajectoryPoint)
    rs.joint_state.position = pt.positions
    rs.joint_state.velocity = pt.velocities

    return rs

class CurveExecutor(object):
    def __init__(self, mg='r1_arm'):
        self.server = actionlib.SimpleActionServer('curve_executor', CurveExecutionAction, self.execute, False)

        self.mg = MoveGroupCommander(mg)
        self.traj_client = TrajectoryActionClient(arm='r1', # TODO: not general
                                                  moveGroupCommander=self.mg,
                                                  topic="/r1/trajectory_controller/follow_joint_trajectory",
                                                  robotCommander=None)

        self.Z_MIN = 0.02

        js_lpos = self.mg.get_named_target_values('L_position')
        self.mg.go(js_lpos)

        # load kdl
        self.eef = 'tool0'
        self._world_frame = 'world'
        self.robot_urdf = URDF.from_parameter_server('robot_description')
        self.kdl_kin = KDLKinematics(self.robot_urdf, self._world_frame, self.eef)

        ####################################################################################################################
        # self.relaxedIK = RelaxedIK.init_from_config(config_file_name)

        # prepare JointTrajectory for relaxedik callback
        self.traj = JointTrajectory()
        self.traj.joint_names = self.mg.get_active_joints()
        self.traj.points.append(JointTrajectoryPoint())

        self.eef_speed = 0.1 # m/s


        self.traj_cmd = rospy.Publisher('/r1/trajectory_controller/command', JointTrajectory, queue_size=1)

        self.relaxedik_pub = rospy.Publisher('/relaxed_ik/ee_pose_goals', EEPoseGoals, queue_size=1)
        self.relaxedik_sub = rospy.Subscriber('/relaxed_ik/joint_angle_solutions', JointAngles, callback=self.rik_cb, queue_size=1)
        # TODO: extend RelaxedIK to accept other weights for the goal and possibly other goals
        # TODO: rust relaxed IK has a message interface
        ####################################################################################################################


        # http://wiki.ros.org/pykdl_utils
        # from urdf_parser_py.urdf import URDF
        # from pykdl_utils.kdl_parser import kdl_tree_from_urdf_model
        # robot = URDF.load_from_parameter_server(verbose=False)
        # tree = kdl_tree_from_urdf_model(robot)
        # print tree.getNrOfSegments()
        # chain = tree.getChain(base_link, end_link)
        # print chain.getNrOfJoints()

        # from urdf_parser_py.urdf import URDF
        # from pykdl_utils.kdl_kinematics import KDLKinematics
        # robot = URDF.from_parameter_server()
        # kdl_kin = KDLKinematics(robot, base_link, end_link)
        # q = kdl_kin.random_joint_angles()
        # pose = kdl_kin.forward(q)  # forward kinematics (returns homogeneous 4x4 numpy.mat)
        # q_ik = kdl_kin.inverse(pose, q + 0.3)  # inverse kinematics
        # if q_ik is not None:
        #     pose_sol = kdl_kin.forward(q_ik)  # should equal pose
        # J = kdl_kin.jacobian(q)
        # print 'q:', q
        # print 'q_ik:', q_ik
        # print 'pose:', pose
        # if q_ik is not None:
        #     print 'pose_sol:', pose_sol
        # print 'J:', J

        self.traj_goal = None
        self.traj_goal_pt = None

        self.relaxedik_clutch = False

        self.relaxed_ik_js = np.array(7*[0])

        self.last_js = JointState()
        self.las_js_diff = 10

        self.js_sub = rospy.Subscriber('/r1/joint_states', JointState, callback=self.js_cb, queue_size=1)

        self.server.start()

    def switch_relaxed_ik(self, on=None):
        if on is None:
            # toggle mode
            on = not self.relaxedik_clutch

        if self.relaxedik_clutch == on:
            rospy.loginfo('relaxed ik clutch is {}, as it should be. Doing noting.'.format(self.relaxedik_clutch))
            return
        #send robot home
        if not self.relaxedik_clutch:
            rospy.loginfo('relaxed ik clutch is disengaged. Preparing for engagement.')
            js_lpos = self.mg.get_named_target_values('L_position')
            self.mg.go(js_lpos)
        else:
            rospy.loginfo('relaxed ik clutch is engaged. Preparing for DISengagement.')
            goal = EEPoseGoals()
            goal.ee_poses.append(Pose(orientation=Quaternion(w=1)))
            self.relaxedik_pub.publish(goal)
            rospy.sleep(3.0)

        if on:
            goal = EEPoseGoals()
            goal.ee_poses.append(Pose(orientation=Quaternion(w=1)))
            self.relaxedik_pub.publish(goal)
            rospy.sleep(0.5)
            rospy.loginfo('engaging relaxed ik clutch.')
            self.relaxedik_clutch = True
        else:
            rospy.loginfo('disengaging relaxed ik clutch.')
            self.relaxedik_clutch = False


    def js_cb(self, msg):
        #type: (JointState) -> None
        # assert isinstance(msg, JointState)
        # if 'r2' in msg.name[0]:
        self.last_js = msg
        self.las_js_diff = np.linalg.norm(np.array(msg.position) - self.relaxed_ik_js)
        # print("received js: {}".format(msg.position))
        # else:
        #     return

    def rik_cb(self, msg):
        assert isinstance(msg, JointAngles)
        # msg.angles.data
        # trajectory_msgs / JointTrajectory

        if not self.relaxedik_clutch:
            return

        if np.any(np.isnan(msg.angles.data)):
            rospy.logwarn('no relaxed ik solution.')
            return

        pt = self.traj.points[0]  # type: JointTrajectoryPoint
        pt.positions = msg.angles.data

        self.relaxed_ik_js = msg.angles.data

        # cur_pose = self.mg.get_current_pose(self.mg.get_end_effector_link())
        # cur_point = np.array([cur_pose])
        cur_js = self.mg.get_current_joint_values()
        cur_pose_mat = self.kdl_kin.forward(q=cur_js, base_link='world',
                                            end_link=self.mg.get_end_effector_link())
        cur_point = translation_from_matrix(cur_pose_mat)

        new_pose_mat = self.kdl_kin.forward(q=msg.angles.data, base_link='world', end_link=self.mg.get_end_effector_link())
        new_point = translation_from_matrix(new_pose_mat)

        dist = euclidean(new_point, cur_point)

        pt.time_from_start = rospy.Duration.from_sec(dist/self.eef_speed)

        self.traj.header.stamp = rospy.Time.now() + rospy.Duration.from_sec(0.05)
        self.traj_cmd.publish(self.traj)


    def get_joint_limits(self, joints):
        lower = []
        upper = []
        vel = []
        effort = []
        for joint in joints:
            lower += [self.robot_urdf.joint_map[joint].limit.lower]
            upper += [self.robot_urdf.joint_map[joint].limit.upper]
            vel += [self.robot_urdf.joint_map[joint].limit.velocity]
            effort += [self.robot_urdf.joint_map[joint].limit.effort]
        return lower, upper, vel, effort

    # def retime(self, plan):
    #     import toppra as ta
    #     assert isinstance(plan, RobotTrajectory)
    #     lower, upper, vel, effort = self.get_joint_limits(self.mg.get_active_joints())
    #
    #     alims = len(lower)* [1.0]
    #
    #     ss = [pt.time_from_start.to_sec()*0.01 for pt in plan.joint_trajectory.points]
    #     way_pts = [list(pt.positions) for pt in plan.joint_trajectory.points]
    #
    #     path = ta.SplineInterpolator(ss, way_pts)
    #
    #     pc_vel = ta.constraint.JointVelocityConstraint(np.array([lower, upper]).transpose())
    #
    #     pc_acc = ta.constraint.JointAccelerationConstraint(np.array(alims))
    #
    #     def inv_dyn(q, qd, qgg):
    #         # use forward kinematic formula and autodiff to get jacobian, then calc velocities from jacobian and joint
    #         # velocities
    #         J = iiwa_jacobian(q)
    #         cart_vel = np.dot(J, qd)
    #         return np.linalg.norm(cart_vel)
    #
    #     def g(q):
    #         return ([-0.2, 0.2])
    #
    #     def F(q):
    #         return np.eye(1)
    #
    #
    #     eef_vel = ta.constraint.SecondOrderConstraint(inv_dyn=inv_dyn, constraint_F=F, constraint_g=g, dof=7)
    #
    #     instance = ta.algorithm.TOPPRA([pc_vel, pc_acc, eef_vel], path)
    #     # print(instance)
    #     # instance2 = ta.algorithm.TOPPRAsd([pc_vel, pc_acc], path)
    #     # instance2.set_desired_duration(60)
    #     jnt_traj = instance.compute_trajectory()
    #
    #     # ts_sample = np.linspace(0, jnt_traj.duration, 10*len(plan.joint_trajectory.points))
    #     ts_sample = np.linspace(0, jnt_traj.duration, np.ceil(100*jnt_traj.duration))
    #     qs_sample = jnt_traj(ts_sample)
    #     qds_sample = jnt_traj(ts_sample, 1)
    #     qdds_sample = jnt_traj(ts_sample, 2)
    #
    #     new_plan = copy.deepcopy(plan)
    #     new_plan.joint_trajectory.points = []
    #
    #     for t, q, qd, qdd in zip(ts_sample, qs_sample, qds_sample, qdds_sample):
    #         pt = JointTrajectoryPoint()
    #         pt.time_from_start = rospy.Duration.from_sec(t)
    #         pt.positions = q
    #         pt.velocities = qd
    #         pt.accelerations = qdd
    #         new_plan.joint_trajectory.points.append(pt)
    #
    #     if rospy.get_param('plot_joint_trajectory', default=False):
    #         import matplotlib.pyplot as plt
    #
    #         fig, axs = plt.subplots(3, 1, sharex=True)
    #         for i in range(path.dof):
    #             # plot the i-th joint trajectory
    #             axs[0].plot(ts_sample, qs_sample[:, i], c="C{:d}".format(i))
    #             axs[1].plot(ts_sample, qds_sample[:, i], c="C{:d}".format(i))
    #             axs[2].plot(ts_sample, qdds_sample[:, i], c="C{:d}".format(i))
    #         axs[2].set_xlabel("Time (s)")
    #         axs[0].set_ylabel("Position (rad)")
    #         axs[1].set_ylabel("Velocity (rad/s)")
    #         axs[2].set_ylabel("Acceleration (rad/s2)")
    #         plt.show()
    #
    #     return new_plan


    def traj_feedback_cb(self, msg):
        goal = self.traj_client._goal
        if not isinstance(goal, FollowJointTrajectoryGoal):
            return
        assert isinstance(msg, FollowJointTrajectoryFeedback)

        points = self.get_traj_pts(goal)
        desired = np.array(msg.desired.positions)

        dist = (np.linalg.norm(points-desired, axis=1))
        idx = np.argmin(dist)

        fb = CurveExecutionFeedback()
        fb.progress = float(idx)/ points.shape[0]
        self.server.publish_feedback(fb)
        # TODO: the feedback could be misleading in the case that the trajectory is crossing over itself.
        return

    def normalize_vectors(self, y_new):
        scale = 1 / np.linalg.norm(y_new, axis=1)
        y_norm = np.array([y_new[:, 0] * scale, y_new[:, 1] * scale, y_new[:, 2] * scale]).transpose()
        return y_norm

    @staticmethod
    def plot_spline(tck_param, start=0.0, end=1.0, points=[]):
        from mpl_toolkits import mplot3d
        # %matplotlib inline
        # import numpy as np
        import matplotlib.pyplot as plt

        # spline = interpolate.BSpline(tck_param[0], tck_param[1], tck_param[2], extrapolate=False)

        fig = plt.figure()
        ax = plt.axes(projection='3d')

        u = np.linspace(start, end, 300)
        x = interpolate.splev(u, tck=tck_param)

        ax.plot3D(x[0], x[1], x[2], 'gray')
        # ax.scatter(x[0], x[1], x[2])

        if len(points) > 0:
            ax.plot(points[:, 0], points[:, 1], points[:, 2], 'ro')

            # ax.scatter(points[:,0], points[:,1], points[:,2], marker='x')

        fig.show()

    def resample_curve(self, curve, normals, pts_per_meter=100):

        idx = np.linspace(0, 1.0, curve.shape[0])
        curve_1d = interpolate.interp1d(idx, curve, axis=0)
        normals_1d = interpolate.interp1d(idx, normals, axis=0)

        dist = np.hstack([np.zeros((1)), np.cumsum(np.linalg.norm(curve[1:] - curve[:-1], axis=1))])
        idx2length = interpolate.interp1d(idx, dist)

        interp_fn2 = lambda x, y: idx2length(x) - y



        from scipy import optimize

        def opt(distance):
            # res = optimize.newton(interp_fn2, x0=0, args=tuple([distance]))
            # interp_fn2 = lambda x: idx2length(x) - distance
            res = optimize.bisect(interp_fn2, a=0.0, b=1.0, args=tuple([distance]))
            return res

        new_idx = [opt(d) for d in np.linspace(0.0, dist[-1] * 0.99, np.ceil(dist[-1] * pts_per_meter))] + [1.0]

        new_curve = curve_1d(new_idx)
        new_normals = normals_1d(new_idx)

        return new_curve, new_normals, dist[-1]

    def trans2world(self, curve, w2table1, w2table2):
        if self.eef == 'tool_r2':
            curve_world = np.dot(w2table2, curve.transpose()).transpose()
        elif self.eef == 'tool0':
            curve_world = np.dot(w2table1, curve.transpose()).transpose()
        else:
            raise ValueError("Don't know transform to workspace for end effector {}.".format(self.eef))
        return curve_world


    def execute(self, goal):
        assert isinstance(goal, CurveExecutionGoal)
        success = False

        self.eef_speed = goal.opt.cart_vel_limit

        exec_engine = goal.opt.executor_engine
        self.switch_relaxed_ik(on=(exec_engine==ExecutionOptions.RELAXEDIK))

        # prepare trajectory
        curve = np.array([[c.x, c.y, c.z, 1.0] for c in goal.curve])
        normals = np.array([[n.x, n.y, n.z] for n in goal.normals])

        # remove duplicate points from data
        to_delete = np.where(np.linalg.norm(curve[1:] - curve[:-1], axis=1) <= 0.0001)
        curve = np.delete(curve, to_delete, axis=0)
        normals = np.delete(normals, to_delete, axis=0)

        curve, normals, curve_len = self.resample_curve(curve, normals, pts_per_meter=250)


        if goal.header.frame_id != "table2":
            if goal.header.frame_id == "":
                # post processing for curves without camera calibration
                curve = curve - curve.mean(axis=0) + np.array([0.0, 0.0, 0.0, 1.0])
                # curve = curve - curve.mean(axis=0) + np.array([0.1, 0.0, 0.2, 1.0])


                # v1*v2 = | v1 | | v2 | cos(angle)
                normal_avg = np.mean(normals, axis=0)
                v2 = np.array([0,0,1])
                angle = np.arccos(np.dot(v2, normal_avg))  # assumes vectors to be normalized!!!

                axis = np.cross(v2, normal_avg)

                R = np.eye(3)
                R_axis_angle(R, axis=axis, angle=angle)

                shift = np.eye(4)
                shift[:3, :3] = R
                curve_new = np.dot(curve, shift)
                normals_new = np.dot( normals, R)

                curve = curve_new
                normals = normals_new

                curve = curve + np.array([0.1, 0.0, 0.2, 0.0])
            else:
                # transform points according to frame
                raise NotImplementedError
        else:
            pass

        # make sure, we don't hit the table
        mask = np.where(curve[:,2]<self.Z_MIN)
        curve[:,2][mask] = self.Z_MIN


        # points are relative to the frame '/table2' and relaxedIK requires the points relative to the reference pose, which
        # is the robot standing straight up: world to r2_joint_tip pos: -0.0019918; -0.85177; 1.2858      quat: -0.0011798; -0.00090396; -0.0031778; 0.99999
        # world to table2: 0.42783; -0.85741; -0.0054846    quat: -0.0011798; -0.0008628; -0.0030205; 0.99999
        # new world2ref
        # 0.39933; -0.85591; 0.58248
        # quat: 0.0030643; 0.99999; -0.0012084; 0.00084322
        # TODO: relaxed IK reference frame changed

        ########################### NEW ####################
        w2ref = translation_matrix(np.array([0.39933, -0.85591, 0.58248]))
        w2ref_rot = quaternion_matrix([0.0030643, 0.99999, -0.0012084, 0.00084322])
        ref2w_rot = np.linalg.inv(w2ref_rot)
        # w2ref[:3,:3] = w2ref_rot[:3,:3]
        # w2ref = translation_matrix(np.array([-0.0019918, -0.85177, 1.2858]))
        ref2w = np.linalg.inv(w2ref)
        w2table2 = translation_matrix(np.array([0.42783, -0.85741, -0.0054846]))
        w2table1 = translation_matrix(np.array([0.42783, -0.0, -0.0054846]))

        ref2table = np.dot(ref2w, w2table2)

        curve_tf = np.dot(ref2table, curve.transpose()).transpose()[:, :3]

        ############## NEW END ###################

        # w2ref = translation_matrix(np.array([0.39933, -0.85591, 0.58248]))
        # w2ref_rot = quaternion_matrix([0.0030643, 0.99999, -0.0012084, 0.00084322])
        # # w2ref[:3,:3] = w2ref_rot[:3,:3]
        # # w2ref = translation_matrix(np.array([-0.0019918, -0.85177, 1.2858]))
        # ref2w = np.linalg.inv(w2ref)
        # w2table2 = translation_matrix(np.array([0.42783, -0.85741, -0.0054846]))
        #
        # transform = np.dot(ref2w, w2table2)
        #
        # curve_tf = np.dot(transform, curve.transpose()).transpose()[:,:3]

        ############# OLD end ############

        # tck_params, xy_as_u = interpolate.splprep(curve_tf.transpose(), k=5, s=0.01, per=0)
        x = curve_tf[:, 0]
        y = curve_tf[:, 1]
        z = curve_tf[:, 2]
        tck_params, xy_as_u = interpolate.splprep([x, y, z], k=5, s=0, per=0)

        # CurveExecutor.plot_spline(tck_params, points=curve_tf)
        pt_dot = np.transpose(np.array(interpolate.splev(xy_as_u, tck_params, der=1)))
        # y_new = np.cross(pt_dot, -normals)
        y_new = np.cross(-normals, pt_dot)
        #TODO: what if the normals and the path direction are not orthogonal?

        scale = 1/np.linalg.norm(y_new,axis=1)
        y_norm = np.array([y_new[:,0] * scale, y_new[:,1] * scale, y_new[:,2] * scale]).transpose()
        x_norm = self.normalize_vectors(pt_dot)
        z_norm = - self.normalize_vectors(normals)

        # M = np.zeros((len(pt_dot),4,4))
        # M[:,0,0:3] = x_norm
        # M[:,1,0:3] = y_norm
        # M[:,2,0:3] = z_norm
        # M[:, 3, 3] = 1

        M = np.zeros((len(pt_dot), 4, 4))
        M[:, 0:3, 0] = x_norm
        M[:, 0:3, 1] = y_norm
        M[:, 0:3, 2] = z_norm
        M[:, 3, 3] = 1

        # quats = quaternion_from_matrix(M[0])

        M_relIK = np.matmul(ref2w_rot, M)

        quat_relaxed_ik = np.array(list(map(quaternion_from_matrix, M_relIK[:])))

        quats_path = np.array(list(map(quaternion_from_matrix, M[:])))

        # quats = np.array([[0,1,0,0] for i in range(M.shape[0])])
        quats = np.zeros(shape=quats_path.shape)
        quats[:,1] = 1.0

        if goal.opt.tool_orientation == ExecutionOptions.USE_TOOL_ORIENTATION:
            quats = quats_path

        if rospy.get_param('make_RGB_HEDGEHOG', default=False):   # 'make_RGB_HEDGEHOG':
            pub1 = rospy.Publisher('/my_Marker', Marker, queue_size=50)

            def pub_arrow(point, direction, idx, ns, color=(1,0,0,1)):
                m = Marker()
                m.header.frame_id = 'table2'
                m.ns = ns
                m.id = idx
                m.type = m.ARROW
                m.action = m.ADD
                m.pose.orientation.w = 1  # to avoid quaternion not initialized warning in rviz
                m.points.append(Point(*point))
                m.points.append(Point(*(point+direction *0.03)))
                m.scale.x = 0.005
                m.scale.y = 0.01

                m.color.r = color[0]
                m.color.g = color[1]
                m.color.b = color[2]
                m.color.a = color[3]

                pub1.publish(m)

            for idx, (point, direct) in enumerate(zip(curve[:], x_norm[:])):
                pub_arrow(point[:3], direct, idx=idx, ns='x', color=[1,0,0,0.3])
                rospy.sleep(0.01)

            for idx, (point, direct) in enumerate(zip(curve[:], y_norm[:])):
                pub_arrow(point[:3], direct, idx=idx, ns='y',color=[0,1,0,0.3])
                rospy.sleep(0.01)
            for idx, (point, direct) in enumerate(zip(curve[:], z_norm[:])):
                pub_arrow(point[:3], direct, idx=idx, ns='z', color=[0,0,1,0.3])
                rospy.sleep(0.01)

        pub2 = rospy.Publisher('/ee_pose', PoseStamped, queue_size=50)

        def pub_pose(point, quat):
            pose = PoseStamped()
            pose.header.frame_id = 'table2'
            pose.pose.position.x = point[0]
            pose.pose.position.y = point[1]
            pose.pose.position.z = point[2]
            pose.pose.orientation.x = quat[0]
            pose.pose.orientation.y = quat[1]
            pose.pose.orientation.z = quat[2]
            pose.pose.orientation.w = quat[3]
            pub2.publish(pose)

        if False:
            for pos, quat in zip(curve[:], quats):
                pub_pose(pos, quat)
                rospy.sleep(0.01)

        #change of basis
        # M = axis of new coordinate system normalized ()

        # https://stackoverflow.com/questions/49635428/tangent-to-curve-interpolated-from-discrete-data

        use_relaxed_ik = (exec_engine == ExecutionOptions.RELAXEDIK)
        if use_relaxed_ik:
            draw = self.draw_path(self.mg)
            rate = rospy.Rate(1000)

            # send pose goal for start pose
            pos = curve_tf[0]
            ee_goal = EEPoseGoals()
            pose = Pose()
            pose.position.x = pos[0]
            pose.position.y = pos[1]
            pose.position.z = pos[2]
            if goal.opt.tool_orientation == ExecutionOptions.USE_FIXED_ORIENTATION:
                pose.orientation.x = 0
                pose.orientation.y = 0
                pose.orientation.z = 0
                pose.orientation.w = 1
            elif goal.opt.tool_orientation == ExecutionOptions.USE_TOOL_ORIENTATION:
                pose.orientation.x = quat_relaxed_ik[0, 0]
                pose.orientation.y = quat_relaxed_ik[0, 1]
                pose.orientation.z = quat_relaxed_ik[0, 2]
                pose.orientation.w = quat_relaxed_ik[0, 3]
            else:
                pose.orientation.x = 0
                pose.orientation.y = 0
                pose.orientation.z = 0
                pose.orientation.w = 1
            # pose.orientation.x = quat[0]
            # pose.orientation.y = quat[1]
            # pose.orientation.z = quat[2]
            # pose.orientation.w = quat[3]
            ee_goal.ee_poses.append(pose)
            self.relaxedik_pub.publish(ee_goal)

            # TODO: wait for traj finish instead of sleep
            rospy.sleep(5.0)
            while self.las_js_diff > 0.0001:
                rospy.sleep(0.001)
                self.relaxedik_pub.publish(ee_goal)

            rospy.sleep(1.0)
            LoggerProxy.logger(action=LoggingTask.LOG, data='start')

            for idx, (pos, quat) in enumerate(zip(curve_tf, quat_relaxed_ik)):
                cur_js_mg = self.mg.get_current_joint_values()
                cur_js = self.last_js.position
                print('joint angle time: {}'.format(rospy.Time.now().to_time() - self.last_js.header.stamp.to_time()))
                print('joint angle disagreement: {}'.format(euclidean(cur_js, cur_js_mg)))
                cur_pose_mat = self.kdl_kin.forward(q=cur_js, base_link='world',
                                                    end_link=self.mg.get_end_effector_link())
                cur_point = np.ones(4)
                cur_point[:3] = translation_from_matrix(cur_pose_mat)
                cur_point_ref = np.dot(ref2w, cur_point)
                dist = euclidean(cur_point_ref[:3], pos)
                ee_goal = EEPoseGoals()
                pose = Pose()
                pose.position.x = pos[0]
                pose.position.y = pos[1]
                pose.position.z = pos[2]
                if goal.opt.tool_orientation == ExecutionOptions.USE_FIXED_ORIENTATION:
                    pose.orientation.x = 0
                    pose.orientation.y = 0
                    pose.orientation.z = 0
                    pose.orientation.w = 1
                elif goal.opt.tool_orientation == ExecutionOptions.USE_TOOL_ORIENTATION:
                    pose.orientation.x = quat[0]
                    pose.orientation.y = quat[1]
                    pose.orientation.z = quat[2]
                    pose.orientation.w = quat[3]
                else:
                    pose.orientation.x = 0
                    pose.orientation.y = 0
                    pose.orientation.z = 0
                    pose.orientation.w = 1
                # pose.orientation.x = quat[0]
                # pose.orientation.y = quat[1]
                # pose.orientation.z = quat[2]
                # pose.orientation.w = quat[3]
                ee_goal.ee_poses.append(pose)
                self.relaxedik_pub.publish(ee_goal)
                print('dist: {} \n sleep: {}'.format(dist, 0.95* dist/self.eef_speed))
                # rospy.sleep(min([0.95* dist/self.eef_speed,  0.2]))
                sleep_until = rospy.Time.now() + rospy.Duration.from_sec(min([0.95* dist/self.eef_speed,  0.2]))

                while True:
                    draw.next()  # draw next line piece
                    if self.server.is_preempt_requested():
                        self.server.set_preempted()
                        success = False
                        self.traj_client.stop()
                        break
                    else:
                        success = True
                    # if rospy.Time.now() > sleep_until:
                    if self.las_js_diff < 0.001:
                        break
                    else:
                        rate.sleep()
            # destroy the painting iterator
            draw.close()

            rospy.sleep(1.0)
            LoggerProxy.logger(action=LoggingTask.LOG, data='end')

            # send robot to home pose
            # send pose goal for start pose
            ee_goal = EEPoseGoals()
            pose = Pose()
            pose.position.x = 0
            pose.position.y = 0
            pose.position.z = 0
            pose.orientation.x = 0
            pose.orientation.y = 0
            pose.orientation.z = 0
            pose.orientation.w = 1
            # pose.orientation.x = quat[0]
            # pose.orientation.y = quat[1]
            # pose.orientation.z = quat[2]
            # pose.orientation.w = quat[3]
            ee_goal.ee_poses.append(pose)
            self.relaxedik_pub.publish(ee_goal)

            rospy.sleep(3.0)

            if success:
                # create result/response message
                self.server.set_succeeded(CurveExecutionResult(True))
                rospy.loginfo('Action successfully completed')
            else:
                self.server.set_aborted(CurveExecutionResult(False))
                rospy.loginfo('Whoops')

            return

        use_descartes_planner = (exec_engine == ExecutionOptions.DESCARTES_TOLERANCED_PLANNING)
        if use_descartes_planner:
            self.mg.set_pose_reference_frame('world')
            curve_world = self.trans2world(curve, w2table1, w2table2)
            wp = []
            for pos, quat in zip(curve_world, quats):
                pose = Pose()
                pose.position.x = pos[0]
                pose.position.y = pos[1]
                pose.position.z = pos[2]
                pose.orientation.x = quat[0]
                pose.orientation.y = quat[1]
                pose.orientation.z = quat[2]
                pose.orientation.w = quat[3]
                wp.append(pose)

            self.mg.set_end_effector_link(self.eef)
            # plan = self.mg.plan(wp[0])
            # # self.mg.go(wp[0])
            # if plan.joint_trajectory.points.__len__() > 1:
            #     self.mg.execute(plan)

            # get trajectory by cartesian planning (MoveIt!)
            self.mg.set_pose_reference_frame('world')
            # traj_pttrn, fraction = self.mg.compute_cartesian_path(wp, eef_step=5e-4, jump_threshold=100.0)

            # descartes on real pattern trial
            get_traj_srv = rospy.ServiceProxy('/planTolerancedTrajecory', PlanTolerancedTrajecory)
            rospy.sleep(0.5)
            req = PlanTolerancedTrajecoryRequest()
            req.header.frame_id = 'base_link'
            # TODO: make descartes work with world coordinates or this more general.
            req.base_to_path.translation.x = w2table2[0,3]
            req.base_to_path.translation.y = 0.0 # w2table2[1,3]
            req.base_to_path.translation.z = 0.0 # w2table2[2,3]
            req.base_to_path.rotation.w = 1.0
            req.poses = [Pose(position=Point(x, y, z), orientation=Quaternion(0, 0, 0, 1)) for x, y, z, _ in
                         curve[:]]

            res = get_traj_srv.call(req)  # type: PlanTolerancedTrajecoryResponse
            plan = RobotTrajectory()
            plan.joint_trajectory = res.traj
            # plan = res.traj
            plan = retime(plan, cart_vel_limit=goal.opt.cart_vel_limit, pt_per_s=30, curve_len=curve_len)
            # plot_plan(plan, title='Descartes and TOPPRA', save='/home/behrejan/traj_complete_log/trajectory_plot.png', show=False, ret=False)

            #move to start
            self.mg.set_end_effector_link('tool0')
            plan0 = self.mg.plan(plan.joint_trajectory.points[0].positions)
            # self.mg.go(wp[0])
            if plan0.joint_trajectory.points.__len__() > 1:
                self.mg.execute(plan0)

            self.traj_client.add_plan(plan)
            LoggerProxy.logger(action=LoggingTask.LOG, data='start')
            self.traj_client.start(feedback_cb=self.traj_feedback_cb)
            success = True
            # it = self.draw_path(self.mg, link='r2_link_7', link2eef_msg=Transform(Vector3(0,0,0.1), Quaternion(0,0,0,1)))
            it = self.draw_path(self.mg, link='tool0')
            rate = rospy.Rate(20)
            while self.traj_client.state() in [actionlib.GoalStatus.ACTIVE,
                                               actionlib.GoalStatus.PENDING] or np.linalg.norm(
                self.last_js.velocity) > 0.001:
                it.next()  # draw next line piece
                if self.server.is_preempt_requested():
                    self.server.set_preempted()
                    success = False
                    self.traj_client.stop()
                    break
                else:
                    success = True
                rate.sleep()

            it.close()
            LoggerProxy.logger(action=LoggingTask.LOG, data='end')

            # sub.unregister()
            rospy.loginfo('trajectory execution ready')



        use_cart_planner = (exec_engine == ExecutionOptions.MOVEIT_CART_PLANNING)
        if use_cart_planner:
            self.mg.set_pose_reference_frame('world')
            # curve_world = np.dot(w2table2, curve.transpose()).transpose()
            curve_world = self.trans2world(curve, w2table1, w2table2)


            wp = []
            for pos, quat in zip(curve_world, quats):
                pose = Pose()
                pose.position.x = pos[0]
                pose.position.y = pos[1]
                pose.position.z = pos[2]
                pose.orientation.x = quat[0]
                pose.orientation.y = quat[1]
                pose.orientation.z = quat[2]
                pose.orientation.w = quat[3]
                wp.append(pose)

            plan = self.mg.plan(wp[0])
            # self.mg.go(wp[0])
            if plan.joint_trajectory.points.__len__() > 1:
                self.mg.execute(plan)

            # get trajectory by cartesian planning (MoveIt!)
            self.mg.set_pose_reference_frame('world')
            # self.mg.set_pose_reference_frame('r1_link_0')

            traj_pttrn, fraction = self.mg.compute_cartesian_path(wp, eef_step=5e-4, jump_threshold=100.0)

            if fraction >= 1.0:
                # self.mg.execute(traj_pttrn)
                # rs = state_from_plan_point(plan=traj_pttrn, point=0)
                # plan = self.mg.retime_trajectory(rs, traj_pttrn, 0.5)
                plan = traj_pttrn
                if plan.joint_trajectory.points[0].time_from_start == plan.joint_trajectory.points[1].time_from_start:
                    plan.joint_trajectory.points = plan.joint_trajectory.points[1:]
                plan = retime(plan, cart_vel_limit=goal.opt.cart_vel_limit)
                plot_plan(plan)
                self.traj_client.add_plan(plan)

                # self.traj_client._client.feedback_cb = self.traj_feedback_cb

                # sub = rospy.Subscriber('/r2/trajectory_controller/follow_joint_trajectory/feedback',
                #                  FollowJointTrajectoryActionFeedback,
                #                  callback=self.traj_feedback_cb,
                #                  queue_size=1)
                LoggerProxy.logger(action=LoggingTask.LOG, data='start')
                self.traj_client.start(feedback_cb=self.traj_feedback_cb)
                success = True
                it = self.draw_path(self.mg)
                rate = rospy.Rate(20)
                while self.traj_client.state() in [actionlib.GoalStatus.ACTIVE, actionlib.GoalStatus.PENDING] or np.linalg.norm(self.last_js.velocity) > 0.001:
                    it.next() # draw next line piece
                    if self.server.is_preempt_requested():
                        self.server.set_preempted()
                        success = False
                        self.traj_client.stop()
                        break
                    else:
                        success = True
                    rate.sleep()

                it.close()
                LoggerProxy.logger(action=LoggingTask.LOG, data='end')


                # sub.unregister()
                rospy.loginfo('trajectory execution ready')

        if success:
            # create result/response message
            self.server.set_succeeded(CurveExecutionResult(True))
            rospy.loginfo('Action successfully completed')
        else:
            self.server.set_aborted(CurveExecutionResult(False))
            rospy.loginfo('Whoops')

        rospy.sleep(1.0)
        js_lpos = self.mg.get_named_target_values('L_position')
        self.mg.go(js_lpos)
        return





        # joint_states = []
        # for pose in wp:
        #     pose = copy.deepcopy(pose)
        #     pose.position.x += 0.42
        #     pose.position.y -= 0.85
        #
        #     # q_0 = self.kdl_kin.random_joint_angles()
        #     js = self.kdl_kin.inverse(pose, maxiter=1000)
        #     joint_states.append(js)
        #
        # print joint_states




        xopt = []
        for i in range(len(curve_tf)):
            goal_pos = [[0,0,0], curve_tf[i]]
            goal_quat = [[1, 0, 0, 0], [quats[i][3]] + list(quats[i][0:3])]
            xopt += [self.relaxedIK.solve(goal_pos, goal_quat)]

        # xopt = np.array(list(map(self.relaxedIK.solve,
        #    [[[0, 0, 0], pos] for pos in curve_tf],
        #    [[[1, 0, 0, 0], quat] for quat in quats])))

        traj = RobotTrajectory()

        traj.joint_trajectory.joint_names = self.mg.get_active_joints()  # ['r2_joint_0', 'r2_joint_1', 'r2_joint_2', 'r2_joint_3', 'r2_joint_4', 'r2_joint_5', 'r2_joint_6']
        traj.joint_trajectory.header.stamp = rospy.Time.now()
        for idx, pt in enumerate(xopt):
            trajpt = JointTrajectoryPoint()
            trajpt.positions = pt[7:14]
            # trajpt.time_from_start = rospy.Duration.from_sec(t[idx])
            traj.joint_trajectory.points.append(trajpt)

        plan = self.mg.retime_trajectory(get_current_robot_state(self.mg), traj, 0.5)

        self.traj_client.add_plan(plan)

        self.traj_client.start()


        # pause = rospy.Duration(0.01)
        # n_points = curve.shape[0]
        # success = True
        # rospy.loginfo("Starting curve execution.")
        #
        # plan = RobotTrajectory()
        # self.traj_client.add_plan(plan=plan)
        # self.traj_client.start(feedback_cb=self.traj_feedback_cb)
        # for i, (c, n) in enumerate(zip(curve, normals)):
        #     if server.is_preempt_requested():
        #         server.set_preempted()
        #         success = False
        #         break
        #
        #     rospy.loginfo("> c: {}\t n: {} ({:.2f}%)".format(c, n, float(i) / n_points * 100))
        #     server.publish_feedback(CurveExecutionFeedback(progress=i))
        #     rospy.sleep(pause)
        # if success:
        #     # create result/response message
        #     server.set_succeeded(CurveExecutionResult(True))
        #     rospy.loginfo('Action successfully completed')
        # else:
        #     server.set_aborted(CurveExecutionResult(False))
        #     rospy.loginfo('Whoops')

    def draw_path(self, mg, size=0.002, color=[ColorRGBA(0, 1, 0, 1), ColorRGBA(1, 0, 0, 1)], speed_range=[0.0, 0.05], ns='path', link='', link2eef_msg=Transform(Vector3(0,0,0), Quaternion(0,0,0,1))):
        publisher = rospy.Publisher('/ee_path', Marker, queue_size=1)
        if link.__len__() > 0:
            ee = link
        else:
            ee = mg.get_end_effector_link()

        link2eef = quaternion_matrix([link2eef_msg.rotation.x, link2eef_msg.rotation.y, link2eef_msg.rotation.z, link2eef_msg.rotation.w])
        link2eef[0,3] = link2eef_msg.translation.x
        link2eef[1, 3] = link2eef_msg.translation.y
        link2eef[2, 3] = link2eef_msg.translation.z
        m = Marker()
        m.header.frame_id = 'world'
        m.header.stamp = rospy.Time.now()
        m.type = m.SPHERE_LIST
        m.pose.orientation.w = 1
        m.scale.x = size
        m.scale.y = size
        m.scale.z = size
        m.color = color[1]
        # m.color.a = 0.9
        # m.color.r = 1.0
        m.action = m.ADD
        m.ns = ns
        m.pose.position.x = 0.0  # robot_info.get_eef2tip_transform(group_name).transform.translation.x
        m.pose.position.y = 0.0  # robot_info.get_eef2tip_transform(group_name).transform.translation.y
        m.pose.position.z = -0.0  # robot_info.get_eef2tip_transform(group_name).transform.translation.z

        last_time = rospy.Time.now()
        last_pose = mg.get_current_pose()
        speed_max = speed_range[-1]
        speed_min = speed_range[0]
        # while ta_client.state() != GoalStatus.SUCCEEDED:
        while True:
        # draw the line
            pose = mg.get_current_pose(ee)  # type: PoseStamped
            mat = quaternion_matrix([pose.pose.orientation.x, pose.pose.orientation.y, pose.pose.orientation.z, pose.pose.orientation.w])
            mat[0,3] = pose.pose.position.x
            mat[1, 3] = pose.pose.position.y
            mat[2, 3] = pose.pose.position.z

            tcp_mat = np.dot(mat, link2eef)
            t_now = rospy.Time.now()
            #
            speed = np.linalg.norm((np.array([pose.pose.position.x, pose.pose.position.y, pose.pose.position.z]) - np.array([last_pose.pose.position.x, last_pose.pose.position.y, last_pose.pose.position.z])) / (t_now - last_time).to_sec())
            speed_max = max(speed, speed_max)
            speed_min = min(speed, speed_min)
            speed_ratio = (speed - speed_min) / (speed_max - speed_min)

        # joint_angles = ta_client.get_current_tp().positions
            # fk_result = fk.getFK(mg.get_end_effector_link(), mg.get_active_joints(), joint_angles)  # type: GetPositionFKResponse
            p = Point()
            p.x = tcp_mat[0,3]
            p.y = tcp_mat[1,3]
            p.z = tcp_mat[2,3]

            c2 = np.array([1,0,0,1])
            c1 = np.array([0,1,0,1])
            c = c1 + speed_ratio * (c2-c1)  # type: np.ndarray
            # m.colors.append(ColorRGBA(m.color.r, m.color.g, m.color.b, m.color.a))
            m.colors.append(ColorRGBA(*c.tolist()))

            m.action = Marker.ADD

            m.points.append(p)

            publisher.publish(m)
            yield True

    def get_traj_pts(self, traj_goal):
        if self.traj_goal_pt is None:
            self.traj_goal_pt = np.array([pt.positions for pt in traj_goal.trajectory.points])
        return self.traj_goal_pt


def execute(goal):
    curve = np.array([[c.x, c.y, c.z] for c in goal.curve])
    normals = np.array([[n.x, n.y, n.z] for n in goal.normals])
    pause = rospy.Duration(0.01)
    n_points = curve.shape[0]
    success = True
    rospy.loginfo("Starting curve execution.")
    for i, (c, n) in enumerate(zip(curve, normals)):
        if server.is_preempt_requested():
            server.set_preempted()
            success = False
            break

        rospy.loginfo("> c: {}\t n: {} ({:.2f}%)".format(c, n, float(i) / n_points * 100))
        server.publish_feedback(CurveExecutionFeedback(progress=i))
        rospy.sleep(pause)
    if success:
        # create result/response message
        server.set_succeeded(CurveExecutionResult(True))
        rospy.loginfo('Action successfully completed')
    else:
        server.set_aborted(CurveExecutionResult(False))
        rospy.loginfo('Whoops')


if __name__ == '__main__':
    rospy.init_node('dummy_controller')

    ce = CurveExecutor()
    # Similarly to service, advertise the action server
    # server = actionlib.SimpleActionServer('curve_executor', CurveExecutionAction, execute, False)
    # server.start()
    rospy.spin()
