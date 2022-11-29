from inverse_kinematics.kinematics_interface import ForwardKinematics, InverseKinematics
import rclpy
from rclpy.node import Node
from os_and_utils.visualizer_lib import VisualizerLib
from std_msgs.msg import Int8, Float64MultiArray, Int32, Bool
from geometry_msgs.msg import Pose, PoseStamped, Point, Quaternion, Vector3, Vector3Stamped, QuaternionStamped
from moveit_msgs.msg import RobotTrajectory
from trajectory_msgs.msg import JointTrajectoryPoint
from teleop_gesture_toolbox.msg import EEPoseGoals, JointAngles
from visualization_msgs.msg import MarkerArray, Marker
from sensor_msgs.msg import JointState

from copy import deepcopy
#import transformations
from os_and_utils import settings
from os_and_utils.utils_ros import extq

class RelaxedIKInterface(Node):
    def __init__(self):
        super().__init__('ik_lib')
        # RelaxedIK publishers
        self.ik_goal_r_pub = self.create_publisher(PoseStamped,'/ik_goal_r', 5)
        self.ik_goal_l_pub = self.create_publisher(PoseStamped,'/ik_goal_l', 5)
        self.goal_pos_pub = self.create_publisher(Vector3Stamped,'vive_position', 5)
        self.goal_quat_pub = self.create_publisher(QuaternionStamped,'vive_quaternion', 5)
        self.ee_pose_goals_pub = self.create_publisher(EEPoseGoals,'/ee_pose_goals',  5)
        self.quit_pub = self.create_publisher(Bool,'/quit', 5)
        self.seq = 1

    @staticmethod
    def relaxik_t(pose1):
        ''' All position goals and orientation goals are specified with respect to specified initial configuration.
            -> This function relates, sets goal poses to origin [0.,0.,0.] with orientation pointing up [0.,0.,0.,1.]
        '''
        pose_ = deepcopy(pose1)
        # 1.
        if settings.robot == 'panda':
            pose_.position.x -= 0.55442+0.04
            pose_.position.y -= 0.0
            pose_.position.z -= 0.62443
            #pose_.orientation = Quaternion(*transformations.quaternion_multiply([1.0, 0.0, 0.0, 0.0], extq(pose_.orientation)))
            #pose_.position.z -= 0.926
            #pose_.position.x -= 0.107
        elif settings.robot == 'iiwa':
            pose_.position.z -= 1.27
        else: raise Exception("Wrong robot name!")
        # 2.
        if settings.robot == 'iiwa':
            pose_.position.y = -pose_.position.y
        return pose_

    @staticmethod
    def relaxik_t_inv(pose1):
        ''' Additional inverse transformation to relaxik_t()
        '''
        raise Exception("TODO!")
        pose_ = deepcopy(pose1)
        if settings.robot == 'panda':
            pose_.position.z += 0.926
            pose_.position.x += 0.088
        #pose_.position.z += 1.27 # iiwa
        if settings.robot == 'iiwa':
            pose_.position.y = -pose_.position.y
        return pose_

    def ik_node_publish(self, pose_r=None, pose_l=None):
        ''' Sends goal poses to topic '/ee_pose_goals'.
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


class IK_bridge():
    ''' PURPOSE: Compare different inverse kinematics solvers
        - MoveIt IK
        -

        What are joints configuration and what is the latest link?
        Because the last link in urdf file should be linked with the tip_point of
    '''
    def __init__(self):
        if not settings.simulator == 'coppelia':

            self.fk = ForwardKinematics(frame_id=settings.base_link)
            self.ikt = InverseKinematics()

        self.relaxedik = RelaxedIKInterface()

    def getFKmoveit(self):
        return self.fk.getCurrentFK(settings.eef).pose_stamped[0]

    def getFKmoveitPose(self):
        return self.fk.getCurrentFK(settings.eef).pose_stamped[0].pose

    def getIKmoveit(self, pose):
        return self.ikt.getIK(self.move_group.get_name(), self.move_group.get_end_effector_link(), pose)
