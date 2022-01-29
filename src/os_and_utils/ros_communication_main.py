import rospy
import settings
import os_and_utils.move_lib as ml
from inverse_kinematics.ik_lib import IK_bridge
import gestures_lib as gl
#gl.init()

from std_msgs.msg import Int8, Float64MultiArray, Int32, Bool
from geometry_msgs.msg import Pose, PoseStamped, Point, Quaternion, Vector3
from moveit_msgs.msg import RobotTrajectory
from trajectory_msgs.msg import JointTrajectoryPoint
from control_msgs.msg import FollowJointTrajectoryGoal, JointTolerance
from relaxed_ik.msg import EEPoseGoals, JointAngles
from visualization_msgs.msg import MarkerArray, Marker
from sensor_msgs.msg import JointState

from leapmotion.frame_lib import Frame
import mirracle_gestures.msg as rosm
from mirracle_gestures.msg import DetectionSolution, DetectionObservations

class ROSComm():
    ''' ROS communication of main thread: Subscribers (init & callbacks) and Publishers
    '''
    def __init__(self):
        # Saving the joint_states
        if settings.simulator == 'coppelia':
            rospy.Subscriber("joint_states_coppelia", JointState, self.joint_states)

            rospy.Subscriber('/pose_eef', Pose, self.coppelia_eef)
            rospy.Subscriber('/coppelia/camera_angle', Vector3, self.camera_angle)
        else:
            rospy.Subscriber("joint_states", JointState, self.joint_states)

        # Saving relaxedIK output
        rospy.Subscriber('/relaxed_ik/joint_angle_solutions', JointAngles, self.ik)
        # Goal pose publisher
        self.ee_pose_goals_pub = rospy.Publisher('/ee_pose_goals', Pose, queue_size=5)

        rospy.Subscriber('/hand_frame', rosm.Frame, self.hand_frame)

        if settings.launch_gesture_detection:
            rospy.Subscriber('/mirracle_gestures/static_detection_solutions', DetectionSolution, self.save_static_detection_solutions)
            self.static_detection_observations_pub = rospy.Publisher('/mirracle_gestures/static_detection_observations', DetectionObservations, queue_size=5)

            rospy.Subscriber('/mirracle_gestures/dynamic_detection_solutions', DetectionSolution, self.save_dynamic_detection_solutions)
            self.dynamic_detection_observations_pub = rospy.Publisher('/mirracle_gestures/dynamic_detection_observations', DetectionObservations, queue_size=5)

        self.controller = rospy.Publisher('/mirracle_gestures/target', Float64MultiArray, queue_size=5)
        self.ik_bridge = IK_bridge()

    def publish_eef_goal_pose(self, goal_pose):
        ''' Publish goal_pose /relaxed_ik/ee_pose_goals to relaxedIK with its transform
            Publish goal_pose /ee_pose_goals the goal eef pose
        '''
        self.ee_pose_goals_pub.publish(goal_pose)
        self.ik_bridge.relaxedik.ik_node_publish(pose_r = self.ik_bridge.relaxedik.relaxik_t(goal_pose))


    @staticmethod
    def ik(data):
        joints = []
        for ang in data.angles:
            joints.append(ang.data)
        ml.md.goal_joints = joints

    @staticmethod
    def coppelia_eef(data):
        ml.md.eef_pose = data

    @staticmethod
    def camera_angle(data):
        ml.md.camera_orientation = data

    @staticmethod
    def hand_frame(data):
        ''' Hand data received by ROS msg is saved
        '''
        f = Frame()
        f.import_from_ros(data)
        ml.md.frames.append(f)

    @staticmethod
    def joint_states(data):
        ''' Saves joint_states to * append array 'settings.joint_in_time' circle buffer.
                                  * latest data 'ml.md.joints', 'ml.md.velocity', 'ml.md.effort'
            Topic used:
                - CoppeliaSim (PyRep) -> topic "/joint_states_coppelia"
                - Other (Gazebo sim / Real) -> topic "/joint_states"

        '''
        ml.md.joint_states.append(data)

        if settings.robot == 'iiwa':
            if data.name[0][1] == '1': # r1 robot
                ml.md.joints = data.position # Position = Angles [rad]
                ml.md.velocity = data.velocity # [rad/s]
                ml.md.effort = data.effort # [Nm]
            ''' Enable second robot arm
            elif data.name[0][1] == '2': # r2 robot
                r2_joint_pos = data.position # Position = Angles [rad]
                r2_joint_vel = data.velocity # [rad/s]
                r2_joint_eff = data.effort # [Nm]
                float(data.header.stamp.to_sec())
            '''
        elif settings.robot == 'panda':
            ml.md.joints = data.position[-7:] # Position = Angles [rad]
            ml.md.velocity = data.velocity[-7:] # [rad/s]
            ml.md.effort = data.effort[-7:] # [Nm]

        else: raise Exception("Wrong robot name!")

    @staticmethod
    def save_static_detection_solutions(data):
        gl.gd.new_record(data, type='static')

    @staticmethod
    def save_dynamic_detection_solutions(data):
        gl.gd.new_record(data, type='dynamic')







#
