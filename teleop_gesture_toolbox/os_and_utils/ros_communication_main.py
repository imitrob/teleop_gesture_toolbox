import rclpy
from rclpy.node import Node
from os_and_utils import settings
import numpy as np
import os_and_utils.move_lib as ml
if __name__ == '__main__': ml.init()
from os_and_utils.parse_yaml import ParseYAML
from inverse_kinematics.ik_lib import IK_bridge
import gesture_classification.gestures_lib as gl
if __name__ == '__main__': gl.init()
import os_and_utils.scenes as sl
if __name__ == '__main__': sl.init()
from os_and_utils.transformations import Transformations as tfm

from std_msgs.msg import Int8, Float64MultiArray, Int32, Bool, MultiArrayDimension, String
from geometry_msgs.msg import Pose, PoseStamped, Point, Quaternion, Vector3
from moveit_msgs.msg import RobotTrajectory
from trajectory_msgs.msg import JointTrajectoryPoint
from teleop_gesture_toolbox.msg import EEPoseGoals, JointAngles
from visualization_msgs.msg import MarkerArray, Marker
from sensor_msgs.msg import JointState

from leapmotion.frame_lib import Frame
import teleop_gesture_toolbox.msg as rosm
from teleop_gesture_toolbox.msg import DetectionSolution, DetectionObservations

try:
    from coppelia_sim_ros_interface.msg import ObjectInfo
except:
    print("WARNING: coppelia_sim_ros_interface package was not found!")
    coppelia_sim_ros_interface = None

from copy import deepcopy

class ROSComm(Node):
    ''' ROS communication of main thread: Subscribers (init & callbacks) and Publishers
    '''
    def __init__(self):
        super().__init__('ros_comm_main')
        # Saving the joint_states
        if settings.simulator == 'coppelia':
            self.create_subscription(JointState, "joint_states_coppelia", self.joint_states, 10)

            self.create_subscription(Pose, '/pose_eef', self.coppelia_eef, 10)
            self.create_subscription(Vector3, '/coppelia/camera_angle', self.camera_angle, 10)
            if coppelia_sim_ros_interface is not None:
                self.create_subscription(ObjectInfo, '/coppelia/object_info', self.object_info_callback, 10)
        else:
            self.create_subscription(JointState, "joint_states", self.joint_states, 10)

        # Saving relaxedIK output
        self.create_subscription(JointAngles, '/joint_angle_solutions', self.ik, 10)
        # Goal pose publisher
        self.ee_pose_goals_pub = self.create_publisher(EEPoseGoals,'/ee_pose_goals', 5)

        self.create_subscription(rosm.Frame, '/hand_frame', ROSComm.hand_frame, 10)

        if settings.launch_gesture_detection:
            self.create_subscription(DetectionSolution, '/teleop_gesture_toolbox/static_detection_solutions', self.save_static_detection_solutions, 10)
            self.static_detection_observations_pub = self.create_publisher(DetectionObservations,'/teleop_gesture_toolbox/static_detection_observations', 5)

            self.create_subscription(DetectionSolution, '/teleop_gesture_toolbox/dynamic_detection_solutions', self.save_dynamic_detection_solutions, 10)
            self.dynamic_detection_observations_pub = self.create_publisher(DetectionObservations, '/teleop_gesture_toolbox/dynamic_detection_observations', 5)

        self.controller = self.create_publisher(Float64MultiArray, '/teleop_gesture_toolbox/target', 5)
        self.ik_bridge = IK_bridge()
        self.hand_mode = settings.get_hand_mode()

        self.gesture_solution_pub = self.create_publisher(String, '/teleop_gesture_toolbox/filtered_gestures', 5)

    def publish_eef_goal_pose(self, goal_pose):
        ''' Publish goal_pose /ee_pose_goals to relaxedIK with its transform
            Publish goal_pose /ee_pose_goals the goal eef pose
        '''
        self.ee_pose_goals_pub.publish(goal_pose)
        self.ik_bridge.relaxedik.ik_node_publish(pose_r = self.ik_bridge.relaxedik.relaxik_t(goal_pose))

    def object_info_callback(self, data):
        ''' Only handles pose
        '''
        if sl.scene:
            object_names = sl.scene.object_names
            object_id = object_names.index(data.name)
            sl.scene.object_poses[object_id] = data.pose

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

    def send_g_data(self):
        ''' Sends appropriate gesture data as ROS msg
            Launched node for static/dynamic detection.
        '''
        if len(ml.md.frames) == 0: return
        hand_mode = self.hand_mode

        msg = DetectionObservations()
        msg.observations = Float64MultiArray()
        msg.sensor_seq = ml.md.frames[-1].seq
        msg.header.stamp.sec = ml.md.frames[-1].sec
        msg.header.stamp.nanosec = ml.md.frames[-1].nanosec

        mad1 = MultiArrayDimension()
        mad1.label = 'time'
        mad2 = MultiArrayDimension()
        mad2.label = 'xyz'

        for key in hand_mode.keys():
            args = gl.gd.static_network_info
            if hand_mode is not None and 'static' in hand_mode[key]:
                if key == 'l':
                    if ml.md.l_present():
                        msg.observations.data = ml.md.frames[-1].l.get_learning_data_static(definition=args['input_definition_version'])
                        msg.header.frame_id = 'l'
                        self.static_detection_observations_pub.publish(msg)
                elif key == 'r':
                    if ml.md.r_present():
                        msg.observations.data = ml.md.frames[-1].r.get_learning_data_static(definition=args['input_definition_version'])
                        msg.header.frame_id = 'r'
                        self.static_detection_observations_pub.publish(msg)


            time_samples = settings.yaml_config_gestures['misc_network_args']['time_samples']
            if hand_mode is not None and 'dynamic' in hand_mode[key] and len(ml.md.frames) > time_samples:
                args = gl.gd.dynamic_network_info
                if getattr(ml.md, key+'_present')():
                    try:
                        ''' Warn when low FPS '''
                        n = ml.md.frames[-1].fps
                        if n < 5: print("Warning: FPS =", n)

                        ''' Creates timestamp indexes starting with [-1, -x, ...] '''
                        time_samples_series = [-1]
                        time_samples_series.extend((n * np.array(range(-1, -time_samples, -1))  / time_samples).astype(int))
                        time_samples_series.sort()

                        ''' Compose data '''
                        data_composition = []
                        for time_sample in time_samples_series:
                            data_composition.append(getattr(ml.md.frames[time_sample], key).get_single_learning_data_dynamic(definition=args['input_definition_version']))

                        ''' Transform to Leap frame id '''
                        data_composition_ = []
                        for point in data_composition:
                            data_composition_.append(tfm.transformLeapToBase(point, out='position'))
                        data_composition = data_composition_

                        ''' Check if the length of composed data is aorund 1sec '''
                        if 0.7 >= ml.md.frames[-1].stamp() - ml.md.frames[int(-n)].stamp() >= 1.3:
                            print("WARNING: data frame composed is not 1sec long")

                        ''' Subtract middle path point from all path points '''

                        #if 'normalize' in args:
                        data_composition_ = []
                        data_composition0 = deepcopy(data_composition[len(data_composition)//2])
                        for n in range(0, len(data_composition)):
                            data_composition_.append(np.subtract(data_composition[n], data_composition0))
                        data_composition = data_composition_

                        ''' Fill the ROS msg '''
                        data_composition = np.array(data_composition, dtype=float)

                        mad1.size = data_composition.shape[0]
                        mad2.size = data_composition.shape[1]
                        data_composition = list(data_composition.flatten())
                        msg.observations.data = data_composition
                        msg.observations.layout.dim = [mad1, mad2]
                        msg.header.frame_id = key
                        self.dynamic_detection_observations_pub.publish(msg)
                    except IndexError:
                        pass


    def detection_thread(self, freq=1., args={}):
        if not ROS: raise Exception("ROS cannot be imported!")

        rclpy.init()

        settings.gesture_detection_on = True
        settings.launch_gesture_detection = True

        roscm = ROSComm()

        configGestures = ParseYAML.load_gesture_config_file(settings.paths.custom_settings_yaml)
        hand_mode_set = configGestures['using_hand_mode_set']
        hand_mode = dict(configGestures['hand_mode_sets'][hand_mode_set])

        rate = roscm.create_rate(freq)

        while rclpy.ok():
            if ml.md.frames:
                send_g_data(roscm, hand_mode, args)

            rate.sleep()


def init():
    global roscm
    rclpy.init(args=None)
    roscm = ROSComm()
