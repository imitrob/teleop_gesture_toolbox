import collections
import numpy as np
import yaml, random
import settings
from copy import deepcopy

from geometry_msgs.msg import Quaternion, Pose, PoseStamped, Point, Vector3
from os_and_utils.utils import ordered_load
import os_and_utils.scenes as sl

class MoveData():
    def __init__(self):
        '''
        > saved in arrays
        - Leap Controller
        - Plan (eef_pose, goal_pose, ...)
        > single data
        - Plan (eef_pose, goal_pose, ...)
        - States (joints, velocity, eff, ... )
        '''

        bfr_len = 300 #settings.configRecording['BufferLen']
        self.frames = collections.deque(maxlen=bfr_len)

        self.goal_pose_array = collections.deque(maxlen=bfr_len)
        self.eef_pose_array = collections.deque(maxlen=bfr_len)

        self.joint_states = collections.deque(maxlen=bfr_len)

        ## Current/Active robot data at the moment
        self.goal_joints = None
        self.goal_pose = None
        self.joints = None
        self.velocity = None
        self.effort = None
        self.eef_pose = Pose()
        # Goal joints -> RelaxedIK output
        # Goal pose -> RelaxedIK input
        # joints -> JointStates topic
        # eef_pose -> from joint_states


        with open(settings.paths.custom_settings_yaml+"robot_move.yaml", 'r') as stream:
            robot_move_data_loaded = ordered_load(stream, yaml.SafeLoader)

        self.leap_axes = robot_move_data_loaded['LEAP_AXES']
        self.ENV_DAT = {}
        for key in robot_move_data_loaded['ENV_DAT']:
            self.ENV_DAT[key] = {}
            self.ENV_DAT[key]['view'] = robot_move_data_loaded['ENV_DAT'][key]['view']
            self.ENV_DAT[key]['ori_axes'] = robot_move_data_loaded['ENV_DAT'][key]['ori_axes']
            self.ENV_DAT[key]['ori_live'] = robot_move_data_loaded['ENV_DAT'][key]['ori_live']
            self.ENV_DAT[key]['axes'] = robot_move_data_loaded['ENV_DAT'][key]['axes']
            self.ENV_DAT[key]['min'] = Point(*robot_move_data_loaded['ENV_DAT'][key]['min'])
            self.ENV_DAT[key]['max'] = Point(*robot_move_data_loaded['ENV_DAT'][key]['max'])
            self.ENV_DAT[key]['start'] = Point(*robot_move_data_loaded['ENV_DAT'][key]['start'])
            self.ENV_DAT[key]['ori'] = Quaternion(*robot_move_data_loaded['ENV_DAT'][key]['ori'])


        # TODO: Remove this condition
        if settings.inverse_kinematics == 'pyrep':
            self.ENV_DAT['above']['ori'] = Quaternion(0.0, 0.0, 1.0, 0.0)
            self.ENV_DAT['wall']['ori']  = Quaternion(0, np.sqrt(2)/2, 0, np.sqrt(2)/2)
            self.ENV_DAT['table']['ori'] = Quaternion(np.sqrt(2)/2, np.sqrt(2)/2., 0.0, 0.0)
        # chosen workspace
        self.ENV = self.ENV_DAT['above']

        # beta
        # angles from camera -> coppelia
        # angles from real worls -> some params
        # TODO: load from YAML
        self.camera_orientation = Vector3(0.,0.,0.)

        self.mode = 'live' # 'play'/'live'/'alternative'
        self.scale = 2 # scaling factor for Leap
        ## interactive
        self.action = False
        self.strict_mode = False
        # if play path
        self.picked_path = 0
        self.attached = False
        self.live_mode = 'default'

        self.gripper = 0.
        self.speed = 5
        self.applied_force = 10

        ### Live mode: gesture data
        self.gestures_goal_pose = Pose()
        self.gestures_goal_pose.position = self.ENV['start']
        self.gestures_goal_pose.orientation = self.ENV['ori']
        self.gestures_goal_stride = 0.1
        self.gestures_goal_rot_stride = np.pi/8
        # The copy of goal_pose in form of active trajectory
        self._goal = None

        self.traj_update_horizon = 0.6

    def r_present(self):
        if self.frames and self.frames[-1] and self.frames[-1].r and self.frames[-1].r.visible:
            return True
        return False

    def l_present(self):
        if self.frames and self.frames[-1] and self.frames[-1].l and self.frames[-1].l.visible:
            return True
        return False

    def modes(self):
        return ['play', 'live', 'alternative']

    def get_random_position(self):
        ''' Get random position (ret pose obj) within environment based on md.ENV['max'|'min'] boundaries
            Orientation is set to default md.ENV['ori']

        Returns:
            Pose (Pose()): Random pose
        '''
        x_len = self.ENV['max'].x - self.ENV['min'].x
        y_len = self.ENV['max'].y - self.ENV['min'].y
        z_len = self.ENV['max'].z - self.ENV['min'].z

        x = random.random()
        y = random.random()
        z = random.random()

        x_ = self.ENV['min'].x + x_len * x
        y_ = self.ENV['min'].y + y_len * y
        z_ = self.ENV['min'].z + z_len * z

        pose = Pose()
        pose.position = Point(x_, y_, z_)
        pose.orientation = self.ENV['ori']
        return pose

    def get_random_joints(self, settings):
        ''' Returns random robot joints within bounds

        Returns:
            Joints (Float[7]): Robot joints float array based on configuration in settings
        '''
        joints_diff = np.array(settings.upper_lim) - np.array(settings.lower_lim)
        joints_diff_rand = [joints_diff[i] * random.random() for i in range(len(settings.upper_lim))]
        return np.add(settings.lower_lim, joints_diff_rand)

    def point_in_env(self, point):
        if self.ENV['min'].x <= point[0] <= self.ENV['max'].x:
          if self.ENV['min'].y <= point[1] <= self.ENV['max'].y:
            if self.ENV['min'].z <= point[2] <= self.ENV['max'].z:
                return True
        return False

    def changePlayPath(self, path_=None):
        for n, path in enumerate(sl.paths):
            if not path_ or path.name == path_: # pick first path if path_ not given
                sl.scenes.make_scene(path.scene)
                self.picked_path = n
                self.ENV = self.ENV_DAT[path.ENV]
                settings.HoldValue = 0
                settings.currentPose = 0
                self.goal_pose = deepcopy(sl.paths[1].poses[1])
                break

    def changeLiveMode(self, text):
        # Reset Gestures
        self.gestures_goal_pose = Pose()
        self.gestures_goal_pose.position = deepcopy(self.ENV['start'])
        self.gestures_goal_pose.orientation.w = 1.0
        if text == "Default":
            self.live_mode = 'default'
        elif text == "Gesture based":
            self.live_mode = 'gesture'
        elif text == "Interactive":
            self.live_mode = 'interactive'



def init():
    global md
    md = MoveData()









#
