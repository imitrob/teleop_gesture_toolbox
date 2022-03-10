import collections
import numpy as np
import yaml, random
import settings
from copy import deepcopy

from geometry_msgs.msg import Quaternion, Pose, PoseStamped, Point, Vector3
from os_and_utils.utils import ordered_load
import os_and_utils.scenes as sl

class MoveData():
    def __init__(self, init_goal_pose=True):
        '''
        > saved in arrays
        - Leap Controller
        - Plan (eef_pose, goal_pose, ...)
        > single data
        - Plan (eef_pose, goal_pose, ...)
        - States (joints, velocity, eff, ... )
        '''

        bfr_len = 1000 #settings.configRecording['BufferLen']
        ''' Leap Controller data saved as circullar buffer '''
        self.frames = collections.deque(maxlen=bfr_len)
        ''' '''
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
        ''' Scaling factor: if self.mode=='live' '''
        self.scale = 2
        ## interactive
        self.action = False
        self.strict_mode = False
        ''' Path ID: if self.mode=='play' '''
        self.picked_path = 0
        ''' Gripper object attached bool '''
        self.attached = False
        ''' Mode about scene interaction - Deprecated '''
        self.live_mode = 'default'

        ''' Builder mode: ['stack', 'wall', 'replace'] '''
        self.build_modes = ['stack', 'wall', 'replace']
        self.build_mode = 'stack'

        self.structures = [] # Structure()

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

        ''' Constant for updating trajectories for real manipulator '''
        self.traj_update_horizon = 0.6

        ''' mouse3d_position: Not fully integrated '''
        self.mouse3d_position = [0.3, 0.0, 0.5]

        self.current_threshold_to_flip_id = 0
        self.object_focus_id = 0
        self.object_touch_id = 0

        if init_goal_pose:
            self.goal_pose = Pose()
            self.goal_pose.orientation = self.ENV_DAT['above']['ori']
            self.goal_pose.position = Point(0.4,0.,1.0)

        # Handle to access simulator
        self.m = None

    def present(self):
        return self.r_present() or self.l_present()

    def r_present(self):
        if self.frames and self.frames[-1] and self.frames[-1].r and self.frames[-1].r.visible:
            return True
        return False

    def l_present(self):
        if self.frames and self.frames[-1] and self.frames[-1].l and self.frames[-1].l.visible:
            return True
        return False

    def get_frame_window_of_last_secs(self, stamp, N_secs):
        ''' Select frames chosen with stamp and last N_secs
        '''
        n = 0
        #     stamp-N_secs       stamp
        # ----|-------N_secs------|
        # ---*************************- <-- chosen frames
        #              <~~~while~~~  |
        #                           self.frames[-1].stamp()
        for i in range(-1, -len(self.frames),-1):
            if stamp-N_secs > self.frames[i].stamp():
                n=i
                break
        print(f"len(self.frames) {len(self.frames)}, n {n}")

        # return frames time window
        frames = []
        for i in range(-1, n, -1):
            frames.append(self.frames[i])
        return frames

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

class Structure():
    '''
    stack order:
    ||| <- id=2
    ||| <- id=1
    ||| <- id=0
    wall order: (right direction from id=0 which is base obj)
    (space between blocks is not representative)
        ||| <- id=6
      ||| ||| <- id=3,5
    ||| ||| ||| <- id=1,2,4

    Example building stack with base object with id=0:
    id_0_position = structure = Structure(type='wall', id=0, size=0.04) # [0.,0.,0.]
    id_1_position = structure.add(id=10) # [0.,0.,0.04]
    id_2_position = structure.add(id=30) # [0.,0.,0.08]
    id_3_position = structure.add(id=20) # [0.,0.,0.12]
    print(f"Number of blocks: {structure.n}") # "Number of blocks: 4"
    print(f"Object IDs within structure: {structure.object_stack}") # "Object IDs within structure: [0,10,30,20]"
    id_3_position = structure.remove() # [0.,0.,0.12]
    id_2_position = structure.remove() # [0.,0.,0.8]
    id_1_position = structure.remove() # [0.,0.,0.4]
    print(f"Number of blocks: {structure.n}") # "Number of blocks: 1"
    print(f"Object IDs within structure: {structure.object_stack}") # "Object IDs within structure: [0]"
    id_0_position = structure.remove() # [0.,0.,0.]

    '''
    def __init__(self, type, id=None, size=0.04, base_position=[0.,0.,0.]):
        self.type = type
        self.object_stack = []
        self.object_size = 0.04 # box size [m]
        self.base_position = base_position

        if id is not None: self.add(id)

    def __getattr__(self, attr):
        if attr == 'n':
            return len(self.object_stack)
        elif attr == 'base_id':
            if self.object_stack:
                return self.object_stack[0]
            else:
                return None

    def add(self, id):
        self.object_stack.append(id)
        return self.get_position(id)

    def remove(self):
        position_remove = self.get_position(self.object_stack[-1])
        self.object_stack.pop(-1)
        return position_remove

    def get_n_block_based_on_object_id(self, id):
        for n,obj in enumerate(self.object_stack):
            if obj == id:
                return n
        raise Exception(f"ID '{id}' of object is not present in given structure, which has IDs '{self.object_stack}'")

    def get_position(self, id):
        if self.type == 'stack': return list(np.array(self.get_relative_position_stack(id=id)) + np.array(self.base_position))
        elif self.type == 'wall': return list(np.array(self.get_relative_position_wall(id=id)) + np.array(self.base_position))
        else: raise Exception(f"Structure type '{self.type}' not found !")

    def get_relative_position_stack(self, id):
        ''' new relative (to id=0) position is on top of all stacked objects '''
        n_block = self.get_n_block_based_on_object_id(id)
        return [0.,0., n_block * self.object_size]

    def get_relative_position_wall(self, id):
        ''' new wall position '''
        n_block = self.get_n_block_based_on_object_id(id)

        space_between_blocks = self.object_size / 4
        z_step = self.object_size
        x_step = self.object_size + space_between_blocks
        x_odd = x_step/2

        if n_block == 0:
            return [0.,0.,0.]
        elif n_block == 1:
            return [x_step, 0., 0.]
        elif n_block == 2:
            return [x_odd, 0., z_step]
        elif n_block == 3:
            return [2*x_step, 0., 0.]
        elif n_block == 4:
            return [x_odd+x_step, 0., z_step]
        elif n_block == 5:
            return [x_odd*2, 0., 2*z_step]
        elif n_block == 6:
            return [3*x_step, 0., 0.]
        elif n_block == 7:
            return [x_odd+2*x_step, 0., z_step]
        elif n_block == 8:
            return [x_odd*2, 0., 2*z_step]


def init():
    global md
    md = MoveData()









#
