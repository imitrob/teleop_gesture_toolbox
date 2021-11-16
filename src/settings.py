#!/usr/bin/env python2
'''
Loads informations about robot, configurations, move data, gestures, ...
All configurations are loaded from:
    - /include/custom_settings/robot_move.yaml
    - /include/custom_settings/gesture_recording.yaml
    - /include/custom_settings/application.yaml
    and
    - /include/custom_settings/*paths*.yaml
    - /include/custom_settings/*scenes*.yaml
    - /include/custom_settings/poses.yaml

Import this file and call init() to access parameters, info and data
'''
import collections
from collections import OrderedDict
import numpy as np
from copy import deepcopy
import os
from os.path import expanduser, isfile
import time
# Needed to load and save rosparams
try:
    ROS = True
    import rospy
    from geometry_msgs.msg import Quaternion, Pose, PoseStamped, Point, Vector3
    from visualization_msgs.msg import MarkerArray, Marker
    from std_msgs.msg import Int8, Float64MultiArray
except ModuleNotFoundError:
    print("[WARN*] ROS cannot be not imported!")
    ROS = False
import yaml
import io
import random

# function for loading dict from file ordered
def ordered_load(stream, Loader=yaml.SafeLoader, object_pairs_hook=OrderedDict):
    class OrderedLoader(Loader):
        pass
    def construct_mapping(loader, node):
        loader.flatten_mapping(node)
        return object_pairs_hook(loader.construct_pairs(node))
    OrderedLoader.add_constructor(
        yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
        construct_mapping)
    return yaml.load(stream, OrderedLoader)

def init(minimal=False):
    ''' Initialize the shared data across threads (Leap, UI, Control)
    Parameters:
        minimal (Bool): Import only file paths and general informations
    '''
    # tmp
    global loopn
    loopn = 0

    ### 1. Initialize File/Folder Paths ###
    #######################################
    global HOME, LEARN_PATH, GRAPHICS_PATH, GESTURE_NAMES, GESTURE_KEYS, NETWORK_PATH, PLOTS_PATH, WS_FOLDER, COPPELIA_SCENE_PATH, MODELS_PATH, CUSTOM_SETTINGS_YAML, NETWORKS_DRIVE_URL, GESTURE_NETWORK_FILE
    HOME = expanduser("~")
    # searches for the WS name + print it
    THIS_FILE_PATH = os.path.dirname(os.path.realpath(__file__))
    THIS_FILE_TMP = os.path.abspath(os.path.join(THIS_FILE_PATH, '..', '..', '..'))
    WS_FOLDER = THIS_FILE_TMP.split('/')[-1]

    LEARN_PATH = HOME+"/"+WS_FOLDER+"/src/mirracle_gestures/include/data/learning/"
    GRAPHICS_PATH = HOME+"/"+WS_FOLDER+"/src/mirracle_gestures/include/graphics/"
    PLOTS_PATH = HOME+"/"+WS_FOLDER+"/src/mirracle_gestures/include/plots/"
    NETWORK_PATH = HOME+"/"+WS_FOLDER+"/src/mirracle_gestures/include/data/Trained_network/"
    COPPELIA_SCENE_PATH = HOME+"/"+WS_FOLDER+"/src/mirracle_sim/include/scenes/"
    MODELS_PATH = HOME+"/"+WS_FOLDER+"/src/mirracle_gestures/include/models/"
    CUSTOM_SETTINGS_YAML = HOME+"/"+WS_FOLDER+"/src/mirracle_gestures/include/custom_settings/"

    ### 2. Loads config. data to gestures  ###
    ###     - gesture_recording.yaml       ###
    ##########################################

    with open(CUSTOM_SETTINGS_YAML+"gesture_recording.yaml", 'r') as stream:
        gestures_data_loaded = ordered_load(stream, yaml.SafeLoader)
        #gestures_data_loaded = yaml.safe_load(stream)

    keys = gestures_data_loaded.keys()
    Gs = []
    Gs_set = gestures_data_loaded['using_set']
    configGestures = deepcopy(gestures_data_loaded['configGestures'])
    configRecording = deepcopy(gestures_data_loaded['Recording'])
    configRecognition = deepcopy(gestures_data_loaded['Recognition'])
    del gestures_data_loaded['using_set']; del gestures_data_loaded['Recording']; del gestures_data_loaded['configGestures']; del gestures_data_loaded['Recognition']

    GESTURE_NAMES = []
    GESTURE_KEYS = []
    # Check if yaml file is setup properly
    try:
        gestures_data_loaded[Gs_set]
    except:
        raise Exception("Error in gesture_recording.yaml, using_set variable, does not point to any available set below!")
    try:
        gestures_data_loaded[Gs_set].keys()
    except:
        raise Exception("Error in gesture_recording.yaml, used gesture set does not have any item!")
    # Setup gesture list
    for key in gestures_data_loaded[Gs_set].keys():
        g = gestures_data_loaded[Gs_set][key]
        GESTURE_NAMES.append(key)
        GESTURE_KEYS.append(g['key'])

    NETWORKS_DRIVE_URL = configGestures['NETWORKS_DRIVE_URL']
    GESTURE_NETWORK_FILE = configGestures['NETWORK_FILE']
    ### 3. Loads config. about robot         ###
    ###     - ROSparams /mirracle_config/*   ###
    ###     - robot_move.yaml.yaml           ###
    ############################################
    global JOINT_NAMES, BASE_LINK, GROUP_NAME, ROBOT_NAME, GRIPPER_NAME, SIMULATOR_NAME, TAC_TOPIC, JOINT_STATES_TOPIC, EEF_NAME, TOPPRA_ON, VIS_ON, IK_SOLVER, GRASPING_GROUP, IK_TOPIC
    global upper_lim, lower_lim, effort_lim, vel_lim
    if ROS:
        ROBOT_NAME = rospy.get_param("/mirracle_config/robot")
        SIMULATOR_NAME = rospy.get_param("/mirracle_config/simulator")
        GRIPPER_NAME = rospy.get_param("/mirracle_config/gripper")
        VIS_ON = rospy.get_param("/mirracle_config/visualize")
        IK_SOLVER = rospy.get_param("/mirracle_config/ik_solver")
        IK_TOPIC = rospy.get_param("/mirracle_config/ik_topic")
    TOPPRA_ON = True

    with open(CUSTOM_SETTINGS_YAML+"robot_move.yaml", 'r') as stream:
        robot_move_data_loaded = ordered_load(stream, yaml.SafeLoader)
        #robot_move_data_loaded = yaml.safe_load(stream)
    vel_lim = robot_move_data_loaded['robots'][ROBOT_NAME]['vel_lim']
    effort_lim = robot_move_data_loaded['robots'][ROBOT_NAME]['effort_lim']
    lower_lim = robot_move_data_loaded['robots'][ROBOT_NAME]['lower_lim']
    upper_lim = robot_move_data_loaded['robots'][ROBOT_NAME]['upper_lim']
    JOINT_NAMES = robot_move_data_loaded['robots'][ROBOT_NAME]['JOINT_NAMES']
    BASE_LINK = robot_move_data_loaded['robots'][ROBOT_NAME]['BASE_LINK']
    GROUP_NAME = robot_move_data_loaded['robots'][ROBOT_NAME]['GROUP_NAME']
    EEF_NAME = robot_move_data_loaded['robots'][ROBOT_NAME]['EEF_NAME']
    GRASPING_GROUP = robot_move_data_loaded['robots'][ROBOT_NAME]['GRASPING_GROUP']
    if SIMULATOR_NAME not in robot_move_data_loaded['simulators']:
        raise Exception("Wrong simualator name")
    TAC_TOPIC = robot_move_data_loaded['robots'][ROBOT_NAME][SIMULATOR_NAME]['TAC_TOPIC']
    JOINT_STATES_TOPIC = robot_move_data_loaded['robots'][ROBOT_NAME][SIMULATOR_NAME]['JOINT_STATES_TOPIC']

    ### HERE ENDS MINIMAL CONFIGURATION      ###
    if minimal:
        return
    print("[Settings] Workspace folder is set to: "+WS_FOLDER)
    ### 4. Init. Data                              ###
    ###   > saved in arrays                        ###
    ###     - Leap Controller                      ###
    ###     - Plan (eef_pose, goal_pose, ...)      ###
    ###   > single data                            ###
    ###     - Plan (eef_pose, goal_pose, ...)      ###
    ###     - States (joints, velocity, eff, ... ) ###
    ##################################################

    # Note: updated in leapmotionlistener.py with Leap frame frequency (~80Hz)
    global BUFFER_LEN, frames, timestamps, frames_adv, goal_pose_array, eef_pose_array, joints_in_time
    BUFFER_LEN = configRecording['BufferLen']
    frames = collections.deque(maxlen=BUFFER_LEN)
    timestamps = collections.deque(maxlen=BUFFER_LEN)
    frames_adv = collections.deque(maxlen=BUFFER_LEN)
    # Note: updated in main.py with rate 10Hz
    goal_pose_array = collections.deque(maxlen=BUFFER_LEN)
    eef_pose_array = collections.deque(maxlen=BUFFER_LEN)

    joints_in_time = collections.deque(maxlen=BUFFER_LEN)

    ## Current/Active robot data at the moment
    global goal_joints, goal_pose, eef_pose, joints, velocity, effort
    goal_joints, goal_pose, joints, velocity, effort, eef_pose = None, None, None, None, None, Pose()
    # Goal joints -> RelaxedIK output
    # Goal pose -> RelaxedIK input
    # joints -> JointStates topic
    # eef_pose -> from joint_states

    ## Publisher holder for ROS topic
    global ee_pose_goals_pub; ee_pose_goals_pub = None

    ### 5. Latest Data                      ###
    ###     - GestureData()                 ###
    ###     - MoveData()                    ###
    ###     - Generated Scenes from YAML    ###
    ###     - Generated Paths from YAML     ###
    ###     - Current scene info            ###
    ###########################################
    global gd, md, rd, sp, mo, ss, scene
    # Active gesture data at the moment
    gd = GestureDataHands()
    # Move robot and control data
    md = MoveData()

    ## Objects for saved scenes and paths
    ss = CustomScene.GenerateFromYAML()
    sp = CustomPath.GenerateFromYAML()
    scene = None # current scene informations
    mo = None # MoveIt object

    ## Fixed Conditions
    global POSITION_MODE, ORIENTATION_MODE, print_path_trace
    # ORIENTATION_MODE options:
    #   - 'fixed', eef has fixed eef orientaion based on chosen environment (md.ENV)
    ORIENTATION_MODE = 'fixed'

    # POSITION_MODE options:
    #   - '', default
    #   - 'sim_camera', uses simulator camera to transform position
    POSITION_MODE = ''
    # When turned on, rViz marker array of executed trajectory is published
    print_path_trace = False

    ### 6. Gesture Recognition              ###
    ###     - Learning settings             ###
    ###########################################
    global pymcin, pymcout, train_args
    pymcout = None
    pymcin = Float64MultiArray()

    # Loaded from gesture_recording.yaml
    train_args = configRecognition['args']

    ## ROS publisher for pymc
    global pymc_in_pub; pymc_in_pub = None

    ### 7. User Interface Data              ###
    ###     - From application.yaml         ###
    ###########################################
    with open(CUSTOM_SETTINGS_YAML+"application.yaml", 'r') as stream:
        app_data_loaded = ordered_load(stream, yaml.SafeLoader)
        #app_data_loaded = yaml.safe_load(stream)
    # Configuration page values
    global NumConfigBars, VariableValues
    NumConfigBars = [app_data_loaded['ConfigurationPage']['Rows'], app_data_loaded['ConfigurationPage']['Columns']]
    VariableValues = np.zeros(NumConfigBars)

    # Status Bar
    global HoldAnchor, HoldPrevState, HoldValue, currentPose, WindowState, leavingAction, w, h, ui_scale, record_with_keys
    WindowState = 0.0  # 0-Main page, 1-Config page
    HoldAnchor = 0.0  # For moving status bar
    HoldPrevState = False  # --||--
    HoldValue = 0
    currentPose = 0
    leavingAction = False
    w, h = 1000, 800 # Will be set dynamically to proper value
    ui_scale = app_data_loaded['Scale']

    record_with_keys = False # Bool, Enables recording with keys in UI

    ### 8. For visualization data holder    ###
    ###########################################
    global viz; viz = None # VisualizerLib obj
    ## Joints visualization
    global sendedPlot; sendedPlot = None
    global realPlot; realPlot = None
    global sendedPlotVel; sendedPlotVel = None
    global realPlotVel; realPlotVel = None
    global point_before_toppra; point_before_toppra = None
    global point_after_toppra; point_after_toppra = None
    global point_after_replace; point_after_replace = None
    global toppraPlan; toppraPlan = None
    # Visualized Number of Joint <0,6>
    global NJ; NJ = 0

    ## Pose visualization
    global dataPosePlot; dataPosePlot = None
    global dataPoseGoalsPlot; dataPoseGoalsPlot = None


    print("[Settings] done")


class FrameAdv():
    ''' Advanced variables derived from frame object
    '''
    def __init__(self):
        self.l = HandAdv()
        self.r = HandAdv()

class HandAdv():
    ''' Advanced variables of hand derived from hand object
    '''
    def __init__(self):
        self.visible = False
        self.conf = 0.0
        self.OC = [0.0] * 5
        self.TCH12, self.TCH23, self.TCH34, self.TCH45 = [0.0] * 4
        self.TCH13, self.TCH14, self.TCH15 = [0.0] * 3
        self.vel = [0.0] * 3
        self.pPose = PoseStamped()
        self.pRaw = [0.0] * 6 # palm pose: x, y, z, roll, pitch, yaw
        self.pNormDir = [0.0] * 6 # palm normal vector and direction vector
        self.rot = Quaternion()
        self.rotRaw = [0.0] * 3
        self.rotRawEuler = [0.0] * 3

        self.time_last_stop = 0.0

        self.grab = 0.0
        self.pinch = 0.0

        ## Data processed for learning
        # direction vectors
        self.wrist_hand_angles_diff = []
        self.fingers_angles_diff = []
        self.pos_diff_comb = []
        #self.pRaw
        self.index_position = []


class GestureDataHands():
    '''
    '''
    def __init__(self):
        self.l = GestureDataHand()
        self.r = GestureDataHand()


class GestureDataHand():
    '''
    '''
    def __init__(self):
        with open(CUSTOM_SETTINGS_YAML+"gesture_recording.yaml", 'r') as stream:
            gestures_data_loaded = ordered_load(stream, yaml.SafeLoader)
        self.conf = False
        self.MIN_CONFIDENCE = gestures_data_loaded['configGestures']['MIN_CONFIDENCE']
        self.SINCE_FRAME_TIME = gestures_data_loaded['configGestures']['SINCE_FRAME_TIME']

        self.tch12, self.tch23, self.tch34, self.tch45 = [False] * 4
        self.tch13, self.tch14, self.tch15 = [False] * 3
        self.TCH_TURN_ON_DIST = gestures_data_loaded['configGestures']['TCH_TURN_ON_DIST']
        self.TCH_TURN_OFF_DIST = gestures_data_loaded['configGestures']['TCH_TURN_OFF_DIST']

        self.oc = [False] * 5
        self.OC_TURN_ON_THRE =  gestures_data_loaded['configGestures']['OC_TURN_ON_THRE']
        self.OC_TURN_OFF_THRE = gestures_data_loaded['configGestures']['OC_TURN_OFF_THRE']


        self.poses = []
        self.gests = []
        for gesture in GESTURE_NAMES:
            using_set = gestures_data_loaded['using_set']
            try:
                if gesture in gestures_data_loaded[using_set].keys():
                    if gestures_data_loaded[using_set][gesture]['dynamic'] == 'false' or gestures_data_loaded[using_set][gesture]['dynamic'] == False:
                        self.poses.append(PoseData(name=gesture, data=gestures_data_loaded[using_set][gesture]))
                    else:
                        self.gests.append(GestureData(name=gesture, data=gestures_data_loaded[using_set][gesture]))
                else:
                    raise Exception("[ERROR* Settings] Information about gesture with name: ", gesture, " could not be found in gesture_recording.yaml file. Make an entry there.")
            except KeyError:
                print("gesture key", gesture, " and keys ", gestures_data_loaded[using].keys())



        # Create links for gestures
        self.POSES = {}
        for n, i in enumerate(self.poses):
            self.POSES[i.NAME] = n
        self.GESTS = {}
        for n, i in enumerate(self.gests):
            self.GESTS[i.NAME] = n

        self.final_chosen_pose = 0
        self.final_chosen_gesture = 0

class PoseData():
    def __init__(self, name, data):
        data = ParseYAML.parseStaticGesture(data)

        self.NAME = name
        self.prob = 0.0
        self.toggle = False
        self.TURN_ON_THRE = data['turnon']
        self.TURN_OFF_THRE = data['turnoff']
        self.time_visible = 0.0
        self.filename = data['filename']

class GestureData():
    def __init__(self, name, data):
        data = ParseYAML.parseDynamicGesture(data)

        self.NAME = name
        if data['var_len'] > 1:
            self.prob = [0.0] * data['var_len']
            self.toggle = [False] * data['var_len']
        else:
            self.prob = 0.0
            self.toggle = False
        self.time_visible = 0.0
        self.in_progress = False
        self.direction = [0.0,0.0,0.0]
        self.speed = 0.0
        self.filename = data['filename']

        # for circle movement
        self.clockwise = False
        self.angle = 0.0
        self.progress = 0.0
        self.radius = 0.0
        # for move_in_axis thresholds
        self.MIN_THRE = data['minthre']
        self.MAX_THRE = data['maxthre']
        ## move in x,y,z, Positive/Negative
        self.move = [False, False, False]

class MoveData():
    def __init__(self):
        with open(CUSTOM_SETTINGS_YAML+"robot_move.yaml", 'r') as stream:
            robot_move_data_loaded = ordered_load(stream, yaml.SafeLoader)
            #robot_move_data_loaded = yaml.safe_load(stream)

        self.LEAP_AXES = robot_move_data_loaded['LEAP_AXES']
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
        if IK_SOLVER == 'pyrep':
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

        self.Mode = 'live' # 'play'/'live'/'alternative'
        self.SCALE = 2 # scaling factor for Leap
        ## interactive
        self.ACTION = False
        self.STRICT_MODE = False
        # if play path
        self.PickedPath = 0
        self.attached = False
        self.liveMode = 'default'

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



# TODO: Make NAME, ENV, Objects lowercase
class CustomPath():
    def __init__(self, data=None, poses_data=None):
        ''' Create your custom path
        '''
        assert data, "No Data!"
        assert len(data.keys()) == 1, "More input Paths!"
        key = list(data.keys())[0]
        path_data = data[key]
        self.NAME = key
        self.poses = []
        self.actions = []
        if not path_data:
            return

        poses = path_data['poses']
        self.n = len(path_data['poses'])
        self.scene = ParseYAML.parseScene(path_data)
        self.ENV = path_data['env']
        for pose_e in poses:
            pose = pose_e['pose']
            self.poses.append(ParseYAML.parsePose(pose, poses_data))

            self.actions.append(ParseYAML.parseAction(pose_e))





    @staticmethod
    def GenerateFromYAML(paths_folder=None, paths_file_catch_phrase='paths', poses_file_catch_phrase='poses'):
        ''' Generates All paths from YAML files

        Parameters:
            paths_folder (Str): folder to specify YAML files, if not specified the default CUSTOM_SETTINGS_YAML folder is used
            paths_file_catch_phrase (Str): Searches for files with this substring (e.g. use 'paths' to load all names -> 'paths1.yaml', paths2.yaml)
                - If not specified then paths_file_catch_phrase='paths'
                - If specified as '', all files are loaded
            poses_file_catch_phrase (Str): Loads poses from YAML file with this substring
        Returns:
            sp (CustomPath()[]): The generated array of paths
        '''
        sp = []
        if not paths_folder:
            paths_folder = CUSTOM_SETTINGS_YAML

        files = os.listdir(paths_folder)

        poses_data_loaded = {}
        for f in files:
            if '.yaml' in f and poses_file_catch_phrase in f:
                with open(paths_folder+f, 'r') as stream:
                    poses_data_loaded = merge_two_dicts(poses_data_loaded, ordered_load(stream, yaml.SafeLoader))

        for f in files:
            if '.yaml' in f and paths_file_catch_phrase in f:
                with open(paths_folder+f, 'r') as stream:
                    data_loaded = ordered_load(stream, yaml.SafeLoader)
                    for key in data_loaded.keys():
                        pickedpath = {key: data_loaded[key]}
                        sp.append(CustomPath(pickedpath, poses_data_loaded))
        return sp



class CustomScene():
    ''' Custom scenes with custom names is defined with custom objects with
        pose and size

    '''
    def __init__(self, data=None, poses_data=None):
        assert data, "No Data!"
        assert len(data.keys()) == 1, "More input Scenes!"

        key = list(data.keys())[0]
        scene_data = data[key]
        self.NAME = key
        self.object_names = []
        self.object_poses = []
        self.object_sizes = []
        self.object_scales = []
        self.object_colors = []
        self.object_shapes = []
        self.object_masses = []
        self.object_frictions = []
        self.object_inertia = []
        self.object_inertiaTransform = []
        self.mesh_trans_origin = []
        self.object_dynamic = []
        self.object_pub_info = []
        self.object_texture_file = []
        self.object_file = []
        if not scene_data:
            return

        objects = scene_data['Objects']
        self.object_names = objects.keys()
        self.mesh_trans_origin = [Vector3(0.,0.,0.)] * len(objects.keys())
        for object in self.object_names:
            pose_vec = objects[object]['pose']

            self.object_poses.append(ParseYAML.parsePose(pose_vec, poses_data))
            self.object_sizes.append(ParseYAML.parsePosition(objects[object], poses_data, key='size'))
            self.object_scales.append(ParseYAML.parseScale(objects[object]))
            self.object_colors.append(ParseYAML.parseColor(objects[object]))
            self.object_shapes.append(ParseYAML.parseShape(objects[object]))
            self.object_frictions.append(ParseYAML.parseFriction(objects[object]))
            self.object_masses.append(ParseYAML.parseMass(objects[object]))
            self.object_inertia.append(ParseYAML.parseInertia(objects[object]))
            self.object_inertiaTransform.append(ParseYAML.parseInertiaTransform(objects[object]))
            self.object_dynamic.append(ParseYAML.parseDynamic(objects[object]))
            self.object_pub_info.append(ParseYAML.parsePubInfo(objects[object]))
            self.object_texture_file.append(ParseYAML.parseTextureFile(objects[object]))
            self.object_file.append(ParseYAML.parseMeshFile(objects[object]))
            if 'mesh_trans_origin' in scene_data.keys():
                if 'axes' in scene_data.keys():
                    self.mesh_trans_origin = TransformWithAxes(scene_data['mesh_trans_origin'], scene_data['axes'])
                else:
                    self.mesh_trans_origin = scene_data['mesh_trans_origin']


    @staticmethod
    def GenerateFromYAML(scenes_folder=None, scenes_file_catch_phrase='scene', poses_file_catch_phrase='poses'):
        ''' Generates All scenes from YAML files

        Parameters:
            scenes_folder (Str): folder to specify YAML files, if not specified the default CUSTOM_SETTINGS_YAML folder is used
            scenes_file_catch_phrase (Str): Searches for files with this substring (e.g. use 'scene' to load all names -> 'scene1.yaml', scene2.yaml)
                - If not specified then scenes_file_catch_phrase='scene'
                - If specified as '', all files are loaded
            poses_file_catch_phrase (Str): Loads poses from YAML file with this substring
        '''
        ss = []
        if not scenes_folder:
            scenes_folder = CUSTOM_SETTINGS_YAML

        files = os.listdir(scenes_folder)

        poses_data_loaded = {}
        for f in files:
            if '.yaml' in f and poses_file_catch_phrase in f:
                with open(scenes_folder+f, 'r') as stream:
                    poses_data_loaded = merge_two_dicts(poses_data_loaded, ordered_load(stream, yaml.SafeLoader))

        for f in files:
            if '.yaml' in f and scenes_file_catch_phrase in f:
                with open(scenes_folder+f, 'r') as stream:
                    data_loaded = ordered_load(stream, yaml.SafeLoader)
                    for key in data_loaded.keys():
                        pickedscene = {key: data_loaded[key]}
                        ss.append(CustomScene(pickedscene, poses_data_loaded))
        return ss


class ParseYAML():
    @staticmethod
    def parseScene(data):
        '''
        '''
        if 'scene' in data.keys():
            assert data['scene'] in getSceneNames(), "[Settings] Path you want to create has not valid scene name"
            return data['scene']
        else:
            return 'empty'

    @staticmethod
    def parseMeshFile(object):
        '''
        '''
        keys = ['file', 'mesh_file', 'init_file', 'mesh', 'meshFile', 'initFile']
        if any(x in object.keys() for x in keys):
            key = None
            for i in keys:
                if i in object.keys():
                    key = i

            if object[key] in ['', 'none']:
                return ''
            elif not any(x in object[key] for x in ['.obj','.dae']):
                raise Exception("Mesh file not in right format, check YAML file!")
            else:
                return object[key]
        else:
            return ''

    @staticmethod
    def parseTextureFile(object):
        '''
        '''
        keys = ['texture_file', 'texture', 'Texture', 'textureFile']
        if any(x in object.keys() for x in keys):
            key = None
            for i in keys:
                if i in object.keys():
                    key = i
            if object[key] in ['wood']:
                return 'wood.jpg'
            ## More defualt textures goes here!
            elif object[key] in ['', 'none']:
                return ''
            elif not any(x in object[key] for x in ['.jpeg','.jpg','.png','.tga','.bmp','.tiff','.gif']):
                raise Exception("Texture file not in right format, check YAML file!")
            else:
                return object[key]
        else:
            return ''

    @staticmethod
    def parseInertiaTransform(object):
        '''
        '''
        keys = ['inertiaTransform', 'inertia_transform', 'InertiaTransform', 'InertiaTransformation','inertiaTransformation']
        if any(x in object.keys() for x in keys):
            key = None
            for i in keys:
                if i in object.keys():
                    key = i

            if isinstance(object[key], (list, tuple, np.ndarray)):
                if len(object[key]) == 12:
                    return object[key]
                else: raise Exception("Inertia from YAML file is not the right length")
            elif object[key] in ['', 'none']:
                return np.zeros(12)
            else: raise Exception("Inertia not in list, check YAML file!")
        else:
            return np.zeros(12)

    @staticmethod
    def parseInertia(object):
        '''
        '''
        keys = ['inertia', 'init_inertia', 'Inertia']
        if any(x in object.keys() for x in keys):
            key = None
            for i in keys:
                if i in object.keys():
                    key = i

            if isinstance(object[key], (list, tuple, np.ndarray)):
                if len(object[key]) == 9:
                    return object[key]
                else: raise Exception("Inertia from YAML file is not the right length")
            elif object[key] in ['', 'none']:
                return np.zeros(9)
            else: raise Exception("Inertia not in list, check YAML file!")
        else:
            return np.zeros(9)


    @staticmethod
    def parseCollision(object):
        '''
        '''
        keys = ['collision', 'init_collision', 'Collision']
        if any(x in object.keys() for x in keys):
            key = None
            for i in keys:
                if i in object.keys():
                    key = i

            if object[key] in ['true', 'yes', 'True', 'Yes', 'y', 1]:
                return 'true'
            elif object[key] in ['false', 'no', 'False', 'No', 'n', 0]:
                return 'false'
            elif object[key] in ['', 'none']:
                return ''
            else: raise Exception("Shape not in list, check YAML file!")
        else:
            return ''

    @staticmethod
    def parsePubInfo(object):
        '''
        '''
        keys = ['pub_info', 'pubInfo', 'info', 'publish_info', 'publishInfo']
        if any(x in object.keys() for x in keys):
            key = None
            for i in keys:
                if i in object.keys():
                    key = i

            if object[key] in ['true', 'yes', 'True', 'Yes', 'y', 1]:
                return 'true'
            elif object[key] in ['false', 'no', 'False', 'No', 'n', 0]:
                return 'false'
            elif object[key] in ['', 'none']:
                return ''
            else: raise Exception("pub_info not in list, check YAML file!")
        else:
            return ''


    @staticmethod
    def parseDynamic(object):
        '''
        '''
        keys = ['dynamic', 'init_dynamic', 'Dynamic', 'dynamics', 'Dynamics']
        if any(x in object.keys() for x in keys):
            key = None
            for i in keys:
                if i in object.keys():
                    key = i

            if object[key] in ['true', 'yes', 'True', 'Yes', 'y', 1]:
                return 'true'
            elif object[key] in ['false', 'no', 'False', 'No', 'n', 0]:
                return 'false'
            elif object[key] in ['', 'none']:
                return ''
            else: raise Exception("Dynamic not in list, check YAML file!")
        else:
            return ''

    @staticmethod
    def parseFriction(object):
        keys = ['friction', 'init_friction', 'Friction']
        if any(x in object.keys() for x in keys):
            key = None
            for i in keys:
                if i in object.keys():
                    key = i

            if isinstance(object[key], (int, float)):
                return object[key]
            elif object[key] in ['no', 'none']:
                return 0
            elif object[key] in ['']:
                return -1
            else: raise Exception("Friction not in list, check YAML file!")
        else:
            return -1

    @staticmethod
    def parseMass(object):
        keys = ['mass', 'weight', 'init_mass', 'init_weight']
        if any(x in object.keys() for x in keys):
            key = None
            for i in keys:
                if i in object.keys():
                    key = i

            if isinstance(object[key], (int, float)):
                return object[key]
            elif object[key] in ['no', 'none']:
                return 0
            elif object[key] in ['']:
                return -1
            else: raise Exception("Mass not in list, check YAML file!")
        else:
            return -1

    @staticmethod
    def parseShape(object):
        ''' Parse shape
        '''
        if 'shape' in object.keys():
            if object['shape'] in ['cube', 'sphere', 'cylinder', 'cone']:
                return object['shape']
            elif object['shape'] in ['cuboid', 'box', 'rectangle']:
                return 'cube'
            elif object['shape'] in ['ball', 'round', 'globe', 'spheroid']:
                return 'sphere'
            elif object['shape'] in ['', 'none', 'no']:
                return ''
            else: raise Exception("Shape not in list, check YAML file!")
        else:
            return ""

    @staticmethod
    def parseAction(pose):
        ''' Parses action from pose
        Parameters:
            pose (Dict): Item from YAML: path['poses'][i], why not link arg. "action" itself? A: It may does not exist.
        '''
        if 'action' in pose.keys():
            return pose['action']
        else:
            return ""

    @staticmethod
    def parseColor(object):
        ''' Parse color from YAML file
        '''
        if 'color' in object.keys():
            if object['color'] in ['r','g','b','c','m','y','k']:
                return object['color']
            elif object['color'] == 'red':
                return 'r'
            elif object['color'] == 'green':
                return 'g'
            elif object['color'] == 'blue':
                return 'b'
            elif object['color'] == 'cyan':
                return 'c'
            elif object['color'] == 'magenta':
                return 'm'
            elif object['color'] == 'yellow':
                return 'y'
            elif object['color'] in ['key','black']:
                return 'k'
            elif object['color'] in ['', 'no', 'none']:
                return ''
            else: raise Exception("Color not in list, check YAML file!")
        else:
            return ''

    @staticmethod
    def parseScale(object):
        ''' Parse scale from YAML file
        '''
        if 'scale' in object.keys():
            return object['scale']
        else:
            return 0

    @staticmethod
    def parsePose(pose_vec, poses_data):
        ''' Parse pose from YAML file
            1. pose is string: saves from poses.yaml file
        '''
        rosPose = Pose()
        if type(pose_vec) == str:
            if pose_vec not in poses_data.keys():
                raise Exception("Saved pose not found! Pose: "+str(pose_vec))
            rosPose.position = Point(*extv(poses_data[pose_vec]['pose']['position']))
            rosPose.orientation = Quaternion(*extq(poses_data[pose_vec]['pose']['orientation']))
        else:
            rosPose.position = ParseYAML.parsePosition(pose_vec, poses_data)
            rosPose.orientation = ParseYAML.parseOrientation(pose_vec, poses_data)
        return rosPose


    @staticmethod
    def parsePosition(pose_vec, poses_data, key='position'):
        '''
            1. position is {'x':0,'y':0,'z':0} or [0,0,0]
            2. position is {'x':[0,1], 'y':[0,1], 'z':[0,1]}
            3. position is saved pose name
        Parameters:
            key (Str): Extract the dict with key
        '''
        # Check if exists:
        if key in pose_vec.keys():
            position = pose_vec[key]
        else:
            print("[WARN*] Position not specified in YAML, Maybe TODO: to defaulf value of env")
            position = [0.5,0.0,0.5]
        # 1.
        if isarray(position) and isnumber(position[0]):
            return Point(*list(position))
        # 2.
        elif isarray(position) and isarray(position[0]):
            assert len(position[0]) < 3 and len(position[1]) < 3 and len(position[2]) < 3, "Reading from YAML read more ranges"
            for i in range(0,3):
                if isnumber(position[i]):
                    pass
                if isarray(position[i]):
                    assert len(position[i]) == 2, "Range length not right, check YAML"
                    assert position[i][0] < position[i][1], "Min. is greater than Max. in YAML read"
                    position[i] = random.uniform(position[i][0], position[i][1])
            return Point(*list(position))

        elif isinstance(position, dict):
            for i in ['x', 'y', 'z']:
                if isnumber(position[i]):
                    pass
                elif isarray(position[i]):
                    assert len(position[i]) == 2, "Range length not right, check YAML"
                    assert position[i][0] < position[i][1], "Min. is greater than Max. in YAML read"
                    position[i] = random.uniform(position[i][0], position[i][1])

            return Point(*extv(position))
        # 3.
        elif type(position) == str:
            if position in poses_data.keys():
                position = poses_data[position]['pose']['position']
                return Point(*extv(position))
            else: raise Exception("Saved pose "+str(position)+" not found!")
        else: raise Exception("Wrong option reading from YAML: "+str(position))

    @staticmethod
    def parseOrientation(pose_vec, poses_data):
        '''
            1. orientation is {'x':0,'y':0,'z':0,'w':0} or [0,0,0,0] (use x,y,z,w notation)
            2. orientation is {'x':[0,1], 'y':[0,1], 'z':[0,1]}
            3. orientation is saved pose name
        '''
        # Check if exists:
        if 'orientation' in pose_vec.keys():
            orientation = pose_vec['orientation']
        else:
            print("[WARN*] Orientation not specified in YAML, Maybe TODO: to defaulf value of env")
            orientation = [0.,0.,0.,1.]
        # 1.
        if isarray(orientation) and isnumber(orientation[0]):
            return Quaternion(*list(orientation))
        # 2.
        elif isarray(orientation) and isarray(orientation[0]):
            for i in range(0,4):
                if isnumber(orientation[i]):
                    pass
                if isarray(orientation[i]):
                    assert len(position[i]) == 2, "Range length not right, check YAML"
                    assert orientation[i][0] < orientation[i][1], "Min. is greater than Max. in YAML read"
                    orientation[i] = random.uniform(orientation[i][0], orientation[i][1])
            return Quaternion(*list(orientation))
        elif isinstance(orientation, dict):
            for i in ['x', 'y', 'z', 'w']:
                if isnumber(orientation[i]):
                    pass
                if isarray(orientation[i]):
                    assert len(orientation[i]) == 2, "Range length not right, check YAML"
                    assert orientation[i][0] < orientation[i][1], "Min. is greater than Max. in YAML read"
                    orientation[i] = random.uniform(orientation[i][0], orientation[i][1])
            return Quaternion(*extq(orientation))
        # 3.
        elif type(orientation) == str:
            if orientation in poses_data.keys():
                orientation = poses_data[orientation]['pose']['orientation']
                return Quaternion(*extq(orientation))
            else: raise Exception("Saved pose not found! Pose: "+str(orientation))
        else: raise Exception("Wrong option reading from YAML: "+str(orientation))

    @staticmethod
    def parseStaticGesture(gesture):
        if 'turnon' not in gesture.keys():
            gesture['turnon']=0.0
        if 'turnoff' not in gesture.keys():
            gesture['turnoff']=0.0
        if 'filename' not in gesture.keys():
            gesture['filename']=""
        return gesture

    @staticmethod
    def parseDynamicGesture(gesture):
        if 'var_len' not in gesture.keys():
            gesture['var_len']=1
        if 'minthre' not in gesture.keys():
            gesture['minthre']=0.9
        if 'maxthre' not in gesture.keys():
            gesture['maxthre']=0.4
        if 'filename' not in gesture.keys():
            gesture['filename']=""
        return gesture

## Helper functions
def indx(ss, str):
    ''' Returns index of scene specified by name as parameter 'str'
    '''
    N = None
    for n,i in enumerate(ss):
        if i.NAME == str:
            N = n
    assert isinstance(N, int), "Scene not found"
    return N

def getSceneNames():
    return [scene.NAME for scene in ss]

def getPathNames():
    return [path.NAME for path in sp]

def getModes():
    return ['live', 'interactive', '']

# Handy functions
def extq(q):
    ''' Extracts Quaternion object
    Parameters:
        q (Quaternion()): From geometry_msgs.msg
    Returns:
        x,y,z,w (Floats tuple[4]): Quaternion extracted
    '''
    if type(q) == type(Quaternion()):
        return q.x, q.y, q.z, q.w
    elif (isinstance(q, dict) and 'w' in q.keys()):
        return q['x'], q['y'], q['z'], q['w']
    else: raise Exception("extq input arg q: Not Quaternion or dict with 'x'..'w' keys!")


def extv(v):
    ''' Extracts Point/Vector3 to Cartesian values
    Parameters:
        v (Point() or Vector3() or dict with 'x'..'z' in keys): From geometry_msgs.msg or dict
    Returns:
        [x,y,z] (Floats tuple[3]): Point/Vector3 extracted
    '''
    if type(v) == type(Point()) or type(v) == type(Vector3()):
        return v.x, v.y, v.z
    elif (isinstance(v, dict) and 'x' in v.keys()):
        return v['x'], v['y'], v['z']
    else: raise Exception("extv input arg v: Not Point or Vector3 or dict!")


def extp(p):
    ''' Extracts pose
    Paramters:
        p (Pose())
    Returns:
        list (Float[7])
    '''
    assert type(p) == type(Pose()), "extp input arg p: Not Pose type!"
    return p.position.x, p.position.y, p.position.z, p.orientation.x, p.orientation.y, p.orientation.z, p.orientation.w

def isnumber(n):
    if type(n) == int or type(n) == float:
        return True
    return False

def isarray(a):
    if type(a) == tuple or type(a) == list:
        return True
    return False

def merge_two_dicts(x, y):
    z = x.copy()   # start with keys and values of x
    z.update(y)    # modifies z with keys and values of y
    return z

def TransformWithAxes(data_to_transform, transform_mat):
    '''
    Parameters:
        data_to_transform (Vector3()[]) or dict
        transform_mat (2D array): size 3x3
    Returns:
        data_transformed (Vector3()[])
    '''

    if isinstance(data_to_transform[0], dict) and 'x' in data_to_transform[0].keys():
        new_data = []
        for vec in data_to_transform:
            new_data.append(Vector3(vec['x'], vec['y'], vec['z']))
        data_to_transform = new_data

    data_transformed = []
    for i in range(0, len(data_to_transform)):
        orig = data_to_transform[i]
        orig_ = Vector3()
        orig_.x = np.dot(transform_mat[0],[orig.x, orig.y, orig.z])
        orig_.y = np.dot(transform_mat[1],[orig.x, orig.y, orig.z])
        orig_.z = np.dot(transform_mat[2],[orig.x, orig.y, orig.z])
        data_transformed.append(Vector3(orig_.x, orig_.y, orig_.z))
    return data_transformed

#TransformWithAxes(data_to_transform=[Vector3(0.,0.,0.), Vector3(0.024574, 0., 0.032766+0.094613+0.015387+0.124468+0.016383), Vector3(0.024574, 0., 0.032766+0.094613+0.015387), Vector3(0.024574, 0., 0.032766)], transform_mat=[[0.,1.,0.],[-1.,0.,0.],[0.,0.,1.]])

'''
def waitForValue(ooo, errormsg="Wait for value error!"):
    def valueLoaded(ooo.value):
        if type(ooo.value) == type(False) or type(ooo.value) == type(None):
            return False
        return True

    if valueLoaded(ooo.value):
        return
    time.sleep(0.4)
    if valueLoaded(ooo.value):
        return
    time.sleep(0.4)
    if valueLoaded(ooo.value):
        return
    time.sleep(0.8)
    if valueLoaded(ooo.value):
        return
    time.sleep(2)
    if valueLoaded(ooo.value):
        return
    time.sleep(2)
    print(errormsg)
    if valueLoaded(ooo.value):
        return
    time.sleep(2)
    print(errormsg)
    if valueLoaded(ooo.value):
        return
    time.sleep(4)
    print(errormsg)
    while True:
        if valueLoaded(ooo.value):
            return True
        time.sleep(4)
        print(errormsg)
'''
