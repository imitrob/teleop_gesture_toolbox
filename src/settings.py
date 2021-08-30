#!/usr/bin/env python
'''
Informations about robot, configurations, move data, ...
Import this file and call init() to access parameters, info and data
'''
import collections
import numpy as np
from copy import deepcopy
import os
from os.path import expanduser, isfile
import time
# Needed to load and save rosparams
import rospy
from geometry_msgs.msg import Quaternion, Pose, PoseStamped, Point, Vector3
from visualization_msgs.msg import MarkerArray, Marker
from std_msgs.msg import Int8, Float64MultiArray

def init(minimal=False):
    ''' Initialize the shared data across threads (Leap, UI, Control)
    Parameters:
        minimal (Bool): Import only file paths and general informations
    '''
    # tmp
    global loopn
    loopn = 0

    ## Files
    global HOME, LEARN_PATH, GRAPHICS_PATH, GESTURE_NAMES, NETWORK_PATH, PLOTS_PATH, WS_FOLDER, COPPELIA_SCENE_PATH, MODELS_PATH
    HOME = expanduser("~")
    # searches for the WS name + print it
    THIS_FILE_PATH = os.path.dirname(os.path.realpath(__file__))
    THIS_FILE_TMP = os.path.abspath(os.path.join(THIS_FILE_PATH, '..', '..', '..'))
    WS_FOLDER = THIS_FILE_TMP.split('/')[-1]

    LEARN_PATH = HOME+"/"+WS_FOLDER+"/src/mirracle_gestures/include/data/learning/"
    GRAPHICS_PATH = HOME+"/"+WS_FOLDER+"/src/mirracle_gestures/include/graphics/"
    PLOTS_PATH = HOME+"/"+WS_FOLDER+"/src/mirracle_gestures/include/plots/"
    NETWORK_PATH = HOME+"/"+WS_FOLDER+"/src/mirracle_gestures/include/data/learned_networks/"
    COPPELIA_SCENE_PATH = HOME+"/"+WS_FOLDER+"/src/mirracle_gestures/include/coppelia_scenes/"
    MODELS_PATH = HOME+"/"+WS_FOLDER+"/src/mirracle_gestures/include/models/"
    GESTURE_NAMES = ['Grab', 'Pinch', 'Point', 'Respectful', 'Spock', 'Rock', 'Victory', 'Italian', 'Rotate', 'Swipe_Up', 'Pin', 'Touch', 'Swipe_Left', 'Swipe_Down', 'Swipe_Right']

    ## robot
    global JOINT_NAMES, BASE_LINK, GROUP_NAME, ROBOT_NAME, GRIPPER_NAME, SIMULATOR_NAME, TAC_TOPIC, JOINT_STATES_TOPIC, EEF_NAME, TOPPRA_ON, VIS_ON, IK_SOLVER, GRASPING_GROUP
    global upper_lim, lower_lim, effort_lim, vel_lim
    # ROBOT_NAME: - 'panda' or 'iiwa'
    ROBOT_NAME = rospy.get_param("/mirracle_config/robot")
    SIMULATOR_NAME = rospy.get_param("/mirracle_config/simulator")
    GRIPPER_NAME = rospy.get_param("/mirracle_config/gripper")
    VIS_ON = rospy.get_param("/mirracle_config/visualize")
    IK_SOLVER = rospy.get_param("/mirracle_config/ik_solver")
    TOPPRA_ON = True


    ## Robot data
    # 1. velocity limits, effort limits, position limits (lower, upper)
    # 2. Important data for MoveIt!
    # 3. TAC - Trajectory Action Client topic and Joint States topic
    if ROBOT_NAME == 'iiwa':
        vel_lim = [ 1.71, 1.71, 1.75, 2.27, 2.44, 3.14, 3.14 ]
        effort_lim = [ 140, 140, 120, 100, 70, 70, 70 ]
        lower_lim = [-2.97, -2.09, -2.97, -2.09, -2.97, -2.09, -3.05]
        upper_lim = [2.97, 2.09, 2.97, 2.09, 2.97, 2.09, 3.05]
        JOINT_NAMES = ['r1_joint_1', 'r1_joint_2', 'r1_joint_3', 'r1_joint_4', 'r1_joint_5', 'r1_joint_6', 'r1_joint_7']
        BASE_LINK = 'base_link'
        GROUP_NAME = "r1_arm"
        EEF_NAME = 'r1_ee'
        GRASPING_GROUP = 'r1_gripper'
        TAC_TOPIC = "/r1/trajectory_controller/follow_joint_trajectory"
        JOINT_STATES_TOPIC = '/r1/joint_states'
    elif ROBOT_NAME == 'panda':
        vel_lim = [2.1750, 2.1750, 2.1750, 2.1750, 2.6100, 2.6100, 2.6100]
        effort_lim = [ 87, 87, 87, 87, 12, 12, 12 ]
        lower_lim = [-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973]
        upper_lim = [2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973]
        JOINT_NAMES = ['panda_joint1', 'panda_joint2', 'panda_joint3', 'panda_joint4', 'panda_joint5', 'panda_joint6', 'panda_joint7']
        BASE_LINK = 'panda_link0'
        GROUP_NAME = "panda_arm"
        EEF_NAME = 'panda_link8'
        GRASPING_GROUP = 'hand'
        if SIMULATOR_NAME == 'gazebo' or SIMULATOR_NAME == 'real':
            TAC_TOPIC = '/position_joint_trajectory_controller/follow_joint_trajectory'
            JOINT_STATES_TOPIC = '/franka_state_controller/joint_states'
        elif SIMULATOR_NAME == 'rviz':
            TAC_TOPIC = '/execute_trajectory'
            JOINT_STATES_TOPIC = '/joint_states'
        elif SIMULATOR_NAME == 'coppelia':
            TAC_TOPIC = '/fakeFCI/joint_state'
            JOINT_STATES_TOPIC = '/vrep/franka/joint_state'
        else: raise Exception("Wrong simualator name")
    else: raise Exception("Wrong robot name")

    if minimal:
        return
    print("[Settings] Workspace folder is set to: "+WS_FOLDER)

    # Data from Leap Controller saved in arrays
    # Note: updated in leapmotionlistener.py with Leap frame frequency (~80Hz)
    global BUFFER_LEN, frames, timestamps, frames_adv, eef_goal, eef_robot, joints_in_time
    BUFFER_LEN = 300
    frames = collections.deque(maxlen=BUFFER_LEN)
    timestamps = collections.deque(maxlen=BUFFER_LEN)
    frames_adv = collections.deque(maxlen=BUFFER_LEN)
    # Note: updated in main.py with rate 10Hz
    eef_goal = collections.deque(maxlen=BUFFER_LEN)
    eef_robot = collections.deque(maxlen=BUFFER_LEN)

    joints_in_time = collections.deque(maxlen=BUFFER_LEN)

    ## Fixed Conditions
    global FIXED_ORI_TOGGLE, print_path_trace
    # When turned on, eef has fixed eef orientaion based on chosen environment (md.ENV)
    FIXED_ORI_TOGGLE = True
    # When turned on, rViz marker array of executed trajectory is published
    print_path_trace = False

    ## Current/Active robot data at the moment
    global goal_joints, goal_pose, eef_pose, joints, velocity, effort
    goal_joints, goal_pose, joints, velocity, effort, eef_pose = None, None, None, None, None, Pose()
    # Goal joints -> RelaxedIK output
    # Goal pose -> RelaxedIK input
    # joints -> JointStates topic
    # eef_pose -> from joint_states

    global gd, md, rd, sp, mo, ss, scene
    # Active gesture data at the moment
    gd = GestureDataHands()
    # Move robot and control data
    md = MoveData()

    ## Objects for saved scenes and paths
    sp, ss = [], []
    GenerateSomeScenes(ss) # Saved scenes
    GenerateSomePaths(sp, ss) # Saved paths
    scene = None # current scene informations
    mo = None # MoveIt object

    ## Learning settings
    global pymcin, pymcout, observation_type, time_series_operation, position
    pymcout = None
    pymcin = Float64MultiArray()
    '''
    param:
        :observation_type == 'user_defined' -> includes finger on/off normalized, fingers distance normalized
                           == 'all_defined' -> includes all differences between fingers and distances
        :time_series_operation == 'average' -> value of observation is averaged of all time
                               == 'middle' -> take middle time sample
                               == 'as_dimesion' -> three dimensional X as output
                               == 'take_every_10' -> every tenth time sample is new recorded sample
                               == 'take_every' -> every time sample is new recorded sample

        :position == '' -> no position
                  == 'absolute' -> position of palm is intercorporated
                  == 'absolute+finger' -> position of palm and pointing finger is intercorporated
    '''
    observation_type='user_defined'
    time_series_operation = 'take_every_10'
    position = ''
    assert observation_type in ['user_defined', 'all_defined'], "Wrong input"
    assert time_series_operation in ['average', 'middle', 'as_dimesion', 'take_every', 'take_every_10'], "Wrong input"
    assert position in ['', 'absolute', 'absolute+finger', 'time_warp'], "Wrong input"

    ## For visualization data holder
    global viz; viz = None # VisualizerLib obj
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

    ## User Interface Data ##
    # Configuration page values
    global NumConfigBars, VariableValues
    NumConfigBars = [2,4]
    VariableValues = np.zeros(NumConfigBars)

    # Status Bar
    global HoldAnchor, HoldPrevState, HoldValue, currentPose, WindowState, leavingAction, w, h, ui_scale
    WindowState = 0.0  # 0-Main page, 1-Config page
    HoldAnchor = 0.0  # For moving status bar
    HoldPrevState = False  # --||--
    HoldValue = 0
    currentPose = 0
    leavingAction = False
    w, h = 1000, 800
    ui_scale = 2000

    ## Global ROS publisher objects
    global pymc_in_pub; pymc_in_pub = None
    global ee_pose_goals_pub; ee_pose_goals_pub = None
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
        self.conf = False
        self.MIN_CONFIDENCE = 0.3
        self.SINCE_FRAME_TIME = 0.5

        self.tch12, self.tch23, self.tch34, self.tch45 = [False] * 4
        self.tch13, self.tch14, self.tch15 = [False] * 3
        self.TCH_TURN_ON_DIST = [0.9, 0.9, 0.9, 0.9,  0.9, 0.9, 0.9]
        self.TCH_TURN_OFF_DIST = [0.5, 0.5, 0.5, 0.5,  0.5, 0.5, 0.5]

        self.oc = [False] * 5
        self.OC_TURN_ON_THRE =  [0.94, 0.8, 0.8, 0.8, 0.8]
        self.OC_TURN_OFF_THRE = [0.85, 0.6, 0.6, 0.6, 0.6]

        self.poses = []
        self.gests = []

        #was: settings.gd.l.pose1_grab_prob -> settings.gd.l.poses[i].prob

        self.poses.append(PoseData(NAME="grab", filename="gesture56.png", turnon=0.9, turnoff=0.3))
        self.poses.append(PoseData(NAME="pinch", filename="gesture33.png", turnon=0.9, turnoff=0.3))
        self.poses.append(PoseData(NAME="pointing", filename="gesture0.png", turnon=0.9, turnoff=0.3))
        self.poses.append(PoseData(NAME="respectful", filename="gesture100.png", turnon=0.9, turnoff=0.3))
        self.poses.append(PoseData(NAME="spock", filename="gesture101.png", turnon=0.9, turnoff=0.3))
        self.poses.append(PoseData(NAME="rock", filename="gesture102.png", turnon=0.9, turnoff=0.3))
        self.poses.append(PoseData(NAME="victory", filename="gesture103.png", turnon=0.9, turnoff=0.3))
        self.poses.append(PoseData(NAME="italian", filename="gesture104.png", turnon=0.9, turnoff=0.3))
        self.POSES = {}
        for n, i in enumerate(self.poses):
            self.POSES[i.NAME] = n

        self.gests.append(GestureData(NAME="circ", filename="gesture34.png"))
        self.gests.append(GestureData(NAME="swipe", filename="gesture64.png"))
        self.gests.append(GestureData(NAME="pin", filename="gesture3.png"))
        self.gests.append(GestureData(NAME="touch", filename="gesture1.png"))
        self.gests.append(GestureData(NAME="move_in_axis", var_len=3, minthre=0.4, maxthre=0.1))
        self.gests.append(GestureData(NAME="rotation_in_axis", var_len=3, minthre=[-0.9,-0.6,0.8], maxthre=[0.5,0.5,0.4]))
        self.GESTS = {}
        for n, i in enumerate(self.gests):
            self.GESTS[i.NAME] = n

        self.final_chosen_pose = 0
        self.final_chosen_gesture = 0

class PoseData():
    def __init__(self, NAME="", turnon=0.0, turnoff=0.0, filename=""):
        self.NAME = NAME
        self.prob = 0.0
        self.toggle = False
        self.TURN_ON_THRE = turnon
        self.TURN_OFF_THRE = turnoff
        self.time_visible = 0.0
        self.filename = filename

class GestureData():
    def __init__(self, NAME="", var_len=1, minthre=0.9, maxthre=0.4, filename=""):
        self.NAME = NAME
        if var_len > 1:
            self.prob = [0.0] * var_len
            self.toggle = [False] * var_len
        else:
            self.prob = 0.0
            self.toggle = False
        self.time_visible = 0.0
        self.in_progress = False
        self.direction = [0.0,0.0,0.0]
        self.speed = 0.0
        self.filename = filename

        # for circle movement
        self.clockwise = False
        self.angle = 0.0
        self.progress = 0.0
        self.radius = 0.0
        # for move_in_axis thresholds
        self.MIN_THRE = minthre
        self.MAX_THRE = maxthre
        ## move in x,y,z, Positive/Negative
        self.move = [False, False, False]

class MoveData():
    '''
    Info about:
    Q(0,0,0,1) -> down
    Q([0.5,0.5,0.5,-0.5]) -> wall
    Q(0.0,1.0,0.0,0.0) -> up

    Panda rviz:
    Q(np.sqrt(2)/2, np.sqrt(2)/2., 0.0, 0.0) -> above
    Q(0,0,0,1) -> down
    Q(0.5,-0.5,-0.5,0.5) -> wall
    '''
    def __init__(self):
        self.LEAP_AXES = [[1,0,0],[0,0,-1],[0,1,0]]

        if IK_SOLVER == 'relaxed_ik':
            self.ENV_DAT = {'above': {  'min': Point(-0.35, -0.35, 0.6),
                                        'max': Point(0.35,0.35,1.15),
                                        'ori': Quaternion(np.sqrt(2)/2, np.sqrt(2)/2., 0.0, 0.0),
                                        'start': Point(0.0, 0.0, 0.85),
                                        'axes': np.eye(3), 'view': [[1,0,0],[0,0,1],[0,1,0]], 'view2': [[1,0,0],[0,0,1],[0,1,0]],"UIx": 'X', "UIy": 'Z',
                                        'ori_type': 'neg_normal', 'ori_axes': [[0,1,0],[-1,0,0],[0,0,0]], 'ori_live': [[0,-1,0],[1,0,0],[0,0,-1]]
                                     },
                             'wall':  { 'min': Point(0.42, -0.2, 0.0),
                                        'max': Point(0.7, 0.2, 0.74),
                                        'ori': Quaternion(0.5,-0.5,-0.5,0.5),
                                        'start': Point(0.7, 0.0, 0.1),
                                        'axes': [[0,1,0],[-1,0,0],[0,0,1]], 'view': [[0,-1,0],[0,0,1],[1,0,0]], 'view2': [[0,-1,0],[0,0,1],[1,0,0]], "UIx": '-Y', "UIy": 'Z',
                                        'ori_type': 'direction', 'ori_axes': [[0,-1,0],[1,0,0],[0,0,0]], 'ori_live': [[-1,0,0],[0,-1,0],[0,0,-1]]
                                        },
                             'table': { 'min': Point(0.4, -0.3, 0.0),
                                        'max': Point(0.7, 0.3, 0.6),
                                        'start': Point(0.5, 0.0, 0.3),
                                        'ori': Quaternion(0.,0.,0.,1.), #Quaternion(np.sqrt(2)/2, np.sqrt(2)/2., 0.0, 0.0),# -> (-np.sqrt(2)/2, np.sqrt(2)/2., 0.0, 0.0)
                                        'axes': [[0,0,1],[-1,0,0],[0,-1,0]], 'view': [[0,-1,0],[1,0,0],[0,0,1]], 'view2': [[0,-1,0],[1,0,0],[0,0,1]], "UIx": '-Y', "UIy": 'X',
                                        'ori_type': 'direction', 'ori_axes': [[-1,0,0],[0,1,0],[0,0,0]], 'ori_live': [[0,-1,0],[1,0,0],[0,0,-1]]
                                        } }
        elif IK_SOLVER == 'pyrep':
            self.ENV_DAT = {'above': {  'min': Point(-0.35, -0.35, 0.6),
                                        'max': Point(0.35,0.35,1.15),
                                        'ori': Quaternion(0.0, 0.0, 1.0, 0.0),
                                        'start': Point(0.0, 0.0, 0.85),
                                        'axes': np.eye(3), 'view': [[1,0,0],[0,0,1],[0,1,0]], 'view2': [[1,0,0],[0,0,1],[0,1,0]],"UIx": 'X', "UIy": 'Z',
                                        'ori_type': 'neg_normal', 'ori_axes': [[0,1,0],[-1,0,0],[0,0,0]], 'ori_live': [[0,-1,0],[1,0,0],[0,0,-1]]
                                     },
                             'wall':  { 'min': Point(0.42, -0.2, 0.0),
                                        'max': Point(0.7, 0.2, 0.74),
                                        'ori': Quaternion(0, np.sqrt(2)/2, 0, np.sqrt(2)/2),
                                        'start': Point(0.7, 0.0, 0.1),
                                        'axes': [[0,1,0],[-1,0,0],[0,0,1]], 'view': [[0,-1,0],[0,0,1],[1,0,0]], 'view2': [[0,-1,0],[0,0,1],[1,0,0]], "UIx": '-Y', "UIy": 'Z',
                                        'ori_type': 'direction', 'ori_axes': [[0,-1,0],[1,0,0],[0,0,0]], 'ori_live': [[-1,0,0],[0,-1,0],[0,0,-1]]
                                        },
                             'table': { 'min': Point(0.4, -0.3, 0.0),
                                        'max': Point(0.7, 0.3, 0.6),
                                        'start': Point(0.5, 0.0, 0.3),
                                        'ori': Quaternion(np.sqrt(2)/2, np.sqrt(2)/2., 0.0, 0.0),
                                        'axes': [[0,0,1],[-1,0,0],[0,-1,0]], 'view': [[0,-1,0],[1,0,0],[0,0,1]], 'view2': [[0,-1,0],[1,0,0],[0,0,1]], "UIx": '-Y', "UIy": 'X',
                                        'ori_type': 'direction', 'ori_axes': [[-1,0,0],[0,1,0],[0,0,0]], 'ori_live': [[0,-1,0],[1,0,0],[0,0,-1]]
                                        } }
        # chosen workspace
        self.ENV = self.ENV_DAT['above']
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


def GenerateSomePaths(sp, ss):
    poses = []
    pose = Pose()
    pose.orientation = md.ENV_DAT['above']['ori']
    n = 100
    for i in range(0,n):
        pose.position = Point(0.2*np.cos(2*np.pi*i/n),-0.2*np.sin(2*np.pi*i/n),0.95)
        poses.append(deepcopy(pose))
    actions = [""] * len(poses)
    sp.append(CustomPath(poses, actions, "", "circle", "above"))

    poses = []
    pose.position = Point(0.4,0.0,0.65)
    pose.orientation = md.ENV_DAT['wall']['ori']
    poses.append(deepcopy(pose))
    pose.position = Point(0.4,0.0,0.75)
    poses.append(deepcopy(pose))
    pose.position = Point(0.4,0.1,0.65)
    poses.append(deepcopy(pose))
    pose.position = Point(0.4,-0.1,1.05)
    poses.append(deepcopy(pose))
    pose.position = Point(0.4,0.0,0.95)
    poses.append(deepcopy(pose))
    actions = [""] * len(poses)
    sp.append(CustomPath(poses, actions, "", "wall", "wall"))

    poses = []
    pose.position = Point(0.5,0.0,0.2)
    pose.orientation = Quaternion(0.,0.,0.,1.0)
    poses.append(deepcopy(pose))
    pose.position = Point(0.5,0.0,0.2)
    pose.orientation = Quaternion(0.,0.,1.,0.0)
    poses.append(deepcopy(pose))
    pose.position = Point(0.5,0.0,0.2)
    pose.orientation = Quaternion(0,1,0,0)
    poses.append(deepcopy(pose))
    pose.position = Point(0.5,0.0,0.2)
    pose.orientation = Quaternion(1,0,0,0)
    poses.append(deepcopy(pose))
    pose.position = Point(0.5,0.0,0.2)
    pose.orientation = Quaternion(-0.5,0.5,0.5,-0.5)
    poses.append(deepcopy(pose))
    pose.position = Point(0.5,0.0,0.2)
    pose.orientation = Quaternion(0.5,-0.5,-0.5,0.5)
    poses.append(deepcopy(pose))
    pose.position = Point(0.5,0.1,0.2)
    poses.append(deepcopy(pose))
    pose.position = Point(0.5,-0.1,0.2)
    poses.append(deepcopy(pose))
    pose.position = Point(0.5,0.0,0.2)
    poses.append(deepcopy(pose))
    actions = [""] * len(poses)
    sp.append(CustomPath(poses, actions, "", "customtest", "wall"))

    ## Object by existing scene
    SCENE = 'pickplace'
    index = indx(ss, SCENE)
    box_pose = deepcopy(ss[index].mesh_poses[0])
    box_dim = deepcopy(ss[index].mesh_sizes[0])
    poses = []
    pose.position = Point(0.0,0.0,1.17)
    pose.orientation = md.ENV_DAT['above']['ori']
    poses.append(deepcopy(pose))

    pose.position = box_pose.position
    pose.position.z += box_dim.z-0.04
    pose.orientation = md.ENV_DAT['table']['ori']
    poses.append(deepcopy(pose))
    pose.position = Point(0.5,0.0,0.75)
    poses.append(deepcopy(pose))
    pose.position = Point(0.5,-0.3,0.075)
    poses.append(deepcopy(pose))
    pose.position = Point(0.,0.,1.25)
    pose.orientation = md.ENV_DAT['above']['ori']
    poses.append(deepcopy(pose))
    actions = ['', 'box', '', 'box', '']
    sp.append(CustomPath(poses, actions, SCENE, "pickplace", "table"))

    ## Drawer path init
    poses = []
    SCENE = 'drawer'
    index = indx(ss, SCENE)
    drawer_pose = deepcopy(ss[index].mesh_poses[0])
    drawer_dim = deepcopy(ss[index].mesh_sizes[0])
    drawer_trans_origin = deepcopy(ss[index].mesh_trans_origin[1])

    pose.position = Point(0.0,0.0,1.25)
    pose.orientation = md.ENV_DAT['above']['ori']
    poses.append(deepcopy(pose))

    pose.position = drawer_pose.position
    pose.position.x -= 0.02
    pose.position.y += drawer_dim.y/2
    pose.position.z += drawer_trans_origin.z+0.05
    pose.orientation = md.ENV_DAT['wall']['ori']
    poses.append(deepcopy(pose))

    for i in range(0,20):
        pose.position.x -= 0.01 # open the socket
        poses.append(deepcopy(pose))

    pose.position.z -= 0.15
    poses.append(deepcopy(pose))

    drawer_pose = deepcopy(ss[index].mesh_poses[2])
    drawer_trans_origin = deepcopy(ss[index].mesh_trans_origin[2])
    pose.position = drawer_pose.position
    pose.position.x -= 0.02
    pose.position.y += drawer_dim.y/2
    pose.position.z += drawer_trans_origin.z+0.05
    pose.orientation = md.ENV_DAT['wall']['ori']
    poses.append(deepcopy(pose))

    for i in range(0,20):
        pose.position.x -= 0.01 # open the socket
        poses.append(deepcopy(pose))

    pose.position = Point(0.0,0.0,1.25)
    pose.orientation = md.ENV_DAT['above']['ori']
    poses.append(deepcopy(pose))
    actions = ['', 'drawer_socket_1', '','','','','','','','','','', '','','','','','','','','', 'drawer_socket_1', '', 'drawer_socket_2', '','','','','','','','','','', '','','','','','','','','', 'drawer_socket_2', '']
    sp.append(CustomPath(poses, actions, SCENE, "drawer", "wall"))

    poses = []
    pose.position = Point(0.0,0.0,1.25)
    pose.orientation = md.ENV_DAT['above']['ori']
    poses.append(deepcopy(pose))
    pose.position =  Point(0.44,-0.05,0.13)
    pose.orientation = md.ENV_DAT['table']['ori']
    poses.append(deepcopy(pose))
    pose.position =  Point(0.44,-0.05,0.1)
    poses.append(deepcopy(pose))
    pose.position = Point(0.0,0.0,1.25)
    pose.orientation = md.ENV_DAT['above']['ori']
    poses.append(deepcopy(pose))
    actions = ['', 'button', 'button', '']
    sp.append(CustomPath(poses, actions, "pushbutton", "pushbutton", "table"))


def GenerateSomeScenes(ss):
    ## 0 - empty
    ss.append(CustomScene('empty'))
    ## 1 - drawer
    mesh_names=['drawer', 'drawer_socket_1', 'drawer_socket_2', 'drawer_socket_3']
    drawer_heights = [0.083617, 0.124468, 0.094613]
    mesh_trans_origin = [Vector3(0.,0.,0.), Vector3(0.024574, 0., 0.032766+0.094613+0.015387+0.124468+0.016383), Vector3(0.024574, 0., 0.032766+0.094613+0.015387), Vector3(0.024574, 0., 0.032766)]
    # transform to workspace
    axes = [[0.,1.,0.],
    [-1.,0.,0.],
    [0.,0.,1.]]
    mesh_trans_origin_ = []
    for i in range(0, len(mesh_trans_origin)):
        orig = mesh_trans_origin[i]
        orig_ = Vector3()
        orig_.x = np.dot(axes[0],[orig.x, orig.y, orig.z])
        orig_.y = np.dot(axes[1],[orig.x, orig.y, orig.z])
        orig_.z = np.dot(axes[2],[orig.x, orig.y, orig.z])
        mesh_trans_origin_.append(Vector3(orig_.x, orig_.y, orig_.z))
    mesh_trans_origin = mesh_trans_origin_
    ##

    drawer_sockets_open = 0.0
    mesh_poses = []
    mesh_sizes = []
    size = Vector3()
    pose = Pose()
    # drawer shell
    pose.position = Point(0.8,0.2,0.)
    pose.orientation = Quaternion(0.5,-0.5,-0.5,0.5)
    mesh_poses.append(deepcopy(pose))
    size = Vector3(0.3,0.5,0.4)
    size = Vector3(0.5,-0.3,0.4)
    mesh_sizes.append(deepcopy(size))
    # drawer sockets
    for i in range(0,3):
        pose.position = Point(0.8-drawer_sockets_open/2,0.2,0.)
        pose.orientation = Quaternion(0.5,-0.5,-0.5,0.5)
        mesh_poses.append(deepcopy(pose))
        size = Vector3(0.25085,0.5,drawer_heights[i])
        size = Vector3(0.5,-0.25085,drawer_heights[i])
        mesh_sizes.append(deepcopy(size))
    ss.append(CustomScene('drawer', mesh_names=mesh_names, mesh_poses=mesh_poses, mesh_sizes=mesh_sizes, mesh_trans_origin=mesh_trans_origin))
    ## 2 - box
    mesh_names=['box']
    mesh_poses = []
    pose.position = Point(0.6,0.1,0.04)
    pose.orientation = Quaternion(0.0,0.0,0.0,1.0)
    mesh_poses.append(deepcopy(pose))
    mesh_sizes = [Vector3(0.075,0.075,0.075)]
    mesh_trans_origin = [Vector3(0.,-0.0, 0.0)]
    ss.append(CustomScene('pickplace', mesh_names=mesh_names, mesh_poses=mesh_poses, mesh_sizes=mesh_sizes))#, mesh_trans_origin=mesh_trans_origin))
    ## 3 - push button
    mesh_names=['button', 'button_out']
    mesh_poses = []
    pose.position = Point(0.6,0.1,0.0)
    pose.orientation = Quaternion(0.5,-0.5,-0.5,0.5)
    mesh_poses.append(deepcopy(pose))

    pose.orientation = Quaternion(0.5,-0.5,-0.5,0.5)
    mesh_poses.append(deepcopy(pose))
    mesh_sizes = [Vector3(0.1,0.1,0.1), Vector3(0.1,0.1,0.13)]
    mesh_trans_origin = [Vector3(0.,-0.6, .0),Vector3(0.,-0.6, 0.)]
    ss.append(CustomScene('pushbutton', mesh_names=mesh_names, mesh_poses=mesh_poses, mesh_sizes=mesh_sizes))#, mesh_trans_origin=mesh_trans_origin))

    ### MORE SCENES
    # drawer2 - 1.
    mesh_names=['drawer', 'drawer_socket_1', 'drawer_socket_2', 'drawer_socket_3','drawer2', 'drawer2_socket_1', 'drawer2_socket_2', 'drawer2_socket_3']
    mesh_trans_origin2 = deepcopy(mesh_trans_origin)
    mesh_trans_origin2.extend(deepcopy(mesh_trans_origin))
    drawer_sockets_open = 0.0
    mesh_poses = []
    mesh_sizes = []
    size = Vector3()
    pose = Pose()

    pose.position = Point(0.8,0.2,0.)
    pose.orientation = Quaternion(0.5,-0.5,-0.5,0.5)
    mesh_poses.append(deepcopy(pose))
    size = Vector3(0.3,0.5,0.4)
    size = Vector3(0.5,-0.3,0.4)
    mesh_sizes.append(deepcopy(size))
    # drawer sockets
    for i in range(0,3):
        pose.position = Point(0.8-drawer_sockets_open/2,0.2,0.)
        pose.orientation = Quaternion(0.5,-0.5,-0.5,0.5)
        mesh_poses.append(deepcopy(pose))
        size = Vector3(0.25085,0.5,drawer_heights[i])
        size = Vector3(0.5,-0.25085,drawer_heights[i])
        mesh_sizes.append(deepcopy(size))
    # 2.
    pose.position = Point(0.6,-0.2,0.)
    pose.orientation = Quaternion(0.5,-0.5,-0.5,0.5)
    mesh_poses.append(deepcopy(pose))
    size = Vector3(0.3,0.5,0.4)
    size = Vector3(0.5,-0.3,0.4)
    mesh_sizes.append(deepcopy(size))
    # drawer sockets
    for i in range(0,3):
        pose.position = Point(0.6-drawer_sockets_open/2,-0.2,0.)
        pose.orientation = Quaternion(0.5,-0.5,-0.5,0.5)
        mesh_poses.append(deepcopy(pose))
        size = Vector3(0.25085,0.5,drawer_heights[i])
        size = Vector3(0.5,-0.25085,drawer_heights[i])
        mesh_sizes.append(deepcopy(size))
    ss.append(CustomScene('drawer2', mesh_names=mesh_names, mesh_poses=mesh_poses, mesh_sizes=mesh_sizes, mesh_trans_origin=mesh_trans_origin2))
    ## 2 - box
    mesh_names=['box', 'box2']
    mesh_poses = []
    pose.position = Point(0.5,0.0,0.04)
    pose.orientation = Quaternion(0.0,0.0,0.0,1.0)
    mesh_poses.append(deepcopy(pose))
    mesh_sizes = [Vector3(0.075,0.075,0.075), Vector3(0.075,0.075,0.075)]
    # box2 - 1.
    pose.position = Point(0.6,-0.2,0.04)
    pose.orientation = Quaternion(0.0,0.0,0.0,1.0)
    mesh_poses.append(deepcopy(pose))
    ss.append(CustomScene('pickplace2', mesh_names=mesh_names, mesh_poses=mesh_poses, mesh_sizes=mesh_sizes))
    ## 3 - push button
    mesh_names=['button', 'button_out', 'button2', 'button2_out', 'button3', 'button3_out']
    mesh_poses = []
    pose.position = Point(0.4,0.0,0.0)
    pose.orientation = Quaternion(0.5,-0.5,-0.5,0.5)
    mesh_poses.append(deepcopy(pose))
    mesh_poses.append(deepcopy(pose))
    pose.position = Point(0.6,0.2,0.0)
    mesh_poses.append(deepcopy(pose))
    mesh_poses.append(deepcopy(pose))
    pose.position = Point(0.7,-0.3,0.0)
    mesh_poses.append(deepcopy(pose))
    mesh_poses.append(deepcopy(pose))
    mesh_sizes = [Vector3(0.1,0.1,0.1), Vector3(0.1,0.1,0.13), [Vector3(0.1,0.1,0.1), Vector3(0.1,0.1,0.13)], [Vector3(0.1,0.1,0.1), Vector3(0.1,0.1,0.13)]]
    ss.append(CustomScene('pushbutton2', mesh_names=mesh_names, mesh_poses=mesh_poses, mesh_sizes=mesh_sizes))


class CustomPath():
    def __init__(self, poses=[], actions=[], scene=None, NAME="", ENV=""):
        self.n = len(poses)
        self.scene = scene
        self.poses = poses
        self.actions = actions
        self.NAME = NAME
        self.ENV = ENV


class CustomScene():
    ''' Custom scenes with custom names is defined with custom objects with
        pose and size
        - mesh_names, string[], name of object
        - mesh_poses, Pose[], position and orientation of object origin
        - mesh_sizes, Vector3[], length x,y,z of object from the origin
    '''
    def __init__(self, NAME="", mesh_names=[], mesh_poses=[], mesh_sizes=[], mesh_trans_origin=[]):
        self.NAME = NAME
        self.mesh_names = mesh_names
        self.mesh_poses = mesh_poses
        self.mesh_sizes = mesh_sizes
        if mesh_trans_origin:
            self.mesh_trans_origin = mesh_trans_origin
        else:
            self.mesh_trans_origin = [Vector3(0.,0.,0.)] * len(mesh_names)

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
    assert type(q) == type(Quaternion()), "extq input arg q: Not Quaternion!"
    return q.x, q.y, q.z, q.w

def extv(v):
    ''' Extracts Point/Vector3 to Cartesian values
    Parameters:
        v (Point() or Vector3()): From geometry_msgs.msg
    Returns:
        [x,y,z] (Floats tuple[3]): Point/Vector3 extracted
    '''
    assert type(v) == type(Point()) or type(v) == type(Vector3()), "extv input arg v: Not Point or Vector3!"
    return v.x, v.y, v.z

def extp(p):
    ''' Extracts pose
    Paramters:
        p (Pose())
    Returns:
        list (Float[7])
    '''
    assert type(p) == type(Pose()), "extp input arg p: Not Pose type!"
    return p.position.x, p.position.y, p.position.z, p.orientation.x, p.orientation.y, p.orientation.z, p.orientation.w
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
###
