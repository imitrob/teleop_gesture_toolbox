import collections
import numpy as np
import math
from geometry_msgs.msg import Quaternion, Pose, PoseStamped, Point, Vector3
from visualization_msgs.msg import MarkerArray, Marker
from std_msgs.msg import Int8, Float64MultiArray
#import modern_robotics as mr
from copy import deepcopy
from os.path import expanduser, isfile
import rospy

def init():
    ''' 1. All the observations leading to gesture recognization
        2. Gesture detection probabilities
        3. Gesture boolean value
        4. Move data
        -> Gesture global parameters

    '''

    # 1. Frames, Processing a frame
    global BUFFER_LEN, frames, timestamps, frames_adv, forward_kinematics
    BUFFER_LEN = 300
    frames = collections.deque(maxlen=BUFFER_LEN)
    timestamps = collections.deque(maxlen=BUFFER_LEN)
    frames_adv = collections.deque(maxlen=BUFFER_LEN)
    forward_kinematics = collections.deque(maxlen=BUFFER_LEN)

    # 2.,3.
    global gd
    gd = GestureDataHands()

    # 4.
    global md
    md = MoveData()

    global rd
    rd = RobotData()

    ## Configuration page values
    global NumConfigBars, VariableValues
    NumConfigBars = [2,4]
    VariableValues = np.zeros(NumConfigBars)

    ## Status Bar
    global HoldAnchor, HoldPrevState, HoldValue, currentPose, WindowState, leavingAction, w, h, ui_scale
    WindowState = 0.0  # 0-Main page, 1-Config page
    HoldAnchor = 0.0  # For moving status bar
    HoldPrevState = False  # --||--
    HoldValue = 0
    currentPose = 0
    leavingAction = False
    w, h = 1000, 800
    ui_scale = 2000

    # Fixed Conditions
    global FIXED_ORI_TOGGLE, print_path_trace
    FIXED_ORI_TOGGLE = False
    print_path_trace = False

    ## iiwa
    global JOINT_NAMES, BASE_LINK, GROUP_NAME, ROBOT_NAME, TAC_TOPIC, JOINT_STATES_TOPIC, EEF_NAME
    # PICK ROBOT: - 'panda' or 'iiwa'
    ROBOT_NAME = 'panda'
    # -------------- #

    if ROBOT_NAME == 'iiwa':
        JOINT_NAMES = ['r1_joint_1', 'r1_joint_2', 'r1_joint_3', 'r1_joint_4', 'r1_joint_5', 'r1_joint_6', 'r1_joint_7']
        BASE_LINK = 'base_link'
        GROUP_NAME = "r1_arm"
        TAC_TOPIC = "/r1/trajectory_controller/follow_joint_trajectory"
        JOINT_STATES_TOPIC = '/r1/joint_states'
        EEF_NAME = 'r1_ee'
    elif ROBOT_NAME == 'panda':
        JOINT_NAMES = ['panda_joint1', 'panda_joint2', 'panda_joint3', 'panda_joint4', 'panda_joint5', 'panda_joint6', 'panda_joint7']
        BASE_LINK = 'panda_link0'
        GROUP_NAME = "panda_arm"
        TAC_TOPIC = '/position_joint_trajectory_controller/follow_joint_trajectory'
        JOINT_STATES_TOPIC = '/franka_state_controller/joint_states'
        EEF_NAME = 'panda_link8'
    else:
        raise Exception("Wrong robot name")


    global goal_joints, goal_pose, sp, mo, ss, scene
    goal_joints, goal_pose = None, None
    sp, ss = [], []
    global joints
    joints = []
    GenerateSomeScenes(ss) # Saved scenes
    GenerateSomePaths(sp, ss) # Saved paths
    scene = None # current scene informations
    mo = None # MoveIt object

    ## MODERN ROBOTICS config
    # iiwa
    '''
    global Glist, Mlist, Slist, Kp, Ki, Kd, g, eint
    M01 = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0.089159], [0, 0, 0, 1]]
    M12 = [[0, 0, 1, 0.28], [0, 1, 0, 0.13585], [-1, 0, 0, 0], [0, 0, 0, 1]]
    M23 = [[1, 0, 0, 0], [0, 1, 0, -0.1197], [0, 0, 1, 0.395], [0, 0, 0, 1]]
    M34 = [[0, 0, 1, 0], [0, 1, 0, 0], [-1, 0, 0, 0.14225], [0, 0, 0, 1]]
    M45 = [[1, 0, 0, 0], [0, 1, 0, 0.093], [0, 0, 1, 0], [0, 0, 0, 1]]
    M56 = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0.09465], [0, 0, 0, 1]]
    M67 = [[1, 0, 0, 0], [0, 0, 1, 0.0823], [0, -1, 0, 0], [0, 0, 0, 1]]
    G1 = np.diag([0.010267495893, 0.010267495893,  0.00666, 3.7, 3.7, 3.7])
    G2 = np.diag([0.22689067591, 0.22689067591, 0.0151074, 8.393, 8.393, 8.393])
    G3 = np.diag([0.049443313556, 0.049443313556, 0.004095, 2.275, 2.275, 2.275])
    G4 = np.diag([0.111172755531, 0.111172755531, 0.21942, 1.219, 1.219, 1.219])
    G5 = np.diag([0.111172755531, 0.111172755531, 0.21942, 1.219, 1.219, 1.219])
    G6 = np.diag([0.0171364731454, 0.0171364731454, 0.033822, 0.1879, 0.1879, 0.1879])
    Glist = [G1, G2, G3, G4, G5, G6]
    Mlist = [M01, M12, M23, M34, M45, M56, M67]
    omegas = np.array([[0, 0, 1], [0, 1, 0], [0, 1, 0], [0, 1, 0], [0, 0, 1], [0, 1, 0]])
    qs_cont = np.array([[0.0, 0.0, 0.089159],[0.0, 0.13585, 0.0], [0.0, -0.1197, 0.425], \
                   [0.0, 0.0, 0.39225], [0.0, 0.093, 0.0], [0.0, 0.0, 0.09465]])
    qs = []
    sum = [0,0,0]
    for i in qs_cont:
        sum += i
        qs.append(deepcopy(sum))
    vs = -np.cross(omegas,qs)
    Slist = np.vstack((omegas.T,vs.T))
    Kp = 1.3
    Ki = 1.2
    Kd = 1.1
    g = np.array([0, 0, -9.81])
    eint = np.array([0.2, 0.2, 0.2, 0.2, 0.2, 0.2]) # n-vector of the time-integral of joint errors
    '''

    ## Files
    global HOME, LEARN_PATH, GRAPHICS_PATH, GESTURE_NAMES, NETWORK_PATH, PLOTS_PATH, WS_FOLDER
    HOME = expanduser("~")
    # searches for the WS name + print it
    THIS_FILE_PATH = os.path.dirname(os.path.realpath(__file__))
    THIS_FILE_TMP = os.path.abspath(os.path.join(THIS_FILE_PATH, '..'))
    WS_FOLDER = THIS_FILE_TMP.split('/')[-1]
    print("[Note] Workspace folder is set to: "+WS_FOLDER)

    LEARN_PATH = HOME+"/"+WS_FOLDER+"/src/mirracle_gestures/include/data/learning/"
    GRAPHICS_PATH = HOME+"/"+WS_FOLDER+"/src/mirracle_gestures/include/graphics/"
    PLOTS_PATH = HOME+"/"+WS_FOLDER+"/src/mirracle_gestures/include/plots/"
    NETWORK_PATH = HOME+"/"+WS_FOLDER+"/src/mirracle_gestures/include/data/learned_networks"
    GESTURE_NAMES = ['Grab', 'Pinch', 'Point', 'Respectful', 'Spock', 'Rock', 'Victory', 'Italian', 'Rotate', 'Swipe_Up', 'Pin', 'Touch', 'Swipe_Left', 'Swipe_Down', 'Swipe_Right']


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



    ## Misc other vars
    global pymc_in_pub
    pymc_in_pub = None # Publisher object
    print("set")


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
    Q(0,0,0,1) -> pointing down
    Q([0.5,0.5,0.5,-0.5]) -> wall

    '''
    def __init__(self):
        self.LEAP_AXES = [[1,0,0],[0,0,-1],[0,1,0]]
        self.ENV_DAT = {'above': {  'min': Point(-0.35, -0.35, 0.6),
                                    'max': Point(0.35,0.35,1.15),
                                    'ori': Quaternion(0.0,1.0,0.0,0.0),
                                    'start': Point(0.0, 0.0, 0.85),
                                    'axes': np.eye(3), 'view': [[1,0,0],[0,0,1],[0,1,0]], 'view2': [[1,0,0],[0,0,1],[0,1,0]],"UIx": 'X', "UIy": 'Z',
                                    'ori_type': 'neg_normal', 'ori_axes': [[0,1,0],[-1,0,0],[0,0,0]], 'ori_live': [[0,-1,0],[1,0,0],[0,0,-1]]
                                 },
                         'wall':  { 'min': Point(0.42, -0.2, 0.0),
                                    'max': Point(0.7, 0.2, 0.74),
                                    'ori': Quaternion(-0.5,0.5,0.5,-0.5),
                                    'start': Point(0.7, 0.0, 0.1),
                                    'axes': [[0,1,0],[-1,0,0],[0,0,1]], 'view': [[0,-1,0],[0,0,1],[1,0,0]], 'view2': [[0,-1,0],[0,0,1],[1,0,0]], "UIx": '-Y', "UIy": 'Z',
                                    'ori_type': 'direction', 'ori_axes': [[0,-1,0],[1,0,0],[0,0,0]], 'ori_live': [[-1,0,0],[0,-1,0],[0,0,-1]]
                                    },
                         'table': { 'min': Point(0.4, -0.3, 0.0),
                                    'max': Point(0.7, 0.3, 0.6),
                                    'start': Point(0.5, 0.0, 0.3),
                                    'ori': Quaternion(0.,0.,0.,1.), # (np.sqrt(2)/2, np.sqrt(2)/2., 0.0, 0.0) -> (-np.sqrt(2)/2, np.sqrt(2)/2., 0.0, 0.0)
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
        self.speed = 1
        self.applied_force = 10

        ### Live mode: gesture data
        self.gestures_goal_pose = Pose()
        self.gestures_goal_pose.position = self.ENV['start']
        self.gestures_goal_pose.orientation = self.ENV['ori']
        self.gestures_goal_stride = 0.1
        self.gestures_goal_rot_stride = np.pi/8
        # trajectory
        self._goal = None
        ### Additional poses

        self.LeapInRviz = Pose()
        self.LeapInRviz.orientation.w = 1.0

        self.pose_default = Pose()
        self.pose_default.position = Point(0.0,0.0,1.0)  # x, y, z
        self.pose_default.orientation = Quaternion(0.0,0.0,0.0,1.0)  # x, y, z, w

        self.pose_saved = Pose()
        self.pose_saved.position = Point(0.6,0.0,0.33)  # x, y, z
        self.pose_saved.orientation = Quaternion(0.5,0.5, 0.5, 0.5)  # x, y, z, w

        self.pose_pick = Pose()
        self.pose_pick.position = Point(0.3,0.0,-1.17)  # x, y, z
        self.pose_pick.orientation = Quaternion(np.sqrt(2)/2, np.sqrt(2)/2., 0.0, 0.0)  # x, y, z, w

        self.pose_place = Pose()
        self.pose_place.position = Point(0.55,0.0,0.7)  # x, y, z
        self.pose_place.orientation = Quaternion(0.5,0.5,0.5,0.5)  # x, y, z, w

        ## Pose Objects
        self.name_cube = 'box'
        self.pose_cube = Pose()
        self.pose_cube.position = Point(0.4,0.0,-1.24)
        self.pose_cube.orientation = Quaternion(0.0,0.0,0.0,1.0)

        ## Limits
        self.vel_lim = [ 1.71, 1.71, 1.75, 2.27, 2.44, 3.14, 3.14 ]
        self.effort_lim = [ 140, 140, 120, 100, 70, 70, 70 ]
        self.lower_lim = [-2.97, -2.09, -2.97, -2.09, -2.97, -2.09, -3.05]
        self.upper_lim = [2.97, 2.09, 2.97, 2.09, 2.97, 2.09, 3.05]

def quaternion_multiply(quaternion1, quaternion0):
    w0, x0, y0, z0 = quaternion0
    w1, x1, y1, z1 = quaternion1
    return np.array([-x1 * x0 - y1 * y0 - z1 * z0 + w1 * w0,
                     x1 * w0 + y1 * z0 - z1 * y0 + w1 * x0,
                     -x1 * z0 + y1 * w0 + z1 * x0 + w1 * y0,
                     x1 * y0 - y1 * x0 + z1 * w0 + w1 * z0], dtype=np.float64)


#quaternion_multiply((np.sqrt(2)/2, np.sqrt(2)/2., 0.0, 0.0),(0,1,0,0))

def GenerateSomePaths(sp, ss):
    poses = []
    pose = Pose()
    pose.orientation = Quaternion(0,0,1,0)
    n = 100
    for i in range(0,n):
        pose.position = Point(0.2*math.cos(2*math.pi*i/n),-0.2*math.sin(2*math.pi*i/n),0.95)
        poses.append(deepcopy(pose))
    actions = [""] * len(poses)
    sp.append(CustomPath(poses, actions, "", "circle", "above"))

    poses = []
    pose.position = Point(0.4,0.0,0.95)
    pose.orientation = Quaternion(-0.5,0.5,0.5,-0.5)
    poses.append(deepcopy(pose))
    pose.position = Point(0.4,0.0,1.05)
    poses.append(deepcopy(pose))
    pose.position = Point(0.4,0.1,1.05)
    poses.append(deepcopy(pose))
    pose.position = Point(0.4,-0.1,1.05)
    poses.append(deepcopy(pose))
    pose.position = Point(0.4,0.0,0.95)
    poses.append(deepcopy(pose))
    actions = [""] * len(poses)
    sp.append(CustomPath(poses, actions, "", "wall", "wall"))

    poses = []
    pose.position = Point(0.5,0.0,0.85)
    pose.orientation = Quaternion(0.,0.,0.,1.0)
    poses.append(deepcopy(pose))
    pose.position = Point(0.5,0.0,0.85)
    pose.orientation = Quaternion(0.,0.,1.,0.0)
    poses.append(deepcopy(pose))
    pose.position = Point(0.5,0.0,0.85)
    pose.orientation = Quaternion(0,1,0,0)
    poses.append(deepcopy(pose))
    pose.position = Point(0.5,0.0,0.85)
    pose.orientation = Quaternion(1,0,0,0)
    poses.append(deepcopy(pose))
    pose.position = Point(0.5,0.0,0.85)
    pose.orientation = Quaternion(-0.5,0.5,0.5,-0.5)
    poses.append(deepcopy(pose))
    pose.position = Point(0.5,0.0,0.85)
    pose.orientation = Quaternion(0.5,-0.5,-0.5,0.5)
    poses.append(deepcopy(pose))
    pose.position = Point(0.5,0.1,0.45)
    poses.append(deepcopy(pose))
    pose.position = Point(0.5,-0.1,0.45)
    poses.append(deepcopy(pose))
    pose.position = Point(0.5,0.0,0.45)
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
    pose.orientation = Quaternion(0.0, 0.0, 0.0, 1.0)
    poses.append(deepcopy(pose))

    pose.position = box_pose.position
    pose.position.z += box_dim.z-0.04
    pose.orientation = Quaternion(np.sqrt(2)/2, np.sqrt(2)/2., 0.0, 0.0)
    poses.append(deepcopy(pose))
    pose.position = Point(0.5,0.0,0.75)
    poses.append(deepcopy(pose))
    pose.position = Point(0.5,-0.3,0.075)
    poses.append(deepcopy(pose))
    pose.position = Point(0.,0.,1.25)
    pose.orientation = Quaternion(0.0, 0.0, 0.0, 1.0)
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
    pose.orientation = Quaternion(0.0, 0.0, 0.0, 1.0)
    poses.append(deepcopy(pose))

    pose.position = drawer_pose.position
    pose.position.x -= 0.02
    pose.position.y += drawer_dim.y/2
    pose.position.z += drawer_trans_origin.z+0.05
    pose.orientation = Quaternion(0.5,0.5,0.5,0.5)
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
    pose.orientation = Quaternion(0.5,0.5,0.5,0.5)
    poses.append(deepcopy(pose))

    for i in range(0,20):
        pose.position.x -= 0.01 # open the socket
        poses.append(deepcopy(pose))

    pose.position = Point(0.0,0.0,1.25)
    pose.orientation = Quaternion(0.0, 0.0, 0.0, 1.0)
    poses.append(deepcopy(pose))
    actions = ['', 'drawer_socket_1', '','','','','','','','','','', '','','','','','','','','', 'drawer_socket_1', '', 'drawer_socket_2', '','','','','','','','','','', '','','','','','','','','', 'drawer_socket_2', '']
    sp.append(CustomPath(poses, actions, SCENE, "drawer", "wall"))

    poses = []
    pose.position = Point(0.0,0.0,1.25)
    pose.orientation = Quaternion(0.0, 0.0, 0.0, 1.0)
    poses.append(deepcopy(pose))
    pose.position =  Point(0.44,-0.05,0.13)
    pose.orientation = Quaternion(np.sqrt(2)/2, np.sqrt(2)/2., 0.0, 0.0)
    poses.append(deepcopy(pose))
    pose.position =  Point(0.44,-0.05,0.1)
    poses.append(deepcopy(pose))
    pose.position = Point(0.0,0.0,1.25)
    pose.orientation = Quaternion(0.0, 0.0, 0.0, 1.0)
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
        self.poses = [relaxik_t(pose) for pose in poses]
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

class RobotData():
    def __init__(self):
        self.eef_pose = Pose()


def indx(ss, str):
    N = None
    for n,i in enumerate(ss):
        if i.NAME == str:
            N = n
    assert isinstance(N, int), "Scene not found"
    return N

## TODO: Duplicite function, move to unit place
def relaxik_t(pose1):
    pose_ = deepcopy(pose1)
    pose_.position.z -= 1.27
    pose_.position.y = -pose_.position.y
    return pose_

def getSceneNames():
    return [scene.NAME for scene in ss]

def getPathNames():
    return [path.NAME for path in sp]

def getModes():
    return ['live', 'interactive', 'nothin']
###
