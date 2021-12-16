#!/usr/bin/env python3.8
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
import numpy as np
from copy import deepcopy
import os, yaml, random
from os.path import expanduser, isfile
import time
from os_and_utils.utils import ordered_load, GlobalPaths, ros_enabled, load_params, merge_two_dicts, isarray, isnumber
from os_and_utils.utils_ros import extv, extp, extq

if ros_enabled():
    import rospy
    from geometry_msgs.msg import Quaternion, Pose, PoseStamped, Point, Vector3
    from visualization_msgs.msg import MarkerArray, Marker
    from std_msgs.msg import Int8, Float64MultiArray
else:
    print("[WARN*] ROS cannot be not imported!")



def init(minimal=False):
    ''' Initialize the shared data across threads (Leap, UI, Control)
    Parameters:
        minimal (Bool): Import only file paths and general informations
    '''
    global Gs, GsK, configGestures, paths
    # 1. Initialize File/Folder Paths
    paths = GlobalPaths()

    # 2. Loads config. data to gestures
    # - gesture_recording.yaml
    configGestures = ParseYAML.load_gesture_config_file(paths.custom_settings_yaml)
    recordingConfig = ParseYAML.load_recording_file(paths.custom_settings_yaml)
    Gs, GsK = ParseYAML.load_gestures_file(paths.custom_settings_yaml)

    # 3. Loads config. about robot
    # - ROSparams /mirracle_config/*
    # - robot_move.yaml.yaml
    global robot, simulator, gripper, plot, inverse_kinematics, inverse_kinematics_topic
    robot, simulator, gripper, plot, inverse_kinematics, inverse_kinematics_topic = load_params()

    global vel_lim, effort_lim, lower_lim, upper_lim, joint_names, base_link, group_name, eef, grasping_group, tac_topic, joint_states_topic
    vel_lim, effort_lim, lower_lim, upper_lim, joint_names, base_link, group_name, eef, grasping_group, tac_topic, joint_states_topic = ParseYAML.load_robot_move_file(paths.custom_settings_yaml, robot, simulator)

    print("[Settings] Workspace folder is set to: "+paths.ws_folder)


    ### 5. Latest Data                      ###
    ###     - Generated Scenes from YAML    ###
    ###     - Generated Paths from YAML     ###
    ###     - Current scene info            ###
    ###########################################
    global sp, mo, ss, scene

    ## Objects for saved scenes and paths
    ss = CustomScene.GenerateFromYAML()
    sp = CustomPath.GenerateFromYAML()
    scene = None # current scene informations

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


    ### 7. User Interface Data              ###
    ###     - From application.yaml         ###
    ###########################################
    with open(paths.custom_settings_yaml+"application.yaml", 'r') as stream:
        app_data_loaded = ordered_load(stream, yaml.SafeLoader)
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
    print("[Settings] done")




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
            paths_folder (Str): folder to specify YAML files, if not specified the default paths.custom_settings_yaml folder is used
            paths_file_catch_phrase (Str): Searches for files with this substring (e.g. use 'paths' to load all names -> 'paths1.yaml', paths2.yaml)
                - If not specified then paths_file_catch_phrase='paths'
                - If specified as '', all files are loaded
            poses_file_catch_phrase (Str): Loads poses from YAML file with this substring
        Returns:
            sp (CustomPath()[]): The generated array of paths
        '''
        sp = []
        if not paths_folder:
            paths_folder = paths.custom_settings_yaml

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
            scenes_folder (Str): folder to specify YAML files, if not specified the default paths.custom_settings_yaml folder is used
            scenes_file_catch_phrase (Str): Searches for files with this substring (e.g. use 'scene' to load all names -> 'scene1.yaml', scene2.yaml)
                - If not specified then scenes_file_catch_phrase='scene'
                - If specified as '', all files are loaded
            poses_file_catch_phrase (Str): Loads poses from YAML file with this substring
        '''
        ss = []
        if not scenes_folder:
            scenes_folder = paths.custom_settings_yaml

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

    @staticmethod
    def load_gesture_config_file(custom_settings_yaml):
        with open(custom_settings_yaml+"gesture_config.yaml", 'r') as stream:
            gesture_config = ordered_load(stream, yaml.SafeLoader)
        return gesture_config

    @staticmethod
    def load_recording_file(custom_settings_yaml):
        with open(custom_settings_yaml+"recording.yaml", 'r') as stream:
            recording = ordered_load(stream, yaml.SafeLoader)
        return recording


    @staticmethod
    def load_gestures_file(custom_settings_yaml, ret=''):
        with open(custom_settings_yaml+"gestures.yaml", 'r') as stream:
            gestures_data_loaded = ordered_load(stream, yaml.SafeLoader)

        if ret=='obj': return gestures_data_loaded

        gesture_config = ParseYAML.load_gesture_config_file(custom_settings_yaml)

        keys = gestures_data_loaded.keys()

        Gs_set = gesture_config['using_set']

        Gs = []
        GsK = []
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
            Gs.append(key)
            GsK.append(g['key'])

        return Gs, GsK

    @staticmethod
    def load_robot_move_file(custom_settings_yaml, robot, simulator):
        with open(paths.custom_settings_yaml+"robot_move.yaml", 'r') as stream:
            robot_move_data_loaded = ordered_load(stream, yaml.SafeLoader)
        vel_lim = robot_move_data_loaded['robots'][robot]['vel_lim']
        effort_lim = robot_move_data_loaded['robots'][robot]['effort_lim']
        lower_lim = robot_move_data_loaded['robots'][robot]['lower_lim']
        upper_lim = robot_move_data_loaded['robots'][robot]['upper_lim']

        joint_names = robot_move_data_loaded['robots'][robot]['joint_names']
        base_link = robot_move_data_loaded['robots'][robot]['base_link']
        group_name = robot_move_data_loaded['robots'][robot]['group_name']
        eef = robot_move_data_loaded['robots'][robot]['eef']
        grasping_group = robot_move_data_loaded['robots'][robot]['grasping_group']
        if simulator not in robot_move_data_loaded['simulators']:
            raise Exception("Wrong simualator name")
        tac_topic = robot_move_data_loaded['robots'][robot][simulator]['tac_topic']
        joint_states_topic = robot_move_data_loaded['robots'][robot][simulator]['joint_states_topic']

        return vel_lim, effort_lim, lower_lim, upper_lim, joint_names, base_link, group_name, eef, grasping_group, tac_topic, joint_states_topic


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
