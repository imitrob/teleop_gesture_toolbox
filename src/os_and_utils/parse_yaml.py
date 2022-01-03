import yaml, random
import numpy as np
from os_and_utils.utils import ordered_load, isarray, isnumber

# ROS dependency only to some functions
try:
    from os_and_utils.utils_ros import extv, extq
    from geometry_msgs.msg import Pose, Point, Quaternion
except:
    pass

class ParseYAML():
    @staticmethod
    def parseScene(data, ss):
        '''
        '''
        if 'scene' in data.keys():
            assert data['scene'] in ss.names(), "[Settings] Path you want to create has not valid scene name"
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
        with open(custom_settings_yaml+"robot_move.yaml", 'r') as stream:
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
