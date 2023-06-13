import sys, os, yaml, inspect
import numpy as np
from collections import OrderedDict

def ros_enabled():
    try:
        from teleop_gesture_toolbox.msg import Frame as Framemsg
        '''
        try:
            rospy.get_param("/mirracle_config/robot", 'panda')
            return True
        except ConnectionRefusedError:
            print("roscore not running -> launching without ros")
            return False
        '''
        return True
    except:
        return False

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

class GlobalPaths():
    '''
    '''
    def __init__(self, change_working_directory=True):
        '''
        Parameters:
            change_working_directory (bool): Changed to '<this_file>/../'
        '''
        self.home = os.path.expanduser("~")
        # searches for the WS name + print it

        THIS_FILE_PATH = os.path.dirname(inspect.getabsfile(inspect.currentframe()))
        THIS_FILE_TMP = os.path.abspath(os.path.join(THIS_FILE_PATH, '..', '..', '..', '..'))
        self.ws_folder = THIS_FILE_TMP.split('/')[-1]

        MG_PATH = os.path.abspath(os.path.join(THIS_FILE_PATH, '..', '..'))
        self.teleop_gesture_toolbox_path = MG_PATH+'/src/'
        self.learn_path = MG_PATH+'/include/data/learning/'
        self.data_export_path = MG_PATH+'/include/data/export/'
        self.graphics_path = MG_PATH+'/include/graphics/'
        self.plots_path = MG_PATH+'/include/plots/'
        self.network_path = MG_PATH+'/include/data/Trained_network/'
        self.models_path = MG_PATH+'/include/models/'
        self.custom_settings_yaml = MG_PATH+'/include/custom_settings/'
        self.UCB_path = MG_PATH+'/include/third_party/UCB/'
        self.promp_sebasutp_path = MG_PATH+'/include/third_party/promp/'
        self.promp_paraschos_path = MG_PATH+'/include/third_party/promps_python/'
        TMP2 = os.path.abspath(os.path.join(MG_PATH, '..'))
        self.coppelia_scene_path = TMP2+"/coppelia_sim_ros_interface/include/scenes/"
        self.coppelia_sim_ros_interface_path = TMP2+"/coppelia_sim_ros_interface/src/"
        if change_working_directory:
            sys.path.append(MG_PATH+'/src')
            os.chdir(MG_PATH+'/src')

def to_bool(str):
    if str in ['true', 'True', 't', '1', True, 1]:
        return True
    elif str in ['false', 'False', 'f', '0', False, 0]:
        return False
    else: raise Exception("[str_to_bool] Wrong input!")

#to_bool('true')

def load_params():
    if ros_enabled():
        try:
            rospy
        except NameError:
            import rospy

        robot = rospy.get_param("/mirracle_config/robot", 'panda')
        simulator = rospy.get_param("/mirracle_config/simulator", 'coppelia')
        gripper = rospy.get_param("/mirracle_config/gripper", 'none')
        plot = to_bool(rospy.get_param("/mirracle_config/visualize", 'false'))
        inverse_kinematics = rospy.get_param("/mirracle_config/ik_solver", 'relaxed_ik')
        inverse_kinematics_topic = rospy.get_param("/mirracle_config/ik_topic", '')
        gesture_detection_on = to_bool(rospy.get_param("/mirracle_config/launch_gesture_detection", 'false'))
        launch_gesture_detection = to_bool(rospy.get_param("/mirracle_config/launch_gesture_detection", 'false'))
        launch_ui = to_bool(rospy.get_param("/mirracle_config/launch_ui", 'false'))
    else:
        robot = 'panda'
        simulator = 'coppelia'
        gripper = 'none'
        plot = False
        inverse_kinematics = 'relaxed_ik'
        inverse_kinematics_topic = ''
        gesture_detection_on = False
        launch_gesture_detection = False
        launch_ui = False
    return robot, simulator, gripper, plot, inverse_kinematics, inverse_kinematics_topic, gesture_detection_on, launch_gesture_detection, launch_ui



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

def distancePoses(p1, p2):
    ''' Returns distance between two pose objects
    Parameters:
        pose1 (type list,tuple,np.ndarray,Pose() from geometry_msgs.msg)
        pose2 (type list,tuple,np.ndarray,Pose() from geometry_msgs.msg)
    Returns:
        distance (Float)
    '''
    try:
        p1.position
        p1 = [p1.position.x, p1.position.y, p1.position.z]
    except AttributeError:
        pass
    try:
        p2.position
        p2 = [p2.position.x, p2.position.y, p2.position.z]
    except AttributeError:
        pass
    return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2 + (p1[2] - p2[2])**2)


def get_object_of_closest_distance(objects_in_scenes, pose_in_scene):
    print(f"Type: objects_in_scenes {objects_in_scenes}, pose_in_scene {pose_in_scene}")
    min_dist, min_id = float('inf'), None
    for n, object_in_scenes in enumerate(objects_in_scenes):
        dist = distancePoses(object_in_scenes, pose_in_scene)
        if dist < min_dist:
            min_dist = dist
            min_id = n
    return n


def point_by_ratio(p1, p2, ratio):
    x1, y1 = p1
    x2, y2 = p2
    return (x2 - x1) * ratio, (y2 - y1) * ratio

#
