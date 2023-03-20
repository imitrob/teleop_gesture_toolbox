import sys, os, yaml, inspect
import numpy as np
from collections import OrderedDict

def ros_enabled():
    try:
        from teleop_gesture_toolbox.msg import Frame as Framemsg
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
        self.teleop_gesture_toolbox_path = MG_PATH+'/teleop_gesture_toolbox/'
        self.learn_path = MG_PATH+'/include/data/learning/'
        self.data_export_path = MG_PATH+'/include/data/export/'
        self.graphics_path = MG_PATH+'/include/graphics/'
        self.plots_path = MG_PATH+'/include/plots/'
        self.network_path = MG_PATH+'/include/data/trained_networks/'
        self.models_path = MG_PATH+'/include/models/'
        self.custom_settings_yaml = MG_PATH+'/include/custom_settings/'
        self.UCB_path = MG_PATH+'/include/third_party/UCB/'
        self.promp_sebasutp_path = MG_PATH+'/include/third_party/promp/'
        self.promp_paraschos_path = MG_PATH+'/include/third_party/promps_python/'
        TMP2 = os.path.abspath(os.path.join(MG_PATH, '..'))
        self.coppelia_scene_path = TMP2+"/coppelia_sim_ros_interface/include/scenes/"
        self.coppelia_sim_ros_interface_path = TMP2+"/coppelia_sim_ros_interface/coppelia_sim_ros_interface/"
        if change_working_directory:
            sys.path.append(MG_PATH+'/teleop_gesture_toolbox')
            os.chdir(MG_PATH+'/teleop_gesture_toolbox')

def to_bool(str):
    if str in ['true', 'True', 't', '1', True, 1]:
        return True
    elif str in ['false', 'False', 'f', '0', False, 0]:
        return False
    else: raise Exception("[str_to_bool] Wrong input!")


def load_params(roscm=None):
    if roscm is not None:
        try:
            rclpy
        except NameError:
            import rclpy
        robot = self.get_parameter("/gtoolbox_config/robot", 'panda')
        simulator = self.get_parameter("/gtoolbox_config/simulator", 'coppelia')
        gripper = self.get_parameter("/gtoolbox_config/gripper", 'none')
        plot = to_bool(self.get_parameter("/gtoolbox_config/visualize", 'false'))
        inverse_kinematics = self.get_parameter("/gtoolbox_config/ik_solver", '')
        inverse_kinematics_topic = self.get_parameter("/gtoolbox_config/ik_topic", '')
        gesture_detection_on = to_bool(self.get_parameter("/gtoolbox_config/launch_gesture_detection", 'false'))
        launch_gesture_detection = to_bool(self.get_parameter("/gtoolbox_config/launch_gesture_detection", 'false'))
        launch_ui = to_bool(self.get_parameter("/gtoolbox_config/launch_ui", 'false'))
    else:
        robot = 'panda'
        simulator = 'coppelia'
        gripper = 'franka_hand'
        plot = False
        inverse_kinematics = ''
        inverse_kinematics_topic = ''
        gesture_detection_on = True
        launch_gesture_detection = True
        launch_ui = True
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


def get_cbgo_path():
    from ament_index_python.packages import get_package_share_directory
    try:
        package_share_directory = get_package_share_directory('context_based_gesture_operation')
    except:
        raise Exception("Package context_based_gesture_operation not found!")
    return "/".join(package_share_directory.split("/")[:-4])+"/src/context_based_gesture_operation/context_based_gesture_operation"

class cc:
    H = '\033[95m'
    OK = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    W = '\033[93m'
    F = '\033[91m'
    E = '\033[0m'
    B = '\033[1m'
    U = '\033[4m'

# Handy functions
def extq(q):
    ''' Extracts Quaternion object
    Parameters:
        q (Quaternion()): From geometry_msgs.msg
    Returns:
        x,y,z,w (Floats tuple[4]): Quaternion extracted
    '''
    if (isinstance(q, dict) and 'w' in q.keys()):
        return q['x'], q['y'], q['z'], q['w']
    else:
        return q.x, q.y, q.z, q.w

def extv(v):
    ''' Extracts Point/Vector3 to Cartesian values
    Parameters:
        v (Point() or Vector3() or dict with 'x'..'z' in keys): From geometry_msgs.msg or dict
    Returns:
        [x,y,z] (Floats tuple[3]): Point/Vector3 extracted
    '''
    if (isinstance(v, dict) and 'x' in v.keys()):
        return v['x'], v['y'], v['z']
    else:
        return v.x, v.y, v.z

def extp(p):
    ''' Extracts pose
    Paramters:
        p (Pose())
    Returns:
        list (Float[7])
    '''
    return p.position.x, p.position.y, p.position.z, p.orientation.x, p.orientation.y, p.orientation.z, p.orientation.w
