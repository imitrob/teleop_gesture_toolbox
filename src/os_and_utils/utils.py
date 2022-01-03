import sys, os, yaml, inspect
from collections import OrderedDict

def ros_enabled():
    try:
        from mirracle_gestures.msg import Frame as Framemsg
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

        self.learn_path = MG_PATH+'/include/data/learning/'
        self.graphics_path = MG_PATH+'/include/graphics/'
        self.plots_path = MG_PATH+'/include/plots/'
        self.network_path = MG_PATH+'/include/data/Trained_network/'
        self.models_path = MG_PATH+'/include/models/'
        self.custom_settings_yaml = MG_PATH+'/include/custom_settings/'
        TMP2 = os.path.abspath(os.path.join(MG_PATH, '..'))
        self.coppelia_scene_path = TMP2+"/mirracle_sim/include/scenes/"
        if change_working_directory:
            os.chdir(MG_PATH+'/src')

def load_params():
    if ros_enabled():
        try:
            rospy
        except NameError:
            import rospy

        robot = rospy.get_param("/mirracle_config/robot", 'panda')
        simulator = rospy.get_param("/mirracle_config/simulator", 'coppelia')
        gripper = rospy.get_param("/mirracle_config/gripper", 'none')
        plot = rospy.get_param("/mirracle_config/visualize", 'false')
        inverse_kinematics = rospy.get_param("/mirracle_config/ik_solver", 'relaxed_ik')
        inverse_kinematics_topic = rospy.get_param("/mirracle_config/ik_topic", '')
        gesture_detection_on = rospy.get_param("/mirracle_config/launch_gesture_detection", 'false')
        launch_gesture_detection = rospy.get_param("/mirracle_config/launch_gesture_detection", 'false')
        launch_ui = rospy.get_param("/mirracle_config/launch_ui", 'false')
    else:
        robot = 'panda'
        simulator = 'coppelia'
        gripper = 'none'
        plot = 'false'
        inverse_kinematics = 'relaxed_ik'
        inverse_kinematics_topic = ''
        gesture_detection_on = 'false'
        launch_gesture_detection = 'false'
        launch_ui = 'false'
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

def distancePoses(self, p1, p2):
    ''' Returns distance between two pose objects
    Parameters:
        pose1 (type Pose() from geometry_msgs.msg)
        pose2 (type Pose() from geometry_msgs.msg)
    Returns:
        distance (Float)
    '''
    p1 = p1.position
    p2 = p2.position
    return np.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2 + (p1.z - p2.z)**2)


#
