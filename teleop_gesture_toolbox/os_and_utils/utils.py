import sys, os, yaml, inspect, itertools, collections
import numpy as np
from collections import OrderedDict, Counter
from scipy.signal import argrelextrema


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
    return p.position.x, p.position.y, p.position.z, p.reject_outliersorientation.x, p.orientation.y, p.orientation.z, p.orientation.w

def reject_outliers(data, m=2):
    return data[abs(data - np.mean(data)) < m * np.std(data)]

def get_local_min_and_max(arr):
    return (argrelextrema(arr, np.greater),argrelextrema(arr, np.less))

def get_median_extreme(arr):
    l = len(arr)
    gr, ls = get_local_min_and_max(arr)
    extrems = (*gr,*ls)

    min(extrems)

'''
>>> # Start with zeros, then move to true value 0.05, finally go back to zeros
>>> test_data = [0.01,0.02,0.01,0.02,0.04,0.05,0.051,0.02,0.01,0.00,0.01]
>>> # const
>>> test_data = [0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5]
>>> # Start with the right value, then move out
>>> test_data = [0.05,0.051,0.049,0.052,0.02,0.01,0.001]

>>> a = np.array([0.,0.5,1.0,1.5,2.2,2.3,2.4,2.0,1.4,1.0,0.0,0.1,0.2,0.5,0.3])
>>> x = np.array(range(len(a)))
>>> get_local_min_and_max(a)
>>> import matplotlib.pyplot as plt
>>> plt.plot(test_data)
'''
def get_dist_by_extremes(y, bufferlen=100, fit_deg=6):
    # y = test_data
    # 1. Fit the data
    x = np.array(range(len(y)))
    polyvals = np.polyfit(x, y, deg=fit_deg)
    # 2. Gen. new fitted data
    x_ = np.linspace(0,len(x),bufferlen)
    y_ = [np.polyval(polyvals, i) for i in x_]
    ''' # possible plot
    plt.plot(x_,y_)
    plt.plot(y)
    '''
    # 3. Get extremes and pick the closest extreme to center
    extremes = np.hstack(get_local_min_and_max(np.array(y_)))[0]
    if len(extremes) == 0: return np.array(y).mean()
    pt = min(extremes-np.array(bufferlen//2), key=abs)+bufferlen//2

    dist = y[round(pt/100*len(x))]
    return dist

'''
fit_degs = [2,3,4,5,6,8,10]
res = []
for fit_deg in fit_degs:
    res.append(get_dist_by_extremes(test_data, fit_deg=fit_deg))
plt.plot(fit_degs, res)
'''


class CustomDeque(collections.deque):
    def __init__(self, *args, **kwargs):
        super().__init__(self, *args, **kwargs)

    def __getitem__(self, slic):
        if len(self) == 0: return None
        if isinstance(slic, slice):
            start, stop, step = slic.start, slic.stop, slic.step
            #print(f"start {start}")
            if isinstance(start, int) and start < 0: start = len(self)+start
            if isinstance(stop, int) and stop < 0: stop = len(self)+stop
            #print(f"start {start}")
            if start is not None: start = np.clip(start, 0, len(self)-1)
            if stop is not None: stop = np.clip(stop, 0, len(self)-1)
            #print(f"start {start} stop {stop}")
            return list(itertools.islice(self, start, stop, step))
            return collections.deque(itertools.islice(self, start, stop, step))
        else:
            try:
                return super(CustomDeque, self).__getitem__(slic)
            except IndexError:
                return None

    def get_last(self, nlast):
        assert nlast>0
        return self[-nlast:]

    def get_last_with_delay(self, nlast, delay):
        assert delay>=0
        assert nlast>0
        return self[-nlast-delay-1:-delay-1]

    def to_ids(self, seq):
        return seq #[i for i in seq]

    def get_last_common(self, nlast, threshold=0.0):
        last_seq = self.to_ids(self.get_last(nlast))
        if last_seq is None: return None
        most_common = Counter(last_seq).most_common(1)[0]
        if float(most_common[1]) / nlast >= threshold:
            return most_common[0]
        else:
            return None

    def get_last_commons(self, nlast, threshold=0.0, most_common=2, delay=0):
        last_seq = self.to_ids(self.get_last_with_delay(nlast, delay))
        most_commons = Counter(last_seq).most_common(2)
        n = len(last_seq)
        for i in range(len(most_commons)):
            most_commons[i] = (most_commons[i][0], most_commons[i][1]/n)

        ret = []
        for i in range(len(most_commons)):
            #if most_commons[i][1] >= threshold:
            ret.append(most_commons[i][0])
        return ret

'''
class CustomCounter():
    def __init__(self, data):
        self.data = data
    def most_common(self, n, fromkey=1):
        counts = {}
        for d in self.data:
            counts[d[fromkey]] += 1

counts = {'a':3, 'b':2, 'c':1, 'e':3}

for i in range(n):
    m = max(counts)
    counts.pop(m)
'''

class GestureQueue(CustomDeque):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def to_ids(self, seq):
        return [i[1] for i in seq]

def customdeque_tester():
    d = CustomDeque(maxlen=5)
    d.append(1)
    d.append(2)
    d.append(3)
    d.append(4)
    d.append(5)
    d.append(6)
    d
    assert d[0] == 2
    d[-1:]
    assert d[-1] == 6
    d[-1:]
    assert d[-1:] == [6]
    d[-2:]
    assert d[-2:] == [5,6]
    d[:]
    assert d[:] == [2,3,4,5,6]
    d[0:]
    assert d[0:] == [2,3,4,5,6]

    d.get_last(3)

    d.append(1)
    d.append(1)
    d
    assert d.get_last_common(3) == 1
    assert d.get_last_common(3, threshold=0.7) is None
    d
    d.get_last_common(20, threshold=0.0)

    d.append(6)
    d.append(6)
    d.append(1)
    d
    assert d.get_last_common(3, threshold=0.0) == 6
    assert d.get_last_common(5, threshold=0.0) == 1


    d = CustomDeque(maxlen=5)
    assert d[-1] is None
    assert d[0] is None
    assert d[100] is None

    assert d[-10:] is None
    assert d.get_last(1) is None

    d.get_last_common(5, threshold=0.0)

    d = CustomDeque(maxlen=5)
    d.append(1)
    d.append(1)
    d.append(2)
    d.append(2)
    d.append(3)
    d.append(3)
    d.append(3)
    d.append(3)
    d.append(4)
    d.append(5)
    d.append(6)
    d.get_last_commons(20, threshold=0.0)



    d[-20:]

    gq = GestureQueue(maxlen=5)
    gq.append((123,'abc',567))
    gq.append((124,'dcb',678))
    gq.append((124,'abc',678))
    gq.append((124,'abc',678))
    gq.get_last_common(10, threshold=0.0)
