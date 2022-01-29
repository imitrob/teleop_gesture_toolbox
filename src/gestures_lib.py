#!/usr/bin/env python3.8
'''
import gestures_lib as gl
gl.init()
Get info:
gl.gd.l.static.info.<g1>.threshold

Get latest data:
gl.gd.l.static[-1].<g1>.probability
(gestures_lib.GestureDataDetection.GestureDataHand.GestureMorphClass.Static.prob)

Get timestamped data:
gl.gd.l.static[<stamp> or index].<g1>.probability
(gestures_lib.GestureDataDetection.GestureDataHand[float or int].static.prob)
'''
import os, collections, time
import gdown
import numpy as np

import settings
if __name__ == '__main__': settings.init()
import os_and_utils.move_lib as ml
if __name__ == '__main__': ml.init()
from os_and_utils.parse_yaml import ParseYAML

if __name__ == '__main__':
    from inverse_kinematics.ik_lib import IK_bridge
    from os_and_utils.ros_communication_main import ROSComm

# Keep independency to ROS
try:
    import rospy
    from mirracle_gestures.msg import DetectionSolution, DetectionObservations
    from std_msgs.msg import Int8, Float64MultiArray
    ROS = True
except ModuleNotFoundError:
    ROS = False

def detection_thread(freq=1., args={}):
    if not ROS: raise Exception("ROS cannot be imported!")

    rospy.init_node("detection_thread", anonymous=True)

    settings.gesture_detection_on = True
    settings.launch_gesture_detection = True

    roscm = ROSComm()

    configGestures = ParseYAML.load_gesture_config_file(settings.paths.custom_settings_yaml)
    hand_mode_set = configGestures['using_hand_mode_set']
    hand_mode = dict(configGestures['hand_mode_sets'][hand_mode_set])

    rate = rospy.Rate(freq)

    while not rospy.is_shutdown():
        if ml.md.frames:
            send_g_data(roscm, hand_mode, args)

        rate.sleep()

def send_g_data(roscm, hand_mode, args):
    ''' Sends appropriate gesture data as ROS msg
        Launched node for static/dynamic detection.
    '''
    msg = DetectionObservations()
    msg.observations = Float64MultiArray()
    msg.sensor_seq = ml.md.frames[-1].seq
    msg.header.stamp.secs = ml.md.frames[-1].secs
    msg.header.stamp.nsecs = ml.md.frames[-1].nsecs

    for key in hand_mode.keys():
        if 'static' in hand_mode[key]:
            if key == 'l':
                if ml.md.l_present():
                    msg.observations.data = ml.md.frames[-1].l.get_learning_data_static(type=args['type'])
                    msg.header.frame_id = 'l'
                    roscm.static_detection_observations_pub.publish(msg)
            elif key == 'r':
                if ml.md.r_present():
                    msg.observations.data = ml.md.frames[-1].r.get_learning_data_static(type=args['type'])
                    msg.header.frame_id = 'r'
                    roscm.static_detection_observations_pub.publish(msg)

        if 'dynamic' in hand_mode[key]:
            if key == 'l':
                if ml.md.l_present():
                    msg.observations.data = [frame.l.get_single_learning_data_dynamic(type=args['type']) for frame in ml.md.frames.copy()]
                    msg.header.frame_id = 'l'
                    roscm.dynamic_detection_observations_pub.publish(msg)
            elif key == 'r':
                if ml.md.r_present():
                    msg.observations.data = [frame.r.get_single_learning_data_dynamic(type=args['type']) for frame in ml.md.frames.copy()]
                    msg.header.frame_id = 'r'
                    roscm.dynamic_detection_observations_pub.publish(msg)


## Class definitions
class GestureDataAtTime():
    ''' Same for static/dynamic, same for left/right hand
    '''
    def __init__(self, probability, biggest_probability=False):
        self.probability = probability
        # logic
        self.above_threshold = False
        self.biggest_probability = biggest_probability
        self.time_visible = 0.0  # [sec]

        self.activated = False
        self.first_activated = False

class GestureMorphClass(object):
    ''' Super cool class
        Initialization: gmc = GestureMorphClass()
                        gmc.<g1> = SomeGestureG1Data() # Assign data
                        gmc.<g2> = SomeGestureG2Data()
        Get number of gestures: gmc.n
        Get names of gestures: gmc.names
        Get gestures over id: gmc[0] # Get g1
                              gmc[1] # Get g2
                              gmc[:] # Get [g1, g2]
        Iterate over gestures: for g in gmc: # g1, g2
                                   g # g1, g2 in a loop

        Used placeholder for: 1. Info about gestures: gl.gd.l.static.info.<g1>.threshold
                                - StaticInfo(), DynamicInfo() classes
                              2. Get real data: gl.gd.l.static[<time>].<g1>.probability
                                - GestureDataAtTime() class
    '''
    def __init__(self):
        self.device = self

    def get_all_attributes(self):
        ''' Get all gestures
        '''
        l = list(self.__dict__.keys())
        l.remove('device')
        try: l.remove('header')
        except ValueError: pass

        return l

    def __iter__(self):
        attrs = self.get_all_attributes()
        objs = []
        for attr in attrs:
            objs.append(getattr(self,attr))
        return iter(objs)

    def __getattr__(self, attr):
        if attr == 'n':
            return len(self.get_all_attributes())
        elif attr == 'names':
            return self.get_all_attributes()
        elif attr == 'filenames':
            attrs = self.get_all_attributes()
            filenames = []
            for attr in attrs:
                filenames.append(getattr(self,attr).filename)
            return filenames
        elif attr == 'biggest_probability':
            attrs = self.get_all_attributes()
            for attr in attrs:
                if getattr(self,attr).biggest_probability == True:
                    return attr
            return 'Does not have biggest_probability set to True'
        elif attr == 'probabilities':
            attrs = self.get_all_attributes()
            ret = []
            for attr in attrs:
                ret.append(getattr(self,attr).probability)
            return ret
        elif attr == 'activates_int':
            attrs = self.get_all_attributes()
            ret = []
            for attr in attrs:
                ret.append(int(getattr(self,attr).activated))
            return ret
        elif attr == 'above_threshold_int':
            attrs = self.get_all_attributes()
            ret = []
            for attr in attrs:
                ret.append(int(getattr(self,attr).above_threshold))
            return ret
        elif attr == 'thresholds':
            attrs = self.get_all_attributes()
            ret = []
            for attr in attrs:
                ret.append(getattr(self,attr).thresholds)
            return ret

    def __getitem__(self, index):
        all_gesutres = self.get_all_attributes()
        if all_gesutres == []:
            return None
        attr = all_gesutres[index]

        ret = None
        if isinstance(attr, list):
            ret = [getattr(self,att) for att in attr]
        else:
            ret = getattr(self,attr)

        if ret == None:
            return []
        return ret


class GHeader():
    def __init__(self, stamp, seq, approach):
        self.stamp = stamp
        self.seq = seq
        self.approach = approach


class GestureMorphClassStamped(GestureMorphClass):
    def __init__(self, data, Gs):
        '''
        Parameters:
            data (DetectionSolutions)
                stamp (stamp): float64
                seq (int)
                approach (str): 'PyMC3', 'PyTorch', 'DTW'
        '''
        super().__init__()
        assert isinstance(data, DetectionSolution)
        stamp = data.header.stamp.secs + data.header.stamp.nsecs*10e-9
        self.header = GHeader(stamp, data.header.seq, data.approach)

        for n,g in enumerate(Gs):
            # gl.gd.l.static[<time>].<g> = GestureDataAtTime(probability, biggest_probability)
            setattr(self, g, GestureDataAtTime(data.probabilities.data[n], data.id == n))

class StaticInfo():
    def __init__(self, name, data=None):
        data = ParseYAML.parseStaticGesture(data)
        # info
        self.name = name
        self.filename = data['filename']
        # config
        if data['thresholds']:
            self.thresholds = data['thresholds']
        else: # Default thresholds
            self.thresholds = [0.9,0.9]

        if data['time_visible_threshold']:
            self.time_visible_threshold = data['time_visible_threshold']
        else: # Default value
            self.time_visible_threshold = 2.0

class DynamicInfo():
    def __init__(self, name, data=None):
        data = ParseYAML.parseDynamicGesture(data)
        # info
        self.name = name
        self.filename = data['filename']
        # for move_in_axis thresholds
        self.min_threshold = data['minthre']
        self.max_threshold = data['maxthre']
        ## move in x,y,z, Positive/Negative
        self.move = [False, False, False]

class TemplateGs():
    def __init__(self, type):
        '''
        Get info about gesture: gl.gd.l.static.info.<g1>
        Get data about gesture at time: gl.gd.l.static[<time index>].<g1>.probability
        '''
        self.info = GestureMorphClass()

        # fill the self.info with gesutre in yaml file: self.info.<g1>. ...
        configGestures = ParseYAML.load_gesture_config_file(settings.paths.custom_settings_yaml)
        gestures = ParseYAML.load_gestures_file(settings.paths.custom_settings_yaml, ret='obj')
        GsSet = gestures[configGestures['using_set']]
        for gesture in GsSet:
            if type == 'static':
                if 'static' in GsSet[gesture] and (GsSet[gesture]['static'] == 'true' or GsSet[gesture]['static'] == True):
                    setattr(self.info, gesture, StaticInfo(name=gesture, data=GsSet[gesture]))
            elif type =='dynamic':
                if 'static' in GsSet[gesture] and (GsSet[gesture]['static'] == 'true' or GsSet[gesture]['static'] == True):
                    pass
                else:
                    setattr(self.info, gesture, DynamicInfo(name=gesture, data=GsSet[gesture]))

        self.data_queue = collections.deque(maxlen=100)

    def get_times(self):
        return [g.stamp for g in self.data_queue]

    def find_nearest(self, array, value):
        array = np.asarray(array)
        idx = (np.abs(array - value)).argmin()
        return idx

    def by_seq(self, seq):
        for n, rec in self.data_queue:
            if seq == rec.header.seq:
                return self.data_queue[n]
        raise Exception("[Gesture Lib] Selecting data based on sequence: The seq. was not found!")

    def __getitem__(self, index):
        '''
            1. index == int -> search via normal index
            2. index == float -> search as time
        '''
        if isinstance(index, float):
            # times are values and index are pointing to gesture_queue
            times = self.get_times()
            closest_index = self.find_nearest(times, index)
            return self.data_queue[closest_index]
        elif isinstance(index, slice):
            dql = len(self.data_queue)
            # Handle None values
            if index.start == None: start = 0
            else: start = index.start
            if index.stop == None: stop = dql
            else: stop = index.stop
            # Make slice indexes positive and lower than seque length
            if start >= dql: start = dql
            elif -start >= dql: start = -dql
            if stop > dql: stop = dql
            elif -stop > dql: stop = -dql
            if start < 0: start = dql + start
            if stop < 0: stop = dql + stop

            # Handle floats -> it is time
            if isinstance(start, float) or isinstance(stop, float):
                times = self.get_times()
                # start/stop times to indexes
                start = self.find_nearest(times, start)
                stop = self.find_nearest(times, stop)
            ret = []
            for i in range(start, stop):
                ret.append(self.data_queue[i])
            return ret
        else:
            return self.data_queue[index]

    def relevant(self, last_secs=0.):
        ''' Searches gestures_queue, and returns the last gesture record, None if no records in that time were made
        '''
        if ROS: abs_time = rospy.Time.now().to_sec() - last_secs
        else: abs_time = time.time() - last_secs

        abs_time = time.time() - last_secs
        # if happened within last_secs interval
        if self.data_queue and self.data_queue[-1].header.stamp > abs_time:
            print( self.data_queue[-1].header.stamp, abs_time)
            return self.data_queue[-1]
        else:
            return None

class StaticGs(TemplateGs):
    def __init__(self, type='static'):
        super().__init__(type)

class DynamicGs(TemplateGs):
    def __init__(self, type='dynamic'):
        super().__init__(type)

class GestureDataHand():
    ''' The 2nd class: gl.gd.l
                       gl.gd.r
    '''
    def __init__(self):
        self.static = StaticGs()
        self.dynamic = DynamicGs()
        print(f"Gestures are: static: {self.static.info.names}, dynamic {self.dynamic.info.names}")

class GestureDataDetection():
    ''' The main class: gl.gd
    '''
    def __init__(self):
        self.l = GestureDataHand()
        self.r = GestureDataHand()
        ## TODO: Additional names for additional gestures computed by comparing left and right hand
        self.combs = None

        self.actions_queue = collections.deque(maxlen=30)

        # Auxiliary
        self.last_seq = 0

    def new_record(self, data, type='static'):
        ''' New gesture data arrived and will be saved
        '''
        if ml.md.frames[-1].seq-data.sensor_seq > 100:
            print(f"[Warning] Program cannot compute gs in time, probably rate is too big!")

        # choose hand with data.header.frame_id
        obj_by_hand = getattr(self, data.header.frame_id)
        # choose static/dynamic with arg: type
        obj_by_type = getattr(obj_by_hand, type)
        obj_by_type.data_queue.append(GestureMorphClassStamped(data, obj_by_type.info.names))

        # Add logic
        self.prepare_postprocessing(obj_by_type[-1], type, data.header.frame_id, data.sensor_seq)
    '''
    LOGIC
    '''
    def prepare_postprocessing(self, g_stamped, type, frame_id, sensor_seq):
        ''' High-level logic

        '''
        hand = getattr(self, frame_id)
        static = getattr(hand, type)
        latest_gs = static[-1]

        for n,g in enumerate(latest_gs):
            # 1. Probability over threshold
            if g.probability > static.info[n].thresholds[0]:
                g.above_threshold = True
            else:
                g.above_threshold = False

            # Right now it is pseudo time
            if g.above_threshold: g.time_visible = static[self.last_seq][n].time_visible + 1
            print("sensor seq", sensor_seq)
            self.last_seq = sensor_seq

            print(f"biggest_probability id {g.biggest_probability}, g.time_visible {g.time_visible}")
            # 1., 2. Biggest probability of all gestures and 3. gesture is visible for some time
            if g.above_threshold and g.biggest_probability and g.time_visible > static.info[n].time_visible_threshold:
                g.activate = True
            elif g.above_threshold == False:
                g.activate = False
            # If at least 10 gesture records were toggled (occured) and now it is not occuring
            if np.array([data[n].above_threshold for data in static[-10:]]).all() \
                and (static[-1].toggle == False):
                g.first_activated = True
                self.actions_queue.append((latest_gs.header.stamp, latest_gs[n]))
            else:
                g.first_activated = False

    def last(self):
        return self.l.static.relevant(), self.r.static.relevant(), self.l.dynamic.relevant(), self.r.dynamic.relevant()

    def var_generate(self, hand, type, g, time):
        vars = {}

        frames = ml.md.frames.get_window_last(1)
        # velocity
        hand_frame = getattr(frames[len(frames)//2], hand)
        vars['velocity'] = np.linalg.norm(hand_frame.palm_velocity())

        # TODO: others

        return vars

    '''
    Static manage methods
    '''
    @staticmethod
    def download_networks_gdrive():
        # get one dir above
        NETWORKS_PATH = '/'.join((settings.paths.network_path).split('/')[:-2])
        gdown.download_folder(settings.configGestures['networks_drive_url'], output=NETWORKS_PATH)

    @staticmethod
    def get_networks():
        ''' Looks at the settings.paths.network_path folder and returns every file with extension *.pkl
        '''
        networks = []
        for file in os.listdir(settings.paths.network_path):
            if file.endswith(".pkl"):
                networks.append(file)
        return networks

    '''
    Old deterministic approach
    '''
    @staticmethod
    def all():
        if settings.frames and settings.mo:
            GestureDataDetection.processTouches()
            GestureDataDetection.processOc()

            if 'grab' in settings.Gs: GestureDataDetection.processPose_grab()
            if 'pinch' in settings.Gs: GestureDataDetection.processPose_pinch()
            if 'point' in settings.Gs: GestureDataDetection.processPose_point()
            if 'respectful' in settings.Gs: GestureDataDetection.processPose_respectful()
            if 'spock' in settings.Gs: GestureDataDetection.processPose_spock()
            if 'rock' in settings.Gs: GestureDataDetection.processPose_rock()
            if 'victory' in settings.Gs: GestureDataDetection.processPose_victory()
            if 'italian' in settings.Gs: GestureDataDetection.processPose_italian()

            if 'move_in_axis' in settings.Gs: GestureDataDetection.processGest_move_in_axis()
            if 'rotation_in_axis' in settings.Gs: GestureDataDetection.processGest_rotation_in_axis()

            if 'move_in_axis' in settings.Gs: GestureDataDetection.processComb_goToConfig()

    @staticmethod
    def processTouches():
        fa = settings.frames[-1]
        if fa.r.visible:
            if fa.r.conf > gd.r.min_confidence:
                gd.r.conf = True
            else:
                gd.r.conf = False

            if fa.r.touch12 > settings.gd.r.touch_turn_on_dist[0] and settings.gd.r.conf:
                settings.gd.r.touch12 = False
            elif fa.r.touch12 < settings.gd.r.touch_turn_off_dist[0]:
                settings.gd.r.touch12 = True
            if fa.r.touch23 > settings.gd.r.touch_turn_on_dist[1] and settings.gd.r.conf:
                settings.gd.r.touch23 = False
            elif fa.r.touch23 < settings.gd.r.touch_turn_off_dist[1]:
                settings.gd.r.touch23 = True
            if fa.r.touch34 > settings.gd.r.touch_turn_on_dist[2] and settings.gd.r.conf:
                settings.gd.r.touch34 = False
            elif fa.r.touch34 < settings.gd.r.touch_turn_off_dist[2]:
                settings.gd.r.touch34 = True
            if fa.r.touch45 > settings.gd.r.touch_turn_on_dist[3] and settings.gd.r.conf:
                settings.gd.r.touch45 = False
            elif fa.r.touch45 < settings.gd.r.touch_turn_off_dist[3]:
                settings.gd.r.touch45 = True

            if fa.r.touch13 > settings.gd.r.touch_turn_on_dist[4] and settings.gd.r.conf:
                settings.gd.r.touch13 = False
            elif fa.r.touch13 < settings.gd.r.touch_turn_off_dist[4]:
                settings.gd.r.touch13 = True
            if fa.r.touch14 > settings.gd.r.touch_turn_on_dist[5] and settings.gd.r.conf:
                settings.gd.r.touch14 = False
            elif fa.r.touch14 < settings.gd.r.touch_turn_off_dist[5]:
                settings.gd.r.touch14 = True
            if fa.r.touch15 > settings.gd.r.touch_turn_on_dist[6] and settings.gd.r.conf:
                settings.gd.r.touch15 = False
            elif fa.r.touch15 < settings.gd.r.touch_turn_off_dist[6]:
                settings.gd.r.touch15 = True

    @staticmethod
    def processOc():
        fa = md.frames[-1]
        if fa.r.visible:
            gd = settings.gd.r
            if fa.r.conf > gd.min_confidence:
                gd.conf = True
            else:
                gd.conf = False

            for i in range(0,5):
                if fa.r.OC[i] > gd.oc_turn_on_thre[i] and gd.conf:
                    gd.oc[i] = True
                elif fa.r.OC[i] < gd.oc_turn_off_thre[i]:
                    gd.oc[i] = False

    @staticmethod
    def processPose_grab():
        fa = md.frames[-1]
        if fa.l.visible:
            gd = settings.gd.l
            g = gd.poses[gd.POSES["grab"]]
            g.prob = fa.l.grab
            # gesture toggle processing
            if fa.l.grab > g.TURN_ON_THRE:
                g.toggle = True
            elif fa.l.grab < g.TURN_OFF_THRE:
                g.toggle = False
        if fa.r.visible:
            gd = settings.gd.r
            g = gd.poses[gd.POSES["grab"]]
            g.prob = fa.r.grab
            # gesture toggle processing
            if fa.r.grab > g.TURN_ON_THRE and gd.conf:
                g.toggle = True
            elif fa.r.grab < g.TURN_OFF_THRE:
                g.toggle = False

    @staticmethod
    def processPose_pinch():
        fa = md.frames[-1]
        if fa.r.visible:
            gd = settings.gd.r
            g = gd.poses[gd.POSES["pinch"]]
            g.prob = fa.r.pinch
            if fa.r.pinch > g.TURN_ON_THRE and gd.conf:
                g.toggle = True
                g.time_visible += 0.01
            elif fa.r.pinch < g.TURN_OFF_THRE:
                g.toggle = False
                g.time_visible = 0.0

    @staticmethod
    def processPose_point():
        ''' touch, open/close functions need to be called before to get fingers O/C
        '''
        fa = md.frames[-1]
        if fa.r.visible:
            gd = settings.gd.r
            g = gd.poses[gd.POSES["point"]]
            if gd.oc[1] is True and gd.oc[2] is False and gd.oc[3] is False and gd.oc[4] is False:
                g.toggle = True
                g.time_visible += 0.01
            elif gd.oc[1] is False or gd.oc[3] is True or gd.oc[4] is True:
                g.toggle = False
                g.time_visible = 0.0

    @staticmethod
    def processPose_respectful():
        ''' touches, open/close functions need to be called before to get fingers O/C
        '''
        fa = md.frames[-1]
        if fa.r.visible:
            gd = settings.gd.r
            g = gd.poses[gd.POSES["respectful"]]
            if gd.oc[0] is False and gd.oc[1] is True and gd.oc[2] is True and gd.oc[3] is True and gd.oc[4] is False:
                g.toggle = True
                g.time_visible = 1
            elif gd.oc[0] is True or gd.oc[1] is False or gd.oc[2] is False or gd.oc[3] is False or gd.oc[4] is True:
                g.toggle = False

    @staticmethod
    def processPose_spock():
        ''' touches, open/close functions need to be called before to get fingers O/C
        '''
        fa = md.frames[-1]
        if fa.r.visible:
            gd = settings.gd.r
            g = gd.poses[gd.POSES["spock"]]
            if gd.oc[1] is True and gd.oc[2] is True and gd.oc[3] is True and gd.oc[4] is True and gd.touch23 is True and gd.touch34 is False and gd.touch45 is True:
                g.toggle = True
                g.time_visible = 1
            elif gd.oc[1] is False or gd.oc[2] is False or gd.oc[3] is False or gd.oc[4] is False or gd.touch23 is False or gd.touch34 is True or gd.touch45 is False:
                g.toggle = False

    @staticmethod
    def processPose_rock():
        ''' touches, open/close functions need to be called before to get fingers O/C
        '''
        fa = md.frames[-1]
        if fa.r.visible:
            gd = settings.gd.r
            g = gd.poses[gd.POSES["rock"]]
            if gd.oc[1] is True and gd.oc[4] is True and gd.oc[2] is False and gd.oc[3] is False:
                g.toggle = True
                g.time_visible = 1
            elif gd.oc[1] is False or gd.oc[2] is True or gd.oc[3] is True or gd.oc[4] is False:
                g.toggle = False

    @staticmethod
    def processPose_victory():
        ''' touches, open/close functions need to be called before to get fingers O/C
        '''
        fa = md.frames[-1]
        if fa.r.visible:
            gd = settings.gd.r
            g = gd.poses[gd.POSES["victory"]]
            if gd.oc[1] is True and gd.oc[2] is True and gd.oc[3] is False and gd.oc[4] is False and gd.oc[0] is False:
                g.toggle = True
                g.time_visible = 1
            elif gd.oc[1] is False or gd.oc[2] is False or gd.oc[3] is True or gd.oc[4] is True or gd.oc[0] is True:
                g.toggle = False

    @staticmethod
    def processPose_italian():
        ''' touches, open/close functions need to be called before to get fingers O/C
        '''
        fa = md.frames[-1]
        if fa.r.visible:
            gd = settings.gd.r
            g = gd.poses[gd.POSES["italian"]]
            if gd.touch12 is True and gd.touch23 is True and gd.touch34 is True and gd.touch45 is True:
                g.toggle = True
                g.time_visible = 1
            elif gd.touch12 is False or gd.touch23 is False or gd.touch34 is False or gd.touch45 is False:
                g.toggle = False

    @staticmethod
    def processComb_goToConfig():
        ''' touches, open/close functions need to be called before to get fingers O/C
        '''
        fa = md.frames[-1]
        g = settings.gd.r.dynamic.move_in_axis
        g_time = settings.gd.r.static.point.time_visible
        if g_time > 2:
            if g.toggle[0] and g.move[0]:
                settings.WindowState = 1
            if g.toggle[0] and not g.move[0]:
                settings.WindowState = 0


    @staticmethod
    def processGest_move_in_axis():
        '''
        '''
        fa = md.frames[-1]
        if fa.r.visible:
            gd = settings.gd.r
            g = gd.gests[gd.GESTS["move_in_axis"]]
            g_tmp = deepcopy(g.toggle)
            if abs(fa.r.vel[0]) > g.MIN_THRE and fa.r.vel[1] < g.MAX_THRE and fa.r.vel[2] < g.MAX_THRE:
                g.toggle[0] = True
                g.time_visible = 1
                g.move[0] = True if fa.r.vel[0] > g.MIN_THRE else False
                if g_tmp[0] == False:
                    settings.mo.gestureGoalPoseUpdate(0, g.move[0])
            else:
                g.toggle[0] = False
            if abs(fa.r.vel[1]) > g.MIN_THRE and fa.r.vel[0] < g.MAX_THRE and fa.r.vel[2] < g.MAX_THRE:
                g.time_visible = 1
                g.toggle[1] = True
                g.move[1] = True if fa.r.vel[1] > g.MIN_THRE else False
                if g_tmp[1] == False:
                    settings.mo.gestureGoalPoseUpdate(1, g.move[1])
            else:
                g.toggle[1] = False
            if abs(fa.r.vel[2]) > g.MIN_THRE and fa.r.vel[0] < g.MAX_THRE and fa.r.vel[1] < g.MAX_THRE:
                g.time_visible = 1
                g.toggle[2] = True
                g.move[2] = True if fa.r.vel[2] > g.MIN_THRE else False
                if g_tmp[2] == False:
                    settings.mo.gestureGoalPoseUpdate(2, g.move[2])
            else:
                g.toggle[2] = False


    @staticmethod
    def processGest_rotation_in_axis():
        '''
        '''
        fa = md.frames[-1]
        if fa.r.visible:
            euler = fa.r.pRaw[3:6]
            gd = settings.gd.r
            g = gd.gests[gd.GESTS["rotation_in_axis"]]
            g_tmp = deepcopy(g.toggle)
            if (euler[0] > g.MAX_THRE[0] or euler[0] < g.MIN_THRE[0]):
                g.time_visible = 1
                g.toggle[0] = True
                g.move[0] = True if euler[0] > g.MAX_THRE[0] else False
                if g_tmp[0] == False:
                    settings.mo.gestureGoalPoseRotUpdate(0, g.move[0])
            else:
                g.toggle[0] = False
            if (euler[1] > g.MAX_THRE[1] or euler[1] < g.MIN_THRE[1]):
                g.toggle[1] = True
                g.time_visible = 1
                g.move[1] = True if euler[1] > g.MAX_THRE[1] else False
                if g_tmp[1] == False:
                    settings.mo.gestureGoalPoseRotUpdate(1, g.move[1])
            else:
                g.toggle[1] = False
            if (euler[1] > g.MAX_THRE[2] or euler[1] < g.MIN_THRE[2]):
                g.toggle[2] = True
                g.time_visible = 1
                g.move[2] = True if euler[2] > g.MAX_THRE[2] else False
                if g_tmp[2] == False:
                    settings.mo.gestureGoalPoseRotUpdate(2, g.move[2])
            else:
                g.toggle[2] = False

def init():
    global gd
    gd = GestureDataDetection()

#### For testing purposes

def main():
    init()
    detection_thread(freq = 1., args={'type': 'old_defined'})

if __name__ == '__main__':
    main()


#
