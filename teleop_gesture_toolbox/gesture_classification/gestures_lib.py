#!/usr/bin/env python3
'''
import gesture_classification.gestures_lib as gl
gl.init()
Get info:
gl.gd.static_info().<g1>.thresholds

Get latest data:
gl.gd.l.static[-1].<g1>.probability
(gestures_lib.GestureDataDetection.GestureDataHand.GestureMorphClass.Static.prob)

Get timestamped data:
gl.gd.l.static[<stamp> or index].<g1>.probability
(gestures_lib.GestureDataDetection.GestureDataHand[float or int].static.prob)
'''
import os, collections, time
try: import gdown
except ModuleNotFoundError: gdown = None
import numpy as np
from copy import deepcopy

from os_and_utils import settings
from os_and_utils.nnwrapper import NNWrapper
if __name__ == '__main__': settings.init()
import os_and_utils.move_lib as ml
if __name__ == '__main__': ml.init()
from os_and_utils.utils import cc

from os_and_utils.parse_yaml import ParseYAML

if __name__ == '__main__':
    from os_and_utils.ros_communication_main import ROSComm

# Keep independency to ROS
try:
    import rclpy
    from teleop_gesture_toolbox.msg import DetectionSolution, DetectionObservations
    from std_msgs.msg import Int8, Float64MultiArray, MultiArrayDimension, String
    import os_and_utils.ros_communication_main as rc
    if __name__ == '__main__': rc.init()
    ROS = True
except ModuleNotFoundError:
    ROS = False

def NormalizeData(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))

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
        self.action_activated = False

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

        Used placeholder for: 1. Info about gestures: gl.gd.static_info().<g1>.threshold
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
        elif attr == 'record_keys':
            attrs = self.get_all_attributes()
            filenames = []
            for attr in attrs:
                filenames.append(getattr(self,attr).record_key)
            return filenames
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
        elif attr == 'biggest_probability_id':
            attrs = self.get_all_attributes()
            for n,attr in enumerate(attrs):
                if getattr(self,attr).biggest_probability == True:
                    return n
            return 'Does not have biggest_probability set to True'
        elif attr == 'probabilities':
            attrs = self.get_all_attributes()
            ret = []
            for attr in attrs:
                ret.append(getattr(self,attr).probability)
            return ret
        elif attr == 'probabilities_norm':
            attrs = self.get_all_attributes()
            ret = []
            for attr in attrs:
                ret.append(getattr(self,attr).probability)
            return NormalizeData(ret)
        elif attr == 'activates':
            attrs = self.get_all_attributes()
            ret = []
            for attr in attrs:
                ret.append(getattr(self,attr).activated)
            return ret
        elif attr == 'activate_id':
            attrs = self.get_all_attributes()
            for n,attr in enumerate(attrs):
                if getattr(self,attr).activated:
                    return n
            return None
        elif attr == 'action_activate_id':
            attrs = self.get_all_attributes()
            for n,attr in enumerate(attrs):
                if getattr(self,attr).action_activated:
                    return n
            return None
        elif attr == 'activate_name':
            attrs = self.get_all_attributes()
            for attr in attrs:
                if getattr(self,attr).activated:
                    return attr
            return None
        elif attr == 'above_threshold':
            attrs = self.get_all_attributes()
            ret = []
            for attr in attrs:
                ret.append(getattr(self,attr).above_threshold)
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

    def __getstate__(self): return self.__dict__
    def __setstate__(self, d): self.__dict__.update(d)

class GHeader():
    def __init__(self, stamp, seq, approach):
        self.stamp = stamp
        self.seq = seq
        self.approach = approach


class GestureMorphClassStamped(GestureMorphClass):
    def __init__(self, data, Gs):
        ''' New detection record
        Parameters:
            data (DetectionSolutions)
                stamp (stamp): float64
                seq (int)
                approach (str): 'PyMC3', 'PyTorch', 'DTW'
        '''
        super().__init__()
        assert isinstance(data, DetectionSolution)
        stamp = data.header.stamp.sec + data.header.stamp.nanosec*1e-9
        self.header = GHeader(stamp, data.seq, data.approach)

        for n,g in enumerate(Gs):
            # gl.gd.l.static[<time>].<g> = GestureDataAtTime(probability, biggest_probability)
            setattr(self, g, GestureDataAtTime(data.probabilities.data[n], data.id == n))

class StaticInfo():
    def __init__(self, name, data=None):
        data = ParseYAML.parseStaticGesture(data)
        # info
        self.name = name
        self.record_key = data['record_key']
        self.filename = data['filename']
        # config
        if data['thresholds']:
            self.thresholds = data['thresholds']
        else: # Default thresholds
            self.thresholds = [0.9,0.9]

        if data['time_visible_threshold']:
            self.time_visible_threshold = data['time_visible_threshold']
        else: # Default value
            self.time_visible_threshold = 8.0

class DynamicInfo():
    def __init__(self, name, data=None):
        data = ParseYAML.parseDynamicGesture(data)
        # info
        self.name = name
        self.record_key = data['record_key']
        self.filename = data['filename']
        # for move_in_axis thresholds
        if data['thresholds']:
            self.thresholds = data['thresholds']
        else: # Default thresholds
            self.thresholds = [0.9,0.9]
        ## move in x,y,z, Positive/Negative
        self.move = [False, False, False]

        if data['time_visible_threshold']:
            self.time_visible_threshold = data['time_visible_threshold']
        else: # Default value
            self.time_visible_threshold = 8.0

class MotionPrimitiveInfo():
    def __init__(self, name, data=None):
        data = ParseYAML.parseDynamicGesture(data)
        # info
        self.name = name
        self.record_key = data['record_key']
        self.filename = data['filename']
        # for move_in_axis thresholds
        if data['thresholds']:
            self.thresholds = data['thresholds']
        else: # Default thresholds
            self.thresholds = [0.9,0.9]
        ## move in x,y,z, Positive/Negative
        self.move = [False, False, False]

class TemplateGs():
    def __init__(self, type, local_data=None):
        '''
        Get info about gesture: gl.gd.static_info().<g1>
        Get data about gesture at time: gl.gd.l.static[<time index>].<g1>.probability
        '''
        self.info = GestureMorphClass()

        if local_data:
            if local_data.type == type:
                if local_data.type == 'static':
                    for i in range(len(local_data.Gs)):
                        data = { 'record_key': local_data.record_keys[i], 'filename': local_data.filenames[i] }
                        setattr(self.info, local_data.Gs[i], StaticInfo(name=local_data.Gs[i], data=data))
                elif local_data.type == 'dynamic':
                    for i in range(len(local_data.Gs)):
                        data = { 'record_key': local_data.record_keys[i], 'filename': local_data.filenames[i] }
                        setattr(self.info, local_data.Gs[i], DynamicInfo(name=local_data.Gs[i], data=data))
        else:
            # fill the self.info with gesutre in yaml file: self.info.<g1>. ...
            configGestures = ParseYAML.load_gesture_config_file(settings.paths.custom_settings_yaml)
            gestures = ParseYAML.load_gestures_file(settings.paths.custom_settings_yaml, ret='obj')
            GsSet = gestures[configGestures['using_config']]
            for gesture in GsSet:
                type_ = ParseYAML.parseGestureType(GsSet[gesture])
                if type == type_:
                    if type == 'static':
                        setattr(self.info, gesture, StaticInfo(name=gesture, data=GsSet[gesture]))
                    elif type == 'dynamic':
                        setattr(self.info, gesture, DynamicInfo(name=gesture, data=GsSet[gesture]))
                    elif type == 'mp':
                        setattr(self.info, gesture, MotionPrimitiveInfo(name=gesture, data=GsSet[gesture]))

        self.data_queue = collections.deque(maxlen=300)

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
            try:
                return self.data_queue[index]
            except IndexError: return
                #print(f"Index {index} not found from len {len(self.data_queue)}")

    def __getattr__(self, attr):
        if attr == 'n':
            return len(self.data_queue)

    def __getstate__(self): return self.__dict__
    def __setstate__(self, d): self.__dict__.update(d)

    def relevant(self, last_secs=1.0):
        ''' Searches gestures_queue, and returns the last gesture record, None if no records in that time were made
        '''
        '''if ROS:
            abs_time = rc.roscm.get_time() - last_secs
        else:'''
        abs_time = time.time() - last_secs

        #abs_time = time.time() - last_secs
        # if happened within last_secs interval
        if self.data_queue and self.data_queue[-1].header.stamp > abs_time:
            return self.data_queue[-1]
        else:
            return None

class StaticGs(TemplateGs):
    def __init__(self, type='static', local_data=None):
        super().__init__(type, local_data=local_data)

class DynamicGs(TemplateGs):
    def __init__(self, type='dynamic', local_data=None):
        super().__init__(type, local_data=local_data)

class MotionPrimitiveGs(TemplateGs):
    def __init__(self, type='mp', local_data=None):
        super().__init__(type, local_data=local_data)

class GestureDataHand():
    ''' The 2nd class: gl.gd.l
                       gl.gd.r
    '''
    def __init__(self):
        self.static = StaticGs()
        self.dynamic = DynamicGs()
        self.mp = MotionPrimitiveGs()

class GestureDataDetection():
    ''' The main class: gl.gd
    '''
    def __init__(self, silent=False):
        self.l = GestureDataHand()
        self.r = GestureDataHand()
        if not silent:
            print(f"Static gestures: {self.static_info().names}, \nDynamic gestures {self.dynamic_info().names}\nMotionPrimitive ids {self.l.mp.info.names}")
        ## TODO: Additional names for additional gestures computed by comparing left and right hand
        self.combs = None

        # TEMP g data:
        self.experimental_move_in_axis_toggle = [False, False, False]

        self.gestures_queue = collections.deque(maxlen=50)

        self.static_network_info, self.dynamic_network_info = self.load_netork_info()
        # Auxiliary
        self.last_seq = 0
        self.rate = settings.yaml_config_gestures['misc']['rate']
        self.activate_length=None # for export only
        self.distance_length=None # for export only

        self.print_error_once_1 = True

        # Save of actions
        self.action_saves = []

        # Gesture prediction
        self.static_gesture_action_prediction = [""] * self.static_info().n
        self.dynamic_gesture_action_prediction = [""] * self.dynamic_info().n

    def static_info(self):
        ''' Note: info is saved both in self.l and self.r as the same
        '''
        return self.l.static.info
    def dynamic_info(self):
        return self.l.dynamic.info
    def mp_info(self):
        return self.l.mp.info

    def load_netork_info(self):
        static_network_file = settings.get_network_file(type='static')
        dynamic_network_file = settings.get_network_file(type='dynamic')

        if static_network_file is not None:
            from os_and_utils.nnwrapper import NNWrapper
            nw = NNWrapper.load_network(settings.paths.network_path, static_network_file)
            static_network_info = nw.args
            static_network_info['accuracy'] = nw.accuracy
        else:
            static_network_info = None

        if dynamic_network_file is not None:
            #nw = NNWrapper.load_network(settings.paths.network_path, dynamic_network_file)
            #dynamic_network_info = nw.args
            dynamic_network_info = {'input_definition_version': 0, 'scene_frame':1}
        else:
            dynamic_network_info = None

        return static_network_info, dynamic_network_info

    def relevant(self, time_now, hand='r', type='static', relevant_time=1.0):
        ''' Returns solutions based on hand and type if not older than relevant_time
        '''
        '''if ROS:
            t = rc.roscm.get_time()
        else: '''
        t = time.time()

        self_h = getattr(self, hand)
        self_h_type = getattr(self_h, type)
        try:
            if self_h_type.n > 0 and (t-self_h_type[-1].stamp) < relevant_time:
                return self_h_type[-1]
            else:
                return None
        except:
            return None


    def gesture_change_srv(self, local_data):
        if local_data.type == 'static':
            self.l.static = StaticGs(local_data=local_data)
            self.r.static = StaticGs(local_data=local_data)
        elif local_data.type == 'dynamic':
            self.l.dynamic = DynamicGs(local_data=local_data)
            self.r.dynamic = DynamicGs(local_data=local_data)

        self.static_network_info, self.dynamic_network_info = self.load_netork_info()

    def __getattr__(self, attr):
        if attr == 'Gs':
            Gs = self.l.static.info.names
            Gs.extend(self.l.dynamic.info.names)
            return Gs
        elif attr == 'GsExt':
            Gs = self.l.static.info.names
            Gs.extend(self.l.dynamic.info.names)
            Gs.extend(self.l.mp.info.names)
            return Gs
        elif attr == 'Gs_static':
            return self.l.static.info.names
        elif attr == 'Gs_dynamic':
            return self.l.dynamic.info.names
        elif attr == 'Gs_keys':
            Gs = self.l.static.info.record_keys
            Gs.extend(self.l.dynamic.info.record_keys)
            return Gs
        elif attr == 'GsExt_keys':
            Gs = self.l.static.info.record_keys
            Gs.extend(self.l.dynamic.info.record_keys)
            Gs.extend(self.l.mp.info.record_keys)
            return Gs
        elif attr == 'Gs_keys_static':
            return self.l.static.info.record_keys
        elif attr == 'Gs_keys_dynamic':
            return self.l.dynamic.info.record_keys

    def __getstate__(self): return self.__dict__
    def __setstate__(self, d): self.__dict__.update(d)

    def new_record(self, data, type='static'):
        ''' New gesture data arrived and will be saved
        '''
        if ml.md.frames[-1].seq-data.sensor_seq > 100:
            print(f"[Warning] Program cannot compute gs in time, probably rate is too big! (or fake data are used)")

        # choose hand with data.header.frame_id
        obj_by_hand = getattr(self, data.header.frame_id)
        # choose static/dynamic with arg: type
        obj_by_type = getattr(obj_by_hand, type)

        if len(data.probabilities.data) != obj_by_type.info.n:
            if self.print_error_once_1:
                print(f"Data received from {type} detection, lengths not match {len(data.probabilities.data)} != {obj_by_type.info.n}")
                print(f"Data received from {type} detection, lengths not match {len(data.probabilities.data)} != {obj_by_type.info.n}")
                print(f"Data received from {type} detection, lengths not match {len(data.probabilities.data)} != {obj_by_type.info.n}")
                self.print_error_once_1 = False
            return
        obj_by_type.data_queue.append(GestureMorphClassStamped(data, obj_by_type.info.names))

        # Add logic
        self.prepare_postprocessing(data.header.frame_id, type, data.sensor_seq)
    '''
    LOGIC
    '''
    def prepare_postprocessing(self, hand_tag, type, sensor_seq, activate_length=1.0, activate_length_dynamic=0.3, distance_length=3.0):
        ''' High-level logic
        Parameters:
            activate_length (Int): [seconds] Time length when gesture is activated in order to make action
            activate_length_dynamic (Float): [seconds] -||- but dynamic gestures
            distance_length (Int): [seconds] Length between the activate_action
        '''
        self.distance_length = distance_length
        if type == 'static':
            self.activate_length = activate_length
        else:
            self.activate_length = activate_length_dynamic
        distance_length = int(self.rate * self.distance_length) # seconds -> seqs
        activate_length = int(self.rate * self.activate_length) # seconds -> seqs

        hand = getattr(self, hand_tag)
        gs = getattr(hand, type)
        latest_gs = gs[-1]
        probabilities = latest_gs.probabilities_norm
        for n,g in enumerate(latest_gs):
            probability = probabilities[n]
            # 1. Probability over threshold
            if probability > gs.info[n].thresholds[0]:
                g.above_threshold = True
            else:
                g.above_threshold = False

            # Right now it is pseudo time
            if g.above_threshold and gs.n>1: g.time_visible = gs[-2][n].time_visible + 1

            # Activate should refer to evaluation within one time sample
            # 1., 2. Biggest probability of all gestures and 3. gesture is visible for some times


            #print("g.above_threshold", g.above_threshold, "g.biggest_probability", g.biggest_probability, "time_visible > time_visible_threshold", g.time_visible > gs.info[n].time_visible_threshold, "time_visible", g.time_visible, "gs.info[n].time_visible_threshold", gs.info[n].time_visible_threshold)

            if g.above_threshold and g.biggest_probability and g.time_visible > gs.info[n].time_visible_threshold:
                g.activated = True
            elif g.above_threshold == False:
                g.activated = False

            # Activate first is higher layer that evaluates more frames and add action to queue
            # If at least 10 gesture records were toggled (occured) and now it is not occuring
            g.action_activated = False
            if gs.n > activate_length: # on start
                if np.array([data[n].activated for data in gs[-activate_length-2:-1]]).all(): # gesture was shown with no interruption
                    if ( True or #gs[-1][n].activated == False or   # now it ended -> action can happen
                        (np.array([data[n].activated for data in gs[-activate_length-2:-1]]).all() and not np.array([data[n].action_activated for data in gs[-distance_length*2-2:-1]]).any())): # or gesture happening for a long time
                        if not np.array([data[n].action_activated for data in gs[-distance_length-2:-1]]).any():
                            g.action_activated = True
                            self.gestures_queue.append((latest_gs.header.stamp, gs.info.names[n], hand_tag))

    def gestures_queue_to_ros(self, rostemplate):
        '''
        GesturesRos()
        '''
        rostemplate.probabilities.data = list(np.array(np.zeros(len(self.Gs)), dtype=float))
        for g in self.gestures_queue:
            rostemplate.probabilities.data[self.Gs.index(g[1])] = 1.0

        return rostemplate

    def last(self):
        return self.l.static.relevant(), self.r.static.relevant(), self.l.dynamic.relevant(), self.r.dynamic.relevant()

    def var_generate(self, hand, stamp):
        vars = {'velocities':[], 'point_direction':[], 'palm_euler': [], 'grab': [], 'pinch': [], 'holding_sphere_radius': []}

        last_secs = settings.yaml_config_gestures['misc']['relevant_time']

        frames = ml.md.get_frame_window_of_last_secs(stamp, last_secs)

        # go through frames and get vars
        for frame in frames:
            hand_frame = getattr(frame, hand)
            vars['velocities'].append(np.linalg.norm(hand_frame.palm_velocity()))
            vars['point_direction'].append(hand_frame.index_direction())
            vars['palm_euler'].append(hand_frame.get_palm_euler())
            vars['grab'].append(hand_frame.grab_strength)
            vars['pinch'].append(hand_frame.pinch_strength)
            vars['holding_sphere_radius'].append(hand_frame.sphere_radius)

        return vars

    def export(self):
        ''' Exports Selected gesture data as new folder, add metadata + plot
        '''
        N_ = 0
        while os.path.isdir(settings.paths.data_export_path+'record_'+str(N_)) == True:
            N_+=1
        os.mkdir(settings.paths.data_export_path+'record_'+str(N_))

        # Data composition
        normalizer = None
        if self.l.static[0] and self.r.dynamic[0]:
            id = np.argmin((self.l.static[0].header.stamp, self.r.dynamic[0].header.stamp))
            normalizer = (self.l.static[0].header.stamp, self.r.dynamic[0].header.stamp)[id]
        if self.l.static[0]:
            if not normalizer: normalizer = self.l.static[0].header.stamp
            norm_seq = self.l.static[0].header.seq
            gesture_data_left_static = np.array([(g.header.stamp-normalizer, g.header.seq-norm_seq, *g.probabilities, *g.activates) for g in self.l.static[:]])
            #gesture_data_left_dynamic = np.array([(g.stamp, g.header.seq, *g.probabilities, *g.probabilities_norm, *g.activates) for g in self.l.dynamic[:]])
            #gesture_data_right_static = np.array([(g.stamp, g.header.seq, *g.probabilities, *g.activates) for g in self.r.static[:]])
        if self.r.dynamic[0]:
            if not normalizer: normalizer = self.r.dynamic[0].header.stamp
            norm_seq = self.r.dynamic[0].header.seq
            gesture_data_right_dynamic = np.array([(g.header.stamp-normalizer, g.header.seq-norm_seq, *g.probabilities, *g.probabilities_norm, *g.activates) for g in self.r.dynamic[:]])

        # Motion Primitives Actions Composition
        if self.action_saves:
            np.save(settings.paths.data_export_path+'record_'+str(N_)+"/executed_MPs.npy", self.action_saves)

        #gesture_data_left_static = gesture_data_left_static[gesture_data_left_static[:, 0].argsort()]
        #gesture_data_right_dynamic = gesture_data_right_dynamic[gesture_data_right_dynamic[:, 0].argsort()]

        # Data save
        if self.l.static[0]:
            np.save(settings.paths.data_export_path+'record_'+str(N_)+"/gesture_data_left_static.npy", gesture_data_left_static)
        if self.r.dynamic[0]:
            np.save(settings.paths.data_export_path+'record_'+str(N_)+"/gesture_data_right_dynamic.npy", gesture_data_right_dynamic)

        file = open(settings.paths.data_export_path+'record_'+str(N_)+"/config.txt","a")
        file.write(f"[Static] Detection approach: {settings.get_detection_approach(type='static')}, Network file {settings.get_network_file(type='static')}\n")
        file.write(f"[Dynamic] Detection approach: {settings.get_detection_approach(type='dynamic')}, Network file {settings.get_network_file(type='dynamic')}\n")
        file.write(f"[Mapping] Left hand: {settings.get_hand_mode()['l']}, Right hand: {settings.get_hand_mode()['r']}\n\n")

        file.write(f"Static network info: {self.static_network_info}\n")
        file.write(f"Dynamic network info: {self.dynamic_network_info}")
        file.write(f" {dict(settings.yaml_config_gestures['misc_network_args'])}\n\n")

        file.write(f"{self.static_info().n} static gestures: {self.static_info().names}\n")
        file.write(f"{self.dynamic_info().n} dynamic gestures: {self.dynamic_info().names}\n")
        file.write(f"{self.mp_info().n} MPs: {self.mp_info().names}\n\n")

        file.write(f"columns static gestures: stamp, seq, probabilities {self.static_info().names}, gesture activates {self.static_info().names}\n")
        file.write(f"columns dynamic gestures: probabilities {self.dynamic_info().names}, norm. probabilities {self.static_info().names}, gesture activates {self.static_info().names}\n\n")
        file.write(f"columns mp activates: id gesture, id_primitive, action_stamp, gesture variables assigned to MP [velocities,point_direction,palm_euler,grab,pinch,holding_sphere_radius], generated path e.g.(3x200) \n\n")

        file.write(f"Logic parameters: activate_length={self.activate_length}s (Time length when gesture is activated in order to make action distance_length={self.distance_length}s Length between the activate_actiondistance_length)\n\n")

        file.close()

        from os_and_utils.visualizer_lib import VisualizerLib
        viz = VisualizerLib()
        viz.visualize_new_fig("Gesture probability through time", dim=2)


        if self.r.dynamic[0]:
            time = gesture_data_right_dynamic[:,0]
            gesture_data_right_dynamic = np.array(gesture_data_right_dynamic)[:,7:12].T
            for n in range(2):#gl.gd.r.dynamic.info.n):
                viz.visualize_2d(list(zip(time, gesture_data_right_dynamic[n])),label=f"{self.r.dynamic.info.names[n]}", xlabel='time [s]', ylabel='Gesture probability', start_stop_mark=False)

        if self.l.static[0]:
            time = gesture_data_left_static[:,0]
            gesture_data_left_static = np.array(gesture_data_left_static)[:,2:10].T
            for n in range(2):#gl.gd.r.static.info.n):
                viz.visualize_2d(list(zip(time, gesture_data_left_static[n])),label=f"{self.r.static.info.names[n]}", xlabel='time [s]', ylabel='Gesture probability', start_stop_mark=False)
        viz.savefig(settings.paths.data_export_path+'record_'+str(N_)+"/plot")

        if self.action_saves:
            viz = VisualizerLib()
            viz.visualize_new_fig("Executed MPs", dim=3)

            # self.action_saves # id, id_primitive, tmp_action_stamp, vars, path
            for id, id_primitive, tmp_action_stamp, vars, path, waypoints in self.action_saves:
                viz.visualize_3d(path,label=f"id: {id}, id_primitive: {id_primitive}, stamp: {tmp_action_stamp}", xlabel='X', ylabel='Y', zlabel='Z')
            viz.savefig(settings.paths.data_export_path+'record_'+str(N_)+"/plot_mp")


        return




    '''
    Static manage methods
    '''
    @staticmethod
    def download_networks_gdrive():
        if not gdown: print("Networks cannot be downloaded! Install gdown: conda install -c conda-forge gdown")
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

    @staticmethod
    def capture_gesture_moment(DEBUG=False):
        if DEBUG: print("[[1]] finish episode")
        while ml.md.present():
            time.sleep(0.01)
        if DEBUG: print("[[2]] approvement init")
        while not ml.md.present():
            time.sleep(0.01)
        if DEBUG: print("[[3]] make hand stable")
        while not ml.md.any_hand_stable():
            time.sleep(0.01)
        if DEBUG: print("[[4]] 1 sec")
        time.sleep(1.0)
        if DEBUG: print("[[5]] cvak")


    @staticmethod
    def approve_handle(printer, type='y/n'):
        print(f"{cc.OKCYAN}{printer}{cc.E}")
        if settings.feedback_mode == 'keyboard':
            y = input()
            return y
        elif settings.feedback_mode == 'gesture':

            gd.capture_gesture_moment()

            if type == 'y/n':
                while True:
                    ag = gd.approve_gesture()
                    if ag != '':
                        print(f"{cc.H}{'Agreement' if ag == 'y' else 'Disagreement'}{cc.E}")
                        return ag
            elif type == 'y':
                while True:
                    ag = gd.approve_gesture()
                    if ag != '':
                        print(f"{cc.H}{'Continue'}{cc.E}")
                        return ag
            elif type == 'number':
                while True:
                    ag = gd.number_gesture()
                    if ag != '':
                        print(f"{cc.H}Number: {ag}{cc.E}")
                        return ag
            else:
                raise Exception("NotImplementedError!")
        else: raise Exception(f"wrong feedback_mode {settings.feedback_mode} not in {settings.feedback_modes}")


    @staticmethod
    def approve_gesture():
        if gd.processPose_approvement():
            return 'y'
        elif gd.processPose_disapprovement():
            return 'n'
        else:
            return ''

    @staticmethod
    def number_gesture():
        while not ml.md.frames: time.sleep(0.1)
        fa = ml.md.frames[-1]
        if fa.r.visible:
            h = 'r'
        elif fa.l.visible:
            h = 'l'
        else: return
        fah = getattr(fa,h)

        fingers = 0
        # TODO: load from config
        THRE = [0.95, 0.7, 0.7, 0.7, 0.7]
        for i in range(5):
            if fah.oc[i] > THRE[i]:
                fingers += 1
        return fingers

    @staticmethod
    def processPose_approvement():
        while not ml.md.frames: time.sleep(0.1)
        fa = ml.md.frames[-1]
        if fa.r.visible:
            h = 'r'
        elif fa.l.visible:
            h = 'l'
        else: return
        fah = getattr(fa,h)

        # Thumbs up gesture
        if fah.oc[1] < 0.3 and fah.oc[2] < 0.3 and fah.oc[3] < 0.3 and fah.oc[4] < 0.3:
            return True
        else:
            return False

    @staticmethod
    def processPose_disapprovement():
        if not ml.md.frames: return

        fa = ml.md.frames[-1]
        if fa.r.visible:
            h = 'r'
        elif fa.l.visible:
            h = 'l'
        else: return
        fah = getattr(fa,h)

        # Thumbs up gesture
        if fah.oc[1] > 0.3 and fah.oc[2] > 0.3 and fah.oc[3] > 0.3 and fah.oc[4] > 0.3:
            return True
        else:
            return False
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
            h = 'r'
        elif fa.l.visible:
            h = 'l'
        else: return

        gd = getattr(settings.gd,h)
        if getattr(fa,h).conf > gd.min_confidence:
            gd.conf = True
        else:
            gd.conf = False

        for i in range(0,5):
            if getattr(fa,h).OC[i] > gd.oc_turn_on_thre[i] and gd.conf:
                gd.oc[i] = True
            elif getattr(fa,h).OC[i] < gd.oc_turn_off_thre[i]:
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


    def processGest_move_in_axis(self):
        '''
        '''
        minthre = 400
        maxthre = 100
        gg_toggle_tmp = deepcopy(self.experimental_move_in_axis_toggle)
        fa = ml.md.frames[-1]
        if fa.r.visible:
            if abs(fa.r.palm_velocity()[0]) > minthre and fa.r.palm_velocity()[1] < maxthre and fa.r.palm_velocity()[2] < maxthre:
                self.experimental_move_in_axis_toggle[0] = True
                move = 'swipe_front_right' if fa.r.palm_velocity()[0] > minthre else 'swipe_left'
                if gg_toggle_tmp[0] == False:
                    return move
            else:
                self.experimental_move_in_axis_toggle[0] = False

            if abs(fa.r.palm_velocity()[1]) > minthre and fa.r.palm_velocity()[0] < maxthre and fa.r.palm_velocity()[2] < maxthre:
                self.experimental_move_in_axis_toggle[1] = True
                move = 'swipe_up' if fa.r.palm_velocity()[1] > minthre else 'swipe_down'
                if gg_toggle_tmp[1] == False:
                    return move
            else:
                self.experimental_move_in_axis_toggle[1] = False
            '''
            if abs(fa.r.palm_velocity()[2]) > minthre and fa.r.palm_velocity()[0] < maxthre and fa.r.palm_velocity()[1] < maxthre:
                self.experimental_move_in_axis_toggle[2] = True
                move = 'right' if fa.r.palm_velocity()[2] > minthre else 'left'
                if gg_toggle_tmp[2] == False:
                    return 'z', move
            else:
                self.experimental_move_in_axis_toggle[2] = False
            '''
        return None

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

def init(silent=False):
    global gd
    gd = GestureDataDetection(silent=silent)

#### For testing purposes
def main():
    init()


if __name__ == '__main__':
    main()

#
