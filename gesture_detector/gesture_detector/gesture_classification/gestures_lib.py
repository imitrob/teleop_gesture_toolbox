#!/usr/bin/env python3
'''
Get info:
self.static_info().<g1>.thresholds

Get latest data:
self.l.static[-1].<g1>.probability
(gestures_lib.GestureDataDetection.GestureDataHand.GestureMorphClass.Static.prob)

Get timestamped data:
self.l.static[<stamp> or index].<g1>.probability
(gestures_lib.GestureDataDetection.GestureDataHand[float or int].static.prob)
'''
import sys, os, collections, time, json
import numpy as np
from copy import deepcopy
import threading
import rclpy
from rclpy.node import Node

from std_msgs.msg import Float64MultiArray, MultiArrayDimension, String

import gesture_detector
from gesture_detector.utils.transformations import Transformations as tfm
from gesture_detector.hand_processing.frame_lib import Frame

from gesture_msgs.msg import DetectionSolution
import gesture_msgs.msg as rosm
from gesture_msgs.msg import DetectionSolution, DetectionObservations
from gesture_msgs.srv import SaveHandRecord, GetModelConfig
from gesture_detector.gesture_classification.episodic_accumulation import AccumulatedGestures
# from gesture_detector.gesture_classification.sentence_creation import GestureSentence

DEBUGSEMAPHORE = False

rossem = threading.Semaphore()

def withsem(func):
    def inner(*args, **kwargs):
        if DEBUGSEMAPHORE: print(f"ACQ, {args}, {kwargs}")
        with rossem:
            ret = func(*args, **kwargs)
        if DEBUGSEMAPHORE: print("---")
        return ret
    return inner



class ROSComm(Node):
    ''' ROS communication of main thread: Subscribers (init & callbacks) and Publishers
    '''
    def __init__(self):
        if not self.topic:
            self.topic = 'ros_comm_main'
        # super().__init__(self.topic) # not sure about this one
        super(ROSComm, self).__init__(self.topic)

        self.create_subscription(rosm.Frame, '/hand_frame', self.hand_frame_callback, 10)


        self.create_subscription(DetectionSolution, '/teleop_gesture_toolbox/static_detection_solutions', self.save_static_detection_solutions_callback, 10)
        self.static_detection_observations_pub = self.create_publisher(DetectionObservations,'/teleop_gesture_toolbox/static_detection_observations', 5)

        self.create_subscription(DetectionSolution, '/teleop_gesture_toolbox/dynamic_detection_solutions', self.save_dynamic_detection_solutions_callback, 10)
        self.dynamic_detection_observations_pub = self.create_publisher(DetectionObservations, '/teleop_gesture_toolbox/dynamic_detection_observations', 5)

        self.save_hand_record_cli = self.create_client(SaveHandRecord, '/save_hand_record')
        if not self.save_hand_record_cli.wait_for_service(timeout_sec=1.0):
            print('save_hand_record service not available!')
        
        self.last_time_livin = time.time()

        self.all_states_pub = self.create_publisher(String, '/teleop_gesture_toolbox/all_states', 5)

        self.get_static_model_config = self.create_client(GetModelConfig, '/teleop_gesture_toolbox/static_detection_info')
        while not self.get_static_model_config.wait_for_service(timeout_sec=1.0):
            print('service not available, waiting again...')
        self.get_dynamic_model_config = self.create_client(GetModelConfig, '/teleop_gesture_toolbox/dynamic_detection_info')
        while not self.get_dynamic_model_config.wait_for_service(timeout_sec=1.0):
            print('service not available, waiting again...')

    def call_static_model_config_service(self):
        self.future = self.get_static_model_config.call_async(GetModelConfig.Request())
        rclpy.spin_until_future_complete(self, self.future)
        return list(self.future.result().gestures)
    
    def call_dynamic_model_config_service(self):
        self.future = self.get_dynamic_model_config.call_async(GetModelConfig.Request())
        rclpy.spin_until_future_complete(self, self.future)
        return list(self.future.result().gestures)

    @withsem
    def save_hand_record(self, dir):
        self.save_hand_record_req = SaveHandRecord.Request()
        self.save_hand_record_req.directory = gesture_detector.gesture_data_path+dir
        self.save_hand_record_req.recording_length = 1.0

        return self.save_hand_record_cli.call_async(self.save_hand_record_req)

    @withsem
    def create_rate_(self, rate):
        return self.create_rate(rate)

    @withsem
    def get_time(self):
        return self.get_clock().now().nanoseconds/1e9

    def spin_once(self, sem=True):
        if sem:
            if DEBUGSEMAPHORE: print("ACQ - spinner")
            with rossem:
                if DEBUGSEMAPHORE: print("*")
                self.last_time_livin = time.time()
                rclpy.spin_once(self)
            if DEBUGSEMAPHORE: print("---")
        else:
            self.last_time_livin = time.time()
            rclpy.spin_once(self)

    def publish_eef_goal_pose(self, goal_pose):
        ''' Publish goal_pose /ee_pose_goals to relaxedIK with its transform
            Publish goal_pose /ee_pose_goals the goal eef pose
        '''
        self.ee_pose_goals_pub.publish(goal_pose)
        #self.ik_bridge.relaxedik.ik_node_publish(pose_r = self.ik_bridge.relaxedik.relaxik_t(goal_pose))



    def hand_frame_callback(self, data):
        ''' Hand data received by ROS msg is saved
        '''
        f = Frame()
        f.import_from_ros(data)
        self.hand_frames.append(f)



    def save_static_detection_solutions_callback(self, data):
        self.new_record(data, type='static')

    def save_dynamic_detection_solutions_callback(self, data):
        self.new_record(data, type='dynamic')

    @withsem
    def send_g_data(self, l_hand_mode, r_hand_mode, dynamic_detection_window=1.5, time_samples = 5):
        ''' Sends appropriate gesture data as ROS msg
            Launched node for static/dynamic detection.
        '''
        if len(self.hand_frames) == 0: return

        msg = DetectionObservations()
        msg.observations = Float64MultiArray()
        msg.sensor_seq = self.hand_frames[-1].seq
        msg.header.stamp.sec = self.hand_frames[-1].sec
        msg.header.stamp.nanosec = self.hand_frames[-1].nanosec

        mad1 = MultiArrayDimension()
        mad1.label = 'time'
        mad2 = MultiArrayDimension()
        mad2.label = 'xyz'

        hand_mode = {'l': l_hand_mode, 'r': r_hand_mode}
        for hand in hand_mode.keys():
            if 'static' in hand_mode[hand]:
                if hand == 'l':
                    if self.l_present():
                        msg.observations.data = self.hand_frames[-1].l.get_learning_data_static(definition=self.gesture_config['static']['input_definition_version'])
                        msg.header.frame_id = 'l'
                        self.static_detection_observations_pub.publish(msg)
                elif hand == 'r':
                    if self.r_present():
                        msg.observations.data = self.hand_frames[-1].r.get_learning_data_static(definition=self.gesture_config['static']['input_definition_version'])
                        msg.header.frame_id = 'r'
                        self.static_detection_observations_pub.publish(msg)


            
            if 'dynamic' in hand_mode[hand] and len(self.hand_frames) > time_samples:
                if getattr(self, hand+'_present')():
                    try:
                        n = 1
                        visibles = []
                        while True:
                            ttt = self.hand_frames[-1].stamp() - self.hand_frames[-n].stamp()
                            visibles.append( getattr(self.hand_frames[-n], hand).visible )
                            if ttt > 1.5: break
                            n += 1
                        if not np.array(visibles).all():
                            return

                        ''' Creates timestamp indexes starting with [-1, -x, ...] '''
                        time_samples_series = [-1]
                        time_samples_series.extend((n * np.array(range(-1, -time_samples, -1))  / time_samples).astype(int))
                        time_samples_series.sort()

                        ''' Compose data '''
                        data_composition = []
                        for time_sample in time_samples_series:
                            data_composition.append(getattr(self.hand_frames[time_sample], hand).get_single_learning_data_dynamic(definition=self.gesture_config['dynamic']['input_definition_version']))

                        ''' Transform to Leap frame id '''
                        data_composition_ = []
                        for point in data_composition:
                            data_composition_.append(tfm.transformLeapToBase(point, out='position'))
                        data_composition = data_composition_

                        ''' Check if the length of composed data is aorund 1sec '''
                        ttt = self.hand_frames[-1].stamp() - self.hand_frames[int(time_samples_series[0])].stamp()
                        if not (0.7 <= ttt <= 2.0):
                            print(f"WARNING: data frame composed is {ttt} long")
                        ''' Subtract middle path point from all path points '''

                        #if 'normalize' in args:
                        data_composition_ = []
                        data_composition0 = deepcopy(data_composition[len(data_composition)//2])
                        for n in range(0, len(data_composition)):
                            data_composition_.append(np.subtract(data_composition[n], data_composition0))
                        data_composition = data_composition_

                        ''' Fill the ROS msg '''
                        data_composition = np.array(data_composition, dtype=float)

                        mad1.size = data_composition.shape[0]
                        mad2.size = data_composition.shape[1]
                        data_composition = list(data_composition.flatten())
                        msg.observations.data = data_composition
                        msg.observations.layout.dim = [mad1, mad2]
                        msg.header.frame_id = hand
                        self.dynamic_detection_observations_pub.publish(msg)
                    except IndexError:
                        pass
    
    
    def send_state(self):
        
        spinner_livin = (time.time()-self.last_time_livin) < 1.0
        
        dict_to_send = {
            "spinner_livin": str(spinner_livin).lower(),
            "fps": -1,
            "seq": -1,
        }
        
        # if ml.md.goal_pose and ml.md.goal_joints:
        #     structures_str = [structure.object_stack for structure in ml.md.structures]
        #     textStatus += f"eef: {str(round(ml.md.eef_pose.position.x,2))} {str(round(ml.md.eef_pose.position.y,2))} {str(round(ml.md.eef_pose.position.z,2))}\ng p: {str(round(ml.md.goal_pose.position.x,2))} {str(round(ml.md.goal_pose.position.y,2))} {str(round(ml.md.goal_pose.position.z,2))}\ng q:{str(round(ml.md.goal_pose.orientation.x,2))} {str(round(ml.md.goal_pose.orientation.y,2))} {str(round(ml.md.goal_pose.orientation.z,2))} {str(round(ml.md.goal_pose.orientation.w,2))}\nAttached: {ml.md.attached}\nbuild_mode {ml.md.build_mode}\nobject_touch and focus_id {ml.md.object_focus_id} {ml.md.object_focus_id}\nStructures: {str(structures_str)}\n"
        if self.present():
            dict_to_send["fps"] = round(self.hand_frames[-1].fps)
            dict_to_send["seq"] = self.hand_frames[-1].seq
            
            # dict_to_send["gesture_type_selected"] = gl.sd.prev_gesture_type
            # dict_to_send["gs_state_action"] = GestureSentence.process_gesture_queue(self.gestures_queue)
            # dict_to_send["gs_state_objects"] = self.target_objects
        
        if self.l.static.relevant():
            static_n = self.static_info().n
            dict_to_send['l_static_names'] = self.Gs_static
            dict_to_send['l_static_probs'] = [self.l.static[-1][n].probability for n in range(static_n)]
            dict_to_send['l_static_activated'] = [str(self.l.static[-1][n].activated).lower() for n in range(static_n)]

        if self.r.static.relevant():
            static_n = self.static_info().n
            dict_to_send['r_static_names'] = self.Gs_static
            dict_to_send['r_static_probs'] = [self.r.static[-1][n].probability for n in range(static_n)]
            dict_to_send['r_static_activated'] = [str(self.r.static[-1][n].activated).lower() for n in range(static_n)]
        
        if self.l.dynamic and self.l.dynamic.relevant():
            dynamic_n = self.dynamic_info().n
            dict_to_send['l_dynamic_names'] = self.Gs_dynamic
            dict_to_send['l_dynamic_probs'] = list(self.l.dynamic[-1].probabilities_norm)
            dict_to_send['l_dynamic_activated'] = [str(self.l.dynamic[-1][n].activated).lower() for n in range(dynamic_n)]
        
        if self.r.dynamic and self.r.dynamic.relevant():
            dynamic_n = self.dynamic_info().n
            dict_to_send['r_dynamic_names'] = self.Gs_dynamic
            dict_to_send['r_dynamic_probs'] = list(self.r.dynamic[-1].probabilities_norm)
            dict_to_send['r_dynamic_activated'] = [str(self.r.dynamic[-1][n].activated).lower() for n in range(dynamic_n)]
            # self.get_logger().info(f"r dynamic enabled {dict_to_send['r_dynamic_probs']}.. {dict_to_send['r_dynamic_activated']}, {dict_to_send['r_dynamic_names']}")

        if self.l.static and self.l.static.relevant() is not None:  
            try:
                dict_to_send['l_static_relevant_biggest_id'] = self.l.static.relevant().biggest_probability_id
            except AttributeError:
                dict_to_send['l_static_relevant_biggest_id'] = -1

        if self.r.static and self.r.static.relevant() is not None:  
            try:
                dict_to_send['r_static_relevant_biggest_id'] = self.r.static.relevant().biggest_probability_id
            except AttributeError:
                dict_to_send['r_static_relevant_biggest_id'] = -1

        if self.l.dynamic and self.l.dynamic.relevant() is not None:
            try:
                dict_to_send['l_dynamic_relevant_biggest_id'] = self.l.dynamic.relevant().biggest_probability_id
            except AttributeError:
                dict_to_send['l_dynamic_relevant_biggest_id'] = -1

        if self.r.dynamic and self.r.dynamic.relevant() is not None:
            try:
                dict_to_send['r_dynamic_relevant_biggest_id'] = self.r.dynamic.relevant().biggest_probability_id
            except AttributeError:
                dict_to_send['r_dynamic_relevant_biggest_id'] = -1
        
        
        # compound_gestures = self.c[-1]
        # dict_to_send['compound_activated'] = ['false'] * len(self.c.info.names)
        # dict_to_send['compound_names'] = list(self.c.info.names)
        
        # if compound_gestures is not None:
        #     dict_to_send['compound_activated'] = [str(a).lower() for a in compound_gestures.activates]
        # print("dict_to_send['compound_activated']", dict_to_send['compound_activated'], "dict_to_send['compound_names']", dict_to_send['compound_names'])
        
        
        data_as_str = str(dict_to_send)
        data_as_str = data_as_str.replace("'", '"')
        
        self.all_states_pub.publish(String(data=data_as_str))


    def spin_until_future_complete_(self, future):
        if rossem is not None:
            while rclpy.spin_until_future_complete(self, future, timeout_sec=0.01) is not None:
                rossem.release()
                time.sleep(0.01)
                rossem.acquire()
        else:
            raise Exception("[ERROR] NotImplementedError!")





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

        Used placeholder for: 1. Info about gestures: self.static_info().<g1>.threshold
                                - StaticInfo(), DynamicInfo() classes
                              2. Get real data: self.l.static[<time>].<g1>.probability
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

    @property
    def n(self):
        return len(self.get_all_attributes())
    @property
    def names(self):
        return self.get_all_attributes()

    @property
    def biggest_probability(self):
        attrs = self.get_all_attributes()
        for attr in attrs:
            if getattr(self,attr).biggest_probability == True:
                return attr
        return 'Does not have biggest_probability set to True'
    @property
    def biggest_probability_id(self):
        attrs = self.get_all_attributes()
        for n,attr in enumerate(attrs):
            if getattr(self,attr).biggest_probability == True:
                return n
        return 'Does not have biggest_probability set to True'
    @property
    def probabilities(self):
        attrs = self.get_all_attributes()
        ret = []
        for attr in attrs:
            ret.append(getattr(self,attr).probability)
        return ret
    @property
    def probabilities_norm(self):
        attrs = self.get_all_attributes()
        ret = []
        for attr in attrs:
            ret.append(getattr(self,attr).probability)
        return NormalizeData(ret)
    @property
    def activates(self):
        attrs = self.get_all_attributes()
        ret = []
        for attr in attrs:
            ret.append(getattr(self,attr).activated)
        return ret
    @property
    def activate_id(self):
        attrs = self.get_all_attributes()
        for n,attr in enumerate(attrs):
            if getattr(self,attr).activated:
                return n
        return None
    @property
    def action_activate_id(self):
        attrs = self.get_all_attributes()
        for n,attr in enumerate(attrs):
            if getattr(self,attr).action_activated:
                return n
        return None
    @property
    def activate_name(self):
        attrs = self.get_all_attributes()
        for attr in attrs:
            if getattr(self,attr).activated:
                return attr
        return None
    @property
    def above_threshold(self):
        attrs = self.get_all_attributes()
        ret = []
        for attr in attrs:
            ret.append(getattr(self,attr).above_threshold)
        return ret
    @property
    def thresholds(self):
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
            # self.l.static[<time>].<g> = GestureDataAtTime(probability, biggest_probability)
            setattr(self, g, GestureDataAtTime(data.probabilities.data[n], data.id == n))


class StaticInfo():
    def __init__(self, name):
        self.name = name
        # config
        self.thresholds = [0.9,0.9]
        self.time_visible_threshold = 8.0

class DynamicInfo():
    def __init__(self, name):
        self.name = name
        
        self.thresholds = [0.9,0.9]
        ## move in x,y,z, Positive/Negative
        self.move = [False, False, False]
        self.time_visible_threshold = 8.0

class TemplateGs():
    def __init__(self, data=None):
        '''
        Get info about gesture: self.static_info().<g1>
        Get data about gesture at time: self.l.static[<time index>].<g1>.probability
        '''
        self.info = GestureMorphClass()

        if data['type'] == 'static':
            for i in range(len(data['Gs'])):
                setattr(self.info, data['Gs'][i], StaticInfo(name=data['Gs'][i]))
        elif data['type'] == 'dynamic':
            for i in range(len(data['Gs'])):
                setattr(self.info, data['Gs'][i], DynamicInfo(name=data['Gs'][i]))

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
            # times are values and index are pointing to gestures_queue
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

    @property
    def n(self):
        return len(self.data_queue)

    def __getstate__(self): return self.__dict__
    def __setstate__(self, d): self.__dict__.update(d)

    def relevant(self, last_secs=1.0):
        ''' Searches gestures_queue, and returns the last gesture record, None if no records in that time were made
        '''
        abs_time = time.time() - last_secs

        #abs_time = time.time() - last_secs
        # if happened within last_secs interval
        if self.data_queue and self.data_queue[-1].header.stamp > abs_time:
            return self.data_queue[-1]
        else:
            return None

    def get_activation_gestures_dict(self):
        return {item.name: item.activation_gestures for item in self.info[:]}

class StaticGs(TemplateGs):
    def __init__(self, data=None):
        super().__init__(data)

class DynamicGs(TemplateGs):
    def __init__(self, data=None):
        super().__init__(data)

class GestureDataHand():
    ''' The 2nd class: self.l
                       self.r
    '''
    def __init__(self, gesture_config):
        self.static = StaticGs(gesture_config['static'])
        self.dynamic = DynamicGs(gesture_config['dynamic'])

class GestureDataDetection(ROSComm):
    ''' The main class: self
    '''
    def __init__(self, silent=False, load_trained=True):
        self.topic = None
        super(GestureDataDetection, self).__init__()
        self.bfr_len = 1000
        ''' Leap Controller hand data saved as circullar buffer '''
        self.hand_frames = collections.deque(maxlen=self.bfr_len)
        
        # TODO: This is temp, -> Read more from the detectors
        self.gesture_config = {
            'static': {
                'type': 'static',
                'Gs': self.call_static_model_config_service(),
                'input_definition_version': 1,
            },
            'dynamic': {
                'type': 'dynamic',
                'Gs': self.call_dynamic_model_config_service(),
                'input_definition_version': 1,
            }
        }

        self.l = GestureDataHand(self.gesture_config)
        self.r = GestureDataHand(self.gesture_config)
        if not silent:
            print(f"Static gestures: {self.static_info().names}, \nDynamic gestures {self.dynamic_info().names}")

        self.gestures_queue = AccumulatedGestures()
        
        # Misc
        self.last_seq = 0
        self.activate_length=None # for export only
        self.distance_length=None # for export only


        # Gesture prediction
        self.static_gesture_action_prediction = [""] * self.static_info().n
        self.dynamic_gesture_action_prediction = [""] * self.dynamic_info().n


    @property
    def frames(self):
        ''' Over-load option for getting hand data'''
        return self.hand_frames
    
    def any_hand_stable(self, time=1.0):
        if self.stopped(h='r', time=time) or self.stopped(h='l', time=time):
            return True
        return False

    def stopped(self, h, time=1.0):
        frames_stopped = self.frames[-1].fps * time
        frames_stopped = np.clip(len(self.frames)-2, 1, frames_stopped)
        test_frames = np.linspace(1,frames_stopped,5, dtype=int)
        for f in test_frames:
            #print(f"frame: {f}, stable: {getattr(self.frames[f], h).stable}")
            ho = self.frames[-f].get_hand(h)
            if ho is None:
                return False
            if not ho.stable:
                return False
        return True

    def present(self):
        return self.r_present() or self.l_present()

    def r_present(self):
        if self.frames and self.frames[-1] and self.frames[-1].r and self.frames[-1].r.visible:
            return True
        return False

    def l_present(self):
        if self.frames and self.frames[-1] and self.frames[-1].l and self.frames[-1].l.visible:
            return True
        return False

    def get_frame_window_of_last_secs(self, stamp, N_secs):
        ''' Select frames chosen with stamp and last N_secs
        '''
        n = 0
        #     stamp-N_secs       stamp
        # ----|-------N_secs------|
        # ---*************************- <-- chosen frames
        #              <~~~while~~~  |
        #                           self.frames[-1].stamp()
        for i in range(-1, -len(self.frames),-1):
            if stamp-N_secs > self.frames[i].stamp():
                n=i
                break
        print(f"len(self.frames) {len(self.frames)}, n {n}")

        # return frames time window
        frames = []
        for i in range(-1, n, -1):
            frames.append(self.frames[i])
        return frames

    def static_info(self):
        ''' Note: info is saved both in self.l and self.r as the same
        '''
        return self.l.static.info
    def dynamic_info(self):
        return self.l.dynamic.info

    def relevant(self, hand='r', type='static', relevant_time=1.0, records=1):
        ''' Returns solutions based on hand and type if not older than relevant_time
        '''
        t = time.time()
        self_h = getattr(self, hand)
        self_h_type = getattr(self_h, type)

        gs = []
        i = 1
        while self_h_type.n > i and (t-self_h_type[-i].header.stamp) < relevant_time:
            gs.append(self_h_type[-i])
            if i > records:
                break
            i+=1
        return gs

    @property
    def Gs(self):
        Gs = self.l.static.info.names
        Gs.extend(self.l.dynamic.info.names)
        return Gs

    @property
    def GsExt(self):
        Gs = self.l.static.info.names
        Gs.extend(self.l.dynamic.info.names)
        Gs.extend(self.l.mp.info.names)
        return Gs

    @property
    def Gs_static(self):
        return self.l.static.info.names

    @property
    def Gs_dynamic(self):
        return self.l.dynamic.info.names

    @property
    def Gs_keys(self):
        Gs = self.l.static.info.record_keys
        Gs.extend(self.l.dynamic.info.record_keys)
        return Gs

    @property
    def GsExt_keys(self):
        Gs = self.l.static.info.record_keys
        Gs.extend(self.l.dynamic.info.record_keys)
        Gs.extend(self.l.mp.info.record_keys)
        return Gs

    @property
    def Gs_keys_static(self):
        return self.l.static.info.record_keys

    @property
    def Gs_keys_dynamic(self):
        return self.l.dynamic.info.record_keys

    def get_gesture_type(self, gesture):
        if gesture in self.Gs_static:
            return 'static'
        elif gesture in self.Gs_dynamic:
            return 'dynamic'
        else: raise Exception(f"gesture {gesture} not known!")

    def __getstate__(self): return self.__dict__
    def __setstate__(self, d): self.__dict__.update(d)

    def new_record(self, data, type='static'):
        ''' New gesture data arrived and will be saved
        '''
        if self.hand_frames[-1].seq-data.sensor_seq > 100:
            print(f"[Warning] Program cannot compute gs in time, probably rate is too big! (or fake data are used)")

        # choose hand with data.header.frame_id
        obj_by_hand = getattr(self, data.header.frame_id)
        # choose static/dynamic with arg: type
        obj_by_type = getattr(obj_by_hand, type)

        if len(data.probabilities.data) != obj_by_type.info.n:
            return
        obj_by_type.data_queue.append(GestureMorphClassStamped(data, obj_by_type.info.names))

        # Add logic
        self.prepare_postprocessing(data.header.frame_id, type, data.sensor_seq)


    def get_static_and_extended_probabilities_norm(self, hand_tag, frame_n=-1):
        ret = []
        hand = getattr(self,hand_tag)
        gesture_static_timeframe = hand.static[frame_n]
        if gesture_static_timeframe is not None:
            ret.extend(gesture_static_timeframe.probabilities_norm)
        else:
            ret.extend([0.] * len(self.Gs_static))
        gesture_dynamic_timeframe = hand.dynamic[frame_n]
        if gesture_dynamic_timeframe is not None:
            ret.extend(gesture_dynamic_timeframe.probabilities_norm)
        else:
            ret.extend([0.] * len(self.Gs_dynamic))

        return ret

    '''
    LOGIC
    '''
    def prepare_postprocessing(self, hand_tag, type, sensor_seq, activate_length=1.0, activate_length_dynamic=0.3, distance_length=3.0, rate=10):
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
        distance_length = int(rate * self.distance_length) # seconds -> seqs
        activate_length = int(rate * self.activate_length) # seconds -> seqs

        hand = getattr(self, hand_tag)
        gs = getattr(hand, type)
        latest_gs = gs[-1]
        probabilities = latest_gs.probabilities_norm
        all_probs = self.get_static_and_extended_probabilities_norm(hand_tag)
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
                            print(f"New Action gesture detected: {gs.info.names[n]}, {hand_tag}")
                            self.gestures_queue.append(
                                {"stamp": latest_gs.header.stamp, 
                                "name": gs.info.names[n],
                                "hand": hand_tag,
                                "probs": all_probs
                            })

    def load_all_relevant_gestures(self, relevant_time=0.5, records=3):
        l_s = self.relevant(hand='l', type='static', relevant_time=relevant_time, records=records)
        l_d = self.relevant(hand='l', type='dynamic', relevant_time=relevant_time, records=records)
        r_s = self.relevant(hand='r', type='static', relevant_time=relevant_time, records=records)
        r_d = self.relevant(hand='r', type='dynamic', relevant_time=relevant_time, records=records)
        
        relevant = []
        if l_s is not None: relevant.extend(l_s)
        if l_d is not None: relevant.extend(l_d)
        if r_s is not None: relevant.extend(r_s)
        if r_d is not None: relevant.extend(r_d)
        
        return relevant

    def load_all_relevant_activated_gestures(self, relevant_time=0.5, records=3):
        relevant = self.load_all_relevant_gestures(relevant_time=relevant_time, records=records)
        # get activated
        activated_gestures = []
        for g in relevant:
            if g is not None and g.activate_name is not None:
                activated_gestures.append(g.activate_name)

        return activated_gestures

    def handle_compound_gestures(self, sensor_seq):
        ''' Load compound gesture definitions '''
        compound_gestures = self.c.get_activation_gestures_dict()

        ''' Load recent activated gestures '''
        activated_gestures = self.load_all_relevant_activated_gestures()

        ''' Check if fulfilled '''
        cgs_activated = []
        for cgk in compound_gestures.keys():
            cg = compound_gestures[cgk]
            cgs_activated.append(True)
            for cgi in cg:
                if not (cgi in activated_gestures):
                    cgs_activated[-1] = False

        if np.array(cgs_activated).any():
            self.c.data_queue.append(CompoundGestureMorphClassStamped(sensor_seq, compound_gestures.keys(), cgs_activated))



    def last(self):
        return self.l.static.relevant(), self.r.static.relevant(), self.l.dynamic.relevant(), self.r.dynamic.relevant()



