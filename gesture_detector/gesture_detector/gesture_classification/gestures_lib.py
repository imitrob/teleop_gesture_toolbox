#!/usr/bin/env python3
'''
Get latest data:
self.l.static[-1].<g1>.probability
(gestures_lib.GestureDataDetection.GestureDataHand.GestureMorphClass.Static.prob)

Get timestamped data:
self.l.static[<stamp> or index].<g1>.probability
(gestures_lib.GestureDataDetection.GestureDataHand[float or int].static.prob)
'''
import collections, json, time
import numpy as np
from copy import deepcopy
import threading
import rclpy
from rclpy.node import Node

from std_msgs.msg import Float64MultiArray, MultiArrayDimension, String

import gesture_detector
from gesture_detector.hand_processing.frame_lib import Frame
from gesture_detector.utils.utils import transform_leap_to_leapdynamicdetector, normalize_data, GHeader
from gesture_detector.gesture_classification import gesture_parametric_extractor

from gesture_msgs.msg import DetectionSolution
import gesture_msgs.msg as rosm
from gesture_msgs.msg import DetectionSolution, DetectionObservations
from gesture_msgs.srv import SaveHandRecord, GetModelConfig
from gesture_detector.gesture_classification.episodic_accumulation import AccumulatedGestures
# from gesture_detector.gesture_classification.sentence_creation import GestureSentence


DEBUGSEMAPHORE = False
MODEL_CONFIG_RESPONSE_TIMEOUT = 2.0

# Tune this first: dynamic_window + activate_length_dynamic/GESTURE_DETECTOR_RATE is
# the time from the user starting the gesture to the action firing, and the window is
# also the longest gesture that still fits in it. Overridable as a ROS parameter in
# gesture_detect.py. The resting threshold that used to live here belongs to the
# model, and is now rest_displacement in the model's json.
DYNAMIC_WINDOW = 0.6             # s, matched to how long a swipe actually takes
DYNAMIC_MIN_VISIBLE_RATIO = 0.9  # tolerate dropped frames rather than restart the window

rossem = threading.Semaphore()


class DetectionFreshness:
    """Choose the freshness clock and sequence policy for live or replay input."""

    def __init__(self, replay_mode=False):
        self.replay_mode = replay_mode

    def accepts_sequence(self, latest_sensor_seq, solution_sensor_seq):
        return self.replay_mode or latest_sensor_seq-solution_sensor_seq <= 100

    def is_relevant(self, record, relevant_time):
        if self.replay_mode:
            return record.header.received_at > time.monotonic()-relevant_time
        return record.header.stamp > time.time()-relevant_time

def dynamic_sample_indices(n, time_samples):
    ''' Frame indices spanning (time_samples-1)/time_samples of the last n frames '''
    indices = [-1]
    indices.extend((n * np.array(range(-1, -time_samples, -1)) / time_samples).astype(int))
    indices.sort()
    return indices

def mode_posture_shown(activated_postures, mode_postures):
    ''' True when a posture that puts the user into a mode appears in the window.

    A mode owns the hand: while the user points, the path of the hand is the
    pointing, not a dynamic gesture, and the swipe that would be read out of it is
    a phantom. Every other posture leaves the two detectors independent, which is
    the point -- a static gesture that was mispredicted or is not in the model
    must not cost the dynamic stream its evidence. '''
    return not set(activated_postures).isdisjoint(mode_postures)

def withsem(func):
    def inner(*args, **kwargs):
        if DEBUGSEMAPHORE: print(f"ACQ, {args}, {kwargs}")
        with rossem:
            ret = func(*args, **kwargs)
        if DEBUGSEMAPHORE: print("---")
        return ret
    return inner

class GestureDataDetection(Node):
    def __init__(self, 
                 silent: bool = False, # No printing
                 activate_length: int = 10, # Time for gesture to activate
                 # depends on gesture_detect.py:GESTURE_DETECTOR_RATE, time_to_activate = activate_length/GESTURE_DETECTOR_RATE
                 activate_length_dynamic: int = 3 # A path stays top-1 for only a few sliding windows
                 ):
        self.activate_length = activate_length
        self.activate_length_dynamic = activate_length_dynamic

        super(GestureDataDetection, self).__init__('ros_comm_main')

        self.declare_parameter('replay_mode', False)
        self.replay_mode = self.get_parameter(
            'replay_mode').get_parameter_value().bool_value
        self.freshness = DetectionFreshness(replay_mode=self.replay_mode)

        self.static_detection_observations_pub = self.create_publisher(DetectionObservations,'/teleop_gesture_toolbox/static_detection_observations', 5)

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

        self.l = GestureDataHand(self.gesture_config, freshness=self.freshness)
        self.r = GestureDataHand(self.gesture_config, freshness=self.freshness)
        if not silent:
            print(f"Static gestures: {self.l.static.Gs}, \nDynamic gestures {self.l.dynamic.Gs}")

        self.gestures_queue = AccumulatedGestures()

        # Do not accept frames or detection results until the model config and
        # per-hand state above are ready. spin_until_future_complete() processes
        # subscriptions, so creating these earlier lets callbacks observe a
        # half-initialized object when a config response is delayed or lost.
        self.create_subscription(rosm.Frame, '/teleop_gesture_toolbox/hand_frame', self.hand_frame_callback, 10)
        self.create_subscription(DetectionSolution, '/teleop_gesture_toolbox/static_detection_solutions', self.save_static_detection_solutions_callback, 10)
        self.create_subscription(DetectionSolution, '/teleop_gesture_toolbox/dynamic_detection_solutions', self.save_dynamic_detection_solutions_callback, 10)

        # Which postures put the user into a mode: read from the sentence maker's
        # 1 Hz config snapshot rather than kept as a second copy of the user's
        # adaptive_setup here, so editing the links file is enough to change it.
        # Empty until that snapshot arrives, which leaves both detectors running.
        self.mode_postures = set()
        self.create_subscription(String, '/teleop_gesture_toolbox/gesture_meaning_info', self.gesture_meaning_info_callback, 1)

        # Gesture prediction
        self.static_gesture_action_prediction = [""] * len(self.l.static.Gs)
        self.dynamic_gesture_action_prediction = [""] * len(self.l.dynamic.Gs)


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

    def relevant(self, hand='r', type='static', relevant_time=1.0, records=1):
        ''' Returns solutions based on hand and type if not older than relevant_time
        '''
        self_h = getattr(self, hand)
        self_h_type = getattr(self_h, type)

        gs = []
        i = 1
        while (self_h_type.n > i and
               self.freshness.is_relevant(self_h_type[-i], relevant_time)):
            gs.append(self_h_type[-i])
            if i > records:
                break
            i+=1
        return gs

    @property
    def Gs(self):
        Gs = self.l.static.Gs.copy()
        Gs.extend(self.l.dynamic.Gs)
        return Gs

    @property
    def Gs_static(self):
        return self.l.static.Gs

    @property
    def Gs_dynamic(self):
        return self.l.dynamic.Gs

    def activate_length_for(self, type):
        return self.activate_length if type == 'static' else self.activate_length_dynamic

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
        if (len(self.hand_frames) > 0 and not self.freshness.accepts_sequence(
                self.hand_frames[-1].seq, data.sensor_seq)):
            print(f"[Warning] Program cannot compute gs in time, probably rate is too big! (or fake data are used)", flush=True)
            return

        # choose hand with data.header.frame_id
        try:
            obj_by_hand = getattr(self, data.header.frame_id)
            # choose static/dynamic with arg: type
            obj_by_type = getattr(obj_by_hand, type)
        except AttributeError:
            print("hand data not available", flush=True)
            return

        if len(data.probabilities.data) != len(obj_by_type.Gs):
            return
        obj_by_type.data_queue.append(GestureMorphClassStamped(data, obj_by_type.Gs))

        # Add logic
        self.activation_postprocessing(data.header.frame_id, type)#, data.sensor_seq)


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
    def activation_postprocessing(self, 
            hand_tag: str, # "l" or "r"
            type: str, # "static" or "dynamic"
            ):
        hand = getattr(self, hand_tag)
        gs = getattr(hand, type)
        latest_gs = gs[-1]
        activate_length = self.activate_length_for(type)
        for n,g in enumerate(latest_gs):
            if g.biggest_probability_flag:
                g_id = n
                break

        g = latest_gs[g_id]
        # gesture was shown with no interruption
        if gs.count_activ_evidence(g_id, activate_length)/activate_length >= 1.0:
            # check if action was already triggered recently
            if not gs.did_action_happened(g_id, 2*activate_length):
                g.action_activated = True
                print(f"New Action gesture detected: {gs.Gs[g_id]}, {hand_tag}", flush=True)
                self.gestures_queue.append({
                    "stamp": latest_gs.header.stamp, 
                    "name": gs.Gs[g_id],
                    "hand": hand_tag,
                    "probs": self.get_static_and_extended_probabilities_norm(hand_tag),
                    "params": {
                        "speed": gesture_parametric_extractor.get_speed(self.hand_frames, hand_tag),
                        "distance": gesture_parametric_extractor.crossed_distance(self.hand_frames, hand_tag),
                        "rotation": gesture_parametric_extractor.get_rotation(self.hand_frames, hand_tag),
                    }
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
            if g is not None and g.activated is not None:
                activated_gestures.append(g.activated)

        return activated_gestures


    def last(self):
        return self.l.static.relevant(), self.r.static.relevant(), self.l.dynamic.relevant(), self.r.dynamic.relevant()


    def call_static_model_config_service(self):
        return self._call_model_config_service(
            self.get_static_model_config,
            '/teleop_gesture_toolbox/static_detection_info',
        )
    
    def call_dynamic_model_config_service(self):
        return self._call_model_config_service(
            self.get_dynamic_model_config,
            '/teleop_gesture_toolbox/dynamic_detection_info',
        )

    def _call_model_config_service(self, client, service_name):
        attempt = 0
        while rclpy.ok():
            attempt += 1
            future = client.call_async(GetModelConfig.Request())
            rclpy.spin_until_future_complete(
                self,
                future,
                timeout_sec=MODEL_CONFIG_RESPONSE_TIMEOUT,
            )

            retry_reason = (
                f"timed out after {MODEL_CONFIG_RESPONSE_TIMEOUT:.1f}s"
            )
            if future.done():
                try:
                    response = future.result()
                except Exception as error:  # noqa: BLE001 - retry transport failures
                    retry_reason = f"request failed: {error}"
                else:
                    if response is not None:
                        return list(response.gestures)
                    retry_reason = "received an empty response"

            client.remove_pending_request(future)
            print(
                f"[GestureDataDetection] {service_name} {retry_reason} "
                f"on attempt {attempt}; retrying",
                flush=True,
            )

        raise RuntimeError(
            f"ROS shutdown while waiting for model config from {service_name}"
        )

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

    def gesture_meaning_info_callback(self, data):
        setup = json.loads(data.data).get('adaptive_setup') or {}
        self.mode_postures = {g for gestures in setup.values() for g in gestures}

    @withsem
    def send_g_data(self, l_hand_mode, r_hand_mode, dynamic_detection_window=DYNAMIC_WINDOW,
                    time_samples = 10):
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
                    if mode_posture_shown(
                            [g.activated for g in self.relevant(
                                hand=hand, type='static',
                                relevant_time=dynamic_detection_window, records=1000)],
                            self.mode_postures):
                        continue # the user is pointing: the path is not a gesture
                    try:
                        n = 1
                        visibles = 0
                        while True:
                            ttt = self.hand_frames[-1].stamp() - self.hand_frames[-n].stamp()
                            visibles += bool(getattr(self.hand_frames[-n], hand).visible)
                            if ttt > dynamic_detection_window: break
                            n += 1
                        if visibles < DYNAMIC_MIN_VISIBLE_RATIO * n:
                            continue

                        time_samples_series = dynamic_sample_indices(n, time_samples)

                        ''' Compose data '''
                        data_composition = []
                        for time_sample in time_samples_series:
                            data_composition.append(getattr(self.hand_frames[time_sample], hand).get_single_learning_data_dynamic(definition=self.gesture_config['dynamic']['input_definition_version']))

                        ''' Transform to Leap frame id '''
                        data_composition_ = []
                        for point in data_composition:
                            data_composition_.append(transform_leap_to_leapdynamicdetector(point)) # input is list
                        data_composition = data_composition_

                        ''' Check the composed data spans about the window '''
                        ttt = self.hand_frames[-1].stamp() - self.hand_frames[int(time_samples_series[0])].stamp()
                        if not (0.5 * dynamic_detection_window <= ttt <= 1.5 * dynamic_detection_window):
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
        
        if self.present():
            dict_to_send["fps"] = round(self.hand_frames[-1].fps)
            dict_to_send["seq"] = self.hand_frames[-1].seq
            
            # The mode is not available here: this node is the detector, while the
            # mode lives in the sentence maker, which publishes it on
            # /teleop_gesture_toolbox/gesture_mode for the dashboard's Doing sign.
            # dict_to_send["gs_state_action"] = GestureSentence.process_gesture_queue(self.gestures_queue)
            # dict_to_send["gs_state_objects"] = self.target_objects
        
        if self.l.static.relevant():
            static_n = len(self.l.static.Gs)
            dict_to_send['l_static_names'] = self.Gs_static
            dict_to_send['l_static_probs'] = [self.l.static[-1][n].probability for n in range(static_n)]
            dict_to_send['l_static_activated'] = [str(self.l.static[-1][n].biggest_probability_flag).lower() for n in range(static_n)]

        if self.r.static.relevant():
            static_n = len(self.l.static.Gs)
            dict_to_send['r_static_names'] = self.Gs_static
            dict_to_send['r_static_probs'] = [self.r.static[-1][n].probability for n in range(static_n)]
            dict_to_send['r_static_activated'] = [str(self.r.static[-1][n].biggest_probability_flag).lower() for n in range(static_n)]
        
        if self.l.dynamic and self.l.dynamic.relevant():
            dynamic_n = len(self.l.dynamic.Gs) 
            dict_to_send['l_dynamic_names'] = self.Gs_dynamic
            dict_to_send['l_dynamic_probs'] = list(self.l.dynamic[-1].probabilities_norm)
            dict_to_send['l_dynamic_activated'] = [str(self.l.dynamic[-1][n].biggest_probability_flag).lower() for n in range(dynamic_n)]
        
        if self.r.dynamic and self.r.dynamic.relevant():
            dynamic_n = len(self.l.dynamic.Gs)
            dict_to_send['r_dynamic_names'] = self.Gs_dynamic
            dict_to_send['r_dynamic_probs'] = list(self.r.dynamic[-1].probabilities_norm)
            dict_to_send['r_dynamic_activated'] = [str(self.r.dynamic[-1][n].biggest_probability_flag).lower() for n in range(dynamic_n)]

        if self.l.static and self.l.static.relevant() is not None:  
            try:
                dict_to_send['l_static_relevant_biggest_id'] = self.l.static.relevant().activated_id
                
            except AttributeError:
                dict_to_send['l_static_relevant_biggest_id'] = -1

            dict_to_send['l_static_evidence'] = self.l.static.count_activ_evidence(self.l.static.relevant().activated_id, self.activate_length)/self.activate_length

        if self.r.static and self.r.static.relevant() is not None:  
            try:
                dict_to_send['r_static_relevant_biggest_id'] = self.r.static.relevant().activated_id
            
            except AttributeError:
                dict_to_send['r_static_relevant_biggest_id'] = -1
        
            dict_to_send['r_static_evidence'] = self.r.static.count_activ_evidence(self.r.static.relevant().activated_id, self.activate_length)/self.activate_length

        if self.l.dynamic and self.l.dynamic.relevant() is not None:
            try:
                dict_to_send['l_dynamic_relevant_biggest_id'] = self.l.dynamic.relevant().activated_id
                dict_to_send['l_dynamic_evidence'] = self.l.dynamic.count_activ_evidence(self.l.dynamic.relevant().activated_id, self.activate_length_dynamic)/self.activate_length_dynamic
            except AttributeError:
                dict_to_send['l_dynamic_relevant_biggest_id'] = -1

        if self.r.dynamic and self.r.dynamic.relevant() is not None:
            try:
                dict_to_send['r_dynamic_relevant_biggest_id'] = self.r.dynamic.relevant().activated_id
                dict_to_send['r_dynamic_evidence'] = self.r.dynamic.count_activ_evidence(self.r.dynamic.relevant().activated_id, self.activate_length_dynamic)/self.activate_length_dynamic

            except AttributeError:
                dict_to_send['r_dynamic_relevant_biggest_id'] = -1
        
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


class GestureDataAtTime():
    ''' Same for static/dynamic, same for left/right hand
    '''
    def __init__(self, probability, biggest_probability_flag=False):
        self.probability = probability
        self.biggest_probability_flag = biggest_probability_flag
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

        Get real data: self.l.static[<time>].<g1>.probability
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
    def activated(self):
        attrs = self.get_all_attributes()
        for attr in attrs:
            if getattr(self,attr).biggest_probability_flag == True:
                return attr
        return 'Does not have biggest_probability set to True'

    @property
    def activated_id(self):
        attrs = self.get_all_attributes()
        for n,attr in enumerate(attrs):
            if getattr(self,attr).biggest_probability_flag == True:
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
        return normalize_data(ret)


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
        self.header.received_at = time.monotonic()

        for n,g in enumerate(Gs):
            # self.l.static[<time>].<g> = GestureDataAtTime(probability, biggest_probability)
            setattr(self, g, GestureDataAtTime(data.probabilities.data[n], data.id == n))

class TemplateGs():
    def __init__(self, data=None, freshness=None):
        '''
        Get data about gesture at time: self.l.static[<time index>].<g1>.probability
        '''
        self.Gs = data['Gs']
        self.freshness = freshness or DetectionFreshness()
        self.data_queue = collections.deque(maxlen=300)

    def get_times(self):
        return [g.stamp for g in self.data_queue]

    def find_nearest(self, array, value):
        array = np.asarray(array)
        idx = (np.abs(array - value)).argmin()
        return idx

    def count_activ_evidence(self, gesture_id, activation_length):
        """How many of the newest solutions, up to activation_length, had this
        gesture as top-1 without interruption.

        Both ways out of the loop used to report the index they stopped at rather
        than the count before it, so a gesture activated one detection early (three
        of them for an activate_length of four) and a queue too short to judge read
        as full evidence."""
        for i in range(activation_length):
            try:
                if not self.data_queue[-(i + 1)][gesture_id].biggest_probability_flag:
                    return float(i)
            except IndexError:
                return float(i)
        return float(activation_length)

    def did_action_happened(self, gesture_id, activation_length):
        """Returns float (0-1) of activated  
        """        
        for i in range(1, activation_length+1):
            try:
                if self.data_queue[-i][gesture_id].action_activated:
                    return True
            except IndexError:
                return False
        return False


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
        if (self.data_queue and
                self.freshness.is_relevant(self.data_queue[-1], last_secs)):
            return self.data_queue[-1]
        else:
            return None

class StaticGs(TemplateGs):
    def __init__(self, data=None, freshness=None):
        super().__init__(data, freshness=freshness)

class DynamicGs(TemplateGs):
    def __init__(self, data=None, freshness=None):
        super().__init__(data, freshness=freshness)

class GestureDataHand():
    ''' The 2nd class: self.l
                       self.r
    '''
    def __init__(self, gesture_config, freshness=None):
        self.static = StaticGs(gesture_config['static'], freshness=freshness)
        self.dynamic = DynamicGs(gesture_config['dynamic'], freshness=freshness)
