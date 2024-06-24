        

from collections import Counter
from copy import deepcopy
import time

from scene_getter.scene_getter import SceneGetter
from gesture_detector.gesture_classification.gestures_lib import GestureDataDetection
import numpy as np
import rclpy
from rclpy.node import Node

from gesture_sentence_maker.aux_gestures import load_auxiliary_parameter, misc_gesture_handle
from gesture_sentence_maker.hricommand_export import export_only_objects_to_HRICommand, export_original_to_HRICommand
from pointing_object_selection.pointing_object_getter import PointingObjectGetter
from gesture_sentence_maker.utils import get_dist_by_extremes

from gesture_msgs.msg import HRICommand
from gesture_detector.utils.utils import CustomDeque

class GestureSentence(PointingObjectGetter, SceneGetter, GestureDataDetection):
    def __init__(self, ignored_gestures = ['point', 'no_moving', 'five', 'pinch'], object_pick_method = 'last'):
        self.topic = "sentence_processor_node"
        
        super(GestureDataDetection, self).__init__()
        super(SceneGetter, self).__init__()
        super(PointingObjectGetter, self).__init__()

        self.gesture_sentence_publisher = self.create_publisher(HRICommand, "/teleop_gesture_toolbox/hricommand_original", 5)

        self.ignored_gestures = ignored_gestures
        self.object_pick_method = object_pick_method
        
        # self.gestures_queue - deque
        # self.gestures_queue_proc - processed queue

        # sentence data
        self.previous_gesture_observed_data_action = None
        self.previous_gesture_observed_data_object_names = CustomDeque()
        self.previous_gesture_observed_data_measurement_distance = []
        self.ap = None # (Dict?)
        self.previous_gesture_observed_data_action = None
        self.previous_gesture_observed_data_object_info = CustomDeque()

        self.target_objects = None # (String[])
        self.target_objects_infos = []
        self.evidence_gesture_type_to_activate_last_added = 0.
        self.evidence_gesture_type_to_activate = CustomDeque()

        self.action_received = False
        print("[GS] Done ")

    def step(self):
        print("step")
        if self.present():
            print("gesture step")
            self.gesturing_step()
        elif len(self.gestures_queue) > 0:
            time.sleep(0.5)
            self.gestures_queue_proc = self.process_gesture_queue(self.gestures_queue)

            # target_object data save
            if self.previous_gesture_observed_data_object_names != []:
                self.save_accumulated_deictic_gesture_data(self.object_pick_method)

            if len(self.gestures_queue_proc) == 0:
                self.gesture_sentence_publisher.publish(export_only_objects_to_HRICommand(self.scene, self.target_objects_infos))
                return

        else:
            self.gesture_sentence_publisher.publish(export_only_objects_to_HRICommand(self.scene, self.target_objects_infos))
        
        # Whenever hand is not seen clearing
        self.clearing(wait=False)

    def gesturing_step(self):
        activated_gestures = self.load_all_relevant_activated_gestures(relevant_time=2.0, records=3)

        activated_gesture_type = AdaptiveSetup.get_adaptive_gesture_type(activated_gestures)

        activated_gesture_type_action = self.activated_gesture_type_to_action(activated_gesture_type)

        self.save_accumulated_data_of_unactivated_gesture_types()

        if activated_gesture_type_action == 'deictic':
            self.step_deictic()
        
        # elif activated_gesture_type_action == 'approvement':
        #     self.step_approvement()

        # elif activated_gesture_type_action == 'measurement_distance':
        #     self.step_measurement()
        
        else:
            # Evaluate when action gesturing ends
            self.previous_gesture_observed_data_action = 'action'

    def activated_gesture_type_to_action(self, activated_gesture_type, rate=10, x=5, y=10, threshold=0.9):
        '''
        Parameters:
            rate (Int): Rate of new frames (Hz)
            x (Int): Gesture type evidence to be activated (frames)
            y (Int): How many frames it must be non activated, before the gesture type is activated
            threshold (Float): accuracy threshiold, >% frames gesture type -> activated
        Returns:
            gesture_type (String/None): If fulfills the conditions or None if not

        --------- | --------- | ---------
             aaaaa|dddddddddddddddddddd aaaaa
                   < -------- x ------> x True
             <-y->|< ----- delay -----> y False

                    |< -------- x ------> x True
               <-y->|< ----- delay -----> y False

        '''
        if (time.time()-self.evidence_gesture_type_to_activate_last_added) > (1/rate):
            self.evidence_gesture_type_to_activate_last_added = time.time()
            self.evidence_gesture_type_to_activate.append(activated_gesture_type)

        gesture_type = self.evidence_gesture_type_to_activate.get_last_common(x, threshold=1.0)
        if gesture_type is not None:
            return gesture_type



    def save_accumulated_data_of_unactivated_gesture_types(self):
        if (self.previous_gesture_observed_data_action != 'deictic' and 
            self.previous_gesture_observed_data_object_names != []):

            self.save_accumulated_deictic_gesture_data(self.object_pick_method)
    
        if (self.previous_gesture_observed_data_action != 'measurement_distance' and 
                self.previous_gesture_observed_data_measurement_distance != []):
            
            self.ap.append(get_dist_by_extremes(np.array(self.previous_gesture_observed_data_measurement_distance)))
            self.previous_gesture_observed_data_measurement_distance = []

    def step_deictic(self):
        ''' Activated gesture enabled Deictic gesture mode.
        '''
        object_name_1, on1_info = self.get_target_object()

        self.previous_gesture_observed_data_action = 'deictic'
        self.previous_gesture_observed_data_object_names.append(object_name_1)
        self.previous_gesture_observed_data_object_info.append(on1_info)
        print(f"Added obj {object_name_1}")

    def step_approvement(self):
        res = misc_gesture_handle(f"Approve? (y/n)", new_episode=False)
        print(f"Added approvement {res}")
        self.ap.append(res)

    def step_measurement(self):
        dist = load_auxiliary_parameter()
        self.previous_gesture_observed_data_action = 'measurement_distance'
        self.previous_gesture_observed_data_measurement_distance.append(dist)

    def get_max_gesture_probs(self):
        # Get all action gesture probs by max
        detected_gestures_probs = [] # 2D (gesture activation, probabilities)
        for detected_gesture in self.gestures_queue:
            stamp, name, hand_tag, all_probs = detected_gesture
            detected_gestures_probs.append(all_probs)
        return self.process_gesture_probs_by_max(detected_gestures_probs)
        

    def make_new_gesture_sentence(self):
        max_gesture_probs = self.get_max_gesture_probs()

        self.gesture_sentence_publisher.publish(export_original_to_HRICommand(
            deepcopy(max_gesture_probs), self.target_object_infos, self.gestures_queue, self.Gs
            ))
        
        self.clearing()
        return 

    def clearing(self, wait=True):
        self.gestures_queue.clear()
        self.gestures_queue_proc = []
        self.evaluate_episode = False

        self.target_object_infos = []
        self.target_objects = []
        self.ap = []

        if wait:
            print("Move hand out to end the episode!")
            while self.present():
                time.sleep(0.1)


    def process_gesture_queue(self, gestures_queue,ignored_gestures=['point', 'no_moving']):
        ''' gestures_queue has combinations of
        Parameters:
            gesture_queue (String[]): Activated action gestures within episode
                - Can be mix static and dynamic ones
        Experimental:
        1. There needs to be some regulation of static and dynamic ones
        2. Weigthing based on when they were generated
        '''
        total_count = len(gestures_queue)
        if total_count <= 0: return []
        gestures_queue = [g[1] for g in gestures_queue]
        sta, dyn = self.get_most_probable_sta_dyn(gestures_queue,2,ignored_gestures)

        if sta == [] and dyn == []: return []
        gestures_queue = [max([*sta, *dyn])]
        return gestures_queue
    
    def process_gesture_probs_by_max(self, gesture_probs):
        ''' max alongsize 0th axis '''
        return np.max(gesture_probs, axis=0)

    def get_most_probable_sta_dyn(self, gesture_queue, n, ignored_gestures):
        ''' Gets the most 'n' occurings from static and dynamic gestures
            - I sorts gestures_queue list into static and dynamic gesture lists
        e.g. gesture_queue = ['apple','apple','banana','banana','banana', 'coco', 'coco', 'coco','coco']
        Returns: for (n=2): ['coco','banana']
        '''
        static_gestures, dynamic_gestures = [], []

        #gesture_queue = ['apple','apple','banana','banana','banana', 'coco', 'coco', 'coco','coco']
        counts = Counter(gesture_queue)
        while len(counts) > 0:

            # get max
            gesture_name = max(counts)
            m = counts.pop(gesture_name)

            gt = self.get_gesture_type(gesture_name)

            if gt == 'static' and len(static_gestures) < n and gesture_name not in ignored_gestures:
                static_gestures.append(gesture_name)
            elif gt == 'dynamic' and len(dynamic_gestures) < n and gesture_name not in ignored_gestures:
                dynamic_gestures.append(gesture_name)
            else: continue

        return static_gestures, dynamic_gestures


    def save_accumulated_deictic_gesture_data(self, object_pick_method):
        if object_pick_method == 'max':
            c = Counter(self.previous_gesture_observed_data_object_names)
            c_max = c.most_common(1)[0]
            self.target_objects.append(c_max[0])
            i = self.previous_gesture_observed_data_object_names.index(c_max[0])
            self.target_object_infos.append(self.previous_gesture_observed_data_object_info[i])
        elif object_pick_method == 'last':
            try: # Filter last 3 time-frames of possible
                self.target_object_infos.append(self.previous_gesture_observed_data_object_info[-4])
                self.gd.target_objects.append(self.previous_gesture_observed_data_object_names[-4])
            except: # If there are less than 4 samples -> use the last
                self.target_objects.append(self.previous_gesture_observed_data_object_names[-1])
                self.target_object_infos.append(self.previous_gesture_observed_data_object_info[-1])
        else: raise Exception()

        print(f"Added obj {Counter(self.previous_gesture_observed_data_object_names)}")
        self.previous_gesture_observed_data_object_names = []




class AdaptiveSetup():
    adaptive_setup = {
        'deictic': ('point'), # TODO: 'steady_point'
        #'approvement': ('thumbsup', 'five'), # steady five
        # 'measurement_distance': ('pinch'), # steady pinch
        #'measurement_rotation': ('five'), # steady pinch
    }

    @staticmethod
    def get_adaptive_gesture_type(activated_gestures):
        activated_gesture_types = []

        as_ = AdaptiveSetup.adaptive_setup
        # activated_gestures = ('point')
        for ag in activated_gestures:
            # as_.keys() = ('deictic', 'approvement', 'measurement')
            for k in as_.keys():
                asi = as_[k]
                # if the adaptive setup item has the activated gesture in its list
                if ag in asi: # gesture which is activated is is adaptive setup gestures
                    # activate the gesture type
                    if k not in activated_gesture_types:
                        activated_gesture_types.append(k)

        if len(activated_gesture_types) > 1:
            # TODO:
            #print(f"[WARNING] More possible gesture types, act: {activated_gesture_types}")
            activated_gesture_type = activated_gesture_types[0]
        elif len(activated_gesture_types) == 0:
            activated_gesture_type = 'action'
        elif len(activated_gesture_types) == 1:
            activated_gesture_type = activated_gesture_types[0]
        else: raise Exception("Cannot happen")

        return activated_gesture_type







if __name__ == '__main__':
    rclpy.init()
    sentence_processor = GestureSentence()

    while rclpy.ok():

        rclpy.spin_once(sentence_processor)
        sentence_processor.step()




    '''
        # Update focus target
        if self.seq % (settings.yaml_config_gestures['misc']['rate'] * 2) == 0: # every sec
            if sl.scene and len(sl.scene.object_poses) > 0:
                try:
                    rc.roscm.r.add_or_edit_object(name='Focus_target', pose=sl.scene.objects[self.object_focus_id].position_real, timeout=0.2)
                except IndexError:
                    print("Detections warning, check objects!")




        self.seq += 1'''