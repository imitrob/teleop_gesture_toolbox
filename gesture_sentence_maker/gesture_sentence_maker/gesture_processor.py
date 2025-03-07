#!/usr/bin/env python
from collections import Counter
from copy import deepcopy
import time

from scene_getter.scene_getting import SceneGetter
from gesture_detector.gesture_classification.gestures_lib import GestureDataDetection
import numpy as np
import rclpy
from rclpy.node import Node

from gesture_sentence_maker.hricommand_export import export_original_to_HRICommand
from pointing_object_selection.pointing_object_getter import PointingObjectGetter
from gesture_sentence_maker.utils import get_dist_by_extremes

from hri_msgs.msg import HRICommand
from gesture_detector.utils.utils import CustomDeque
from gesture_sentence_maker.segmentation_task.deictic_solutions_plot import deictic_solutions_plot_save
from gesture_sentence_maker.segmentation_task.deictic_segment import find_pointed_objects_timewindowmax

from hri_msgs.msg import HRICommand as HRICommandMSG
from rclpy.qos import QoSProfile, QoSReliabilityPolicy

class GestureSentence(PointingObjectGetter, SceneGetter, GestureDataDetection):
    def __init__(self,
                 ignored_gestures = ['point', 'no_moving'],
                 step_period = 0.2, # seconds
                 ):
        """
        Args:
            ignored_gestures (list, optional): Activation of these gesture names do not trigger gesture sentence (publisher is not sending). Defaults to ['point', 'no_moving', 'five', 'pinch'].
        """        
        self.topic = "sentence_processor_node"
        super(GestureSentence, self).__init__()

        self.gesture_sentence_publisher = self.create_publisher(HRICommand, "/teleop_gesture_toolbox/hricommand_original", qos_profile=QoSProfile(depth=10, reliability=QoSReliabilityPolicy.BEST_EFFORT))

        # sentence data
        self.prev_gesture_type = None
        self.prev_deictic_solutions = CustomDeque()
        self.prev_auxgesture_solutions = CustomDeque()
        
        self.target_object_solutions = CustomDeque() # Queue of Dicts
        self.target_auxgesture_solutions = CustomDeque() # Queue of (?)

        self.evidence_gesture_type_to_activate_last_added = 0.
        self.evidence_gesture_type_to_activate = CustomDeque()

        self.ignored_gestures = ignored_gestures

        self.step_period = step_period

        self.continue_episode = self.present
        print(f"[Gesture Processor] Note that gesture processor is discarding gestures: {ignored_gestures}")
        print("[GS] Done ")

    def step(self):
        time.sleep(self.step_period)
        if self.continue_episode(): # Hand not visible is condition for Episode to End 
            self.gesturing_step()
        
        else: # End episode
            if not self.gestures_queue.empty:
                time.sleep(0.5)

                # target_object data save
                if not self.prev_deictic_solutions.empty:
                    self.save_accumulated_deictic_gesture_data()

                publish, max_probs, max_timestamps, params = self.gestures_queue.processing(self.ignored_gestures)

                if publish > 0:
                    self.gesture_sentence_publisher.publish(export_original_to_HRICommand(
                        self.scene, self.target_object_solutions, gesture_probabilities=max_probs, gesture_timestamps=max_timestamps, gesture_names=self.Gs, params=params
                        ))
                    self.clearing()
                elif len(self.target_object_solutions) > 0:
                    self.gesture_sentence_publisher.publish(export_original_to_HRICommand(self.scene, self.target_object_solutions))
                    self.clearing()
                    return

            elif len(self.target_object_solutions) > 0:
                self.gesture_sentence_publisher.publish(export_original_to_HRICommand(self.scene, self.target_object_solutions))
        
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

        
        else:
            # Evaluate when action gesturing ends
            self.prev_gesture_type = 'action'

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
        '''
        if (time.time()-self.evidence_gesture_type_to_activate_last_added) > (1/rate):
            self.evidence_gesture_type_to_activate_last_added = time.time()
            self.evidence_gesture_type_to_activate.append(activated_gesture_type)

        gesture_type = self.evidence_gesture_type_to_activate.get_last_common(x, threshold=1.0)
        if gesture_type is not None:
            return gesture_type



    def save_accumulated_data_of_unactivated_gesture_types(self):
        ''' Handle for each gesture type '''

        if (self.prev_gesture_type != 'deictic' and 
            not self.prev_deictic_solutions.empty):

            self.save_accumulated_deictic_gesture_data()
    
        if (self.prev_gesture_type != 'measurement_distance' and 
                not self.prev_auxgesture_solutions.empty):
            
            self.target_auxgesture_solutions.append(get_dist_by_extremes(np.array(self.prev_auxgesture_solutions)))
            self.prev_auxgesture_solutions = CustomDeque()

    def step_deictic(self):
        ''' Activated gesture enabled Deictic gesture mode.
        '''
        deictic_solution = self.get_target_object()

        self.prev_gesture_type = 'deictic'
        self.prev_deictic_solutions.append(deictic_solution)

    def step_approvement(self):
        res = misc_gesture_handle(f"Approve? (y/n)", new_episode=False)
        print(f"Added approvement {res}")
        self.target_auxgesture_solutions.append(res)

    def clearing(self, wait=True):
        self.gestures_queue.clear()
        self.evaluate_episode = False

        self.target_object_solutions = CustomDeque()
        self.target_auxgesture_solutions = CustomDeque()

        if wait:
            print("Move hand out to end the episode!")
            while self.present():
                time.sleep(0.1)
            print("Episode finished!")
            print("=================")
            print("")


    def save_accumulated_deictic_gesture_data(self):

        if self.prev_deictic_solutions.empty:
            print("No object to be added, returning")
            return
        elif len(self.prev_deictic_solutions)<4: # If there are less than 4 samples -> use the last
            self.target_object_solutions.append(self.prev_deictic_solutions[-1])
        else:
            self.target_object_solutions.append(self.prev_deictic_solutions[-4])
        
        print(f"New scene object selected: {self.target_object_solutions[-1]['target_object_name']}")
        self.prev_deictic_solutions = CustomDeque()




class AdaptiveSetup():
    adaptive_setup = {
        'deictic': ('point', 'two'),
        #'approvement': ('thumbsup', 'five'),
        # 'measurement_distance': ('pinch'),
        #'measurement_rotation': ('five'),
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


import threading

def spinning_threadfn(gd):
    while rclpy.ok():
        gd.spin_once(sem=True)
        time.sleep(0.01)


def main():
    rclpy.init()
    sentence_processor = GestureSentence()
    spinning_thread = threading.Thread(target=spinning_threadfn, args=(sentence_processor, ), daemon=True)
    spinning_thread.start()
    while rclpy.ok():
        sentence_processor.step()


if __name__ == '__main__':
    main()