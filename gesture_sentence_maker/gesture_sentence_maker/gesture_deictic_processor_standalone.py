#!/usr/bin/env python
import time, rclpy, json

from scene_getter.scene_getting import SceneGetter
from pointing_object_selection.pointing_object_getter import PointingObjectGetter
from gesture_detector.utils.utils import CustomDeque
from gesture_sentence_maker.hricommand_export import export_only_objects_to_HRICommand
from gesture_sentence_maker.segmentation_task.deictic_solutions_plot import deictic_solutions_plot_save
from gesture_sentence_maker.segmentation_task.deictic_segment import find_pointed_objects_timewindowmax

from hri_msgs.msg import HRICommand as HRICommandMSG
from rclpy.qos import QoSProfile, QoSDurabilityPolicy

from rclpy.node import Node
import threading
rossem = threading.Semaphore()
class RosNode(Node):
    def __init__(self):
        super().__init__("gesture_deictic_sentence_node")
        self.last_time_livin = time.time()

    def spin_once(self, sem=True):
        if sem:
            with rossem:
                self.last_time_livin = time.time()
                rclpy.spin_once(self)
        else:
            self.last_time_livin = time.time()
            rclpy.spin_once(self)

class GestureDeicticSentence(PointingObjectGetter, SceneGetter, RosNode):
    def __init__(self, step_period: float = 0.2):
        self.topic = "sentence_processor_node"
        super(GestureDeicticSentence, self).__init__()

        self.gesture_sentence_publisher = self.create_publisher(HRICommand, "/teleop_gesture_toolbox/hricommand_deictic", 5)

        # sentence data
        self.deictic_solutions = CustomDeque()
        
        self.step_period = step_period
        self.continue_episode = self._continue_episode
        self.episode_end_flag = False
        qos_profile = QoSProfile(
            depth=5,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL
        )
        self.create_subscription(HRICommandMSG, "/modality/nlp", self.signal_episode_end, qos_profile)
        
    def _continue_episode(self):
        return not self.episode_end_flag

    def signal_episode_end(self, msg):
        self.episode_end_flag = True
        s = msg.data[0]
        s = s.replace("'", '"')
        msg_dict = json.loads(s)
        # self.nlp_target_objects = msg_dict['target_object']
        self.nlp_target_objects_stamp = msg_dict['target_obect_stamp']
        print("received nlp")

    def step(self):
        time.sleep(self.step_period)
        if self.continue_episode():
            if self.target_object_valid():
                # Point gesture precondition
                #if activated_gesture_type_action == 'deictic':
                if True:
                    self.deictic_solutions.append(self.get_target_object())
        
        else: # End episode
            self.episode_end_flag = False
            if not self.deictic_solutions.empty:
                time.sleep(0.5)

                deictic_solutions_plot_save(self.deictic_solutions, nlp_timestamps=self.nlp_target_objects_stamp)
                pointed_objects = find_pointed_objects_timewindowmax(
                    self.deictic_solutions, 
                    target_pointings_stamp=self.nlp_target_objects_stamp
                )
                

                print("pointed_objects: ", pointed_objects)
                self.gesture_sentence_publisher.publish(export_only_objects_to_HRICommand(pointed_objects))
                self.deictic_solutions = CustomDeque()

def export_only_objects_to_HRICommand(pointed_objects):

    sentence_as_dict = {
        'target_object': pointed_objects,
        # 'objects': target_object_names, # This should be all object names detected on the scene
        # 'object_probs': list(target_object_probs), # This should be all object likelihoods 
    }
    data_as_str = str(sentence_as_dict)
    data_as_str = data_as_str.replace("'", '"')

    return HRICommand(data=[str(data_as_str)])   

import threading

def spinning_threadfn(gd):
    while rclpy.ok():
        gd.spin_once(sem=True)
        time.sleep(0.01)

def main():
    rclpy.init()
    sentence_processor = GestureDeicticSentence()
    spinning_thread = threading.Thread(target=spinning_threadfn, args=(sentence_processor, ), daemon=True)
    spinning_thread.start()
    
    while rclpy.ok():
        sentence_processor.step()
        

if __name__ == '__main__':
    main()
