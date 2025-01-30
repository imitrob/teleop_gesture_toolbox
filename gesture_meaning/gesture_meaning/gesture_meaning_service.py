
from gesture_sentence_maker.hricommand_export import import_original_HRICommand_to_dict
import rclpy
from rclpy.node import Node

from gesture_msgs.srv import GestureToMeaning
from hri_msgs.msg import HRICommand
import numpy as np

from gesture_meaning.gesture_names_call import GestureListService
from rclpy.qos import QoSProfile, QoSReliabilityPolicy
from std_msgs.msg import String
import threading
import time

class RosNode(Node):
    def __init__(self):
        super(RosNode, self).__init__("gestures_to_actions_node")

class GestureToMeaningNode(GestureListService, RosNode): 
    def __init__(self):
        super(GestureToMeaningNode, self).__init__()
        # service callback
        self.create_service(GestureToMeaning, '/teleop_gesture_toolbox/get_meaning', self.G2I_service_callback)
        # OR subscription
        self.create_subscription(HRICommand, "/teleop_gesture_toolbox/hricommand_original", self.G2I_message_callback, qos_profile=QoSProfile(depth=10, reliability=QoSReliabilityPolicy.BEST_EFFORT))
        self.pub = self.create_publisher(HRICommand, "/modality/gestures", 5)

        self.user = self.declare_parameter("user_name", "").get_parameter_value().string_value

        self.meaning_info_pub = self.create_publisher(String, "/teleop_gesture_toolbox/gesture_meaning_info", 5)
        self.links_str = {}
        thr = threading.Thread(target=self.send_info_thread, daemon=True)    
        thr.start()

        if self.user == "":
            print("No user specified!", flush=True)

    def send_info_thread(self):
        
        while rclpy.ok():
            time.sleep(1.0)

            self.links_str["user"] = self.user

            data_as_str = str(self.links_str)
            data_as_str = data_as_str.replace("'", '"')
        
            self.meaning_info_pub.publish(String(data=data_as_str))

    def G2I_message_callback(self, msg):
        d = import_original_HRICommand_to_dict(msg)

        if 'gesture_names' not in d.keys() or 'gesture_probs' not in d.keys():
            print("HRICommand wihtout gesture solutions, returning", flush=True)
            return

        gesture_names = d['gesture_names']
        gesture_probs = d['gesture_probs']

        assert self.Gs == gesture_names, f"Gesture's set names must match!\nRequest gestures:{gesture_names}\nSampler gestures:\n{self.Gs}"
        
        target_action_probs = self.sample(gesture_probs)
        target_action_name = self.A[np.argmax(target_action_probs)]
        
        self.pub.publish(export_mapped_to_HRICommand(d, list(target_action_probs), self.A, target_action_name))

    def G2I_service_callback(self, request, response):
        assert self.Gs == request.gestures.gesture_names, f"Gesture's set names must match!\nRequest gestures:{request.gestures.gesture_names}\nSampler gestures:\n{self.Gs}"
        
        probs = self.sample(request.gestures.gesture_probs)
        a = self.A[np.argmax(probs)]
        response.intent.target_action = a

        # response.intent.action_probs = list(np.array(probs, dtype=float))
        # response.intent.action_names = self.A

        return response


def export_mapped_to_HRICommand(
        d: dict,
        target_action_probs: list,
        target_action_names: list,
        target_action_name: list
    ):
    assert isinstance(target_action_probs, list)
    
    # keep dict as it is
    d
    # add action solutions
    d["target_action"] = target_action_name
    d["action_names"] = target_action_names
    print("here: ", str(target_action_probs))
    d["action_probs"] = target_action_probs

    data_as_str = str(d)
    data_as_str = data_as_str.replace("'", '"')

    return HRICommand(data=[str(data_as_str)])

class OneToOneMapping(GestureToMeaningNode):
    '''
        self.Gs names obtained via service calling gesture_detector
        Gesture names self.Gs are in order    
        Every gesture needs target action defined or specified empty string
    '''
    A = ['move_up', 'release', 'stop', 'pick_up', 'push', 'unglue', 'pour', 'put', 'stack']
    mapping = {
        #'gesture': -- to --> 'action'
        'swipe_up':           'move_up', 
        'swipe_left':         'push', 
        'swipe_down':         'put', 
        'swipe_right':        'push', 
        'swipe_front_right':  'push', 
        'pinch':              'pick_up',
        'grab':               '', 
        'point':              '', 
        'two':                'unglue', 
        'three':              'stack', 
        'four':               'release',
        'five':               'stop',
        'thumbsup':           'pour',
        'no_moving':          '',
        'thumb':              '',
    }

    def __init__(self):
        super(OneToOneMapping, self).__init__()
        self.T = np.zeros((len(self.A), len(self.Gs)))
        for i, g in enumerate(self.Gs):
            a = self.mapping[g]
            if a != '':
                self.T[self.A.index(a),i] = 1
            
    def sample(self, x):
        return np.max(self.T * x, axis=1)

class OneToOneCompoundMapping(GestureToMeaningNode): # Compound = Combination of Static and Dynamic Gesture
    '''
        self.Gs names obtained via service calling gesture_detector
        Gesture names self.Gs are in order    
        Every gesture needs target action defined or specified empty string
    '''
    A = ['move_up', 'release', 'stop', 'pick_up', 'push', 'unglue', 'pour', 'put', 'stack']
    mapping = {
        #'static gesture' + 'dynamic gesture' = 'action' 
        # NOT IMPLEMENTED:
        'swipe_up':           'move_up', 
        'swipe_left':         'push', 
        'swipe_down':         'put', 
        'swipe_right':        'push', 
        'swipe_front_right':  'push', 
        'pinch':              'pick_up',
        'grab':               '', 
        'point':              '', 
        'two':                'unglue', 
        'three':              'stack', 
        'four':               'release',
        'five':               'stop',
        'thumbsup':           'pour',
        'no_moving':          '',
        'thumb':              '',
    }

    def __init__(self):
        super(OneToOneCompoundMapping, self).__init__()
        self.T = np.zeros((len(self.A), len(self.Gs)))
        for i, g in enumerate(self.Gs):
            a = self.mapping[g]
            if a != '':
                self.T[self.A.index(a),i] = 1
            
    def sample(self, x):
        return np.max(self.T * x, axis=1)

import yaml

class OneToOneCompoundUserMapping(GestureToMeaningNode): # Compound = Combination of Static and Dynamic Gesture
    def __init__(self):
        super(OneToOneCompoundUserMapping, self).__init__()

        import hri_manager
        links_dict = yaml.safe_load(open(f"{hri_manager.package_path}/links/{self.user}_links.yaml", mode='r'))
        self.A = links_dict['actions']
        print(f"Actions: {self.A}", flush=True)
        print(f"Gestures: {self.Gs}, static: {self.Gs_static}, dynamic: {self.Gs_dynamic}", flush=True)
        mapping = []
        self.T = np.zeros((len(self.Gs_static), len(self.Gs_dynamic), len(self.A)))
        
        
        for name,link in links_dict['links'].items():
            self.links_str[name] = link['action_words']
            link['user'] # "melichar"
            link['action_template'] # "push"
            link['object_template'] # "cube_template"
            link['action_words'] # "push"
            assert len(link['action_gestures']) == 1, "There are more more mapping from gesture to action! NotImplementedError"
            static_action_gesture, dynamic_action_gesture = link['action_gestures'][-1] # [grab, swipe right]

            mapping.append([static_action_gesture, dynamic_action_gesture, link['action_template']])

            static_action_gesture_id = self.Gs_static.index(static_action_gesture)
            dynamic_action_gesture_id = self.Gs_dynamic.index(dynamic_action_gesture)
            action_template_id = self.A.index(link['action_template'])

            self.T[static_action_gesture_id, dynamic_action_gesture_id, action_template_id] = 1
        
    def sample(self, x):
        print(self.Gs)
        # split static and dynamic gestures
        static_gesture_p = np.array(x[:len(self.Gs_static)])
        dynamic_gesture_p = np.array(x[len(self.Gs_static):])

        static_gesture_p_expanded = static_gesture_p[:, np.newaxis, np.newaxis]  # Shape: (x1_dim, 1, 1)
        dynamic_gesture_p_expanded = dynamic_gesture_p[np.newaxis, :, np.newaxis]  # Shape: (1, x2_dim, 1)

        # Element-wise multiplication and summation over x1 and x2
        return np.sum(self.T * static_gesture_p_expanded * dynamic_gesture_p_expanded, axis=(0, 1))
        
    
def one_to_one_mapping():
    rclpy.init()
    g2i = OneToOneMapping()

    rclpy.spin(g2i)

def compound_gesture_meaning():
    rclpy.init()
    g2i = OneToOneCompoundMapping()

    rclpy.spin(g2i)

def compound_gesture_user_meaning():
    rclpy.init()
    g2i = OneToOneCompoundUserMapping()

    rclpy.spin(g2i)

if __name__ == "__main__":
    one_to_one_mapping()