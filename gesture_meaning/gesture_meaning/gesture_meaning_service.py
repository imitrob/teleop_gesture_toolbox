
from gesture_sentence_maker.hricommand_export import import_original_HRICommand_to_dict
import rclpy
from rclpy.node import Node

from gesture_msgs.srv import GestureToMeaning
from gesture_msgs.msg import HRICommand
import numpy as np

from gesture_meaning.gesture_names_call import GestureListService

class RosNode(Node):
    def __init__(self):
        super(RosNode, self).__init__("gestures_to_actions_node")

class GestureToMeaningNode(GestureListService, RosNode): 
    def __init__(self):
        super(GestureToMeaningNode, self).__init__()
        # service callback
        self.create_service(GestureToMeaning, '/teleop_gesture_toolbox/get_meaning', self.G2I_service_callback)
        # OR subscription
        self.create_subscription(HRICommand, "/teleop_gesture_toolbox/hricommand_original", self.G2I_message_callback, 10)
        self.pub = self.create_publisher(HRICommand, "/modality/gestures", 5)

    def G2I_message_callback(self, msg):
        d = import_original_HRICommand_to_dict(msg)

        if 'gestures' not in d.keys() or 'gesture_probs' not in d.keys():
            print("HRICommand wihtout gesture solutions, returning")
            return

        gesture_names = d['gestures']
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
    d["actions"] = target_action_names
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

def main():
    rclpy.init()
    g2i = OneToOneMapping()

    rclpy.spin(g2i)

if __name__ == "__main__":
    main()
