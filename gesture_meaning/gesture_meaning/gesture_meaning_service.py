
from gesture_sentence_maker.hricommand_export import import_original_HRICommand_to_dict
import rclpy
from rclpy.node import Node

from gesture_msgs.srv import GestureToMeaning
from gesture_msgs.msg import HRICommand
import numpy as np


class GestureToMeaningNode(Node): 
    def __init__(self):
        super().__init__("gestures_to_actions_node")
        # service callback
        self.create_service(GestureToMeaning, '/teleop_gesture_toolbox/get_meaning', self.G2I_service_callback)
        # OR subscription
        self.create_subscription(HRICommand, "/teleop_gesture_toolbox/hricommand_original", self.G2I_message_callback, 10)
        self.pub = self.create_publisher(HRICommand, "/hri/command", 5)

        self.sampler = OneToOne_Sample()
    
    def G2I_message_callback(self, msg):
        d = import_original_HRICommand_to_dict(msg)

        if 'gestures' not in d.keys() or 'gesture_probs' not in d.keys():
            print("HRICommand wihtout gesture solutions, returning")
            return

        gesture_names = d['gestures']
        gesture_probs = d['gesture_probs']

        assert self.sampler.G == gesture_names, "Gesture's set names must match!"
        
        target_action_probs = self.sampler.sample(gesture_probs)
        target_action_name = self.sampler.A[np.argmax(target_action_probs)]
        
        self.pub.publish(export_mapped_to_HRICommand(d, target_action_probs, self.sampler.A, target_action_name))

    def G2I_service_callback(self, request, response):
        assert self.sampler.G == request.gestures.gesture_names, "Gesture's set names must match!"
        
        probs = self.sampler.sample(request.gestures.gesture_probs)
        a = self.sampler.A[np.argmax(probs)]
        response.intent.target_action = a

        # response.intent.action_probs = list(np.array(probs, dtype=float))
        # response.intent.action_names = self.sampler.A

        return response


def export_mapped_to_HRICommand(d, target_action_probs, target_action_names, target_action_name):
    # keep dict as it is
    d
    # add action solutions
    d["target_action"] = target_action_name
    d["actions"] = target_action_names
    d["action_probs"] = target_action_probs

    data_as_str = str(d)
    print(data_as_str)
    data_as_str = data_as_str.replace("'", '"')

    return HRICommand(data=[str(data_as_str)])

class OneToOne_Sample():
    G = ['swipe_up', 'swipe_left', 'swipe_down', 'swipe_right', 'pinch','grab', 'point', 'two', 'three', 'four', 'five', 'thumbsup']
    A = ['move_up', 'release', 'stop', 'pick_up', 'push', 'unglue', 'pour', 'put', 'stack']

    def __init__(self):
        "# gestures:           up,left,down,rght,pinh,grab,point,two,thre,four,five,thumbsup\n",
        self.T =    np.array([[ 1,   0,   0,   0,   0, 0,    0,  0,   0,   0,   0,  0],  # move_up
                              [ 0,   0,   0,   0,   0, 0,    0,  0,   0,   1,   0,  0],  # release
                              [ 0,   0,   0,   0,   0, 0,    0,  0,   0,   0,   1,  0],  # stop
                              [ 0,   0,   0,   0,   1, 0,    0,  0,   0,   0,   0,  0],  # pick_up
                              [ 0,   1,   0,   1,   0, 0,    0,  0,   0,   0,   0,  0],  # push
                              [ 0,   0,   0,   0,   0, 0,    0,  1,   0,   0,   0,  0],  # unglue
                              [ 0,   0,   0,   0,   0, 0,    0,  0,   0,   0,   0,  1],  # pour
                              [ 0,   0,   1,   0,   0, 0,    0,  0,   0,   0,   0,  0],  # put
                              [ 0,   0,   0,   0,   0, 0,    0,  0,   1,   0,   0,  0]], dtype=float) # stack
    
    def sample(self, x):
        return np.max(self.T * x, axis=1)

def main():
    rclpy.init()
    g2i = GestureToMeaningNode()

    rclpy.spin(g2i)

if __name__ == "__main__":
    main()
