
import rclpy
from rclpy.node import Node

from teleop_msgs.srv import G2I
import numpy as np


class GestureToMeaningNode(Node): 
    def __init__(self):
        super().__init__("gestures_to_actions_node")
        self.create_service(G2I, '/teleop_gesture_toolbox/get_meaning', self.G2I_service_callback)


        self.sampler = OneToOne_Sample()
        
    def G2I_service_callback(self, request, response):

        assert self.sampler.G == request.gesture_names, "Gesture's set names must match!"
        

        probs = self.sampler.sample(request.gesture_probs)[0]
        a = self.A[np.argmax(probs)]
        

        response.intent.target_action = a
        
        response.intent.action_probs = list(np.array(probs, dtype=float))
        response.intent.action_names = self.sampler.A

        return response



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
    
    def sample(self, data):
        return np.max(self.T * data.data, axis=1)


if __name__ == "__main__":
    rclpy.init()
    g2i = GestureToMeaningNode()

    rclpy.spin(g2i)