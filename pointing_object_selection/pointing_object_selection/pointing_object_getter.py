
from gesture_msgs.msg import DeicticSolution
import numpy as np
import time

MAX_DELAY = 0.5

class PointingObjectGetter():
    def __init__(self):
        super(PointingObjectGetter, self).__init__()

        self.create_subscription(DeicticSolution, "/teleop_gesture_toolbox/deictic_solution", self.receive_deictic_solution, 5)
        self.last_stamp = 0.0

    def receive_deictic_solution(self, msg):
        self.last_stamp = time.time()
        
        self.deictic_last_stamp = msg.header.stamp.sec+msg.header.stamp.nanosec*1e-9
        self.deictic_target_object_id = msg.object_id
        self.deictic_target_object_name = msg.object_name
        self.deictic_object_names = msg.object_names
        self.deictic_distances_from_line = msg.distances_from_line
        self.deictic_line_point_1 = msg.line_point_1
        self.deictic_line_point_2 = msg.line_point_2
        self.deictic_target_object_position = msg.target_object_position
        self.deictic_hand_velocity = msg.hand_velocity

    def target_object_valid(self):
        """ Target object is valid if the last solution was received recently """
        if (time.time() - self.last_stamp) < MAX_DELAY:
            return True
        else:
            return False

    def get_target_object(self):
        if self.last_stamp == 0.0: return None

        return {
            "target_object_stamp": self.deictic_last_stamp,
            "target_object_id": self.deictic_target_object_id,
            "target_object_name": self.deictic_target_object_name,
            "object_names": self.deictic_object_names,
            "object_likelihoods": 1/np.array(self.deictic_distances_from_line),
            "object_distances": self.deictic_distances_from_line,
            "line_points": [self.deictic_line_point_1, self.deictic_line_point_2],
            "target_object_position": self.deictic_target_object_position,
            "hand_velocity": self.deictic_hand_velocity,
        }
        

