
from gesture_msgs.msg import DeicticSolution as DeicticSolutionMSG
import numpy as np
import time
from collections import deque

MAX_DELAY = 0.5

from pointing_object_selection.deictic_lib import DeicticSolution, Point3D, Line3D

class PointingObjectGetter():
    def __init__(self):
        super(PointingObjectGetter, self).__init__()

        self.create_subscription(DeicticSolutionMSG, "/teleop_gesture_toolbox/deictic_solution", self.receive_deictic_solution, 5)
        self.last_stamp = 0.0

        self.target_object_solutions = deque(maxlen=50)

    def receive_deictic_solution(self, msg):
        self.last_stamp = time.time()
        
        self._target_object = DeicticSolution(
            target_object_stamp = msg.header.stamp.sec+msg.header.stamp.nanosec*1e-9,
            target_object_id = msg.target_object_id,
            target_object_name = msg.target_object_name,
            object_names = msg.object_names,
            object_distances = msg.object_distances,
            object_likelihoods = 1/np.array(msg.object_distances),
            line_points = Line3D(start=Point3D(x=msg.line_point_1.x, y=msg.line_point_1.y, z=msg.line_point_1.z),
                                 end=Point3D(x=msg.line_point_2.x, y=msg.line_point_2.y, z=msg.line_point_2.z)),
            target_object_position = Point3D(x=msg.target_object_position.x, y=msg.target_object_position.y, z=msg.target_object_position.z),
            hand_velocity = msg.hand_velocity,
        )
        self.target_object_solutions.append(self._target_object)
    def target_object_valid(self):
        """ Target object is valid if the last solution was received recently """
        if (time.time() - self.last_stamp) < MAX_DELAY:
            return True
        else:
            return False

    def get_target_object(self):
        if self.last_stamp == 0.0: return None
        return self._target_object

    def target_object_age(self):
        return time.time() - self.last_stamp