
from gesture_msgs.msg import DeicticSolution


class PointingObjectGetter():
    def __init__(self):
        super(PointingObjectGetter, self).__init__()

        self.create_subscriber(DeicticSolution, "/teleop_gesture_toolbox/deictic_solution", self.receive_deictic_solution, 5)

        self.deictic_target_object_id = None
        self.deictic_target_object_name = None
        print("[POG] Done")

    def receive_deictic_solution(self, msg):
        self.deictic_target_object_id = msg.object_id
        self.deictic_target_object_name = msg.object_name
        # self.deictic_object_names = msg.object_names
        # self.deictic_distances_from_line = msg.distances_from_line
        # self.deictic_line_point_1 = msg.line_point_1
        # self.deictic_line_point_2 = msg.line_point_2
        # self.deictic_target_object_position = msg.target_object_position

    def get_target_object(self):
        return self.deictic_target_object_name
        

