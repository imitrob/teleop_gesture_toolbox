
from typing import Iterable
from gesture_detector.hand_processing.hand_listener import HandListener
import numpy as np

from scene_getter.scene_getter import SceneGetter
from pointing_object_selection.transform import transform_leap_to_base

from rclpy.node import Node
from geometry_msgs.msg import Point
from gesture_msgs.msg import DeicticSolution

class DeiticLib(HandListener):
    def __init__(self):
        super(DeiticLib, self).__init__()
        self.disabled = False

    def get_closest_point_to_line(self,line_points, test_point):
        ''' Line points
        '''
        p1, p2 = line_points
        p3 = test_point

        x1, y1, z1 = p1
        x2, y2, z2 = p2
        x3, y3, z3 = p3
        dx, dy, dz = x2-x1, y2-y1, z2-z1
        det = dx*dx + dy*dy + dz*dz
        a = (dy*(y3-y1)+dx*(x3-x1)+dz*(z3-z1))/det
        return x1+a*dx, y1+a*dy, z1+a*dz

    def get_id_of_closest_point_to_line(self, line_points, test_points, max_dist=np.inf):
        '''
        Embedding: line_points are start and end of last bone (distal) from pointing finger (2nd finger)
        test_points:
        '''
        assert test_points != [], "test_points empty"
        distances_from_line = []
        for test_point in test_points:
            closest_point = self.get_closest_point_to_line(line_points, test_point)
            norm_distance = np.linalg.norm(np.array(closest_point)-np.array(test_point))
            distances_from_line.append(norm_distance)

        if np.min(distances_from_line) > max_dist: return None, np.min(distances_from_line)
        print(f"[Deictic] Chosen: {np.argmin(distances_from_line)}, {np.min(distances_from_line)}, {distances_from_line}")
        return int(np.argmin(distances_from_line)), np.min(distances_from_line), distances_from_line

    def step(self, f, h: str, object_poses: Iterable[list], object_names: Iterable[str]):
        """
        Args:
            f (Frame): self.hand_frames[-1] (take last detection frame)
            h (str): hand - 'l' left, 'r', right
            object_poses (Iterable[list]): sl.scene.object_poses
            object_names (Iterable[str]): _description_

        Returns:
            dict: deictic_solution
        """
        if len(object_poses) == 0: return None

        if h == 'lr':
            if f.l.visible:
                h = 'l'
            elif f.r.visible:
                h = 'r'
            else:
                print("[WARNING] Deictic has no visible hand!")
                h = 'l'
        if not isinstance(h, str): 
            print(f"h is not string, h is {type(h)}")
        hand = getattr(f, h)

        p1, p2 = np.array(hand.palm_position()), np.array(hand.palm_position())+np.array(hand.direction())
        #p1, p2 = np.array(hand.fingers[1].bones[3].prev_joint()), np.array(hand.fingers[1].bones[3].next_joint())
        p1s = np.array(transform_leap_to_base(p1))
        p2s = np.array(transform_leap_to_base(p2))
        v = 1000*(p2s-p1s)
        line_points = [list(p1s), list(p2s+v)]

        if isinstance(object_poses[0], (list, tuple, np.ndarray)):
            object_positions = [[pose[0],pose[1],pose[2]] for pose in object_poses]
        else:
            object_positions = [[pose.position.x,pose.position.y,pose.position.z] for pose in object_poses]
        idobj, _, distances_from_line = self.get_id_of_closest_point_to_line(line_points, object_positions, max_dist=np.inf)
        
        assert len(object_poses) == len(distances_from_line)

        deictic_solution = {
            "object_id": idobj,
            "object_name": object_names[idobj],
            "object_names": object_names,
            "distances_from_line": distances_from_line,
            "line_point_1": line_points[0],
            "line_point_2": line_points[1],
            "target_object_position": object_positions[idobj],
        }
        return deictic_solution
    
    @staticmethod
    def deictic_solution_to_ros(deictic_solution: dict):
        line_point_1 = deictic_solution['line_point_1']
        line_point_2 = deictic_solution['line_point_2']
        to_position = deictic_solution['target_object_position']

        return DeicticSolution(
            object_id = deictic_solution['object_id'],
            object_name = deictic_solution['object_name'],
            object_names = deictic_solution['object_names'],
            distances_from_line = deictic_solution['distances_from_line'],
            line_point_1 = Point(x=line_point_1[0], y=line_point_1[1], z=line_point_1[2]),
            line_point_2 = Point(x=line_point_2[0], y=line_point_2[1], z=line_point_2[2]),
            target_object_position = Point(x=to_position[0],y=to_position[1],z=to_position[2]),
        )


    def set_focus_logic(self, hand):
        '''
        Set focus action can be enabled again only when:
            1.) done grab gesture (hand.grab_strength > 0.9)
            2.) hand out of scene (triggered externally in main script)
        '''
        if not self.disabled:
            self.disabled = True
            return True
        else:
            if hand.grab_strength > 0.9:
                self.enable()
            return False

    def enable(self):
        self.disabled = False


class DeicticLibRos(DeiticLib, SceneGetter, Node):
    def __init__(self):
        Node.__init__(self, "deictic_ros_node")
        DeiticLib.__init__(self)
        SceneGetter.__init__(self)
        
        self.deictic_solutions_pub = self.create_publisher(DeicticSolution, "/teleop_gesture_toolbox/deictic_solution", 5)

