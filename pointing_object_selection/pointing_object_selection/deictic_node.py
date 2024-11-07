import argparse
import time
from pointing_object_selection.deictic_lib import DeiticLib
import rclpy

from scene_getter.scene_getter import SceneGetter
from rclpy.node import Node
from gesture_msgs.msg import DeicticSolution
from gesture_detector.hand_processing.hand_listener import HandListener
from geometry_msgs.msg import Point

from pointing_object_selection.transform_ros_getter import TransformUpdater

class NamedRosNode(Node):
    def __init__(self):
        super(NamedRosNode, self).__init__("deitic_ros_node")

class DeicticLibRos(DeiticLib, TransformUpdater, HandListener, SceneGetter, NamedRosNode):
    def __init__(self, hand):
        self.hand = hand
        super(DeicticLibRos, self).__init__()
        
        self.deictic_solutions_pub = self.create_publisher(DeicticSolution, "/teleop_gesture_toolbox/deictic_solution", 5)

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

    def step(self):
        if self.latest_transform is None:
            print("Waiting for TF")
            return

        s = self.get_scene()

        if s.object_positions == []: 
            print("Empty scene!")
            return
        if len(self.hand_frames) == 0: 
            print("No hand data!")
            return

        deictic_solution = self.compute_deictic_solution(
            self.hand_frames[-1], 
            self.hand, 
            s.object_positions, 
            s.object_names,
            self.apply_transform,
        )
        if deictic_solution is None:
            return None
        
        self.deictic_solutions_pub.publish(
            self.deictic_solution_to_ros(deictic_solution)
        )
        return deictic_solution


def main(args):
    rclpy.init()
    print("Test deictic started")
    dl = DeicticLibRos(hand=args['hand'])

    print("[Info] Ctrl+C to leave")
    try:
        while True:
            rclpy.spin_once(dl)
            time.sleep(1/args['frequency'])
            dl.step()
            
    except KeyboardInterrupt:
        print("KeyboardInterrupt: Test deictic ended\n\n")

def run_node_default():
    main(args = {
        'hand': "lr",
        'frequency': 2,
    })

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog="Run Deictic node",
        description="",
        epilog="",
    )
    parser.add_argument(
        "--hand",
        default="lr",
        choices=["l", "r", "lr"],
    )
    parser.add_argument(
        "--frequency",
        default=2,
        type=int,
    )

    main(vars(parser.parse_args()))