import argparse
import time
from pointing_object_selection.deictic_lib import DeiticLib, DeicticSolution
import rclpy

from scene_getter.scene_getting import SceneGetter
from rclpy.node import Node
from gesture_msgs.msg import DeicticSolution as DeicticSolutionMSG
from gesture_detector.hand_processing.hand_listener import HandListener
from geometry_msgs.msg import Point
from std_msgs.msg import Header
from pointing_object_selection.transform_ros_getter import TFBaseLeapworld

class NamedRosNode(Node):
    def __init__(self):
        super(NamedRosNode, self).__init__("deitic_ros_node")

class DeicticLibRos(DeiticLib, TFBaseLeapworld, HandListener, SceneGetter, NamedRosNode):
    # Note: TFBaseLeapworld (Leapworld frame) exists for both Leap and RealSense 
    def __init__(self, hand):
        self.hand = hand
        super(DeicticLibRos, self).__init__()
        
        self.deictic_solutions_pub = self.create_publisher(DeicticSolutionMSG, "/teleop_gesture_toolbox/deictic_solution", 5)

    def deictic_solution_to_ros(self, deictic_solution: DeicticSolution):
        line_point_1 = deictic_solution.line_points.start
        line_point_2 = deictic_solution.line_points.end
        to_position =  deictic_solution.target_object_position

        return DeicticSolutionMSG(
            header = Header(stamp=self.get_clock().now().to_msg(), frame_id="leapworld"),
            target_object_id = deictic_solution.target_object_id,
            target_object_name = deictic_solution.target_object_name,
            object_names = deictic_solution.object_names,
            object_distances = deictic_solution.object_distances,
            line_point_1 = Point(x=line_point_1.x, y=line_point_1.y, z=line_point_1.z),
            line_point_2 = Point(x=line_point_2.x, y=line_point_2.y, z=line_point_2.z),
            target_object_position = Point(x=to_position.x,y=to_position.y,z=to_position.z),
            hand_velocity = deictic_solution.hand_velocity,
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
            self.apply_transform, # tf to base received by TFBaseLeapworld
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
        'frequency': 10,
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
        default=10,
        type=int,
    )

    main(vars(parser.parse_args()))