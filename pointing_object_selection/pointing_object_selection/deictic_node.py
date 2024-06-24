import argparse
import time
# from panda_ros import Panda
from pointing_object_selection.deictic_lib import DeicticLibRos
import rclpy


# class DeicticLibPandaRos(Panda, DeicticLibRos):
#     pass


def main(args):
    rclpy.init()
    print("Test deictic started")
    dl = DeicticLibRos()

    print("[Info] Ctrl+C to leave")
    try:
        while True:
            rclpy.spin_once(dl)
            time.sleep(1/args['frequency'])
            s = dl.get_scene()

            if s.object_positions == []: 
                print("Empty scene!")
                continue
            
            deictic_solution = dl.step(dl.hand_frames[-1], args['hand'], s.object_positions, s.object_names)
            
            dl.deictic_solutions_pub.publish(
                dl.deictic_solution_to_ros(deictic_solution)
            )
            # dl.go_on_top_of_object(nameobj, s)
        

    except KeyboardInterrupt:
        print("KeyboardInterrupt: Test deictic ended\n\n")


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