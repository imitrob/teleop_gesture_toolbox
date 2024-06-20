import argparse
import time
from panda_ros import Panda
from teleop_gesture_toolbox.pointing_object_selection.deitic_lib import DeicticLibRos




class DeicticLibPandaRos(Panda, DeicticLibRos):
    pass


def main(hand: str):

    print("Test deictic started")
    dl = DeicticLibRos()

    try:
        while True:
            time.sleep(0.5)
            s = dl.get_scene()
            if s.object_positions_real == []: 
                print("Empty scene!")
                continue
            
            idobj, distances_from_line, line_data = dl.step(dl.hand_frames[-1], hand, s.object_positions_real, plot_line=True)
            
            object_distances = list(zip(s.object_names, distances_from_line))
            nameobj = s.object_names[idobj]

            dl.publish_deictic_data.publish(line_data)

            dl.go_on_top_of_object(nameobj, s)
            print("[Info] Ctrl+C to leave")

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

    main(vars(parser.parse_args()))