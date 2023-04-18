#!/usr/bin/env python
import sys, os, time, threading
sys.path.append(os.path.join(os.path.abspath(__file__), "..", '..', 'python3.9', 'site-packages', 'teleop_gesture_toolbox'))
import numpy as np
import rclpy

# Init functions create global placeholder objects for data
import os_and_utils.settings as settings; settings.init()
import os_and_utils.move_lib as ml; ml.init()
import gesture_classification.gestures_lib as gl; gl.init()
import os_and_utils.ros_communication_gestures as rc; rc.init()

def main():
    rate = rc.roscm.create_rate(settings.yaml_config_gestures['misc']['rate']) # Hz
    while rclpy.ok():
        # Send gesture data based on hand mode
        if ml.md.frames and settings.gesture_detection_on:
            rc.roscm.send_g_data()

        if len(gl.gd.gestures_queue) > 0:
            action = gl.gd.gestures_queue.pop()
            print(action[1])
        rate.sleep()
    print("quit")

if __name__ == '__main__':
    spinning_thread = threading.Thread(target=rclpy.spin, args=(rc.roscm, ), daemon=True)
    spinning_thread.start()
    main()
    print('[Main] Interrupted')
    try:
        sys.exit(0)
    except SystemExit:
        os._exit(0)

Describe paper Attention Is All You Need from 2017.