#!/usr/bin/env python
import sys, os, time, threading
sys.path.append(os.path.join(os.path.abspath(__file__), "..", '..', 'python3.9', 'site-packages', 'teleop_gesture_toolbox'))
import numpy as np
import rclpy

# Init functions create global placeholder objects for data
import os_and_utils.settings as settings; settings.init()
import gesture_classification.gestures_lib as gl; gl.init()
import os_and_utils.ros_communication_main as rc; rc.init()

from std_msgs.msg import String

from gesture_classification.sentence_creation import GestureSentence

def main():
    #rate = rc.roscm.create_rate_(settings.yaml_config_gestures['misc']['rate']) # Hz
    gspub = rc.roscm.create_publisher(String, "/gesture_sentence", 5)
    while rclpy.ok():
        t1 = time.perf_counter()
        # Send gesture data based on hand mode
        if gl.gd.hand_frames and settings.gesture_detection_on:
            rc.roscm.send_g_data()
            
            GestureSentence.adaptive_eee(None, s=None, mode='modular')
        print(f"{time.perf_counter()-t1}")
        #rate.sleep()
    print("quit")

if __name__ == '__main__':
    ''' Default main has three threads: 1. ROS spin, 2. GUI (optional), 3. main
    '''
    settings.launch_ui = False
    # Spin in a separate thread
    spinning_thread = threading.Thread(target=rclpy.spin, args=(rc.roscm, ), daemon=True)
    spinning_thread.start()
    
    main()
    print('[Main] Interrupted')
    try:
        sys.exit(0)
    except SystemExit:
        os._exit(0)
