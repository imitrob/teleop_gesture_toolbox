#!/usr/bin/env python
import sys, os, time, threading
sys.path.append(os.path.join(os.path.abspath(__file__), "..", '..', 'python3.9', 'site-packages', 'teleop_gesture_toolbox'))
import numpy as np
import rclpy

# Init functions create global placeholder objects for data
import os_and_utils.settings as settings; settings.init()
import os_and_utils.move_lib as ml; ml.init()
import os_and_utils.scenes as sl; sl.init()
import gesture_classification.gestures_lib as gl; gl.init()
import os_and_utils.ui_lib as ui
import os_and_utils.ros_communication_main as rc; rc.init()

from std_msgs.msg import String

def main():
    rate = rc.roscm.create_rate_(settings.yaml_config_gestures['misc']['rate']) # Hz
    gspub = rc.roscm.create_publisher(String, "/gesture_sentence", 5)
    while rclpy.ok():
        # Send gesture data based on hand mode
        if ml.md.frames and settings.gesture_detection_on:
            rc.roscm.send_g_data()

        if len(gl.gd.gestures_queue) > 0:
            action = gl.gd.gestures_queue.pop()
            if action[1] != "no_moving": print(action[1])
            gspub.publish(String(data=action[1]))
        rate.sleep()
    print("quit")

if __name__ == '__main__':
    ''' Default main has three threads: 1. ROS spin, 2. GUI (optional), 3. main
    '''
    if len(sys.argv)>1 and sys.argv[1] == 'noui': settings.launch_ui = False
    # Spin in a separate thread
    spinning_thread = threading.Thread(target=rclpy.spin, args=(rc.roscm, ), daemon=True)
    spinning_thread.start()
    if settings.launch_ui:
        thread_main = threading.Thread(target = main, daemon=True)
        thread_main.start()
        try:
            app = ui.QApplication(sys.argv)
            ex = ui.Example()
            sys.exit(app.exec_())
        except KeyboardInterrupt:
            pass
    else:
        main()
    print('[Main] Interrupted')
    try:
        sys.exit(0)
    except SystemExit:
        os._exit(0)
