#!/usr/bin/env python
import sys, os, time, threading
import numpy as np
import rclpy

# Init functions create global placeholder objects for data
from os_and_utils import settings; settings.init()
import os_and_utils.move_lib as ml; ml.init()
import os_and_utils.scenes as sl; sl.init()
import gesture_classification.gestures_lib as gl; gl.init()
import os_and_utils.ui_lib as ui
import os_and_utils.ros_communication_main as rc; rc.init('coppelia')

# ProMP path generator
from promps.promp_lib import ProMPGenerator, map_to_primitive_gesture, get_id_motionprimitive_type

# TEMP:
from geometry_msgs.msg import Point, Pose, Quaternion

def main():
    # If gesture detection not enabled in ROSparam -> enable it manually
    #ml.md.eef_pose.position = Point(0.3, 0.0, 0.3)

    #path_generator = ProMPGenerator(promp='sebasutp')
    path_generator = None
    #sl.scenes.make_scene('pickplace3')

    rate = rc.roscm.create_rate(settings.yaml_config_gestures['misc']['rate'])
    try:
        while rclpy.ok():
            ml.md.main_handle_step(path_generator)
            rate.sleep()
    except KeyboardInterrupt:
        pass

    gl.gd.export()
    print("[Main] Ended")

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
