#!/usr/bin/env python
''' Sends gesture data and receive results in GUI
'''
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
#settings.simulator =
import os_and_utils.ros_communication_main as rc; rc.init('real')

def main():
    with rc.rossem:
        rate = rc.roscm.create_rate(1.0)

    #spinnin_thread = threading.Thread(target=rclpy.spin, daemon=True, args=(rc.roscm,))
    #spinnin_thread.start()
    spinning_thread = threading.Thread(target=spinning_threadfn2, args=(), daemon=True)
    spinning_thread.start()

    try:
        while rclpy.ok():
            with rc.rossem:
                ml.md.main_handle_step(None)
                #rc.roscm.send_g_data()
            time.sleep(0.01)#rate.sleep()
    except KeyboardInterrupt:
        pass

    gl.gd.export()
    print("[Main] Ended")

def spinning_threadfn():
    while rclpy.ok():
        with rc.rossem:
            rclpy.spin_once(rc.roscm)
        time.sleep(0.001)

def spinning_threadfn2():
    while rclpy.ok():
        rclpy.spin_once(rc.roscm)
        time.sleep(0.001)

if __name__ == '__main__':
    ''' Default main has three threads: 1. ROS spin, 2. GUI (optional), 3. main
    '''
    if len(sys.argv)>1 and sys.argv[1] == 'noui': settings.launch_ui = False
    # Spin in a separate thread
    #spinning_thread = threading.Thread(target=spinning_threadfn, args=(), daemon=True)
    #spinning_thread.start()
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
