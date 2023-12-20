#!/usr/bin/env python
import sys, os, time, threading
import numpy as np
import rclpy

# Init functions create global placeholder objects for data
import os_and_utils.settings as settings; settings.init()
import gesture_classification.gestures_lib as gl; gl.init()
import os_and_utils.move_lib as ml; ml.init()
import os_and_utils.ros_communication_main as rc; rc.init(robot_interface='coppelia', setup='mirracle')
import os_and_utils.scenes as sl; sl.init()
import os_and_utils.deitic_lib as dl; dl.init()
import os_and_utils.ui_lib as ui

from gesture_classification.sentence_creation import GestureSentence

def main():
    # No action execution when running as "gesture detect node"
    settings.action_execution = False

    rate = rc.roscm.create_rate_(10) #settings.yaml_config_gestures['misc']['rate']) # Hz
    while rclpy.ok():
        # TODO: Execute action from GUI         
        # if len(self.todoactionqueue) > 0:
        #     action, o1, o2 = self.todoactionqueue.pop()
        #     getattr(rral,action)((o1,o2))

        t1 = time.perf_counter()
        # Send gesture data based on hand mode
        if gl.gd.present() and not gl.gd.any_hand_stable(time=1.0):
            rc.roscm.send_g_data()
            
        if len(gl.gd.gestures_queue) > 0:
            GestureSentence.adaptive_eee(None, s=rc.roscm.update_scene(), mode='modular', s_func=rc.roscm.update_scene)
        
        rc.roscm.send_state()
        rate.sleep()
        
        print(f"{time.perf_counter()-t1}")
    print("quit")

def spinning_threadfn():
    while rclpy.ok():
        rc.roscm.spin_once(sem=True)
        time.sleep(0.01)

def main():
    ''' Default main has three threads: 1. ROS spin, 2. GUI (optional), 3. main
    '''
    if 'noui' in sys.argv: settings.launch_ui = False
    # Spin in a separate thread
    spinning_thread = threading.Thread(target=spinning_threadfn, args=(), daemon=True)
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

if __name__ == '__main__':
    main()