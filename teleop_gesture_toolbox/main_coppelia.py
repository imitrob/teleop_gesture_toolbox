#!/usr/bin/env python
import sys, os, time, threading
sys.path.append(os.path.join(os.path.abspath(__file__), "..", '..', 'python3.9', 'site-packages', 'teleop_gesture_toolbox'))
import numpy as np
import rclpy

# Init functions create global placeholder objects for data
from os_and_utils import settings; settings.init()
import os_and_utils.move_lib as ml; ml.init()
import os_and_utils.scenes as sl; sl.init()
import gesture_classification.gestures_lib as gl; gl.init()
import os_and_utils.ui_lib as ui
import os_and_utils.ros_communication_main as rc; rc.init('coppelia')
import os_and_utils.deitic_lib as dl; dl.init()

# ProMP path generator
from promps.promp_lib import ProMPGenerator, map_to_primitive_gesture, get_id_motionprimitive_type


def main():
    path_generator = None
    path_generator = ProMPGenerator(promp='sebasutp')
    sl.scenes.make_scene_from_yaml('pickplace3')

    #static_network_file = settings.get_network_file(type='static')
    #dynamic_network_file = 
    #print("Static network file: ", settings.get_network_file(type='static'))
    #print("Static network info: ", gl.gd.static_network_info)

    rate = rc.roscm.create_rate_(settings.yaml_config_gestures['misc']['rate'])
    try:
        while rclpy.ok():
            if ml.md.frames and settings.gesture_detection_on:
                ml.md.main_handle_step(path_generator)
                if ml.md.frames:
                    f = ml.md.frames[-1]
                    if f.l.visible and f.l.grab_strength < 0.1:
                        dl.dd.main_deitic_fun(ml.md.frames[-1], 'l', sl.scene.object_poses)
                    if not f.l.visible:
                        dl.dd.enable() # enabled focusing on new object
                    if f.l.visible and f.l.grab_strength > 0.9:
                        if f.l.pinch_strength > 0.9:
                            rc.roscm.r.pick_object(object=sl.scene.object_names[ml.md.object_focus_id])
                        elif f.l.pinch_strength < 0.1:
                            rc.roscm.r.release_object()

            rate.sleep()
    except KeyboardInterrupt:
        pass

    gl.gd.export()
    print("[Main] Ended")

def spinning_threadfn():
    while rclpy.ok():
        rc.roscm.spin_once()
        time.sleep(0.001)

if __name__ == '__main__':
    ''' Default main has three threads: 1. ROS spin, 2. GUI (optional), 3. main
    '''
    if len(sys.argv)>1 and sys.argv[1] == 'noui': settings.launch_ui = False
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
