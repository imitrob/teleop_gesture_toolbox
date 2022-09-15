#!/usr/bin/env python3.8
import sys, os, time, threading
import numpy as np
import rospy
from os_and_utils.nnwrapper import NNWrapper
# Init functions create global placeholder objects for data
import settings
settings.init()
import os_and_utils.move_lib as ml
ml.init()
import os_and_utils.scenes as sl
sl.init()
import gestures_lib as gl
gl.init()



import ui_lib as ui
from os_and_utils.ros_communication_main import ROSComm
from os_and_utils.parse_yaml import ParseYAML

sys.path.append(settings.paths.mirracle_sim_path)
from coppelia_sim_ros_client import CoppeliaROSInterface

from promps.promp_lib import ProMPGenerator

def main():
    # If gesture detection not enabled in ROSparam -> enable it manually
    settings.gesture_detection_on = True
    settings.launch_gesture_detection = True
    # Vizdata
    vizdataact = []
    vizdataabovethre = []
    viztime_dyn = []
    vizdata_dyn = []
    viztime_sta = []
    vizdata_sta = []

    roscm = ROSComm()
    prompg = ProMPGenerator(promp='sebasutp')
    sl.scenes.make_scene(None, 'pickplace')
    rate = rospy.Rate(settings.yaml_config_gestures['misc']['rate']) # Hz
    try:
        while not rospy.is_shutdown():
            if ml.md.present():
                # 1. Send gesture data based on hand mode
                if ml.md.frames and settings.gesture_detection_on:
                    roscm.send_g_data()

                # 2. Info + Save plot data
                print(f"fps {ml.md.frames[-1].fps}, id {ml.md.frames[-1].seq}")
                print(f"actions queue {[act[1] for act in gl.gd.actions_queue]}")

                # Printing presented here represent current mapping
                if gl.gd.r.dynamic.relevant():
                    print(f"Right Dynamic relevant info: Biggest prob. gesture: {gl.gd.r.dynamic.relevant().biggest_probability}")
                    viztime_dyn.append(ml.md.frames[-1].seq)
                    vizdata_dyn.append(gl.gd.r.dynamic.relevant().probabilities_norm)
                if gl.gd.l.static.relevant():
                    viztime_sta.append(ml.md.frames[-1].seq)
                    vizdata_sta.append(gl.gd.l.static.relevant().probabilities)
                    print(f"Left Static relevant info: Biggest prob. gesture: {gl.gd.l.static.relevant().biggest_probability}")

            # 3. Handle gesture activation
            if len(gl.gd.actions_queue) > 0:
                prompg.handle_action_queue(gl.gd.actions_queue.pop())

            rate.sleep()
    except KeyboardInterrupt:
        pass


    vizdata_dyn = np.array(vizdata_dyn).T
    vizdata_sta = np.array(vizdata_sta).T
    #vizdataact = np.array(vizdataact).T
    #vizdataabovethre = np.array(vizdataabovethre).T

    from os_and_utils.visualizer_lib import VisualizerLib
    viz = VisualizerLib()
    viz.visualize_new_fig("Gesture probability through time", dim=2)
    if vizdata_dyn.any():
        for n in range(2):#gl.gd.r.dynamic.info.n):
            viz.visualize_2d(list(zip(viztime_dyn, vizdata_dyn[n])),label=f"{gl.gd.dynamic_info().names[n]}", xlabel='leap seq [-] (~100 seq/sec)', ylabel='Gesture probability', start_stop_mark=False)

    if vizdata_sta.any():
        for n in range(2):#gl.gd.r.static.info.n):
            viz.visualize_2d(list(zip(viztime_sta, vizdata_sta[n])),label=f"{gl.gd.static_info().names[n]}", xlabel='leap seq [-] (~100 seq/sec)', ylabel='Gesture probability', start_stop_mark=False)

    #viz.visualize_new_fig("Gesture activates through time", dim=2)
    #for n in range(gl.gd.r.dynamic.info.n):
    #    viz.visualize_2d(list(zip(viztime, vizdataact[n])),label=f"{gl.gd.r.dynamic.info.names[n]}", xlabel='time, seq [-] (~100 seq/sec)', ylabel='Gesture activates')
    #viz.visualize_new_fig("Gesture above threshold through time", dim=2)
    #for n in range(gl.gd.r.dynamic.info.n):
    #    viz.visualize_2d(list(zip(viztime, vizdataabovethre[n])),label=f"{gl.gd.r.dynamic.info.names[n]}", xlabel='time, seq [-] (~100 seq/sec)', ylabel='Gesture above threshold')

    gl.gd.export()
    print("[Main] Ended")


if __name__ == '__main__':
    if len(sys.argv)>1 and sys.argv[1] == 'ui':
        settings.launch_ui = True

    rospy.init_node('main_manager', anonymous=True)

    if settings.launch_ui:
        thread_main = threading.Thread(target = main)
        thread_main.daemon=True
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
