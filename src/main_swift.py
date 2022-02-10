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
from os_and_utils.swift_interface import Swift

from promps.promp_lib import ProMPGenerator

def main():
    # If gesture detection not enabled in ROSparam -> enable it manually
    settings.gesture_detection_on = True
    settings.launch_gesture_detection = True

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
                if gl.gd.l.static.relevant():
                    print(f"Left Static relevant info: Biggest prob. gesture: {gl.gd.l.static.relevant().biggest_probability}")

            # 3. Handle gesture activation
            if False: #len(gl.gd.actions_queue) > 0:
                path = prompg.handle_action_queue(gl.gd.actions_queue.pop())
                print(f"path {path[0]}")
                swft = Swift()
                swft.trajectory(path[0])

            rate.sleep()
    except KeyboardInterrupt:
        pass

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
