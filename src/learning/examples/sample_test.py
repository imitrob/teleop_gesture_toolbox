#!/usr/bin/env python3.8
import sys, os, time
import numpy as np
import rospy

sys.path.append("../..")
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

def main():
    # If gesture detection not enabled in ROSparam -> enable it manually
    settings.gesture_detection_on = True
    settings.launch_gesture_detection = True

    rospy.init_node('main_manager', anonymous=True)
    roscm = ROSComm()

    configGestures = ParseYAML.load_gesture_config_file(settings.paths.custom_settings_yaml)
    hand_mode_set = configGestures['using_hand_mode_set']
    hand_mode = dict(configGestures['hand_mode_sets'][hand_mode_set])

    rate = rospy.Rate(1.)
    while not rospy.is_shutdown():
        if ml.md.r_present():
            print(f"fps {ml.md.frames[-1].fps}, id {ml.md.frames[-1].seq}")

        # Send gesture data based on hand mode
        if ml.md.frames and settings.gesture_detection_on:
            gl.send_g_data(roscm, hand_mode, args={'type': 'old_defined'})
        # Need arrived data in gl.gd.l[0.0].static.prob

        rate.sleep()


if __name__ == '__main__':
    main()
    while not rospy.is_shutdown():
        time.sleep(1)
    print('[Main] Interrupted')
    try:
        sys.exit(0)
    except SystemExit:
        os._exit(0)
