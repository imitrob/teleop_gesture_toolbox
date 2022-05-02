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
#from os_and_utils.parse_yaml import ParseYAML

from promps.promp_lib import ProMPGenerator, map_to_primitive_gesture, get_id_motionprimitive_type

sys.path.append(settings.paths.mirracle_sim_path)
from coppelia_sim_ros_lib import CoppeliaROSInterface

# TEMP:
from geometry_msgs.msg import Point, Pose, Quaternion

def main():
    # If gesture detection not enabled in ROSparam -> enable it manually
    settings.gesture_detection_on = True
    settings.launch_gesture_detection = True
    ml.md.eef_pose.position = Point(0.3, 0.0, 0.3)

    roscm = ROSComm()
    prompg = ProMPGenerator(promp='sebasutp')
    print("[Main] Waiting for Coppelia to be initialized")

    ml.md.m = cop = CoppeliaROSInterface()
    print("[Main] Coppelia initialized!")
    sl.scenes.make_scene(cop, 'pickplace3')
    CoppeliaROSInterface.add_or_edit_object(name="Focus_target", pose=Pose(Point(0.0,0.0,0.0),Quaternion(0.0,0.0,0.0,1.0)))

    rate = rospy.Rate(settings.yaml_config_gestures['misc']['rate']) # Hz
    seq = 0
    try:
        while not rospy.is_shutdown():
            ml.md.main_handle_step(cop, roscm, prompg, seq)
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
