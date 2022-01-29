#!/usr/bin/env python3.8
import sys, os, time
import numpy as np
import rospy

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

from os_and_utils.utils import GlobalPaths
sys.path.append(GlobalPaths().home+"/"+GlobalPaths().ws_folder+'/src/mirracle_sim/src')
from coppelia_sim_ros_lib import CoppeliaROSInterface

from promps.promp_lib import ProMPGenerator

from os_and_utils.visualizer_lib import VisualizerLib

def main():
    # If gesture detection not enabled in ROSparam -> enable it manually
    settings.gesture_detection_on = True
    settings.launch_gesture_detection = True

    rospy.init_node('main_manager', anonymous=True)
    roscm = ROSComm()

    configGestures = ParseYAML.load_gesture_config_file(settings.paths.custom_settings_yaml)
    hand_mode_set = configGestures['using_hand_mode_set']
    hand_mode = dict(configGestures['hand_mode_sets'][hand_mode_set])

    vizdata = []
    vizdataact = []
    vizdataabovethre = []
    viztime = []


    prompg = ProMPGenerator(promp='sebasutp')
    rate = rospy.Rate(10) # Hz
    while not rospy.is_shutdown():
        print(ml.md.present())
        if ml.md.present():
            # Send gesture data based on hand mode
            if ml.md.frames and settings.gesture_detection_on:
                gl.send_g_data(roscm, hand_mode, args={'type': 'old_defined'})

            print(f"fps {ml.md.frames[-1].fps}, id {ml.md.frames[-1].seq}")
            print(f"actions queue {gl.gd.actions_queue}")
            if gl.gd.r.static.relevant():
                #print(f"The main gesture {gl.gd.r.static.relevant().biggest_probability}")
                #print(gl.gd.r.static.relevant().probabilities)

                viztime.append(ml.md.frames[-1].seq)
                vizdata.append(gl.gd.r.static.relevant().probabilities)
                vizdataact.append(gl.gd.r.static.relevant().activates_int)
                vizdataabovethre.append(gl.gd.r.static.relevant().above_threshold_int)

        # 1. make function for using this data to get gesture activation

        '''
        if gl.gd.r.static.relevant():
            print(f"grab is the best? {gl.gd.r.static.relevant().grab.biggest_probability} \n\
                    grab is above thre? {gl.gd.r.static.relevant().grab.above_threshold} \n\
                    grab is activated? {gl.gd.r.static.relevant().grab.activated}")
                    '''
        '''
        Data struktura vysledku z ruznych generatoru gest.

        Aby sli dobre vybirat i podle time stampu.

        Info: gl.gd.l.static.info.<g>
        Data: gl.gd.l.static.data_queue[-1].<g>
        Get all latest data: gl.gd.last()
        Get relevant data: gl.gd.l.static.relevant()
        Post classification toggle: gl.gd.l[-1].static.<g>.toggle = True
        Post activation: gl.gd.l[-1].static.<g>.toggle

        - [ ] Combinations of static and dynamic gestures: gl.gd.l.combs
        - [ ] Combinations of left and right hand gestures: gl.gd.lr
        Voleni v YAML filu.

        Vars generator: gl.gd.vars_gen(static,<g>,time) -> it goes and searches in ml.md.frames[]

        - [ ] Handle time differences



        Prace s logikou
        '''

        '''
        if False and len(gl.gd.actions_queue) > 0:
            action = gl.gd.actions_queue.pop()
            path = prompg.generate_path(action[1], vars={})

            execute_path(path)

            print(f"Executing gesture id: {action[1]}, time diff perform to actiovation: {rospy.Time.now().to_sec()-action[0]}")
        '''
        rate.sleep()

    print(f"saodsa static.info[n].threshold {gl.gd.l.static.info.thresholds}")
    vizdata = np.array(vizdata).T
    vizdataact = np.array(vizdataact).T
    vizdataabovethre = np.array(vizdataabovethre).T
    viz = VisualizerLib()
    print(f"vizdataact {vizdataact}")
    print(f"vizdataabovethre {vizdataabovethre}")
    viz.visualize_new_fig("Gesture probability through time", dim=2)
    for n in range(7):
        viz.visualize_2d(list(zip(viztime, vizdata[n])),label=f"{gl.gd.r.static.info.names[n]}", xlabel='time, seq [-] (~100 seq/sec)', ylabel='Gesture probability')
    viz.visualize_new_fig("Gesture activates through time", dim=2)
    for n in range(7):
        viz.visualize_2d(list(zip(viztime, vizdataact[n])),label=f"{gl.gd.r.static.info.names[n]}", xlabel='time, seq [-] (~100 seq/sec)', ylabel='Gesture activates')
    viz.visualize_new_fig("Gesture above threshold through time", dim=2)
    for n in range(7):
        viz.visualize_2d(list(zip(viztime, vizdataabovethre[n])),label=f"{gl.gd.r.static.info.names[n]}", xlabel='time, seq [-] (~100 seq/sec)', ylabel='Gesture above threshold')

    viz.show()


if __name__ == '__main__':
    main()
    while not rospy.is_shutdown():
        time.sleep(1)
    print('[Main] Interrupted')
    try:
        sys.exit(0)
    except SystemExit:
        os._exit(0)
