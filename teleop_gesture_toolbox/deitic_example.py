#!/usr/bin/env python
''' This file represents script that tracks human hand and extracts point finger,
resp. the 1) point finger direction.
The new scene is created. The 2) positions of objects are received from Coppelia.
Using 1) and 2), the closest object to pointing direction 3D line is selected.

Right hand has priority over left one.

Note:
Point gesture direction swapped to whole palm direction as it has better precision.
'''

import sys, os, time, threading
import numpy as np
import rclpy

# Init functions create global placeholder objects for data
import os_and_utils.settings as settings; settings.init()
import os_and_utils.move_lib as ml; ml.init()
import os_and_utils.scenes as sl; sl.init()
import gesture_classification.gestures_lib as gl; gl.init()
import os_and_utils.ui_lib as ui
import os_and_utils.ros_communication_main as rc; rc.init("coppelia")

from os_and_utils.deitic_lib import get_id_of_closest_point_to_line
from os_and_utils.transformations import Transformations as tfm

def spinning_threadfn():
    while rclpy.ok():
        with rc.rossem:
            rclpy.spin_once(rc.roscm)
        time.sleep(0.01)

def main():
    sl.scenes.make_scene('pickplace3')
    # Add Leap Motion object to the scene
    with rc.rossem:
        rc.roscm.r.add_or_edit_object(name="leap", pose=tfm.transformLeapToBase__CornerConfig_translation, color='c', shape='cube', size=[0.03, 0.1, 0.01])
        #rate = rc.roscm.create_rate(2) # Hz

    while rclpy.ok():
        if ml.md.frames and settings.gesture_detection_on:
            with rc.rossem:
                rc.roscm.send_g_data()

        if ml.md.frames:
            f = ml.md.frames[-1]
            if f.r.visible: h = 'r'
            elif f.l.visible: h = 'l'
            else: h = ''

            if h in ['r', 'l']:
                hand = getattr(f, h)
                p1, p2 = np.array(hand.palm_position()), np.array(hand.palm_position())+np.array(hand.direction())
                #p1, p2 = np.array(hand.fingers[1].bones[3].prev_joint()), np.array(hand.fingers[1].bones[3].next_joint())
                p1s = np.array(tfm.transformLeapToBase__CornerConfig(p1))
                p2s = np.array(tfm.transformLeapToBase__CornerConfig(p2))
                v = 1000*(p2s-p1s)
                line_points = (p1s, p2s+v)

                object_positions = [[pose.position.x,pose.position.y,pose.position.z] for pose in sl.scene.object_poses]
                idobj, _ = get_id_of_closest_point_to_line(line_points, object_positions, max_dist=np.inf)

                with rc.rossem:
                    rc.roscm.r.add_line(name='line1', points=line_points)
                    ml.md.object_focus_id = idobj
                    
        #rate.sleep()
        time.sleep(0.1)
    print("quit")



if __name__ == '__main__':
    ''' Default main has three threads: 1. ROS spin, 2. GUI (optional), 3. main
    '''
    if len(sys.argv)>1 and sys.argv[1] == 'noui': settings.launch_ui = False
    # Spin in a separate thread
    spinning_thread = threading.Thread(target=spinning_threadfn, args=( ), daemon=True)
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
