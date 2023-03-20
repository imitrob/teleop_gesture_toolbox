
'''
CoppeliaSim and lab scenes should match

Leap Motion Controller is mounted on the lab robot table.

9 points are marked on the table.
- Closest point is related to robot base, in x direction
- Distances between markers are 20 cm.
|-------------|
|             |
|      R      |
|      |      |
|  x - x - x  |
|  |   |   |  |
|  x - x - x  |
|  |   |   |  |
|  x - x - x  |
|             |
|             |
|-------------|

User gets to point to each marker.
- The distance from 3D point line to contact point is measured
(markers could be 9 invisible objects)
-

'''

#!/usr/bin/env python
import sys, os, time, threading
sys.path.append("/".join(os.path.abspath(__file__).split("/")[:-3])) # teleop_gesture_toolbox based folder
import numpy as np
import rclpy

# Init functions create global placeholder objects for data
from os_and_utils import settings; settings.init()
import os_and_utils.move_lib as ml; ml.init()
import os_and_utils.scenes as sl; sl.init()
import gesture_classification.gestures_lib as gl; gl.init()
import os_and_utils.ui_lib as ui
import os_and_utils.ros_communication_main as rc; rc.init("coppelia")
import os_and_utils.deitic_lib as dl; dl.init()

def main():
    sl.scenes.make_scene_from_yaml('deitic_accuracy_config_scene')
    rate = rc.roscm.create_rate_(10)
    i = 0
    id_obj = None
    try:
        while rclpy.ok():
            if ml.md.frames:
                #ml.md.main_handle_step(None)
                rc.roscm.send_g_data()
                f = ml.md.frames[-1]
                if f.l.visible and f.l.grab_strength < 0.1:
                    id_obj = dl.dd.main_deitic_fun(ml.md.frames[-1], 'l', sl.scene.object_positions_real)
                '''if i % 10 == 0 and id_obj is not None: # every second
                    p = sl.scene.object_poses[id_obj]
                    p.orientation.x = 0.0
                    p.orientation.y = 1.0
                    p.orientation.z = 0.0
                    p.orientation.w = 0.0
                    p.position.z = 0.2
                    rc.roscm.r.go_to_pose(pose=p)'''
            time.sleep(0.06)
            i += 1
    except KeyboardInterrupt:
        pass


    for i in range(9):
        print("Point to the red box.")
        rc.roscm.add_or_edit_object(name=sl.scene.object_names[i], color='r')

        # Pull the information from deitic lib

        input("Press enter to Save and Move to next!")
        rc.roscm.add_or_edit_object(name=sl.scene.object_names[i], color='k')

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


"""
#!/usr/bin/env python
import sys, os
sys.path.append("/".join(os.path.abspath(__file__).split("/")[:-3])) # teleop_gesture_toolbox based folder
import main_thread

import rclpy
import os_and_utils.move_lib as ml
import os_and_utils.ros_communication_main as rc
import os_and_utils.deitic_lib as dl
import os_and_utils.scenes as sl
def main():
    sl.scenes.make_scene_from_yaml('deitic_accuracy_config_scene')
    rate = rc.roscm.create_rate_(10)
    try:
        while rclpy.ok():
            if ml.md.frames:
                ml.md.main_handle_step(None)
                f = ml.md.frames[-1]
                if f.l.visible and f.l.grab_strength < 0.1:
                    dl.dd.main_deitic_fun(ml.md.frames[-1], 'l', sl.scene.object_poses)

            rate.sleep()
    except KeyboardInterrupt:
        pass


    for i in range(9):
        print("Point to the red box.")
        rc.roscm.add_or_edit_object(name=sl.scene.object_names[i], color='r')

        # Pull the information from deitic lib

        input("Press enter to Save and Move to next!")
        rc.roscm.add_or_edit_object(name=sl.scene.object_names[i], color='k')

    print("[Main] Ended")

if __name__ == '__main__':
    main_thread.init(main, 'coppelia') # launches the main function
"""
