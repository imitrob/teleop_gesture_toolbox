#!/usr/bin/env python
''' Up-to-Date Pipeline
'''
raise Exception("CHECKUP NEEDED")
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
''' Real setup or Coppelia sim or default (Real setup) '''
if len(sys.argv)>1 and sys.argv[1] == 'ros1armer': import os_and_utils.ros_communication_main as rc; rc.init('ros1armer')
elif len(sys.argv)>1 and sys.argv[1] == 'coppelia': import os_and_utils.ros_communication_main as rc; rc.init('coppelia')
else: import os_and_utils.ros_communication_main as rc; rc.init('ros1armer')
import os_and_utils.deitic_lib as dl; dl.init()

from geometry_msgs.msg import Pose, Point, Quaternion
from std_msgs.msg import String

from os_and_utils.utils import cc, get_cbgo_path
from os_and_utils.pathgen_dummy import PathGenDummy

from context_based_gesture_operation.msg import Scene as SceneRos
from context_based_gesture_operation.msg import Gestures as GesturesRos
from context_based_gesture_operation.srv import BTreeSingleCall
from teleop_gesture_toolbox.srv import ChangeNetwork, SaveHandRecord

sys.path.append(get_cbgo_path())
from context_based_gesture_operation.srcmodules.Actions import Actions
from context_based_gesture_operation.srcmodules.Scenes import Scene
from context_based_gesture_operation.srcmodules.Objects import Object, Cup, Drawer

from spatialmath import UnitQuaternion
import spatialmath as sm

def main():
    ml.rral.cautious = True
    path_gen = PathGenDummy()

    # Scene initialization
    if len(sys.argv) > 2 and sys.argv[2] == 'drawer':
        s = Scene(init='drawer', random=False)
        s.r.eef_position = np.array([2,2,2])
        s.drawer.position = np.array([2.,0.,0.])
        sl.scene = s
    else:
        s = Scene(init='', random=False)
        s.r.eef_position = np.array([2,2,2])
        sl.scene = s

    time.sleep(5.)
    try:
        while rclpy.ok():
            print("=== New query ===")
            object_name_1 = None
            print("= 1. Object 1 ")
            s=ml.RealRobotConvenience.update_scene()
            if ml.md.real_or_sim_datapull:
                object_name_1 = ml.RealRobotConvenience.get_target_objects(1, s)[0]
            else: # Fake data from GUI
                object_name_1 = ml.md.comboMovePagePickObject1Picked

            print("= 2. Gather action gestures ")
            while len(gl.gd.gestures_queue) == 0:
                while not ml.md.present():
                    time.sleep(0.1)
                while ml.md.present():
                    ml.md.main_handle_step(path_gen=path_gen)
                    time.sleep(0.01)

            print("= 3. Execution ")
            episode_evaluation_and_execution(s=ml.RealRobotConvenience.update_scene(), gesture_sentence_type = '1st-object-action-rest', object_name_1=object_name_1)
            gl.gd.gestures_queue.clear()
            ml.md.evaluate_episode = False
            print("Move hand out to end the episode!")
            while ml.md.present():
                time.sleep(0.1)

    except KeyboardInterrupt:
        pass


def spinning_threadfn():
    while rclpy.ok():
        rc.roscm.spin_once(sem=True)
        time.sleep(0.01)

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
