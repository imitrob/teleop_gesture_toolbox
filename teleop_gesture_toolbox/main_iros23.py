#!/usr/bin/env python
''' Up-to-Date Pipeline
When run with python: Checks if given arguments exists
argv: 'noui'
argv: 'ros1armer' - real (default), 'coppelia' - sim
argv: 'tester'
argv: 'drawer'
argv: 'no_action_execution'
'''
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
if 'ros1armer' in sys.argv: import os_and_utils.ros_communication_main as rc; rc.init('ros1armer')
elif 'coppelia' in sys.argv: import os_and_utils.ros_communication_main as rc; rc.init('coppelia')
else: import os_and_utils.ros_communication_main as rc; rc.init('coppelia')
import os_and_utils.deitic_lib as dl; dl.init()

from os_and_utils.utils import cc, get_cbgo_path
from os_and_utils.pathgen_dummy import PathGenDummy
from gesture_classification.sentence_creation import GestureSentence
from os_and_utils.tests.toolbox_tester import test_toolbox

sys.path.append(get_cbgo_path())
from context_based_gesture_operation.srcmodules.Actions import Actions
from context_based_gesture_operation.srcmodules.Scenes import Scene

def main():
    if 'no_action_execution' in sys.argv: 
        settings.action_execution = False
    if 'action_exeuction' in sys.argv:
        settings.action_execution = True

    if not settings.action_execution: print(f"\n{cc.W}Action execution is turned off!{cc.E}")

    if 'cautious' in sys.argv:
        ml.rral.cautious = True
    if 'incautious' in sys.argv:
        ml.rral.cautious = False
    
    path_gen = PathGenDummy()

    # Scene initialization
    if 'drawer' in sys.argv:
        s = Scene(init='drawer', random=False)
        s.r.eef_position = np.array([2, 2, 2])
        s.drawer.position = np.array([2.,0.,0.])
        sl.scene = s
    else:
        s = Scene(init='', random=False)
        s.r.eef_position = np.array([2,2,2])
        sl.scene = s

    # Trigger testing
    if 'tester' in sys.argv: test_toolbox()

    try:
        while rclpy.ok():
            # Collect action gestures
            ml.md.main_handle_step(path_gen=path_gen)
            time.sleep(0.01)
            
    except KeyboardInterrupt:
        pass

def spinning_threadfn():
    while rclpy.ok():
        rc.roscm.spin_once(sem=True)
        time.sleep(0.01)

if __name__ == '__main__':
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
