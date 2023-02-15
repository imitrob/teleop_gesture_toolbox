#!/usr/bin/env python
''' Sends gesture data and receive results in GUI
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
import os_and_utils.ros_communication_main as rc; rc.init('coppelia') # 'coppelia'
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

def main():
    ml.RealRobotActionLib.cautious = False
    path_gen = PathGenDummy()

    sl.scenes.make_scene_from_yaml('pickplace')
    try:
        while rclpy.ok():
            ml.md.main_handle_step(path_gen=path_gen)
            time.sleep(0.01)
            if ml.md.evaluate_episode:
                episode_evaluation_and_execution(s=ml.RealRobotConvenience.update_scene())
                gl.gd.gestures_queue.clear()
                ml.md.evaluate_episode = False
    except KeyboardInterrupt:
        pass

    gl.gd.export()
    print("[Main] Ended")


def episode_evaluation_and_execution(s):
    ''' Get focus point first
    '''
    object_name_1 = None
    if ml.md.real_or_sim_datapull:
        object_name_1 = ml.RealRobotConvenience.get_target_objects(1, s)[0]
    else: # Fake data from GUI
        object_name_1 = ml.md.comboMovePagePickObject1Picked


    ''' Object not given -> use eef position '''
    if object_name_1 is None:
        focus_point = s.r.eef_position_real
        print("object_name_1 none", focus_point, " object_name_1", object_name_1)
    else:
        print("object_name_1", object_name_1, object_name_1 is None, type(object_name_1))
        if s.get_object_by_name(object_name_1) is None:
            print(f"{cc.W}Object target is not in the scene!{cc.E}")
            return
        focus_point = s.get_object_by_name(object_name_1).position_real

    ''' # TODO: Distinguish between 'object' and 'location'
    '''
    sr = s.to_ros(SceneRos())
    sr.focus_point = np.array(focus_point, dtype=float)
    print(f"[INFO] Aggregated gestures: {list(gl.gd.gestures_queue)}")

    time.sleep(0.01)

    req = BTreeSingleCall.Request()
    req.gestures = gl.gd.gestures_queue_to_ros(GesturesRos())

    req.scene = sr

    target_action_sequence = rc.roscm.call_tree_singlerun(req)

    if len(target_action_sequence) > 1:
        print(f"{cc.OK}BTree generated {len(target_action_sequence)} actions!{cc.E}")

    target_actions = [ta.target_action for ta in target_action_sequence]

    def load_reamining_object_names(s, target_action):
        if ml.md.real_or_sim_datapull:
            num_of_objects = getattr(ml.RealRobotActionLib, target_action+"_deictic_params")
            if num_of_objects > 1:
                return ml.RealRobotConvenience.get_target_objects(num_of_objects-1, s)
            else:
                return []
        else: # Fake data from GUI
            return [ml.md.comboMovePagePickObject2Picked]

    for n,t in enumerate(target_action_sequence):
        target_action = t.target_action
        target_objects = [t.target_object]
        ''' SIMPLIFICATION: remaining object targets obtained from last generated action '''
        if len(target_action_sequence) == n-1:
            target_objects = [object_name_1] + load_reamining_object_names(s, target_actions[-1])

        ''' # TODO: Add checking with multiple objects '''
        if len(target_objects) == 1 and not s.check_semantic_feasibility(target_action, target_objects[0]):
            print(f"{cc.W}Action is not feasible to do!{cc.E}")
            print(f"{target_action}: target_action, {target_objects}: target_objects, scene {s}")
            return

        if ml.RealRobotActionLib.cautious and input(f"[{n}] Execute action(s): {target_action}, {target_objects}? (y) ") != 'y':
            print(f"{cc.W}Returning{cc.E}")
            return

        if not (target_action in dir(ml.RealRobotActionLib)):
            print(f"{cc.W}Action not defined{cc.E}")
            return

        print(f"{cc.H}Executing{cc.E}")
        getattr(ml.RealRobotActionLib, target_action)(target_objects)

        ml.md.actions_done.append((target_action, target_objects))

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
