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
import os_and_utils.scenes as sl; sl.init(approach=1)
import gesture_classification.gestures_lib as gl; gl.init()
import os_and_utils.ui_lib as ui
import os_and_utils.ros_communication_main as rc; rc.init('coppelia')
import os_and_utils.deitic_lib as dl; dl.init()

from geometry_msgs.msg import Pose, Point, Quaternion
from std_msgs.msg import String

from ament_index_python.packages import get_package_share_directory
try:
    package_share_directory = get_package_share_directory('context_based_gesture_operation')
except:
    raise Exception("Package context_based_gesture_operation not found!")
sys.path.append("/".join(package_share_directory.split("/")[:-4])+"/src/context_based_gesture_operation/context_based_gesture_operation")

from context_based_gesture_operation.srcmodules.Scenes import Scene
from context_based_gesture_operation.srcmodules.Objects import Object
from context_based_gesture_operation.agent_nodes.g2i import G2IRosNode

### TMP ###
# g2i_tester was global function
g2i_tester = G2IRosNode( load_model='M3v8_D4_1.pkl')#M3v10_D5.pkl')

def main():

    sl.scenes.make_scene("pickplace", approach=1)

    with rc.rossem:
        rate = rc.roscm.create_rate(1.0)

    if rc.roscm.is_real:
        spinning_threadfn_ = spinning_threadfn2
    else:
        spinning_threadfn_ = spinning_threadfn

    spinning_thread = threading.Thread(target=spinning_threadfn_, args=(), daemon=True)
    spinning_thread.start()

    ml.RealRobotConvenience.mark_objects(s=ml.RealRobotConvenience.update_scene())

    global evaluate_episode
    evaluate_episode = False
    try:
        while rclpy.ok():
            with rc.rossem:
                if len(sys.argv)>1 and sys.argv[1] == 'live':
                    ml.md.live_handle_step()
                else:
                    main_step_local()
            time.sleep(0.01)

            if evaluate_episode:
                with rc.rossem:
                    s = ml.RealRobotConvenience.update_scene()
                    action, object_names = episode_evaluation(s, g2i_tester)

                    print(f"Episode evaluated! ta: {action}, to: {object_names}")
                    if action in dir(ml.RealRobotActionLib):
                        print("Action is defined! -> executing")
                        getattr(ml.RealRobotActionLib,action)(object_names)
                    else:
                        print("Action not defined")
                        gl.gd.actions_queue.clear()
                evaluate_episode = False

    except KeyboardInterrupt:
        pass

    gl.gd.export()
    print("[Main] Ended")

def episode_evaluation(s, g2i_tester):
    print("Executing actions: ", )

    focus_point = np.array([0.3,0.0,0.3])

    target_action, target_object = g2i_tester.predict_with_list_of_gestures(s, gl.gd.actions_queue, focus_point, scene_def_id=8)
    if target_action is None:
        return (None, None)

    # Empty the queue
    gl.gd.actions_queue.clear()

    num_of_objects = getattr(ml.RealRobotActionLib, target_action+"_deictic_params")
    object_names = ml.RealRobotConvenience.get_target_objects(num_of_objects, s)

    return target_action, object_names

def main_step_local():
    global evaluate_episode

    if ml.md.present(): # If any hand visible
        # Send gesture data based on hand mode
        if ml.md.frames and settings.gesture_detection_on:
            rc.roscm.send_g_data()
    else:
        if len(gl.gd.actions_queue) > 0:
            evaluate_episode = True

    # Handle gesture update activation
    if ml.md.frames:
        ml.md.handle_action_update()

    # Update focus target
    '''
    if ml.md.seq % (settings.yaml_config_gestures['misc']['rate'] * 2) == 0: # every sec
        if sl.scene and len(sl.scene.object_poses) > 0:
            if rc.roscm.r is not None:
                rc.roscm.r.add_or_edit_object(name='Focus_target', pose=sl.scene.object_poses[ml.md.object_focus_id], timeout=0.2)
    '''
    ml.md.seq += 1


def spinning_threadfn():
    while rclpy.ok():
        with rc.rossem:
            rclpy.spin_once(rc.roscm)
        time.sleep(0.001)

def spinning_threadfn2():
    while rclpy.ok():
        rclpy.spin_once(rc.roscm)
        time.sleep(0.001)

if __name__ == '__main__':
    ''' Default main has three threads: 1. ROS spin, 2. GUI (optional), 3. main
    '''
    if len(sys.argv)>1 and sys.argv[1] == 'noui': settings.launch_ui = False
    # Spin in a separate thread
    #spinning_thread = threading.Thread(target=spinning_threadfn, args=(), daemon=True)
    #spinning_thread.start()
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
