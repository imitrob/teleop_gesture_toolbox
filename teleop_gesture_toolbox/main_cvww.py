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
    ml.RealRobotActionLib.cautious = True
    path_gen = PathGenDummy()


    #sl.scenes.make_scene_from_yaml('pickplace')
    if len(sys.argv) > 2 and sys.argv[2] == 'drawer':
        s = Scene(init='drawer', random=False)
        s.r.eef_position = np.array([2,2,2])
        s.drawer.position = np.array([2.,0.,0.])
        sl.scene = s
    else:
        s = Scene(init='', random=False)
        s.r.eef_position = np.array([2,2,2])
        sl.scene = s


    """
    def test_rot_ap(rot_angle):
        rot_angle = -rot_angle
        ''' Bounding - Safety feature '''
        rot_angle = np.clip(rot_angle, -30, 0)
        x,y,z,w = [0.,1.,0.,0.]
        q = UnitQuaternion(sm.base.troty(rot_angle, 'deg') @ UnitQuaternion([w, x,y,z])).vec_xyzs
        ml.md.goal_pose = Pose(position=Point(x=0.5, y=0.0, z=0.3), orientation=Quaternion(x=q[0],y=q[1],z=q[2],w=q[3]))
        print(ml.md.goal_pose)
        ml.RealRobotConvenience.move_sleep()
    '''
    test_rot_ap(0.0)
    input("Asdoinan1")
    test_rot_ap(10.0)
    input("Asdoinan2")
    test_rot_ap(20.0)
    input("Asdoinan3")
    test_rot_ap(0.0)
    input("Asdoinan4")
    '''
    while not ml.md.frames: time.sleep(0.1)
    while True:
        ''' when hand stabilizes, it takes the measurement '''
        prev_rot = np.inf
        final_rot = np.inf
        while True:
            if not ml.md.l_present(): time.sleep(1); continue
            direction_vector = np.cross(ml.md.frames[-1].l.palm_normal(), ml.md.frames[-1].l.palm_direction())
            xy = list(direction_vector[0:2])
            xy.reverse()
            rot = np.rad2deg(np.arctan2(*xy))
            print(f"{rot}")
            test_rot_ap(rot)

            if abs(rot - prev_rot) < 5:
                 final_rot = (rot + prev_rot)/2
                 break
            prev_rot = rot
        print(f"final rot: {final_rot}")
        test_rot_ap(0.0)
        time.sleep(2.0)
    input("ASdaiosnsoidniosnidsoinsdnasd")
    """
    if False:
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

    '''
    while True:
        print("Testing feedback approvement!")
        gl.gd.approve_handle(f"Do you agree? (y/n)", type='number')
    '''

    #  print(s)
    #test_toolbox_features()
    gesture_sentence_type = 'action-objects'
    try:
        while rclpy.ok():


            ml.md.main_handle_step(path_gen=path_gen)
            time.sleep(0.01)
            if gesture_sentence_type == 'action-objects':

                if ml.md.evaluate_episode:
                    episode_evaluation_and_execution(s=ml.RealRobotConvenience.update_scene(), gesture_sentence_type = 'action-objects')
                    gl.gd.gestures_queue.clear()
                    ml.md.evaluate_episode = False
                    print("Move hand out to end the episode!")
                    while ml.md.present():
                        time.sleep(0.1)

            elif gesture_sentence_type == '1st-object-action-rest':
                object_name_1 = None
                if ml.md.real_or_sim_datapull:
                    object_name_1 = ml.RealRobotConvenience.get_target_objects(1, s)[0]
                else: # Fake data from GUI
                    object_name_1 = ml.md.comboMovePagePickObject1Picked

                while not ml.md.present():
                    time.sleep(0.1)
                while ml.md.present():
                    ml.md.main_handle_step(path_gen=path_gen)
                    time.sleep(0.01)

                episode_evaluation_and_execution(s=ml.RealRobotConvenience.update_scene(), gesture_sentence_type = '1st-object-action-rest')
                gl.gd.gestures_queue.clear()
                ml.md.evaluate_episode = False
                print("Move hand out to end the episode!")
                while ml.md.present():
                    time.sleep(0.1)

            elif gesture_sentence_type == 'adaptive':
                object_names = []
                while ml.md.present():

                    ml.md.main_handle_step(path_gen=path_gen)
                    time.sleep(0.01)

                    if gl.gd.l.static[-1].point.activated:
                        ''' '''
                        object_name_1 = None
                        if ml.md.real_or_sim_datapull:
                            object_name_1 = ml.RealRobotConvenience.get_target_objects(1, s)[0]
                        else: # Fake data from GUI
                            object_name_1 = ml.md.comboMovePagePickObject1Picked
                        object_names.append(object_name_1)

                adaptive_episode_evaluation_and_execution(s, object_names)

    except KeyboardInterrupt:
        pass


def episode_evaluation_and_execution(s, gesture_sentence_type, object_name_1=None):
    ''' Get focus point first
    '''
    ''' SIMPLIFICATION choose last only '''

    if gesture_sentence_type == 'action-objects':
        gl.gd.gestures_queue = [gl.gd.gestures_queue[-1]]
        if gl.gd.approve_handle(f"Action gesture: {gl.gd.gestures_queue[-1][1]}, continue? (y/n)") == 'n':
            return None

        object_name_1 = None
        if ml.md.real_or_sim_datapull:
            object_name_1 = ml.RealRobotConvenience.get_target_objects(1, s)[0]
        else: # Fake data from GUI
            object_name_1 = ml.md.comboMovePagePickObject1Picked

    ''' No objects on scene '''
    if object_name_1 == 'q':
        return None

    ''' Object not given -> use eef position '''
    if object_name_1 is None or object_name_1 == 'n':
        s = ml.RealRobotConvenience.update_scene()
        focus_point = s.objects[0].position_real
        #focus_point = s.r.eef_position_real
        #print("object_name_1 none", focus_point, " object_name_1", object_name_1)
    else:
        #print("object_name_1", object_name_1, object_name_1 is None, type(object_name_1))
        if s.get_object_by_name(object_name_1) is None:
            print(f"{cc.W}Object target is not in the scene!{cc.E}, object name: {object_name_1} objects: {s.O}")
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

    ex = ''
    if len(target_action_sequence) > 1:
        ex = gl.gd.approve_handle(f"{cc.OK}BTree generated {len(target_action_sequence)} actions!{cc.E}, execute? (y/n)")

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

    def test_rot_ap(rot_angle):
        rot_angle = -rot_angle
        ''' Bounding - Safety feature '''
        rot_angle = np.clip(rot_angle, -30, 0)
        x,y,z,w = [0.,1.,0.,0.]
        q = UnitQuaternion(sm.base.troty(rot_angle, 'deg') @ UnitQuaternion([w, x,y,z])).vec_xyzs

        ml.md.goal_pose = Pose(position=Point(x=0.5, y=0.0, z=0.3), orientation=Quaternion(x=q[0],y=q[1],z=q[2],w=q[3]))
        ml.RealRobotConvenience.move_sleep()


    def load_auxiliary_parameters(s, target_action, num_of_auxiliary_params):
        if num_of_auxiliary_params in ['', 'q', 'r']:
            return
        num_of_auxiliary_params = int(num_of_auxiliary_params)

        if num_of_auxiliary_params == 0:
            return
        elif num_of_auxiliary_params > 1:
            raise NotImplementedError()
        ''' when hand stabilizes, it takes the measurement '''
        prev_rot = np.inf
        final_rot = np.inf
        while True:
            if not ml.md.l_present(): time.sleep(1); continue
            direction_vector = np.cross(ml.md.frames[-1].l.palm_normal(), ml.md.frames[-1].l.palm_direction())
            xy = list(direction_vector[0:2])
            xy.reverse()
            rot = np.rad2deg(np.arctan2(*xy))
            print(f"{rot}")
            test_rot_ap(rot)

            if abs(rot - prev_rot) < 5:
                 final_rot = (rot + prev_rot)/2
                 break
            prev_rot = rot
        print(f"final rot: {final_rot}")
        ap = {'rotation': final_rot}
        test_rot_ap(0.0)
        return ap



    for n,t in enumerate(target_action_sequence):
        if ex == 'n' and len(target_action_sequence) != n+1: continue
        target_action = t.target_action
        target_objects = [t.target_object]
        ''' SIMPLIFICATION: remaining object targets obtained from last generated action '''
        if len(target_action_sequence) == n+1:
            num_of_objects_ = gl.gd.approve_handle(f"Show how many other objects? ", type='number')
            if num_of_objects_ == '':
                rem_objs = load_reamining_object_names(s, target_actions[-1])
            else:
                rem_objs = ml.RealRobotConvenience.get_target_objects(int(num_of_objects_), s)
            target_objects = [object_name_1] + rem_objs

        ''' '''
        num_of_auxiliary_params = gl.gd.approve_handle(f"Show how many metric parameters? ", type='number')
        ap = load_auxiliary_parameters(s, target_actions[-1], num_of_auxiliary_params)

        ''' TODO: Add checking with multmultipleiple objects '''
        if len(target_objects) == 1 and not s.check_semantic_feasibility(target_action, target_objects[0]):
            print(f"{cc.W}Action is not feasible to do!{cc.E}")
            print(f"{target_action}: target_action, {target_objects}: target_objects, scene {s}")
            y = gl.gd.approve_handle(f"Execute anyway (y/n)")
            if y == 'n':
                return

        print(f"{cc.H}[{n}] Execute action(s): {target_action}, {target_objects}? (y) {cc.E }")
        #if ml.RealRobotActionLib.cautious:
        #    print(f"{cc.W}Returning{cc.E}")
        #    return

        if not (target_action in dir(ml.RealRobotActionLib)):
            print(f"{cc.W}Action not defined{cc.E}")
            return

        print(f"{cc.H}Executing{cc.E}")
        getattr(ml.RealRobotActionLib, target_action)(target_objects, ap)

        Actions.do(sl.scene, (target_action, target_objects[0]), ignore_location=True)
        ml.md.actions_done.append((target_action, target_objects))

def test_toolbox_features():
    s = Scene(init='drawer', random=False)
    s.r.eef_position = np.array([2,2,2])
    s.drawer.position = np.array([2,0,0])
    s.drawer.position_real

    s.drawer.position = np.array([3.,0.,0.])
    sl.scene = s
    y = input(f"[Tester] Open Drawer: {s}")
    target_action = 'open'
    target_objects = ['drawer']
    getattr(ml.RealRobotActionLib, target_action)(target_objects)
    y = input(f"[Tester] Close Drawer: {s}")
    target_action = 'close'
    target_objects = ['drawer']
    getattr(ml.RealRobotActionLib, target_action)(target_objects)

    '''
    y = input("[Tester] Close & Open the gripper")
    rc.roscm.r.set_gripper(position=0.0)
    input("[Tester] Close & Open the gripper")
    rc.roscm.r.set_gripper(position=1.0)


    pose = [0.5,0.0,0.3,  0.,1.,0.,0.]
    y = input(f"[Tester] Move to position: {pose[0:3]} + 0.1m in y axis,\n\tquaternion: {pose[3:7]}")
    rc.roscm.r.go_to_pose(pose)
    time.sleep(1.)
    pose[1] += 0.1
    rc.roscm.r.go_to_pose(pose)
    '''
    num_of_objects = 1
    s = ml.RealRobotConvenience.update_scene()

    y = input("[Tester] Mark an object by pointing to it")
    ml.RealRobotConvenience.get_target_objects(num_of_objects, s, hand='r')
    y = input("[Tester] Mark an object by selecting it")
    ml.RealRobotConvenience.get_target_objects(num_of_objects, s, hand='r')
    y = input("[Tester] Test marking objects")
    ml.RealRobotConvenience.mark_objects(s=ml.RealRobotConvenience.update_scene())
    y = input("[Tester] Test deictic gesture")
    ml.RealRobotConvenience.test_deictic()

    print(f"{cc.H}Tester ended{cc.E}")


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




def adaptive_episode_evaluation_and_execution(s, object_names):
    ''' Get focus point first
    '''
    ''' SIMPLIFICATION choose last only '''

    gl.gd.gestures_queue = [gl.gd.gestures_queue[-1]]
    if gl.gd.approve_handle(f"Action gesture: {gl.gd.gestures_queue[-1][1]}, continue? (y/n)") == 'n':
        return None

    ''' Object not given -> use eef position '''
    if object_names[0] is None:
        focus_point = s.r.eef_position_real
        #print("object_name_1 none", focus_point, " object_name_1", object_name_1)
    else:
        #print("object_name_1", object_name_1, object_name_1 is None, type(object_name_1))
        if s.get_object_by_name(object_names[0]) is None:
            print(f"{cc.W}Object target is not in the scene!{cc.E}, object name: {object_names[0]} objects: {s.O}")
            return
        focus_point = s.get_object_by_name(object_names[0]).position_real

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

    for n,t in enumerate(target_action_sequence):
        target_action = t.target_action
        target_objects = [t.target_object]
        ''' SIMPLIFICATION: remaining object targets obtained from last generated action '''
        if len(target_action_sequence) == n-1:
            target_objects = object_names

        ''' # TODO: Add checking with multmultipleiple objects '''
        if len(target_objects) == 1 and not s.check_semantic_feasibility(target_action, target_objects[0]):
            print(f"{cc.W}Action is not feasible to do!{cc.E}")
            print(f"{target_action}: target_action, {target_objects}: target_objects, scene {s}")
            y = gl.gd.approve_handle(f"Execute anyway (y/n)")
            if y == 'n':
                return

        if ml.RealRobotActionLib.cautious and gl.gd.approve_handle(f"[{n}] Execute action(s): {target_action}, {target_objects}? (y) ") != 'y':
            print(f"{cc.W}Returning{cc.E}")
            return

        if not (target_action in dir(ml.RealRobotActionLib)):
            print(f"{cc.W}Action not defined{cc.E}")
            return

        print(f"{cc.H}Executing{cc.E}")
        getattr(ml.RealRobotActionLib, target_action)(target_objects)

        Actions.do(sl.scene, (target_action, target_objects[0]), ignore_location=True)
        ml.md.actions_done.append((target_action, target_objects))
