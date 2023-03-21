
import sys, os, time, threading
sys.path.append(os.path.join(os.path.abspath(__file__), "..", '..', 'python3.9', 'site-packages', 'teleop_gesture_toolbox'))
import numpy as np
import rclpy

# Init functions create global placeholder objects for data
import os_and_utils.settings as settings
import os_and_utils.move_lib as ml
import os_and_utils.scenes as sl
import gesture_classification.gestures_lib as gl
import os_and_utils.ui_lib as ui
''' Real setup or Coppelia sim or default (Real setup) '''
import os_and_utils.ros_communication_main as rc
import os_and_utils.deitic_lib as dl

from os_and_utils.utils import cc, get_cbgo_path
from os_and_utils.pathgen_dummy import PathGenDummy
from gesture_classification.sentence_creation import GestureSentence
from os_and_utils.tests.toolbox_tester import test_toolbox

sys.path.append(get_cbgo_path())
from context_based_gesture_operation.srcmodules.Actions import Actions
from context_based_gesture_operation.srcmodules.Scenes import Scene

def test_toolbox():
    for i in range(3):
        print(f"[{i}/3] Testing feedback approvement!")
        gl.gd.approve_handle(f"Do you agree? (y/n)")

    for i in range(3):
        print(f"[{i}/3] Testing feedback continue!")
        gl.gd.approve_handle(f"Continue. (y)", type='y')

    for i in range(3):
        print(f"[{i}/3] Testing feedback!")
        gl.gd.approve_handle(f"How many? (number)", type='number')


    y = input(f"[Tester] Test rotation auxiliary parameter:")
    def test_rot_ap(rot_angle):
        rot_angle = -rot_angle
        ''' Bounding - Safety feature '''
        rot_angle = np.clip(rot_angle, -30, 0)
        x,y,z,w = [0.,1.,0.,0.]
        q = UnitQuaternion(sm.base.troty(rot_angle, 'deg') @ UnitQuaternion([w, x,y,z])).vec_xyzs
        ml.md.goal_pose = Pose(position=Point(x=0.5, y=0.0, z=0.3), orientation=Quaternion(x=q[0],y=q[1],z=q[2],w=q[3]))
        print(ml.md.goal_pose)
        ml.RealRobotConvenience.move_sleep()

    input("[0/4] 0° >>>")
    test_rot_ap(0.0)
    input("[1/4] 10° >>>")
    test_rot_ap(10.0)
    input("[2/4] 20° >>>")
    test_rot_ap(20.0)
    input("[3/4] 0° >>>")
    test_rot_ap(0.0)

    input("[4/4] Leap measure >>>")

    while not ml.md.frames: time.sleep(0.1)
    while True:
        ''' when hand stabilizes, it takes the measurement '''
        prev_rot = np.inf
        final_rot = np.inf
        while True:
            # Any Hand visible on the scene
            if not ml.md.present(): time.sleep(1); continue

            # Hand has thumbsup gesture
            if not ml.md.frames[-1].gd_static_thumbsup(): time.sleep(1); continue

            # Get Angle in XY projection
            angle = ml.md.frames[-1].palm_thumb_angle()


            print(f"Rotation angle: {angle}")
            test_rot_ap(angle)

            if abs(angle - prev_rot) < 5:
                 final_rot = (angle + prev_rot)/2
                 break
            prev_rot = angle
        print(f"final angle: {final_rot}")
        test_rot_ap(0.0)
        time.sleep(2.0)
    input("ASdaiosnsoidniosnidsoinsdnasd")

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
