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
from os_and_utils.parse_yaml import ParseYAML
from os_and_utils.utils import point_by_ratio
from os_and_utils.transformations import Transformations as tfm

from os_and_utils.path_def import Waypoint

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
    ml.md.m = cop = CoppeliaROSInterface()
    pose = Pose()
    pose.position = Point(0.0, 0.0, 0.0)
    pose.orientation = Quaternion(0.0, 0.0, 0.0, 1.0)
    sl.scenes.make_scene(cop, 'pickplace3')
    CoppeliaROSInterface.add_or_edit_object(name="Focus_target", pose=pose)

    rate = rospy.Rate(settings.yaml_config_gestures['misc']['rate']) # Hz
    seq = 0
    try:
        while not rospy.is_shutdown():
            if ml.md.present():
                # 1. Send gesture data based on hand mode
                if ml.md.frames and settings.gesture_detection_on:
                    roscm.send_g_data()

                # 2. Info + Save plot data
                #print(f"fps {ml.md.frames[-1].fps}, id {ml.md.frames[-1].seq}")
                #print(f"actions queue {[act[1] for act in gl.gd.actions_queue]}")
                #print(f"point diretoin {ml.md.frames[-1].l.point_direction()}")
                # Printing presented here represent current mapping
                #if gl.gd.r.dynamic.relevant():
                #    print(f"Right Dynamic relevant info: Biggest prob. gesture: {gl.gd.r.dynamic.relevant().biggest_probability}")
                #    #print(ml.md.frames[-1].r.get_learning_data(definition=1))

                #if gl.gd.l.static.relevant():
                #    print(f"Left Static relevant info: Biggest prob. gesture: {gl.gd.l.static.relevant().biggest_probability}")

                #    #print(ml.md.frames[-1].l.get_learning_data(definition=1))
            # 3. Handle gesture activation
            if len(gl.gd.actions_queue) > 0:
                print("===================== ACTION ==========================")
                path, waypoints = prompg.handle_action_queue(gl.gd.actions_queue.pop())
                print("$$ The last waypoints", waypoints)
                cop.execute_trajectory_with_waypoints(path, waypoints)
            if ml.md.frames:
                handle_action_update(cop)

            if seq % settings.yaml_config_gestures['misc']['rate']: # every sec
                cop.add_or_edit_object(name='Focus_target', pose=sl.scene.object_poses[ml.md.object_focus_id])
            rate.sleep()
    except KeyboardInterrupt:
        pass

    gl.gd.export()
    print("[Main] Ended")

def handle_action_update(cop):
    ''' Edited for only left hand classification and right hand metrics
    '''
    for h in ['l']:#, 'r']:
        if getattr(gl.gd, h).static.relevant():
            action_name = getattr(gl.gd, h).static.relevant().activate_name
            if action_name:
                id_primitive = map_to_primitive_gesture(action_name)
                if id_primitive == 'gripper' and getattr(ml.md.frames[-1], 'r').visible:
                    wps = {1.0: Waypoint()}
                    #grr = 0.0 # float(input("Write gripper posiiton: "))
                    #wps[1.0].gripper = grr #1-getattr(ml.md.frames[-1], 'r').pinch_strength #h).pinch_strength
                    wps[1.0].gripper = 1-getattr(ml.md.frames[-1], 'r').pinch_strength #h).pinch_strength
                    cop.execute_trajectory_with_waypoints(None, wps)
                if id_primitive == 'build_switch' and getattr(ml.md.frames[-1], 'r').visible:
                    xe,ye,_ = tfm.transformLeapToUIsimple(ml.md.frames[-1].l.elbow_position(), out='list')
                    xw,yw,_ = tfm.transformLeapToUIsimple(ml.md.frames[-1].l.wrist_position(), out='list')
                    xp,yp,_ = tfm.transformLeapToUIsimple(ml.md.frames[-1].r.point_position(), out='list')

                    min_dist, min_id = 99999, -1
                    distance_threshold = 1000
                    nBuild_modes = len(ml.md.build_modes)
                    for n, i in enumerate(ml.md.build_modes):
                        x_bm, y_bm = point_by_ratio((xe,ye),(xw,yw), 0.5+0.5*(n/nBuild_modes))
                        x_bm = xe + x_bm
                        y_bm = ye + y_bm
                        distance = (x_bm - xp) ** 2 + (y_bm - yp) ** 2
                        if distance < distance_threshold and distance < min_dist:
                            min_dist = distance
                            min_id = n
                    if min_id != -1:
                        ml.md.build_mode = ml.md.build_modes[min_id]

                if id_primitive == 'focus' and getattr(ml.md.frames[-1], 'r').visible:
                    direction = getattr(ml.md.frames[-1], 'r').point_direction() #h).pinch_strength
                    if direction[0] < 0: # user wants to move to next item
                        # check if previously direction was left, if so, ml.md.current_threshold_to_flip_id will be zeroed
                        if ml.md.current_threshold_to_flip_id < 0: ml.md.current_threshold_to_flip_id = 0
                        ml.md.current_threshold_to_flip_id += 1

                    else:
                        if ml.md.current_threshold_to_flip_id > 0: ml.md.current_threshold_to_flip_id = 0
                        ml.md.current_threshold_to_flip_id -= 1

                    if ml.md.current_threshold_to_flip_id > settings.yaml_config_gestures['misc']['rate']:
                        # move next
                        ml.md.current_threshold_to_flip_id = 0
                        ml.md.object_focus_id += 1
                        if ml.md.object_focus_id == sl.scene.n: ml.md.object_focus_id = 0
                        cop.add_or_edit_object(name='Focus_target', pose=sl.scene.object_poses[ml.md.object_focus_id])
                    elif ml.md.current_threshold_to_flip_id < -settings.yaml_config_gestures['misc']['rate']:
                        # move prev
                        ml.md.current_threshold_to_flip_id = 0
                        ml.md.object_focus_id -= 1
                        if ml.md.object_focus_id == -1: ml.md.object_focus_id = sl.scene.n-1
                        cop.add_or_edit_object(name='Focus_target', pose=sl.scene.object_poses[ml.md.object_focus_id])
                    print("ml.md.current_threshold_to_flip_id", ml.md.current_threshold_to_flip_id, "ml.md.object_focus_id", ml.md.object_focus_id)
                '''
                _, mp_type = get_id_motionprimitive_type(id_primitive)
                try:
                    robot_promps = self.robot_promps[self.Gs.index(id_primitive)]
                except ValueError:
                    robot_promps = None # It is static gesture
                '''
                #path = mp_type().update_by_id(robot_promps, id_primitive, self.approach, vars)




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
