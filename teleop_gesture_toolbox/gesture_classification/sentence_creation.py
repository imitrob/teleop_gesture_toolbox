import sys, os, time
sys.path.append(os.path.join(os.path.abspath(__file__), "..", '..', 'python3.9', 'site-packages', 'teleop_gesture_toolbox'))
import numpy as np

# Init functions create global placeholder objects for data
import os_and_utils.settings as settings
import os_and_utils.move_lib as ml
import os_and_utils.scenes as sl
import gesture_classification.gestures_lib as gl
import os_and_utils.ros_communication_main as rc

from geometry_msgs.msg import Pose, Point, Quaternion

from os_and_utils.utils import cc, get_cbgo_path, reject_outliers, get_dist_by_extremes

from context_based_gesture_operation.msg import Scene as SceneRos
from context_based_gesture_operation.msg import Gestures as GesturesRos
from context_based_gesture_operation.srv import BTreeSingleCall

sys.path.append(get_cbgo_path())
from context_based_gesture_operation.srcmodules.Actions import Actions
from context_based_gesture_operation.srcmodules.Scenes import Scene
from context_based_gesture_operation.srcmodules.Objects import Object, Cup, Drawer

from spatialmath import UnitQuaternion
import spatialmath as sm
from collections import Counter

class GestureSentence():
    @staticmethod
    def clearing():
        gl.gd.gestures_queue.clear()
        gl.gd.gestures_queue_proc = []
        ml.md.evaluate_episode = False

        gl.gd.target_objects = []
        gl.gd.ap = []

        print("Move hand out to end the episode!")
        while ml.md.present():
            time.sleep(0.1)
    
    @staticmethod
    def clearing_silent():
        gl.gd.gestures_queue.clear()
        gl.gd.gestures_queue_proc = []
        ml.md.evaluate_episode = False

        gl.gd.target_objects = []
        gl.gd.ap = []

    @staticmethod
    def process_gesture_queue(gestures_queue):
        ''' gestures_queue has combinations of
        Parameters:
            gesture_queue (String[]): Activated action gestures within episode
                - Can be mix static and dynamic ones
        Experimental:
        1. There needs to be some regulation of static and dynamic ones
        2. Weigthing based on when they were generated
        '''
        total_count = len(gestures_queue)
        if total_count <= 0: return []
        gestures_queue = [g[1] for g in gestures_queue]
        sta, dyn = GestureSentence.get_most_probable_sta_dyn(gestures_queue,2)

        '''
        precision = {}
        for gesture,count in counts:
            precision[gesture] = round(count / total_count, 2) * 100
        #print(f"..... {precision}")
        '''
        if sta == [] and dyn == []: return []
        gestures_queue = [max([*sta, *dyn])]
        return gestures_queue

    @staticmethod
    def get_most_probable_sta_dyn(gesture_queue, n):
        static_gestures, dynamic_gestures = [], []

        ignore_gestures = ['point', 'no_moving', 'five', 'pinch']
        #gesture_queue = ['apple','apple','banana','banana','banana', 'coco', 'coco', 'coco','coco']
        counts = Counter(gesture_queue)
        #print("o1", gesture_queue)
        #print("o2", counts)
        while len(counts) > 0:

            # get max
            gesture_name = max(counts)
            m = counts.pop(gesture_name)

            gt = gl.gd.get_gesture_type(gesture_name)
            #print("o3", gt == 'dynamic', len(dynamic_gestures) < n, gesture_name not in ignore_gestures)
            if gt == 'static' and len(static_gestures) < n and gesture_name not in ignore_gestures:
                static_gestures.append(gesture_name)
            elif gt == 'dynamic' and len(dynamic_gestures) < n and gesture_name not in ignore_gestures:
                dynamic_gestures.append(gesture_name)
            else: continue
        #print("o4", dynamic_gestures)

        return static_gestures, dynamic_gestures



    @staticmethod
    def get_target_objects__wrapper(n, s):
        object_name_1 = None
        if ml.md.real_or_sim_datapull:
            return ml.RealRobotConvenience.get_target_objects(n, s)[0]
        else: # Fake data from GUI
            return ml.md.comboMovePagePickObject1Picked
    @staticmethod
    def get_target_object__wrapper_non_blocking(s):
        object_name_1 = None
        if ml.md.real_or_sim_datapull:
            return ml.RealRobotConvenience.get_target_object_non_blocking(s)
        else: # Fake data from GUI
            return ml.md.comboMovePagePickObject1Picked

    @staticmethod
    def target_objects_to_focus_point(object_names, s):
        ''' Choose one object as attention '''
        if len(object_names) == 0: # no object added, no pointing was made
            ''' Object not given -> use eef position '''
            s = ml.RealRobotConvenience.update_scene()
            focus_point = s.r.eef_position_real
        else:
            object_name_1 = object_names[0]
            if s.get_object_by_name(object_name_1) is None:
                print(f"{cc.W}Object target is not in the scene!{cc.E}, object name: {object_name_1} objects: {s.O}")
                GestureSentence.clearing()
                return None
            focus_point = s.get_object_by_name(object_name_1).position_real
        return focus_point

    @staticmethod
    def btsingle_call__wrapper(gestures_queue, focus_point, s):
        sr = s.to_ros(SceneRos())
        sr.focus_point = np.array(focus_point, dtype=float)
        print(f"[INFO] Aggregated gestures: {list(gestures_queue)}")

        time.sleep(0.01)

        req = BTreeSingleCall.Request()
        req.gestures = gl.gd.gestures_queue_to_ros(gestures_queue, GesturesRos())

        req.scene = sr

        return rc.roscm.call_tree_singlerun(req)

    @staticmethod
    def load_reamining_object_names(s, target_action):
        if ml.md.real_or_sim_datapull:
            num_of_objects = getattr(ml.rral, target_action+"_deictic_params")
            if num_of_objects > 1:
                return ml.RealRobotConvenience.get_target_objects(num_of_objects-1, s)
            else:
                return []
        else: # Fake data from GUI
            return [ml.md.comboMovePagePickObject2Picked]

    @staticmethod
    def test_rot_ap(rot_angle):
        rot_angle = -rot_angle
        ''' Bounding - Safety feature '''
        rot_angle = np.clip(rot_angle, -30, 0)
        x,y,z,w = [0.,1.,0.,0.]
        q = UnitQuaternion(sm.base.troty(rot_angle, 'deg') @ UnitQuaternion([w, x,y,z])).vec_xyzs

        ml.md.goal_pose = Pose(position=Point(x=0.5, y=0.0, z=0.3), orientation=Quaternion(x=q[0],y=q[1],z=q[2],w=q[3]))
        ml.RealRobotConvenience.move_sleep()

    @staticmethod
    def test_dist_ap(dist):
        ''' Bounding - Safety feature '''
        dist = np.clip(dist, 0, 0.14)
        rc.roscm.r.set_gripper(position=dist)
        ml.RealRobotConvenience.move_sleep()

    @staticmethod
    def load_auxiliary_parameters(s, num_of_auxiliary_params, type='rotation', robot_feedback=True):
        if num_of_auxiliary_params in ['', 'q', 'r']:
            return
        num_of_auxiliary_params = int(num_of_auxiliary_params)

        if num_of_auxiliary_params == 0:
            return
        elif num_of_auxiliary_params > 1:
            raise NotImplementedError()
        ''' when hand stabilizes, it takes the measurement '''
        if type == 'rotation':
            prev_rot = np.inf
            final_rot = np.inf
            while True:
                if not ml.md.present(): time.sleep(1); continue
                direction_vector = np.cross(ml.md.frames[-1].palm_normal(), ml.md.frames[-1].palm_direction())
                xy = list(direction_vector[0:2])
                xy.reverse()
                rot = np.rad2deg(np.arctan2(*xy))
                print(f"{rot}")
                if robot_feedback:
                    GestureSentence.test_rot_ap(rot)
                else:
                    time.sleep(0.5)

                if abs(rot - prev_rot) < 5:
                     final_rot = (rot + prev_rot)/2
                     break
                prev_rot = rot
            print(f"final rot: {final_rot}")
            ap = {'rotation': final_rot}
            if robot_feedback: GestureSentence.test_rot_ap(0.0)
        elif type == 'distance':
            prev_dist = np.inf
            final_dist = np.inf
            while True:
                if not ml.md.present(): time.sleep(1); continue
                dist = ml.md.frames[-1].touch12
                print(f"{dist}")
                if robot_feedback:
                    GestureSentence.test_dist_ap(dist)
                else:
                    time.sleep(0.5)

                if abs(dist - prev_dist) < 5:
                     final_dist = (dist + prev_dist)/2
                     break
                prev_dist = dist
            print(f"final dist: {final_dist}")
            ap = {'distance': final_dist}
            if robot_feedback: GestureSentence.test_dist_ap(0.0)
        return ap
    @staticmethod
    def load_auxiliary_parameter_non_blocking(s, type='rotation', robot_feedback=True):
        if type == 'rotation':
            direction_vector = np.cross(ml.md.frames[-1].palm_normal(), ml.md.frames[-1].palm_direction())
            xy = list(direction_vector[0:2])
            xy.reverse()
            rot = np.rad2deg(np.arctan2(*xy))
            if robot_feedback:
                GestureSentence.test_rot_ap(rot)
            else:
                time.sleep(0.5)

            ap = rot
            if robot_feedback: GestureSentence.test_rot_ap(0.0)
        elif type == 'distance':

            dist = ml.md.frames[-1].touch12
            if robot_feedback:
                GestureSentence.test_dist_ap(dist)
            else:
                time.sleep(0.5)

            ap = dist
            if robot_feedback: GestureSentence.test_dist_ap(0.0)
        return ap

    @staticmethod
    def execute_action_update(s, target_action, target_objects, ap, n, n_actions):
        ''' TODO: Add checking with multmultipleiple objects '''
        if len(target_objects) == 1 and not s.check_semantic_feasibility(target_action, target_objects[0]):
            print(f"{cc.W}Action is not feasible to do!{cc.E}")

        print(f"{cc.H}[{n}/{n_actions}] Execute action(s): {target_action}, {target_objects}? (y) {cc.E }")

        if not (target_action in dir(ml.rral)):
            print(f"{cc.W}Action not defined{cc.E}")
            GestureSentence.clearing()
            return None

        ''' Execute in real '''
        print(f"{cc.H}Executing{cc.E}")
        getattr(ml.rral, target_action)(target_objects, ap)

        ''' Update semantic scene '''
        if len(target_objects) == 0: 
            to = None
        else: 
            to = target_objects[0]
            to = sl.scene.get_object_by_name(to)
        ret = Actions.do(sl.scene, (target_action, to))
        if ret == False:
            print(f"{cc.W}DEBUG Action which is considered unfeasible:{cc.E}")
            print("sl.scene:", sl.scene)
            print((target_action, to))
            print(s.check_semantic_feasibility(target_action, target_objects[0]))
        # just for record
        ml.md.actions_done.append((target_action, target_objects))
        return True

    @staticmethod
    def episode_evaluation_and_execution(s, object_name_1=None):
        '''
        '''
        gl.gd.gestures_queue_proc = GestureSentence.process_gesture_queue(gl.gd.gestures_queue)
        if gl.gd.misc_gesture_handle(f"Action gesture: {gl.gd.gestures_queue_proc[-1][1]}, continue? (y/n)") == 'n':
            GestureSentence.clearing()
            return None

        object_name_1 = GestureSentence.get_target_objects__wrapper(1, s)
        # DEPRECATED
        #if object_name_1 == 'q': # No objects on the scene
        #    GestureSentence.clearing()
        #    return None

        focus_point = GestureSentence.target_objects_to_focus_point([object_name_1], s)
        if focus_point is None: return

        ''' # TODO: Distinguish between 'object' and 'location'
        '''
        target_action_sequence = GestureSentence.btsingle_call__wrapper(gl.gd.gestures_queue_proc, focus_point, s)
        print(f"{cc.OK}BTree generated {len(target_action_sequence)} actions!{cc.E}")
        target_actions = [ta.target_action for ta in target_action_sequence]

        n_actions = len(target_action_sequence)
        ''' Fix preconditions '''
        for n,t in enumerate(target_action_sequence[:-1]):
            target_action = t.target_action
            target_objects = [t.target_object]

            ret = GestureSentence.execute_action_update(s, target_action, target_objects, None, n, n_actions)
            if ret is None: return

        ''' Execute last action '''
        target_action = target_action_sequence[-1].target_action
        target_objects = [target_action_sequence[-1].target_object]

        num_of_objects_ = gl.gd.misc_gesture_handle(f"Show how many other objects? ", type='number')
        if num_of_objects_ == '':
            rem_objs = GestureSentence.load_reamining_object_names(s, target_actions[-1])
        else:
            rem_objs = ml.RealRobotConvenience.get_target_objects(int(num_of_objects_), s)
        target_objects = [object_name_1] + rem_objs

        num_of_auxiliary_params = gl.gd.misc_gesture_handle(f"Show how many metric parameters? ", type='number')
        ap = GestureSentence.load_auxiliary_parameters(s, num_of_auxiliary_params)

        ret = GestureSentence.execute_action_update(s, target_action, target_objects, ap, n_actions, n_actions)
        if ret is None: return

        ''' Clear queue '''
        GestureSentence.clearing()

    @staticmethod
    def adaptive_episode_evaluation_and_execution(s, gestures_queue_proc=[], object_names=[], ap=[]):
        if len(s.objects) == 0:
            GestureSentence.clearing()
            print(f"Adaptive Episode Evaluation - No objects of scene -> returning")
            return

        print(f"(0/5) Adaptive Episode Evaluation")
        focus_point = GestureSentence.target_objects_to_focus_point(object_names, s)
        if focus_point is None: return
        print(f"(1/5) Adaptive Episode Evaluation - target object to focus point {object_names} -> {focus_point}")

        ''' # TODO: Distinguish between 'object' and 'location'
        '''
        target_action_sequence = GestureSentence.btsingle_call__wrapper(gestures_queue_proc, focus_point, s)
        target_actions = [ta.target_action for ta in target_action_sequence]
        n_actions = len(target_action_sequence)
        print(f"(2/5) Adaptive Episode Evaluation - BT generated {n_actions} target actions {target_actions}")

        ''' Fix preconditions '''
        for n,t in enumerate(target_action_sequence[:-1]):
            target_action = t.target_action
            target_objects = [t.target_object]

            ret = GestureSentence.execute_action_update(s, target_action, target_objects, None, n, n_actions)
            if ret is None: return
            print(f"(3/5) Adaptive Episode Evaluation - Precondition {n} ({target_action}, {target_objects}) completed")

        ''' Execute last action '''
        target_action = target_action_sequence[-1].target_action
        # If len(object_names) >
        dpn = getattr(ml.rral, f'{target_action}_deictic_params')
        if len(object_names) < dpn:
            print(f"{cc.W}Not enough objects {cc.E} {len(object_names)} {dpn} {object_names}")
        print(f"{dpn} object selected, object names: {object_names[:dpn]} selected from {object_names}")
        ret = GestureSentence.execute_action_update(s, target_action, object_names[:dpn], ap, n_actions, n_actions)
        if ret is None: 
            print(f"(4/5) Adaptive Episode Evaluation - Final action {target_action} can't be done")
            return
        else:
            print(f"(4/5) Adaptive Episode Evaluation - Final action {target_action} done")

        ''' Clear queue '''
        GestureSentence.clearing()
        print(f"(5/5) Adaptive Episode Evaluation - Done")

    adaptive_setup = {
        'deictic': ('point'), # TODO: 'steady_point'
        #'approvement': ('thumbsup', 'five'), # steady five
        'measurement_distance': ('pinch'), # steady pinch
        #'measurement_rotation': ('five'), # steady pinch
    }

    @staticmethod
    def get_adaptive_gesture_type(activated_gestures):
        activated_gesture_types = []

        as_ = GestureSentence.adaptive_setup
        # activated_gestures = ('point')
        for ag in activated_gestures:
            # as_.keys() = ('deictic', 'approvement', 'measurement')
            for k in as_.keys():
                asi = as_[k]
                # if the adaptive setup item has the activated gesture in its list
                if ag in asi: # gesture which is activated is is adaptive setup gestures
                    # activate the gesture type
                    if k not in activated_gesture_types:
                        activated_gesture_types.append(k)

        if len(activated_gesture_types) > 1:
            # TODO:
            #print(f"[WARNING] More possible gesture types, act: {activated_gesture_types}")
            activated_gesture_type = activated_gesture_types[0]
        elif len(activated_gesture_types) == 0:
            activated_gesture_type = 'action'
        elif len(activated_gesture_types) == 1:
            activated_gesture_type = activated_gesture_types[0]
        else: raise Exception("Cannot happen")

        return activated_gesture_type

    @staticmethod
    def activated_gesture_type_to_action(activated_gesture_type, rate=10, x=5, y=10, threshold=0.9, blocking=False):
        '''
        Parameters:
            rate (Int): Rate of new frames (Hz)
            x (Int): Gesture type evidence to be activated (frames)
            y (Int): How many frames it must be non activated, before the gesture type is activated
            threshold (Float): accuracy threshiold, >% frames gesture type -> activated
        Returns:
            gesture_type (String/None): If fulfills the conditions or None if not

        --------- | --------- | ---------
             aaaaa|dddddddddddddddddddd aaaaa
                   < -------- x ------> x True
             <-y->|< ----- delay -----> y False

                    |< -------- x ------> x True
               <-y->|< ----- delay -----> y False

        '''
        if (time.time()-gl.gd.evidence_gesture_type_to_activate_last_added) > (1/rate):
            gl.gd.evidence_gesture_type_to_activate_last_added = time.time()
            gl.gd.evidence_gesture_type_to_activate.append(activated_gesture_type)
            #print(f"added {activated_gesture_type}")

        gesture_type = gl.gd.evidence_gesture_type_to_activate.get_last_common(x, threshold=1.0)
        #print(f"gesture type: {activated_gesture_type} {gl.gd.evidence_gesture_type_to_activate[-30:]}")

        if blocking:
            not_in = gl.gd.evidence_gesture_type_to_activate.get_last_commons(y, most_common=2, delay=x, threshold=1.0)
            #print(f"not_in {not_in}")
            if gesture_type is not None and \
                gesture_type not in not_in:
                return gesture_type
        else:
            if gesture_type is not None:
                return gesture_type


    @staticmethod
    def adaptive_eee(path_gen, s, object_pick_method='last'):
        ''' Episode evaluation & execution
        Parameters:
            object_pick_method ('last','max'): Aggregated objects list selected by user, option how to choose
        '''
        # Episode started
        if ml.md.present():

            ## WHEN THE OPTION IS NON BLOCKING THIS MAY NOT FIND ANY GESTURES
            activated_gestures = gl.gd.load_all_relevant_activated_gestures(relevant_time=2.0, records=3)

            activated_gesture_type = GestureSentence.get_adaptive_gesture_type(activated_gestures)

            blocking = False
            activated_gesture_type_action = GestureSentence.activated_gesture_type_to_action(activated_gesture_type, blocking=blocking)

            ''' When no longer Gesture type activated -> save the accumulated data '''
            if ml.md.act_prev_tmp[0] != 'deictic' and ml.md.act_prev_tmp[2] != []:

                if object_pick_method == 'max':
                    gl.gd.target_objects.append(max(ml.md.act_prev_tmp[2]))
                elif object_pick_method == 'last':
                    gl.gd.target_objects.append(ml.md.act_prev_tmp[2][-1])
                else: raise Exception()

                print(f"{cc.H}Added obj {Counter(ml.md.act_prev_tmp[2])}{cc.E}")
                ml.md.act_prev_tmp[2] = []

            if ml.md.act_prev_tmp[0] != 'measurement_distance' and ml.md.act_prev_tmp[3] != []:
                gl.gd.ap.append(get_dist_by_extremes(np.array(ml.md.act_prev_tmp[3])))
                print(f"{cc.H}Added dist {get_dist_by_extremes(np.array(ml.md.act_prev_tmp[3]))}{cc.E}")

                ml.md.act_prev_tmp[3] = []

            object_name_1 = None
            if activated_gesture_type_action == 'deictic':

                if blocking:
                    object_name_1 = GestureSentence.get_target_objects__wrapper(1, s)
                    gl.gd.target_objects.append(object_name_1)
                    print(f"{cc.H}Added obj {object_name_1}{cc.E}")
                else:
                    object_name_1 = GestureSentence.get_target_object__wrapper_non_blocking(s)

                    '''
                    if ml.md.any_hand_stable(time=1000):
                        gl.gd.target_objects.append(object_name_1)
                        print(f"{cc.H}Added obj {object_name_1}{cc.E}")
                        ml.md.act_prev_tmp = [activated_gesture_type_action, 'written', object_name_1]
                    else:'''
                    if object_name_1 not in ['q']:
                        ml.md.act_prev_tmp[0] = activated_gesture_type_action
                        ml.md.act_prev_tmp[2].append(object_name_1)
                        print(f"{cc.H}Added obj {object_name_1}{cc.E}")

            elif activated_gesture_type_action == 'approvement':

                res = gl.gd.misc_gesture_handle(f"Approve? (y/n)", new_episode=False)
                print(f"{cc.H}Added approvement {res}{cc.E}")
                gl.gd.ap.append(res)
            elif activated_gesture_type_action == 'measurement_distance':

                if blocking:
                    dist = GestureSentence.load_auxiliary_parameters(s, num_of_auxiliary_params=1, type='distance', robot_feedback=False)
                    print(f"{cc.H}Added dist.m. {dist}{cc.E}")
                    gl.gd.ap.append(dist)
                else:
                    dist = GestureSentence.load_auxiliary_parameter_non_blocking(s, type='distance', robot_feedback=False)
                    ml.md.act_prev_tmp[0] = activated_gesture_type_action
                    ml.md.act_prev_tmp[3].append(dist)
            else:
                activated_gesture_type_action = 'action'


                # Evaluate when action gesturing ends
                ml.md.act_prev_tmp[0] = activated_gesture_type_action
        else:
            if len(gl.gd.gestures_queue) > 0:
                time.sleep(0.5)
                gl.gd.gestures_queue_proc = GestureSentence.process_gesture_queue(gl.gd.gestures_queue)
                if len(gl.gd.gestures_queue_proc) > 0:
                    print(f"{cc.OK}EEE\t{gl.gd.gestures_queue_proc}\t{gl.gd.target_objects}\t{gl.gd.ap}{cc.E}")
                    if not settings.action_execution:
                        GestureSentence.clearing()
                    else:
                        GestureSentence.adaptive_episode_evaluation_and_execution(s=ml.RealRobotConvenience.update_scene(), \
                        gestures_queue_proc=gl.gd.gestures_queue_proc, object_names=gl.gd.target_objects, ap=gl.gd.ap)
                    
            # Whenever hand is not seen clearing
            GestureSentence.clearing_silent()