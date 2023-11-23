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

from teleop_msgs.msg import Scene as SceneRos
from teleop_msgs.msg import Gestures as GesturesRos
from teleop_msgs.msg import HRICommand
from teleop_msgs.msg import Intent as IntentRos
from teleop_msgs.srv import BTreeSingleCall

sys.path.append(get_cbgo_path())
from context_based_gesture_operation.srcmodules.Actions import Actions
from context_based_gesture_operation.srcmodules.Scenes import Scene
from context_based_gesture_operation.srcmodules.Objects import Object, Cup, Drawer

from spatialmath import UnitQuaternion
import spatialmath as sm
from collections import Counter

class GestureSentence():
    ''' Gesture Sentence Data globally stored in gestures_lib.GestureDetections (gl.gd)
    gl.gd.gestures_queue - deque
    gl.gd.gestures_queue_proc - processed queue
    gl.sd.evaluate_episode 

    gl.gd.target_objects (String[])
    gl.gd.ap (Dict?)

    '''

    @staticmethod
    def clearing():
        gl.gd.gestures_queue.clear()
        gl.gd.gestures_queue_proc = []
        gl.sd.evaluate_episode = False

        gl.gd.target_objects = []
        gl.gd.ap = []

        print("Move hand out to end the episode!")
        while gl.gd.present():
            time.sleep(0.1)
    
    @staticmethod
    def clearing_silent():
        gl.gd.gestures_queue.clear()
        gl.gd.gestures_queue_proc = []
        gl.sd.evaluate_episode = False

        gl.gd.target_objects = []
        gl.gd.ap = []

    @staticmethod
    def process_gesture_queue(gestures_queue,ignored_gestures=['point', 'no_moving', 'five', 'pinch']):
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
        sta, dyn = GestureSentence.get_most_probable_sta_dyn(gestures_queue,2,ignored_gestures)

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
    def process_gesture_probs_by_max(gesture_probs):
        ''' max alongsize 0th axis '''
        return np.max(gesture_probs, axis=0)

    @staticmethod
    def get_most_probable_sta_dyn(gesture_queue, n, ignored_gestures=['point', 'no_moving', 'five', 'pinch']):
        ''' Gets the most 'n' occurings from static and dynamic gestures
            - I sorts gestures_queue list into static and dynamic gesture lists
        e.g. gesture_queue = ['apple','apple','banana','banana','banana', 'coco', 'coco', 'coco','coco']
        Returns: for (n=2): ['coco','banana']
        '''
        static_gestures, dynamic_gestures = [], []

        
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
            if gt == 'static' and len(static_gestures) < n and gesture_name not in ignored_gestures:
                static_gestures.append(gesture_name)
            elif gt == 'dynamic' and len(dynamic_gestures) < n and gesture_name not in ignored_gestures:
                dynamic_gestures.append(gesture_name)
            else: continue
        #print("o4", dynamic_gestures)

        return static_gestures, dynamic_gestures



    @staticmethod
    def get_target_objects__wrapper(n, s, mode):
        if mode == 'modular': raise Exception("Not implemented")
        object_name_1 = None
        if ml.md.real_or_sim_datapull:
            return ml.RealRobotConvenience.get_target_objects(n, s)[0]
        else: # Fake data from GUI
            return ml.md.comboMovePagePickObject1Picked
    @staticmethod
    def get_target_object__wrapper_non_blocking(s, mode):
        object_name_1 = None
        if ml.md.real_or_sim_datapull:
            return ml.RealRobotConvenience.get_target_object_non_blocking(s, mode=mode)
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
                if not gl.gd.present(): time.sleep(1); continue
                direction_vector = np.cross(gl.gd.hand_frames[-1].palm_normal(), gl.gd.hand_frames[-1].palm_direction())
                xy = list(direction_vector[0:2])
                xy.reverse()
                rot = np.rad2deg(np.arctan2(*xy))
                print(f"ap rot: {rot}")
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
                if not gl.gd.present(): time.sleep(1); continue
                dist = gl.gd.hand_frames[-1].touch12
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
            direction_vector = np.cross(gl.gd.hand_frames[-1].palm_normal(), gl.gd.hand_frames[-1].palm_direction())
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

            dist = gl.gd.hand_frames[-1].touch12
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
        ''' OLD
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
    def adaptive_eee(path_gen=None, s=None, object_pick_method='last', blocking=False, mode='modular', ignored_gestures=['point', 'no_moving', 'five', 'pinch'], s_func=None):
        ''' Episode evaluation & execution
        Parameters:
            object_pick_method ('last','max'): Aggregated objects list selected by user, option how to choose
            blocking (bool): Alternative function for getting objects or auxiliary parameters - waits for confirmation
            mode (str): '' - original, 'modular' - get object positions from topic?
            ignored_gestures (list): Gestures which belong to type activation category: e.g. point for deictic gesture type, pinch for mearusing gesture type, etc. 
        '''
        
        # Episode started
        if gl.gd.present():
            
            ## WHEN THE OPTION IS NON BLOCKING THIS MAY NOT FIND ANY GESTURES
            activated_gestures = gl.gd.load_all_relevant_activated_gestures(relevant_time=2.0, records=3)

            activated_gesture_type = GestureSentence.get_adaptive_gesture_type(activated_gestures)

            activated_gesture_type_action = GestureSentence.activated_gesture_type_to_action(activated_gesture_type, blocking=blocking)

            ''' When no longer Gesture type activated -> save the accumulated data '''
            if gl.sd.previous_gesture_observed_data_action != 'deictic' and gl.sd.previous_gesture_observed_data_object_names != []:

                if object_pick_method == 'max':
                    c = Counter(gl.sd.previous_gesture_observed_data_object_names)
                    c_max = c.most_common(1)[0]
                    gl.gd.target_objects.append(c_max)
                    i = gl.sd.previous_gesture_observed_data_object_names.index(c_max)
                    gl.gd.target_object_infos.append(gl.sd.previous_gesture_observed_data_object_info[i])
                elif object_pick_method == 'last':
                    gl.gd.target_objects.append(gl.sd.previous_gesture_observed_data_object_names[-1])
                    gl.gd.target_object_infos.append(gl.sd.previous_gesture_observed_data_object_info[-1])
                else: raise Exception()

                print(f"{cc.H}Added obj {Counter(gl.sd.previous_gesture_observed_data_object_names)}{cc.E}")
                gl.sd.previous_gesture_observed_data_object_names = []

            if gl.sd.previous_gesture_observed_data_action != 'measurement_distance' and gl.sd.previous_gesture_observed_data_measurement_distance != []:
                gl.gd.ap.append(get_dist_by_extremes(np.array(gl.sd.previous_gesture_observed_data_measurement_distance)))
                print(f"{cc.H}Added dist {get_dist_by_extremes(np.array(gl.sd.previous_gesture_observed_data_measurement_distance))}{cc.E}")

                gl.sd.previous_gesture_observed_data_measurement_distance = []

            object_name_1 = None
            if activated_gesture_type_action == 'deictic':
                ''' Activated gesture enabled Deictic gesture mode.
                    Here: We pick object(s)

                when using blocking option:
                    The program stays in get_target_objects__wrapper, until finds specified number of objects (usually 1)
                '''
                if blocking:
                    object_name_1, on1_info = GestureSentence.get_target_objects__wrapper(1, s, mode)
                    gl.gd.target_objects.append(object_name_1)
                    gl.gd.target_objects_info.append(on1_info)
                    print(f"{cc.H}Added obj {object_name_1}{cc.E}")
                else:
                    object_name_1, on1_info = GestureSentence.get_target_object__wrapper_non_blocking(s, mode)

                    '''
                    if gl.gd.any_hand_stable(time=1000):
                        gl.gd.target_objects.append(object_name_1)
                        print(f"{cc.H}Added obj {object_name_1}{cc.E}")
                        gl.sd.previous_gesture_observed_data = [activated_gesture_type_action, 'written', object_name_1]
                    else:'''
                    if object_name_1 not in ['q']:
                        gl.sd.previous_gesture_observed_data_action = activated_gesture_type_action
                        gl.sd.previous_gesture_observed_data_object_names.append(object_name_1)
                        gl.sd.previous_gesture_observed_data_object_info.append(on1_info)
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
                    gl.sd.previous_gesture_observed_data_action = activated_gesture_type_action
                    gl.sd.previous_gesture_observed_data_measurement_distance.append(dist)
            else:
                activated_gesture_type_action = 'action'


                # Evaluate when action gesturing ends
                gl.sd.previous_gesture_observed_data_action = activated_gesture_type_action
        else:
            if len(gl.gd.gestures_queue) > 0:
                time.sleep(0.5)
                gl.gd.gestures_queue_proc = GestureSentence.process_gesture_queue(gl.gd.gestures_queue,ignored_gestures)
                if len(gl.gd.gestures_queue_proc) > 0:
                    print(f"{cc.OK}EEE\t{gl.gd.gestures_queue_proc}\t{gl.gd.target_objects}\t{gl.gd.ap}{cc.E}")
                    if mode == 'modular':
                        rc.roscm.gesture_sentence_publisher_original.publish(GestureSentence.export_original_to_HRICommand(s))
                        
                        def map_action_inner_func(s, gestures_queue_proc, object_names, ap):
                            ''' Behaviour tree is in passive mode here. It functions only as generator of precondition actions
                                which needs to be taken to execute the main (final) action. 
                                When I choose the last action, e.g. 'return target_action_sequence[-1]',
                                behaviour tree has no effects. BT functionality is left here for potentional future cases.
                                - Note that mapping gestures to actions as called inside the behaviour tree wrapper
                            '''
                            # if len(s.objects) == 0:
                            #     GestureSentence.clearing()
                            #     # No objects of scene -> Send empty command
                            #     return

                            if len(object_names) == 0: # no object added, no pointing was made
                                ''' Object not given -> use eef position '''
                                s = s_func()
                                focus_point = s.r.eef_position_real
                            else:
                                object_name_1 = object_names[0]
                                if s.get_object_by_name(object_name_1) is None:
                                    print(f"{cc.W}Object target is not in the scene!{cc.E}, object name: {object_name_1} objects: {s.O}")
                                    GestureSentence.clearing()
                                    return None
                                focus_point = s.get_object_by_name(object_name_1).position_real
                            
                            if focus_point is None: return
                            # target object to focus point {object_names} -> {focus_point}

                            target_action_sequence = GestureSentence.btsingle_call__wrapper(gestures_queue_proc, focus_point, s)
                            return target_action_sequence[-1]
                            
                            # BT generated {n_actions} target actions {target_actions}"
                        ret = map_action_inner_func(s, gl.gd.gestures_queue_proc, gl.gd.target_objects, gl.gd.ap)
                        if ret is None:
                            rc.roscm.gesture_sentence_publisher_mapped.publish(HRICommand(data=['unsuccessful']))
                        else:
                            rc.roscm.gesture_sentence_publisher_mapped.publish(GestureSentence.export_mapped_to_HRICommand(s, ret))
                        
                        GestureSentence.clearing()
                        return 
                    if not settings.action_execution:
                        GestureSentence.clearing()
                    else:
                        GestureSentence.adaptive_episode_evaluation_and_execution(s=ml.RealRobotConvenience.update_scene(), \
                        gestures_queue_proc=gl.gd.gestures_queue_proc, object_names=gl.gd.target_objects, ap=gl.gd.ap)
                    
            # Whenever hand is not seen clearing
            GestureSentence.clearing_silent()

    @staticmethod
    def export_original_to_HRICommand(s):

        detected_gestures_probs = [] # 2D (gesture activation, probabilities)
        for detected_gesture in gl.gd.gestures_queue:
            stamp, name, hand_tag, all_probs = detected_gesture
            detected_gestures_probs.append(all_probs)

        max_gesture_probs = GestureSentence.process_gesture_probs_by_max(detected_gestures_probs)

        gesture_timestamp     = gl.gd.gestures_queue[-1][0]


        gl.gd.ap # All detected auxiliary parameters


        # get object names, this can be easily obtained from the function deictic this definitely was using that 
        # from that I can easily get object classes 
        if len(gl.gd.target_object_infos) > 0:
            target_object_infos = list(gl.gd.target_object_infos[0])
            target_object_infos = np.array(target_object_infos)
        else:
            target_object_infos = []
        target_object_names = [o[0] for o in target_object_infos]
        target_object_probs = [o[1] for o in target_object_infos]

        object_types = []
        for object_name in target_object_names:
            object_types.append(s.get_object_by_name(object_name).type)

        # Collect the data
        sentence_as_dict = {
            'gestures': gl.gd.Gs, # Gesture names 
            'gesture_probs': max_gesture_probs, # Gesture probabilities 
            'gesture_timestamp': gesture_timestamp, # One timestamp
            # (Note: I can get timestamp for every activation)
            'objects': target_object_names, # This should be all object names detected on the scene
            'object_probs': target_object_probs, # This should be all object likelihoods 
            # 'object_timestamps': None, # TODO
            'object_classes': object_types, # Object type names as cbgo types 
            # Each object type should reference to object class
            # 'storages': [''], # TODO: some objects are storages, received by Ontology get function
            # 'storage_probs': [],
            # 'storage_timestamps': [],
            'parameters': gl.gd.ap, 
            # 'parameter_values': [], 
            # 'parameter_timestamps': [],
        }

        return HRICommand(data=[str(sentence_as_dict)])

    @staticmethod
    def export_mapped_to_HRICommand(s, intent):
        action_names = intent.target_action
        action_probs = 1.
        action_timestamp = 0.
        
        object_names = s.O
        
        object_types = []
        for object_name in object_names:
            object_types.append(s.get_object_by_name(object_name).type)

        object_probs = [0.] * len(object_names)
        object_probs[object_names.index(intent.target_object)] = 1.0

        # Collect the data
        sentence_as_dict = {
            'actions': action_names, # Gesture names 
            'action_probs': action_probs, # Gesture probabilities 
            'action_timestamp': action_timestamp, # One timestamp
            # (Note: I can get timestamp for every activation)
            'objects': object_names, # This should be all object names detected on the scene
            'object_probs': object_probs, # This should be all object likelihoods 
            # 'object_timestamps': None, # TODO
            'object_classes': object_types, # Object type names as cbgo types 
            # Each object type should reference to object class
            # 'storages': [''], # TODO: some objects are storages, received by Ontology get function
            # 'storage_probs': [],
            # 'storage_timestamps': [],
            'parameters': intent.auxiliary_parameters, 
            # 'parameter_values': [], 
            # 'parameter_timestamps': [],
        }

        return HRICommand(data=[str(sentence_as_dict)])
