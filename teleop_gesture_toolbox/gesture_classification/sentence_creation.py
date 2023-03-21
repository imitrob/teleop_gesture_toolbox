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

from os_and_utils.utils import cc, get_cbgo_path

from context_based_gesture_operation.msg import Scene as SceneRos
from context_based_gesture_operation.msg import Gestures as GesturesRos
from context_based_gesture_operation.srv import BTreeSingleCall

sys.path.append(get_cbgo_path())
from context_based_gesture_operation.srcmodules.Actions import Actions
from context_based_gesture_operation.srcmodules.Scenes import Scene
from context_based_gesture_operation.srcmodules.Objects import Object, Cup, Drawer

from spatialmath import UnitQuaternion
import spatialmath as sm

class GestureSentence():
    @staticmethod
    def clearing():
        gl.gd.gestures_queue.clear()
        ml.md.evaluate_episode = False
        print("Move hand out to end the episode!")
        while ml.md.present():
            time.sleep(0.1)

    @staticmethod
    def process_gesture_queue():
        ''' SIMPLIFICATION choose last only '''
        gl.gd.gestures_queue = [gl.gd.gestures_queue[-1]]

        if gl.gd.approve_handle(f"Action gesture: {gl.gd.gestures_queue[-1][1]}, continue? (y/n)") == 'n':
            GestureSentence.clearing()
            return None

    @staticmethod
    def get_target_objects__wrapper(n, s):
        object_name_1 = None
        if ml.md.real_or_sim_datapull:
            return ml.RealRobotConvenience.get_target_objects(n, s)[0]
        else: # Fake data from GUI
            return ml.md.comboMovePagePickObject1Picked

    @staticmethod
    def target_object_to_focus_point(object_name_1, s):
        ''' Object not given -> use eef position '''
        if object_name_1 is None or object_name_1 == 'n':
            s = ml.RealRobotConvenience.update_scene()
            focus_point = s.objects[0].position_real
            #focus_point = s.r.eef_position_real
        else:
            if s.get_object_by_name(object_name_1) is None:
                print(f"{cc.W}Object target is not in the scene!{cc.E}, object name: {object_name_1} objects: {s.O}")
                GestureSentence.clearing()
                return
            focus_point = s.get_object_by_name(object_name_1).position_real
        return focus_point

    @staticmethod
    def btsingle_call__wrapper(focus_point, s):
        sr = s.to_ros(SceneRos())
        sr.focus_point = np.array(focus_point, dtype=float)
        print(f"[INFO] Aggregated gestures: {list(gl.gd.gestures_queue)}")

        time.sleep(0.01)

        req = BTreeSingleCall.Request()
        req.gestures = gl.gd.gestures_queue_to_ros(GesturesRos())

        req.scene = sr

        return rc.roscm.call_tree_singlerun(req)

    @staticmethod
    def load_reamining_object_names(s, target_action):
        if ml.md.real_or_sim_datapull:
            num_of_objects = getattr(ml.RealRobotActionLib, target_action+"_deictic_params")
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
            GestureSentence.test_rot_ap(rot)

            if abs(rot - prev_rot) < 5:
                 final_rot = (rot + prev_rot)/2
                 break
            prev_rot = rot
        print(f"final rot: {final_rot}")
        ap = {'rotation': final_rot}
        GestureSentence.test_rot_ap(0.0)
        return ap

    @staticmethod
    def execute_action_update(s, target_action, target_objects, ap, n, n_actions):
        ''' TODO: Add checking with multmultipleiple objects '''
        if len(target_objects) == 1 and not s.check_semantic_feasibility(target_action, target_objects[0]):
            print(f"{cc.W}Action is not feasible to do!{cc.E}")

        print(f"{cc.H}[{n}/{n_actions}] Execute action(s): {target_action}, {target_objects}? (y) {cc.E }")

        if not (target_action in dir(ml.RealRobotActionLib)):
            print(f"{cc.W}Action not defined{cc.E}")
            GestureSentence.clearing()
            return

        ''' Execute in real '''
        print(f"{cc.H}Executing{cc.E}")
        getattr(ml.RealRobotActionLib, target_action)(target_objects, ap)

        ''' Update semantic scene '''
        Actions.do(sl.scene, (target_action, target_objects[0]))
        # just for record
        ml.md.actions_done.append((target_action, target_objects))


    @staticmethod
    def episode_evaluation_and_execution(s, object_name_1=None):
        '''
        '''
        GestureSentence.process_gesture_queue()

        object_name_1 = GestureSentence.get_target_objects__wrapper(1, s)
        # DEPRECATED
        #if object_name_1 == 'q': # No objects on the scene
        #    GestureSentence.clearing()
        #    return None

        focus_point = GestureSentence.target_object_to_focus_point(object_name_1, s)

        ''' # TODO: Distinguish between 'object' and 'location'
        '''
        target_action_sequence = GestureSentence.btsingle_call__wrapper(focus_point, s)
        print(f"{cc.OK}BTree generated {len(target_action_sequence)} actions!{cc.E}")
        target_actions = [ta.target_action for ta in target_action_sequence]

        n_actions = len(target_action_sequence)
        ''' Fix preconditions '''
        for n,t in enumerate(target_action_sequence[:-1]):
            target_action = t.target_action
            target_objects = [t.target_object]

            GestureSentence.execute_action_update(s, target_action, target_objects, None, n, n_actions)

        ''' Execute last action '''
        target_action = target_action_sequence[-1].target_action
        target_objects = [target_action_sequence[-1].target_object]

        num_of_objects_ = gl.gd.approve_handle(f"Show how many other objects? ", type='number')
        if num_of_objects_ == '':
            rem_objs = GestureSentence.load_reamining_object_names(s, target_actions[-1])
        else:
            rem_objs = ml.RealRobotConvenience.get_target_objects(int(num_of_objects_), s)
        target_objects = [object_name_1] + rem_objs

        num_of_auxiliary_params = gl.gd.approve_handle(f"Show how many metric parameters? ", type='number')
        ap = GestureSentence.load_auxiliary_parameters(s, target_actions[-1], num_of_auxiliary_params)

        GestureSentence.execute_action_update(s, target_action, target_objects, ap, n_actions, n_actions)

        ''' Clear queue '''
        GestureSentence.clearing()
