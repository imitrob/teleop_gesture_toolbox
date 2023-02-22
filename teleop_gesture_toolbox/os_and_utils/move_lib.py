import collections
import numpy as np
import yaml, random, time, sys
from os_and_utils import settings
from copy import deepcopy

from std_msgs.msg import String
from geometry_msgs.msg import Quaternion, Pose, PoseStamped, Point, Vector3
from os_and_utils.utils import ordered_load, point_by_ratio, get_cbgo_path
import os_and_utils.scenes as sl
import gesture_classification.gestures_lib as gl
from os_and_utils.transformations import Transformations as tfm
from promps.promp_lib import ProMPGenerator, map_to_primitive_gesture, get_id_motionprimitive_type
from os_and_utils.path_def import Waypoint
from os_and_utils.utils_ros import samePoses
import os_and_utils.ros_communication_main as rc
import os_and_utils.deitic_lib as dl

import spatialmath as sm
from spatialmath import UnitQuaternion

sys.path.append(get_cbgo_path())
import context_based_gesture_operation as cbgo
from context_based_gesture_operation.srcmodules.Actions import Actions
from context_based_gesture_operation.srcmodules.ConditionedActions import ConditionedAction
import os_and_utils.ycb_data as ycb_data

from context_based_gesture_operation.msg import Scene as SceneRos
from context_based_gesture_operation.msg import Gestures as GesturesRos
from context_based_gesture_operation.srv import BTreeSingleCall

try:
    import spatialmath as sm
    from spatialmath import UnitQuaternion
except ModuleNotFoundError:
    sm = None

class MoveData():
    def __init__(self, init_goal_pose=True, init_env='table'):
        '''
        > saved in arrays
        - Leap Controller
        - Plan (eef_pose, goal_pose, ... )
        > single data
        - Plan (eef_pose, goal_pose, ... )
        - States (joints, velocity, eff, ... )
        '''

        bfr_len = 1000 #settings.configRecording['BufferLen']
        ''' Leap Controller data saved as circullar buffer '''
        self.frames = collections.deque(maxlen=bfr_len)
        ''' '''
        self.goal_pose_array = collections.deque(maxlen=bfr_len)
        self.eef_pose_array = collections.deque(maxlen=bfr_len)

        self.joint_states = collections.deque(maxlen=bfr_len)

        ## Current/Active robot data at the moment
        self.goal_joints = None
        self.goal_pose = None
        self.eef_rot = 0.0 #None # 7th joint rotation [rad] (overwrites joints)
        self.eef_rot_scene = 0.0
        self.joints = None
        self.velocity = None
        self.effort = None
        self.eef_pose = Pose()
        # Goal joints -> RelaxedIK output
        # Goal pose -> RelaxedIK input
        # joints -> JointStates topic
        # eef_pose -> from joint_states


        with open(settings.paths.custom_settings_yaml+"robot_move.yaml", 'r') as stream:
            robot_move_data_loaded = ordered_load(stream, yaml.SafeLoader)

        self.leap_axes = robot_move_data_loaded['LEAP_AXES']
        self.ENV_DAT = {}
        for key in robot_move_data_loaded['ENV_DAT']:
            self.ENV_DAT[key] = {}
            self.ENV_DAT[key]['view'] = robot_move_data_loaded['ENV_DAT'][key]['view']
            self.ENV_DAT[key]['ori_axes'] = robot_move_data_loaded['ENV_DAT'][key]['ori_axes']
            self.ENV_DAT[key]['ori_live'] = robot_move_data_loaded['ENV_DAT'][key]['ori_live']
            self.ENV_DAT[key]['axes'] = robot_move_data_loaded['ENV_DAT'][key]['axes']
            x,y,z = robot_move_data_loaded['ENV_DAT'][key]['min']
            self.ENV_DAT[key]['min'] = Point(x=x,y=y,z=z)
            x,y,z = robot_move_data_loaded['ENV_DAT'][key]['max']
            self.ENV_DAT[key]['max'] = Point(x=x,y=y,z=z)
            x,y,z = robot_move_data_loaded['ENV_DAT'][key]['start']
            self.ENV_DAT[key]['start'] = Point(x=x,y=y,z=z)
            x,y,z,w=robot_move_data_loaded['ENV_DAT'][key]['ori']
            self.ENV_DAT[key]['ori'] = Quaternion(x=x,y=y,z=z,w=w)


        # TODO: Remove this condition
        if settings.inverse_kinematics == 'pyrep':
            self.ENV_DAT['above']['ori'] = Quaternion(x=0.0, y=0.0, z=1.0, w=0.0)
            self.ENV_DAT['wall']['ori']  = Quaternion(x=0, y=np.sqrt(2)/2, z=0, w=np.sqrt(2)/2)
            self.ENV_DAT['table']['ori'] = Quaternion(x=np.sqrt(2)/2, y=np.sqrt(2)/2., z=0.0, w=0.0)
        # chosen workspace
        self.ENV = self.ENV_DAT[init_env]

        # beta
        # angles from camera -> coppelia
        # angles from real worls -> some params
        # TODO: load from YAML
        self.camera_orientation = Vector3(x=0.,y=0.,z=0.)

        self.mode = 'gesture' # 'play'/'live'/'gesture'
        ''' Scaling factor: if self.mode=='live' '''
        self.scale = 2
        ## interactive
        self.action = False
        self.strict_mode = False
        ''' Path ID: if self.mode=='play' '''
        self.picked_path = 0
        self.time_on_one_pose = 0.0
        self.leavingAction = False
        self.HoldValue = 0.0
        self.HoldPrevState = False
        self.currentPose = 0
        self.HoldAnchor = 0 # For moving status bar
        ''' Gripper object attached bool '''
        self.attached = False
        ''' Mode about scene interaction - Deprecated '''
        self.live_mode = 'With eef rot'

        ''' Builder mode: ['stack', 'wall', 'replace'] '''
        self.build_modes = ['stack', 'wall', 'replace']
        self.build_mode = 'stack'

        self.structures = [] # Structure()

        self.gripper = 0.
        self.speed = 5
        self.applied_force = 10

        ### Live mode: gesture data
        self.gestures_goal_pose = Pose()
        self.gestures_goal_pose.position = self.ENV['start']
        self.gestures_goal_pose.orientation = self.ENV['ori']
        self.gestures_goal_stride = 0.1
        self.gestures_goal_rot_stride = np.pi/8
        # The copy of goal_pose in form of active trajectory
        self._goal = None

        ''' Constant for updating trajectories for real manipulator '''
        self.traj_update_horizon = 0.6

        ''' mouse3d_position: Not fully integrated '''
        self.mouse3d_position = [0.3, 0.0, 0.5]

        self.current_threshold_to_flip_id = 0
        self.object_focus_id = 0
        self.object_touch_id = 0

        ''' live mode drawing '''
        self.live_mode_drawing = False
        self.live_mode_drawing_anchor = []
        self.live_mode_drawing_eef_rot_anchor = []

        if init_goal_pose:
            self.goal_pose = Pose()
            self.goal_pose.position = self.ENV['start']
            self.goal_pose.orientation = self.ENV['ori']

        self.seq = 0

        ''' Misc variables '''

        self.real_or_sim_datapull = False
        self.comboMovePagePickObject1Picked = None
        self.comboMovePagePickObject2Picked = None

        self.last_time_livin = time.time()
        self.evaluate_episode = False

        self.actions_done = []

    def present(self):
        return self.r_present() or self.l_present()

    def r_present(self):
        if self.frames and self.frames[-1] and self.frames[-1].r and self.frames[-1].r.visible:
            return True
        return False

    def l_present(self):
        if self.frames and self.frames[-1] and self.frames[-1].l and self.frames[-1].l.visible:
            return True
        return False

    def get_frame_window_of_last_secs(self, stamp, N_secs):
        ''' Select frames chosen with stamp and last N_secs
        '''
        n = 0
        #     stamp-N_secs       stamp
        # ----|-------N_secs------|
        # ---*************************- <-- chosen frames
        #              <~~~while~~~  |
        #                           self.frames[-1].stamp()
        for i in range(-1, -len(self.frames),-1):
            if stamp-N_secs > self.frames[i].stamp():
                n=i
                break
        print(f"len(self.frames) {len(self.frames)}, n {n}")

        # return frames time window
        frames = []
        for i in range(-1, n, -1):
            frames.append(self.frames[i])
        return frames

    def modes(self):
        return ['play', 'live', 'alternative']

    def get_random_position(self):
        ''' Get random position (ret pose obj) within environment based on md.ENV['max'|'min'] boundaries
            Orientation is set to default md.ENV['ori']

        Returns:
            Pose (Pose()): Random pose
        '''
        x_len = self.ENV['max'].x - self.ENV['min'].x
        y_len = self.ENV['max'].y - self.ENV['min'].y
        z_len = self.ENV['max'].z - self.ENV['min'].z

        x = random.random()
        y = random.random()
        z = random.random()

        x_ = self.ENV['min'].x + x_len * x
        y_ = self.ENV['min'].y + y_len * y
        z_ = self.ENV['min'].z + z_len * z

        pose = Pose()
        pose.position = Point(x=x_, y=y_, z=z_)
        pose.orientation = self.ENV['ori']
        return pose

    def get_random_joints(self, settings):
        ''' Returns random robot joints within bounds

        Returns:
            Joints (Float[7]): Robot joints float array based on configuration in settings
        '''
        joints_diff = np.array(settings.upper_lim) - np.array(settings.lower_lim)
        joints_diff_rand = [joints_diff[i] * random.random() for i in range(len(settings.upper_lim))]
        return np.add(settings.lower_lim, joints_diff_rand)

    def point_in_env(self, point):
        if self.ENV['min'].x <= point[0] <= self.ENV['max'].x:
          if self.ENV['min'].y <= point[1] <= self.ENV['max'].y:
            if self.ENV['min'].z <= point[2] <= self.ENV['max'].z:
                return True
        return False

    def changePlayPath(self, path_=None):
        for n, path in enumerate(sl.paths):
            if not path_ or path.name == path_: # pick first path if path_ not given
                sl.scenes.make_scene_from_yaml(path.scene)
                self.picked_path = n
                self.ENV = self.ENV_DAT[path.ENV]
                self.HoldValue = 0
                self.currentPose = 0
                self.goal_pose = deepcopy(sl.paths[1].poses[1])
                break

    def changeLiveMode(self, text):
        # Reset Gestures
        self.gestures_goal_pose = Pose()
        self.gestures_goal_pose.position = deepcopy(self.ENV['start'])
        self.gestures_goal_pose.orientation.w = 1.0

        self.live_mode = text

    def live_handle_step(self, mod=3):
        if self.seq % mod == 0:
            if self.r_present():
                self.do_live_mode(h='r', type='drawing with collision detection', link_gesture='grab')
            else:
                self.live_mode_drawing = False
        if self.l_present():
            if self.seq % mod == 0:
                self.grasp_on_basic_grab_gesture(hnds=['l'])



    def main_handle_step(self, path_gen, mod=3, action_execution=True):
        self.last_time_livin = time.time()

        ## live mode control
        # TODO: Mapped to right hand now!
        if self.mode == 'live':
            self.live_handle_step(mod)
        if self.mode == 'play':
            # controls everything:
            #self.HoldValue
            ## -> Path that will be performed
            pp = self.picked_path
            ## -> HoldValue (0-100) to targetPose number (0,len(path))
            targetPose = int(self.HoldValue / (100/len(sl.paths[pp].poses)))
            if targetPose >= len(sl.paths[pp].poses):
                targetPose = len(sl.paths[pp].poses)-1
            #diff_pose_progress = 100/len(sl.sp[pp].poses)
            if targetPose == self.currentPose:
                self.time_on_one_pose = 0.0
            else:
                ## 1 - Forward, -1 - Backward
                direction = 1 if targetPose - self.currentPose > 0 else -1
                ## Attaching/Detaching when moving backwards
                if self.leavingAction and self.time_on_one_pose <= 0.1 and direction == -1 and sl.paths[pp].actions[self.currentPose] != "":
                    rc.roscm.r.toggle_object(self.attached)
                    self.attached = not self.attached

                    self.leavingAction = False

                ## Set goal_pose one pose in direction
                self.goal_pose = deepcopy(sl.paths[pp].poses[self.currentPose+direction])
                ## On new pose target or over time limit
                if self.time_on_one_pose > 10.0 or samePoses(self.eef_pose, sl.paths[pp].poses[self.currentPose+direction]):
                    self.leavingAction = True
                    self.currentPose = self.currentPose+direction
                    ## Attaching/Detaching when moving forward
                    if sl.paths[pp].actions[self.currentPose] != "" and direction == 1:

                        rc.roscm.r.toggle_object(self.attached)
                        self.attached = not self.attached

                    self.time_on_one_pose = 0.0
                self.time_on_one_pose += 1
            rc.roscm.r.go_to_pose(self.goal_pose)

        if self.mode == 'gesture':
            if self.present(): # If any hand visible
                # Send gesture data based on hand mode
                if self.frames and settings.gesture_detection_on:
                    rc.roscm.send_g_data()
            else:
                if len(gl.gd.gestures_queue) > 0:
                    self.evaluate_episode = True

            # Handle gesture activation
            '''
            if len(gl.gd.gestures_queue) > 0:
                action = gl.gd.gestures_queue.pop()
                #rc.roscm.gesture_solution_publish(String(data=action[1]))
                if action_execution:
                    if action[1] in ['nothing_dyn', 'no_gesture']:
                        pass #print(f"===================== ACTION {action[1]} ========================")
                    else:
                        print(f"===================== ACTION {action[1]} ========================")
                        path_ = path_gen.handle_action_queue(action)
                        if path_ is not None:
                            path, waypoints = path_
                            rc.roscm.r.execute_trajectory_with_waypoints(path, waypoints)
                            if np.array(path).any() and path is not None:
                                pose = Pose()
                                pose.position = Point(x=path[-1][0],y=path[-1][1],z=path[-1][2])
                                pose.orientation = Quaternion(x=0.0, y=1.0, z=0.0, w=0.0)
                                self.goal_pose = pose
            # Handle gesture update activation
            if self.frames and action_execution:
                self.handle_action_update()
            '''
        # Update focus target
        if self.seq % (settings.yaml_config_gestures['misc']['rate'] * 2) == 0: # every sec
            if sl.scene and len(sl.scene.object_poses) > 0:

                rc.roscm.r.add_or_edit_object(name='Focus_target', pose=sl.scene.objects[self.object_focus_id].position_real, timeout=0.2)



        # TODO: Possibility to print some info
        if False and self.present(): # If any hand visible
            # 2. Info + Save plot data
            print(f"fps {self.frames[-1].fps}, id {self.frames[-1].seq}")
            print(f"actions queue {[act[1] for act in gl.gd.gestures_queue]}")
            print(f"point diretoin {self.frames[-1].l.point_direction()}")
            # Printing presented here represent current mapping
            if gl.gd.r.dynamic.relevant():
                print(f"Right Dynamic relevant info: Biggest prob. gesture: {gl.gd.r.dynamic.relevant().biggest_probability}")
                #print(self.frames[-1].r.get_learning_data(definition=1))

            if gl.gd.l.static.relevant():
                print(f"Left Static relevant info: Biggest prob. gesture: {gl.gd.l.static.relevant().biggest_probability}")

                #print(self.frames[-1].l.get_learning_data(definition=1))

        self.seq += 1


    def handle_action_update(self):
        ''' Edited for only left hand classification and right hand metrics
        '''
        for h in ['l']:#, 'r']:
            if getattr(gl.gd, h).static.relevant():
                action_name = getattr(gl.gd, h).static.relevant().activate_name
                if action_name:
                    id_primitive = map_to_primitive_gesture(action_name)
                    if id_primitive == 'gripper':# and getattr(self.frames[-1], 'r').visible:
                        wps = {1.0: Waypoint()}
                        #grr = 0.0 # float(input("Write gripper posiiton: "))
                        #wps[1.0].gripper = grr #1-getattr(self.frames[-1], 'r').pinch_strength #h).pinch_strength
                        if action_name in ['grab', 'closed_hand']:
                            wps[1.0].gripper = 0.0 #h).pinch_strength
                        elif action_name in ['thumbsup', 'opened_hand']:
                            wps[1.0].gripper = 1.0 #h).pinch_strength

                        #wps[1.0].gripper = 1-getattr(self.frames[-1], 'l').grab_strength #h).pinch_strength
                        rc.roscm.r.execute_trajectory_with_waypoints(None, wps)

                    if id_primitive == 'build_switch' and getattr(self.frames[-1], 'r').visible:
                        xe,ye,_ = tfm.transformLeapToUIsimple(self.frames[-1].l.elbow_position(), out='list')
                        xw,yw,_ = tfm.transformLeapToUIsimple(self.frames[-1].l.wrist_position(), out='list')
                        xp,yp,_ = tfm.transformLeapToUIsimple(self.frames[-1].r.point_position(), out='list')

                        min_dist, min_id = 99999, -1
                        distance_threshold = 1000
                        nBuild_modes = len(self.build_modes)
                        for n, i in enumerate(self.build_modes):
                            x_bm, y_bm = point_by_ratio((xe,ye),(xw,yw), 0.5+0.5*(n/nBuild_modes))
                            x_bm = xe + x_bm
                            y_bm = ye + y_bm
                            distance = (x_bm - xp) ** 2 + (y_bm - yp) ** 2
                            if distance < distance_threshold and distance < min_dist:
                                min_dist = distance
                                min_id = n
                        if min_id != -1:
                            self.build_mode = self.build_modes[min_id]

                    if id_primitive == 'focus':# and getattr(self.frames[-1], 'r').visible:
                        if action_name == 'point':
                            self.object_focus_id = 0
                        elif action_name == 'two':
                            self.object_focus_id = 1
                        elif action_name == 'three':
                            self.object_focus_id = 2

                        '''
                        direction = getattr(self.frames[-1], 'r').point_direction() #h).pinch_strength
                        if direction[0] < 0: # user wants to move to next item
                            # check if previously direction was left, if so, self.current_threshold_to_flip_id will be zeroed
                            if self.current_threshold_to_flip_id < 0: self.current_threshold_to_flip_id = 0
                            self.current_threshold_to_flip_id += 1

                        else:
                            if self.current_threshold_to_flip_id > 0: self.current_threshold_to_flip_id = 0
                            self.current_threshold_to_flip_id -= 1

                        if self.current_threshold_to_flip_id > settings.yaml_config_gestures['misc']['rate']:
                            # move next
                            self.current_threshold_to_flip_id = 0
                            self.object_focus_id += 1
                            if self.object_focus_id == sl.scene.n: self.object_focus_id = 0
                            #rc.roscm.r.add_or_edit_object(name='Focus_target', pose=sl.scene.object_poses[self.object_focus_id])
                        elif self.current_threshold_to_flip_id < -settings.yaml_config_gestures['misc']['rate']:
                            # move prev
                            self.current_threshold_to_flip_id = 0
                            self.object_focus_id -= 1
                            if self.object_focus_id == -1: self.object_focus_id = sl.scene.n-1
                            #rc.roscm.r.add_or_edit_object(name='Focus_target', pose=sl.scene.object_poses[self.object_focus_id])
                        #print("self.current_threshold_to_flip_id", self.current_threshold_to_flip_id, "self.object_focus_id", self.object_focus_id)
                        '''
                    '''
                    _, mp_type = get_id_motionprimitive_type(id_primitive)
                    try:
                        robot_promps = self.robot_promps[self.Gs.index(id_primitive)]
                    except ValueError:
                        robot_promps = None # It is static gesture
                    '''
                    #path = mp_type().update_by_id(robot_promps, id_primitive, self.approach, vars)
        '''
        for h in ['r']:
            if settings.get_detection_approach(type='dynamic') == 'deterministic':
                if getattr(self.frames[-1], h).visible:
                    ## TEMP: Experimental
                    move = gl.gd.processGest_move_in_axis()
                    if move:
                        gl.gd.gestures_queue.append((rc.roscm.get_clock().now().to_sec(),move,h))
        '''

    def do_live_mode(self, h='r', type='drawing', link_gesture='grab'):
        '''
        Parameters:
            rc.roscm.r (obj): coppelia/swift or other
            h (Str): read hand 'r', 'l'
            type (Str): 'simple' - position based, 'absolute', 'relavive'
            link_gesture (Str): ['<static_gesture>', 'grab'] - live mode activated when using given static gesture
        '''

        # update focused object based on what is closest
        if sl.scene is not None:
            self.object_focus_id = sl.scene.get_closest_object(self.goal_pose)

        if type == 'simple': return DirectTeleoperation.simple_teleop_step(self)
        '''
        Live mode is enabled only, when link_gesture is activated
        '''
        relevant = getattr(gl.gd, h).static.relevant()
        now_actived_gesture = None

        if relevant: now_actived_gesture = relevant.activate_name
        a = False # Activated

        # Check if activated gesture is the gesture which triggers the live mode
        if now_actived_gesture and now_actived_gesture == link_gesture:
            a = True
        # Gesture is 'grab', it is not in list, but it is activated externally
        elif link_gesture == 'grab' and getattr(self.frames[-1], h).grab_strength > 0.8:
            a = True

        if type == 'absolute':
            DirectTeleoperation.absolute_teleop_step(a, self, h)
        elif type == 'relative':
            DirectTeleoperation.relative_teleop_step(a, self, h)
        elif type == 'drawing':
            DirectTeleoperation.drawing_teleop_step(a, self, h)
        elif type == 'drawing with collision detection':
            DirectTeleoperation.drawing_mode_with_collision_detection_step(a,self,h)
        else: raise Exception(f"Wrong parameter type ({type}) not in ['simple','absolute','relative']")


    def grasp_on_grab_gesture(self, hnds=['l']):
        for h in hnds:
            if getattr(gl.gd, h).static.relevant():
                action_name = getattr(gl.gd, h).static.relevant().activate_name
                if action_name:
                    id_primitive = map_to_primitive_gesture(action_name)
                    if id_primitive == 'gripper':
                        wps = {1.0: Waypoint()}
                        if action_name in ['grab', 'closed_hand']:
                            wps[1.0].gripper = 0.0
                        elif action_name in ['thumbsup', 'opened_hand']:
                            wps[1.0].gripper = 1.0

                        rc.roscm.r.execute_trajectory_with_waypoints(None, wps)

    def grasp_on_basic_grab_gesture(self, hnds=['l']):
        for h in hnds:
            grab_strength = getattr(self.frames[-1], h).grab_strength
            if grab_strength > 0.5:
                if not rc.roscm.is_real and sl.scene is not None and not (sl.scene.object_names is None) and self.object_focus_id is not None:
                    rc.roscm.r.pick_object(object=sl.scene.object_names[self.object_focus_id])
                else:
                    rc.roscm.r.close_gripper()
            else:
                rc.roscm.r.release_object()

    def predict_handle(self):
        static_predictions = []
        dynamic_predictions = []
        s = sl.scene
        for gesture in gl.gd.Gs_static:
            static_predictions.append(self.predict_handle__(s, gesture))
        for gesture in gl.gd.Gs_dynamic:
            dynamic_predictions.append(self.predict_handle__(s, gesture))
        return static_predictions, dynamic_predictions


    def predict_handle__(self, s, gesture):
        object_name_1 = None
        if self.real_or_sim_datapull:
            object_name_1 = RealRobotConvenience.get_target_objects(1, s)[0]
        else: # Fake data from GUI
            object_name_1 = self.comboMovePagePickObject1Picked

        ''' Object not given -> use eef position '''
        if object_name_1 is None:
            focus_point = s.r.eef_position_real
        else:
            if s.get_object_by_name(object_name_1) is None:
                print(f"{cc.W}Object target is not in the scene!{cc.E}")
                return
            focus_point = s.get_object_by_name(object_name_1).position_real

        req = BTreeSingleCall.Request()

        gros = GesturesRos()
        gros.probabilities.data = list(np.array(np.zeros(len(gl.gd.Gs)), dtype=float))
        for g in [gesture]:
            gros.probabilities.data[gl.gd.Gs.index(g)] = 1.0
        req.gestures = gros

        sr = s.to_ros(SceneRos())
        sr.focus_point = np.array(focus_point, dtype=float)
        req.scene = sr


        target_action_sequence = rc.roscm.call_tree_singlerun(req)
        if len(target_action_sequence)>0:
            if len(target_action_sequence)==1:
                return f"{target_action_sequence[-1].target_action}\n{target_action_sequence[-1].target_object}"
            else:
                return f"{len(target_action_sequence)},{target_action_sequence[-1].target_action}\n{target_action_sequence[-1].target_object}"
        else:
            return f"none"

class DirectTeleoperation():
    """ TODO: Add remaining function to th
    """
    @staticmethod
    def simple_teleop_step(self):
        self.goal_pose = tfm.transformLeapToScene(getattr(self.frames[-1],h).palm_pose(), self.ENV, self.scale, self.camera_orientation)
        rc.roscm.r.go_to_pose(self.goal_pose)

    @staticmethod
    def absolute_teleop_step(a, self, h):
        if a:
            self.goal_pose = tfm.transformLeapToScene(getattr(self.frames[-1],h).palm_pose(), self.ENV, self.scale, self.camera_orientation)
            rc.roscm.r.go_to_pose(self.goal_pose)

    @staticmethod
    def relative_teleop_step(a, self, h):
        raise Exception("Not Implemented")

    def drawing_teleop_step(a, self, h):
        if a:

            mouse_3d = tfm.transformLeapToScene(getattr(self.frames[-1],h).palm_pose(), self.ENV, self.scale, self.camera_orientation)

            if self.live_mode == 'With eef rot':
                x,y = self.frames[-1].r.direction()[0:2]
                angle = np.arctan2(y,x)

            if not self.live_mode_drawing: # init anchor
                self.live_mode_drawing_anchor = mouse_3d
                self.live_mode_drawing_anchor_scene = deepcopy(self.goal_pose)
                self.live_mode_drawing = True

                if self.live_mode == 'With eef rot':
                    self.eef_rot_scene = deepcopy(self.eef_rot)
                    self.live_mode_drawing_eef_rot_anchor = angle

            #self.goal_pose = self.goal_pose + (mouse_3d - self.live_mode_drawing_anchor)
            self.goal_pose = deepcopy(self.live_mode_drawing_anchor_scene)
            self.goal_pose.position.x += (mouse_3d.position.x - self.live_mode_drawing_anchor.position.x)
            self.goal_pose.position.y += (mouse_3d.position.y - self.live_mode_drawing_anchor.position.y)
            self.goal_pose.position.z += (mouse_3d.position.z - self.live_mode_drawing_anchor.position.z)

            if self.live_mode == 'With eef rot':
                self.eef_rot = deepcopy(self.eef_rot_scene)
                self.eef_rot += (angle - self.live_mode_drawing_eef_rot_anchor)

            q = UnitQuaternion([0.0,0.0,1.0,0.0])
            rot = sm.SO3(q.R) * sm.SO3.Rz(self.eef_rot)
            qx,qy,qz,qw = UnitQuaternion(rot).vec_xyzs
            self.goal_pose.orientation = Quaternion(x=qx, y=qy, z=qz, w=qw)
        else:
            self.live_mode_drawing = False

        rc.roscm.r.go_to_pose(self.goal_pose)


    @staticmethod
    def drawing_mode_with_collision_detection_step(a, self, h):
        if a:

            mouse_3d = tfm.transformLeapToScene(getattr(self.frames[-1],h).palm_pose(), self.ENV, self.scale, self.camera_orientation)

            if self.live_mode == 'With eef rot':
                x,y = self.frames[-1].r.direction()[0:2]
                angle = np.arctan2(y,x)

            if not self.live_mode_drawing: # init anchor
                self.live_mode_drawing_anchor = mouse_3d
                self.live_mode_drawing_anchor_scene = deepcopy(self.goal_pose)
                self.live_mode_drawing = True

                if self.live_mode == 'With eef rot':
                    self.eef_rot_scene = deepcopy(self.eef_rot)
                    self.live_mode_drawing_eef_rot_anchor = angle

            #self.goal_pose = self.goal_pose + (mouse_3d - self.live_mode_drawing_anchor)
            self.goal_pose = deepcopy(self.live_mode_drawing_anchor_scene)
            self.goal_pose.position.x += (mouse_3d.position.x - self.live_mode_drawing_anchor.position.x)
            self.goal_pose.position.y += (mouse_3d.position.y - self.live_mode_drawing_anchor.position.y)
            self.goal_pose.position.z += (mouse_3d.position.z - self.live_mode_drawing_anchor.position.z)

            mouse_3d_list = [mouse_3d.position.x- self.live_mode_drawing_anchor.position.x,
            mouse_3d.position.y- self.live_mode_drawing_anchor.position.y, mouse_3d.position.z- self.live_mode_drawing_anchor.position.z]
            anchor_list = [self.live_mode_drawing_anchor_scene.position.x, self.live_mode_drawing_anchor_scene.position.y, self.live_mode_drawing_anchor_scene.position.z]

            anchor, goal_pose = DirectTeleoperation.damping_difference(anchor=anchor_list,
                                        eef=mouse_3d_list, objects=np.array([]))
            self.goal_pose = deepcopy(self.live_mode_drawing_anchor_scene)
            self.goal_pose.position.x += (goal_pose[0])
            self.goal_pose.position.y += (goal_pose[1])
            self.goal_pose.position.z += (goal_pose[2])

            if self.live_mode == 'With eef rot':
                self.eef_rot = deepcopy(self.eef_rot_scene)
                self.eef_rot += (angle - self.live_mode_drawing_eef_rot_anchor)

            q = UnitQuaternion([0.0,0.0,1.0,0.0])
            rot = sm.SO3(q.R) * sm.SO3.Rz(self.eef_rot)
            qx,qy,qz,qw = UnitQuaternion(rot).vec_xyzs
            self.goal_pose.orientation = Quaternion(x=qx, y=qy, z=qz, w=qw)
        else:
            self.live_mode_drawing = False

        rc.roscm.r.go_to_pose(self.goal_pose)

    @staticmethod
    def compute_closest_pointing_object(hand_trajectory_vector, goal_pose):
        ''' TODO!
        '''
        raise NotImplementedError
        goal_pose + hand_trajectory_vector
        return object_name, object_distance_to_bb

    @staticmethod
    def damping_compute(position, objects):
        '''
        Parameters:
            eef (Vector[3]): xyz of eef position
            objects (Vector[o][4]): where xyz + approx. ball collision diameter
        Return:
            damping (0 - 1): rate

        Linear damping with parameter 0.1m
        '''

        def t__(position):
            return np.clip(10 * (position[2]-0.1), 0, 1)

        def o__(position, p):
            d = np.linalg.norm(position - p[0:3])
            return np.clip(10 * (d - p[3]), 0, 1)

        v__ = []
        v__.append(t__(position))
        #for o in objects:
        #    v__.append(o__(position, o))

        v = np.min(v__)
        return v

    @staticmethod
    def damping_difference(anchor, eef, objects):
        eef = np.array(eef)
        anchor = np.array(anchor)

        path_points = [True, True]
        for i in range(0,100,1):
            i = i/100
            v = anchor + i * (eef)
            r = DirectTeleoperation.damping_compute(v, objects)

            if r < 1.0 and path_points[0] is True:
                path_points[0] = i
            if r == 0.0 and path_points[1] is True:
                path_points[1] = i
        if path_points[0] is True: path_points[0] = 1.0
        if path_points[1] is True: path_points[1] = 1.0
        v = anchor + path_points[0] * (eef)
        v_ = anchor + path_points[1] * (eef)
        damping_factor = (DirectTeleoperation.damping_compute(v_, objects) + DirectTeleoperation.damping_compute(v, objects)) / 2
        v2 = damping_factor * (path_points[1] - path_points[0]) * (eef)
        return anchor, (path_points[0] * (eef))+v2

    '''
    def __test(objects=np.array([])):
        toplot = []

        x = np.arange(10,-10,-1)
        x = x/10
        for i in x:
            toplot.append(damping_difference(np.array([1.0,0.0,0.5]), np.array([1.0,0.0,i]), objects))
        toplot = np.array(toplot)

        plt.plot(x, toplot[:,2])


    __test()
    __test(np.array([[0.0,0.0,0.05,0.05]]))
    '''

    def live_mode_with_damping(mouse_3d):
        '''
        '''
        anchor = np.array([0.0,0.0,0.5])
        #mouse_3d = np.array([0.0,0.0,0.4])

        goal_pose_prev = anchor + (mouse_3d - anchor)



        hand_trajectory_vector = (mouse_3d - anchor)/np.linalg.norm(mouse_3d-anchor)

        #object_name, object_distance_to_bb = compute_closest_pointing_object(hand_trajectory_vector, goal_pose)
        object_name, object_distance_to_bb = 'box1', 0.2


        mode, magn = live_mode_damp_scaler(object_name, object_distance_to_bb)

        if mode == 'damping':
            goal_pose = anchor + magn * (mouse_3d - anchor)
        elif mode == 'interact':
            pass
        return goal_pose,magn

    def live_mode_damp_scaler(object_name, object_distance_to_bb):
        # different value based on object_name & type
        magn = sigmoid(object_distance_to_bb)
        if False: #damping <= 0.0:
            return 'interact', magn
        else:
            return 'damping', magn

    def sigmoid(x, center=0.14, tau=40):
        ''' Inverted sigmoid. sigmoid(x=0)=1, sigmoid(x=center)=0.5
        '''
        return 1 / (1 + np.exp((x-center)*(-tau)))

    '''
    mouse_3d = np.array([[0.0,0.0,0.4],[0.0,0.0,0.35],[0.0,0.0,0.3],[0.0,0.0,0.25],[0.0,0.0,0.2],[0.0,0.0,0.15],[0.0,0.0,0.1],[0.0,0.0,0.08], [0.0,0.0,0.06], [0.0,0.0,0.04], [0.0,0.0,0.02], [0.0,0.0,0.0]])

    magns = []
    for pmouse_3d in mouse_3d:
        edited,magn = live_mode_with_damping(pmouse_3d)
        magns.append(magn)
    magns

    x = np.array(range(12))/10
    y = sigmoid(x)
    import matplotlib.pyplot as plt
    x
    plt.plot(mouse_3d[:,2], edited[:,2])
    '''


class RealRobotConvenience():
    @staticmethod
    def testMovements():
        sleep_time = 0.5
        md.goal_pose = Pose(position=Point(x=0.5, y=0.0, z=0.3), orientation=Quaternion(x=0.0, y=1.0, z=0.0, w=0.0))
        rc.roscm.r.go_to_pose(md.goal_pose)
        '''
        patience = 8 # s
        while not samePoses(md.eef_pose, md.goal_pose, accuracy=0.01):
            time.sleep(0.1)
            patience -= 0.1
            if patience < 0:
                print("[ERROR] Move didn't end up -> reseting")
                return
        print(np.linalg.norm(np.array([md.eef_pose.position.x, md.eef_pose.position.y, md.eef_pose.position.z])-np.array([md.goal_pose.position.x, md.goal_pose.position.y, md.goal_pose.position.z])))
        '''
        input("Now marking objects!")
        RealRobotConvenience.mark_objects(s=RealRobotConvenience.update_scene())



    @staticmethod
    def update_scene_and_print():
        s = RealRobotConvenience.update_scene()
        print(s)
        return s

    @staticmethod
    def deictic():
        s = RealRobotConvenience.update_scene()
        s.object_poses
        if s.object_poses == []: return None, None
        poses = []
        for pose in s.object_poses:
            pose = np.array(pose, dtype=float)
            poses.append( Pose(position=Point(x=pose[0], y=pose[1], z=pose[2]), orientation=Quaternion(x=pose[3], y=pose[4], z=pose[5], w=pose[6])) )

        idobj = dl.dd.main_deitic_fun(md.frames[-1], 'l', poses, plot_line=False)
        return idobj, s.object_names[idobj]

    @staticmethod
    def move_sleep():
        ''' Condition of move completed

        '''
        if rc.roscm.is_real:
            sleep_time = 0.5
            rc.roscm.r.go_to_pose(md.goal_pose)
            time.sleep(sleep_time)
            rc.roscm.r.go_to_pose(md.goal_pose)
            time.sleep(sleep_time)
            rc.roscm.r.go_to_pose(md.goal_pose)
            time.sleep(sleep_time)
        else:
            rc.roscm.r.go_to_pose(md.goal_pose)
            patience = 10 # s
            while not samePoses(md.eef_pose, md.goal_pose, accuracy=0.01):
                time.sleep(0.1)
                patience -= 0.1
                if patience < 0:
                    print("[ERROR] Move didn't end up!")
                    return
            return

        '''
        patience = 0
        while not RealRobotConvenience.same_poses(md.goal_pose, ml.md.joints):
             time.sleep(0.01)
             patience += 1
             if patience > 1000: raise Exception("Move not reached destination!")
        '''

    @staticmethod
    def test_deictic():
        print("Test deictic started")
        try:
            while True:
                time.sleep(0.5)
                _, nameobj = RealRobotConvenience.deictic()
                print(nameobj)
                s = RealRobotConvenience.update_scene()
                RealRobotConvenience.go_on_top_of_object(nameobj, s)
        except KeyboardInterrupt:
            print("KeyboardInterrupt: Test deictic ended\n\n")

    @staticmethod
    def update_scene():
        assert cbgo is not None, "move_lib.py script doesn't have access to context_based_gesture_operation package!"
        if rc.roscm.is_real:
            objects = rc.roscm.r.get_object_positions()

            s = None
            s = cbgo.srcmodules.Scenes.Scene(init='', objects=[], random=False)

            if objects is None: objects = []

            for object in objects:
                type = (ycb_data.NAME2TYPE[ycb_data.COSYPOSE2NAME[object['label']]]).capitalize()

                o = getattr(cbgo.srcmodules.Objects, type)(name=ycb_data.COSYPOSE2NAME[object['label']], position_real=np.array(object['pose'][0]))
                o.quaternion = np.array(object['pose'][1])

                s.objects.append(o)
            #print("Updating Scene REAL!:", s)
            sl.scene = s
            return s
        else:
            position = [md.eef_pose.position.x, md.eef_pose.position.y, md.eef_pose.position.z]
            sl.scene.r.eef_position = sl.scene.pos_real_to_grid(position)
            while md.actions_done != []:
                target_action, target_objects = md.actions_done.pop(0)

                if len(target_objects) == 1:
                    Actions.do(sl.scene, (target_action, target_objects[0]), out=True, ignore_location=True)
                else:
                    raise NotImplementedError("Not implemented")
                    ConditionedActions.do(sl.scene, (target_action, target_objects[0]), out=True, ignore_location=True)
            #print("Updating Scene SIMULATOR!:", sl.scene)
            return sl.scene

    @staticmethod
    def mark_objects(s):

        print("Mark all visible objects procedure started")
        for object in s.objects:
            q_final = RealRobotConvenience.get_quaternion_eef(object.quaternion, object.name)
            p = object.position_real
            input(f"p: {p}, q_final {q_final}")
            md.goal_pose = Pose(position=Point(x=p[0], y=p[1], z=p[2]+0.3), orientation=Quaternion(x=q_final[0],y=q_final[1],z=q_final[2],w=q_final[3]))
            RealRobotConvenience.move_sleep()

            print(f"This is name: {object.name} and type: {object.type}")
            time.sleep(2)

        print("Procedure ended\n\n")


    @staticmethod
    def get_quaternion_eef(q_, name):
        ''' Based on CosyPose, where each object (name) has OFFSET
        '''
        assert sm is not None, "spatialmath package couldn't be imported!"
        try:
            offset_z_rot = ycb_data.OFFSETS_Z_ROT[name.replace("_"," ")]
        except KeyError: # Use simulator
            print(f"get_quaternion_eef - Not found object name: {name}")
            offset_z_rot = 0.

        q = UnitQuaternion([0.0,0.0,1.0,0.0])
        q_2 = UnitQuaternion([q_[3], *q_[0:3]])

        rot = sm.SO3(q.R) * sm.SO3.Rz(q_2.rpy()[2]-np.pi/2+offset_z_rot)
        return UnitQuaternion(rot).vec_xyzs

    @staticmethod
    def go_on_top_of_object(nameobj, s):
        object = s.get_object_by_name(nameobj)
        q_final = RealRobotConvenience.get_quaternion_eef(object.quaternion, object.name)

        p = object.position_real
        md.goal_pose = Pose(position=Point(x=p[0], y=p[1], z=p[2]+0.3), orientation=Quaternion(x=q_final[0],y=q_final[1],z=q_final[2],w=q_final[3]))
        RealRobotConvenience.move_sleep()

    @staticmethod
    def get_target_objects(num_of_objects, s):
        object_names = []
        ret = 'n'
        while ret != 'y':
            for i in range(num_of_objects):
                input("start deictic")
                for i in [3,2,1]:
                    time.sleep(0.3)
                    _, nameobj_ = RealRobotConvenience.deictic()
                    if nameobj_ is None:
                        print("No objects on scene!")
                        return

                    print(i)
                    RealRobotConvenience.go_on_top_of_object(nameobj_, s)

                time.sleep(1.0)
                _, nameobj_ = RealRobotConvenience.deictic()

                print(f"Object {i} picked: {nameobj_}")
                object_names.append(nameobj_)
            ret = input("RIGHT object (y)")
            if ret == 'y': break
            if ret == 'c': continue
            print("new name",[s.objects[int(ret)].name])
            return [s.objects[int(ret)].name]
        return object_names[0]

    @staticmethod
    def manual_check_object6dof_pose():
        if rc.roscm.is_real:
            correct = input("Check correctness of CosyPose and press Enter")
            if correct != "":
                print("Operation aborted!")
                return True
        return False


def printout(func):
    def inner(*args, **kwargs):
        print("-------------- STARTED ------------------")
        ret = func(*args, **kwargs)
        print("==============  DONE  ==================")
        return ret
    return inner

class RealRobotActionLib():
    '''
    High-Level action is sequence of Low-level actions (+init)
    Low-level action is sequence of Moves (+init)
    Move is go_to_pose goal + gripper move

    After every Move, there Behaviour Tree might be called to check the state
    of the system. Conditions which are needed may be written down for checking.
    '''
    cautious = True

    @staticmethod
    def get_available_actions():
        actions = dir(RealRobotActionLib)
        # Discard the standard functions (__...__)
        actions_ = []
        discard_functions = ["get_available_actions", "get_offset_for_name", "check_object_args", 'cautious']
        for action in actions:
            if action[0:2] != "__" and action[-6:] != "params":
                if action not in discard_functions:
                    actions_.append(action)
        return actions_

    @staticmethod
    def check_object_args(args, num_params):
        args = list(args)
        args_ = []
        for arg in args:
            if arg:
                args_.append(arg)

        if len(args_) == num_params:
            return False
        else:
            print("Action has different argument number, Returning!")
            return True



    @staticmethod
    def get_offset_for_name(name):
        try:
            return ycb_data.OFFSETS[name.replace("_"," ")]
        except KeyError:
            return 0.

    place_deictic_params = 0
    @staticmethod
    @printout
    def place(object_names=None):
        ''' Place the object on random free place
        '''
        def init():
            s = RealRobotConvenience.update_scene()
            p = s.get_random_position_in_scene(constraint='on_ground,x-cond,free')
            pr = s.position_real(p)
            if RealRobotActionLib.cautious: input(f"[Place] position grid: {p}, position real: {pr}")
            return pr
        def move_1(pr):
            ''' above location '''
            md.goal_pose = Pose(position=Point(x=pr[0], y=pr[1], z=pr[2]+0.3), orientation=Quaternion(x=0.0,y=1.0,z=0.0,w=0.0))
            RealRobotConvenience.move_sleep()
        def move_2(pr):
            ''' above closer to location '''
            md.goal_pose = Pose(position=Point(x=pr[0], y=pr[1], z=pr[2]+0.1), orientation=Quaternion(x=0.0,y=1.0,z=0.0,w=0.0))
            RealRobotConvenience.move_sleep()
            # gripper move 1
            rc.roscm.r.release_object()

        pr = init()
        check_vars_1 = move_1(pr)
        #checker(check_vars_1)
        check_vars_2 = move_2(pr)
        #checker(check_vars_2)

    move_up_deictic_params = 0
    @staticmethod
    @printout
    def move_up(object_names=None):
        def init():
            s = RealRobotConvenience.update_scene()
            p = s.r.eef_position_real
            return p
        def move_1(p):
            md.goal_pose = Pose(position=Point(x=p[0], y=p[1], z=p[2]+0.2), orientation=Quaternion(x=0.0,y=1.0,z=0.0,w=0.0))
            RealRobotConvenience.move_sleep()

        p = init()
        move_1(p)

    move_left_deictic_params = 0
    @staticmethod
    @printout
    def move_left(object_names=None):
        def init():
            s = RealRobotConvenience.update_scene()
            p = s.r.eef_position_real
            return p
        def move_1(p):
            md.goal_pose = Pose(position=Point(x=p[0], y=p[1]-0.2, z=p[2]), orientation=Quaternion(x=0.0,y=1.0,z=0.0,w=0.0))
            RealRobotConvenience.move_sleep()

        p = init()
        move_1(p)


    move_right_deictic_params = 0
    @staticmethod
    @printout
    def move_right(object_names=None):
        def init():
            s = RealRobotConvenience.update_scene()
            p = s.r.eef_position_real
            return p
        def move_1(p):
            md.goal_pose = Pose(position=Point(x=p[0], y=p[1]+0.2, z=p[2]), orientation=Quaternion(x=0.0,y=1.0,z=0.0,w=0.0))
            RealRobotConvenience.move_sleep()

        p = init()
        move_1(p)

    move_backdown_deictic_params = 0
    @staticmethod
    @printout
    def move_backdown(object_names=None):
        def init():
            s = RealRobotConvenience.update_scene()
            p = s.r.eef_position_real
            return p
        def move_1(p):
            md.goal_pose = Pose(position=Point(x=p[0], y=p[1], z=p[2]-0.2), orientation=Quaternion(x=0.0,y=1.0,z=0.0,w=0.0))
            RealRobotConvenience.move_sleep()

        p = init()
        move_1(p)

    move_down_deictic_params = 0
    @staticmethod
    def move_down(object_names=None):
        RealRobotActionLib.move_backdown(object_names=object_names)

    pick_up_deictic_params = 1
    @staticmethod
    @printout
    def pick_up(object_names):
        def init():
            ''' check vars - objs, update scene '''
            if RealRobotActionLib.check_object_args(object_names, RealRobotActionLib.pick_up_deictic_params): return
            objname = object_names[0]

            s = RealRobotConvenience.update_scene()
            print(s.object_positions_real)
            object = s.get_object_by_name(objname)
            return object
        def move_1(object):
            ''' Move above the picking object'''
            p = np.array(object.position_real)
            p[2]+=0.2
            q = np.array(RealRobotConvenience.get_quaternion_eef(object.quaternion, object.name))
            if RealRobotActionLib.cautious: input(f"Check p: {p.round(2)}, q: {q.round(2)}")
            rc.roscm.r.release_object()
            md.goal_pose = Pose(position=Point(x=p[0], y=p[1], z=p[2]), orientation=Quaternion(x=q[0],y=q[1],z=q[2],w=q[3]))
            RealRobotConvenience.move_sleep()
            if RealRobotConvenience.manual_check_object6dof_pose(): return
        def move_2(object):
            ''' Move to object grasp point'''
            of = RealRobotActionLib.get_offset_for_name(object.name)
            p = np.array(object.position_real)
            p[2]+=of
            q = np.array(RealRobotConvenience.get_quaternion_eef(object.quaternion, object.name))
            if RealRobotActionLib.cautious: input(f"Check p: {p.round(2)}, q: {q.round(2)}, z of {of}")
            md.goal_pose = Pose(position=Point(x=p[0], y=p[1], z=p[2]), orientation=Quaternion(x=q[0],y=q[1],z=q[2],w=q[3]))
            RealRobotConvenience.move_sleep()
            rc.roscm.r.pick_object(object=object.name)
            time.sleep(2.)
        def move_3():
            ''' Move up littlebit'''
            md.goal_pose.position.z += 0.04
            RealRobotConvenience.move_sleep()

        object = init()
        move_1(object)
        move_2(object)
        move_3()

    put_deictic_params = 1
    @staticmethod
    @printout
    def put(object_names=None):
        ''' Same meaning as put_on '''
        RealRobotActionLib.put_on(object_names)

    put_on_deictic_params = 1
    @staticmethod
    @printout
    def put_on(object_names):
        def init():
            if RealRobotActionLib.check_object_args(object_names, RealRobotActionLib.put_on_deictic_params): return
            objname = object_names[0]

            s = RealRobotConvenience.update_scene()
            object = s.get_object_by_name(objname)
            if RealRobotActionLib.cautious: input(f"Put on object {objname}")
            return object
        def move_1(object):
            p = object.position_real
            p[2]+=0.3
            q = RealRobotConvenience.get_quaternion_eef(object.quaternion, object.name)
            if RealRobotActionLib.cautious: input(f"Check p: {p.round(2)}, q: {q.round(2)}, z of {0.3}")
            md.goal_pose = Pose(position=Point(x=p[0], y=p[1], z=p[2]), orientation=Quaternion(x=q[0],y=q[1],z=q[2],w=q[3]))
            RealRobotConvenience.move_sleep()
        def move_2(object):
            of = RealRobotActionLib.get_offset_for_name(object.name)
            p = object.position_real
            q = RealRobotConvenience.get_quaternion_eef(object.quaternion, object.name)
            p[2]-=0.15
            if RealRobotActionLib.cautious: input(f"Check p: {p.round(2)}, q: {q.round(2)}, z of {0.2}")
            md.goal_pose = Pose(position=Point(x=p[0], y=p[1], z=p[2]), orientation=Quaternion(x=q[0],y=q[1],z=q[2],w=q[3]))
            RealRobotConvenience.move_sleep()
            rc.roscm.r.release_object()

        object = init()
        move_1(object)
        move_2(object)

    replace_deictic_params = 2
    @staticmethod
    @printout
    def replace(object_names):
        raise Exception("FIX to new version!")
        def init():
            if RealRobotActionLib.check_object_args(object_names, RealRobotActionLib.replace_deictic_params): return
            objname = object_names[0]
            objname2 = object_names[1]

            s = RealRobotConvenience.update_scene()
            print(s)
            print("picking: ", objname)
            object = s.get_object_by_name(objname)
            return object
        def move_1(object):
            p = object.position_real
            q = get_quaternion_eef(object.quaternion, object.name)
            md.goal_pose = Pose(position=Point(x=p[0], y=p[1], z=p[2]+0.3), orientation=Quaternion(x=q[0],y=q[1],z=q[2],w=q[3]))
            RealRobotConvenience.move_sleep()
        def move_2(object):
            of = RealRobotActionLib.get_offset_for_name(object.name)
            md.goal_pose = Pose(position=Point(x=p[0], y=p[1], z=p[2]+0.2), orientation=Quaternion(x=q[0],y=q[1],z=q[2],w=q[3]))
            RealRobotConvenience.move_sleep()
            input("cont?")
            rc.roscm.r.release_object()

        object = init()
        move_1(object)
        move_2(object)

    put_on_fake_deictic_params = 2
    @staticmethod
    @printout
    def put_on_fake(object_names):
        raise Exception("Fix to new version!")
        def init():
            if RealRobotActionLib.check_object_args(object_names, RealRobotActionLib.put_on_fake_deictic_params): return
            objname = object_names[0]
            objname2 = object_names[1]

            s = RealRobotConvenience.update_scene()
            object = s.get_object_by_name(objname2)

        mindist = np.inf
        bestobj = None
        assert s.object_positions != []
        for object2 in s.object_positions:
            d = np.linalg.norm(np.array(object2)-np.array(object.position_real))
            if d == 0: continue # same object
            if d < mindist:
                bestobj = object2.name
                mindist = d
        print("bestobj: ", bestobj)
        input("cont... >>")
        def move_1():
            RealRobotActionLib.pick_up(objname='mustard bottle')
        def move_2():
            RealRobotActionLib.move_backdown(objname='mustard bottle')
            rc.roscm.r.release_object()
        def move_3():
            RealRobotActionLib.pick_up(objname=objname)
        def move_4():
            RealRobotActionLib.put_on(objname=objname2)

        init()
        move_1()
        move_2()
        move_3()
        move_4()

    pour_deictic_params = 1
    @staticmethod
    @printout
    def pour(object_names):
        def init():
            if RealRobotActionLib.check_object_args(object_names, RealRobotActionLib.replace_deictic_params): return
            objname = object_names[0]

            s = RealRobotConvenience.update_scene()
            print(s)
            if RealRobotActionLib.cautious: input(f"Pouring: {objname}")
            object = s.get_object_by_name(objname)
        def move_1():
            p = object.position_real
            q = get_quaternion_eef(object.quaternion, object.name)
            if RealRobotActionLib.cautious: input(f"Check p: {p.round(2)}, q: {q.round(2)}, z of {0.3}")
            md.goal_pose = Pose(position=Point(x=p[0], y=p[1], z=p[2]+0.3), orientation=Quaternion(x=q[0],y=q[1],z=q[2],w=q[3]))
            RealRobotConvenience.move_sleep()
        def move_2():
            of = RealRobotActionLib.get_offset_for_name(object.name)
            if RealRobotActionLib.cautious: input(f"Check p: {p.round(2)}, q: {q.round(2)}, z of {0.2}")
            ''' Rotate to pour finished pose '''
            rot_angle = 30
            x,y,z,w = q
            q = UnitQuaternion(sm.base.troty(rot_angle, 'deg') @ UnitQuaternion([w, x,y,z])).vec_xyzs

            md.goal_pose = Pose(position=Point(x=p[0], y=p[1], z=p[2]+0.2), orientation=Quaternion(x=q[0],y=q[1],z=q[2],w=q[3]))
            RealRobotConvenience.move_sleep()

        init()
        move_1()
        move_2()

class Structure():
    '''
    stack order:
    ||| <- id=2
    ||| <- id=1
    ||| <- id=0
    wall order: (right direction from id=0 which is base obj)
    (space between blocks is not representative)
        ||| <- id=6
      ||| ||| <- id=3,5
    ||| ||| ||| <- id=1,2,4

    Example building stack with base object with id=0:
    id_0_position = structure = Structure(type='wall', id=0, size=0.04) # [0.,0.,0.]
    id_1_position = structure.add(id=10) # [0.,0.,0.04]
    id_2_position = structure.add(id=30) # [0.,0.,0.08]
    id_3_position = structure.add(id=20) # [0.,0.,0.12]
    print(f"Number of blocks: {structure.n}") # "Number of blocks: 4"
    print(f"Object IDs within structure: {structure.object_stack}") # "Object IDs within structure: [0,10,30,20]"
    id_3_position = structure.remove() # [0.,0.,0.12]
    id_2_position = structure.remove() # [0.,0.,0.8]
    id_1_position = structure.remove() # [0.,0.,0.4]
    print(f"Number of blocks: {structure.n}") # "Number of blocks: 1"
    print(f"Object IDs within structure: {structure.object_stack}") # "Object IDs within structure: [0]"
    id_0_position = structure.remove() # [0.,0.,0.]

    '''
    def __init__(self, type, id=None, size=0.04, base_position=[0.,0.,0.]):
        self.type = type
        self.object_stack = []
        self.object_size = 0.04 # box size [m]
        self.base_position = base_position

        if id is not None: self.add(id)

    def __getattr__(self, attr):
        if attr == 'n':
            return len(self.object_stack)
        elif attr == 'base_id':
            if self.object_stack:
                return self.object_stack[0]
            else:
                return None

    def add(self, id):
        self.object_stack.append(id)
        return self.get_position(id)

    def remove(self):
        position_remove = self.get_position(self.object_stack[-1])
        self.object_stack.pop(-1)
        return position_remove

    def get_n_block_based_on_object_id(self, id):
        for n,obj in enumerate(self.object_stack):
            if obj == id:
                return n
        raise Exception(f"ID '{id}' of object is not present in given structure, which has IDs '{self.object_stack}'")

    def get_position(self, id):
        if self.type == 'stack': return list(np.array(self.get_relative_position_stack(id=id)) + np.array(self.base_position))
        elif self.type == 'wall': return list(np.array(self.get_relative_position_wall(id=id)) + np.array(self.base_position))
        else: raise Exception(f"Structure type '{self.type}' not found !")

    def get_relative_position_stack(self, id):
        ''' new relative (to id=0) position is on top of all stacked objects '''
        n_block = self.get_n_block_based_on_object_id(id)
        return [0.,0., n_block * self.object_size]

    def get_relative_position_wall(self, id):
        ''' new wall position '''
        n_block = self.get_n_block_based_on_object_id(id)

        space_between_blocks = self.object_size / 4
        z_step = self.object_size
        x_step = self.object_size + space_between_blocks
        x_odd = x_step/2

        if n_block == 0:
            return [0.,0.,0.]
        elif n_block == 1:
            return [x_step, 0., 0.]
        elif n_block == 2:
            return [x_odd, 0., z_step]
        elif n_block == 3:
            return [2*x_step, 0., 0.]
        elif n_block == 4:
            return [x_odd+x_step, 0., z_step]
        elif n_block == 5:
            return [x_odd*2, 0., 2*z_step]
        elif n_block == 6:
            return [3*x_step, 0., 0.]
        elif n_block == 7:
            return [x_odd+2*x_step, 0., z_step]
        elif n_block == 8:
            return [x_odd*2, 0., 2*z_step]


def init():
    global md
    md = MoveData()









#
