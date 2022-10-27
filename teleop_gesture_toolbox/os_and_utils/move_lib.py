import collections
import numpy as np
import yaml, random
from os_and_utils import settings
from copy import deepcopy

## TEMP: to test
import time

from std_msgs.msg import String
from geometry_msgs.msg import Quaternion, Pose, PoseStamped, Point, Vector3
from os_and_utils.utils import ordered_load, point_by_ratio
import os_and_utils.scenes as sl
import gesture_classification.gestures_lib as gl
from os_and_utils.transformations import Transformations as tfm
from promps.promp_lib import ProMPGenerator, map_to_primitive_gesture, get_id_motionprimitive_type
from os_and_utils.path_def import Waypoint
from os_and_utils.utils_ros import samePoses
import os_and_utils.ros_communication_main as rc

class MoveData():
    def __init__(self, init_goal_pose=True, init_env='table'):
        '''
        > saved in arrays
        - Leap Controller
        - Plan (eef_pose, goal_pose, ...)
        > single data
        - Plan (eef_pose, goal_pose, ...)
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

        self.mode = 'play' # 'play'/'live'/'gesture'
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
        self.live_mode = 'Default'

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
        self.live_mode_drawing_rot = False
        self.live_mode_drawing_eef_rot_anchor = []

        if init_goal_pose:
            self.goal_pose = Pose()
            self.goal_pose.position = self.ENV['start']
            self.goal_pose.orientation = self.ENV['ori']

        self.seq = 0

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
                sl.scenes.make_scene(rc.roscm.r, path.scene)
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

    def main_handle_step(self, path_gen, mod=3, action_execution=True):
        ## live mode control
        # TODO: Mapped to right hand now!
        if self.mode == 'live':
            if self.seq % mod == 0:
                if self.r_present():
                    self.do_live_mode(rc.roscm.r, h='r', type='drawing', link_gesture='grab')
                else:
                    self.live_mode_drawing = False
                    self.live_mode_drawing_rot = False
            if self.l_present():
                if self.seq % mod == 0:
                    self.grasp_on_basic_grab_gesture(rc.roscm.r, hnds=['l'])

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
                    if rc.roscm.r is not None: rc.roscm.r.toggle_object(self.attached)
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

                        if rc.roscm.r is not None: rc.roscm.r.toggle_object(self.attached)
                        self.attached = not self.attached

                    self.time_on_one_pose = 0.0
                self.time_on_one_pose += 1
            if rc.roscm.r is not None: rc.roscm.r.go_to_pose(self.goal_pose)

        if self.mode == 'gesture':
            if self.present(): # If any hand visible
                # Send gesture data based on hand mode
                if self.frames and settings.gesture_detection_on:
                    rc.roscm.send_g_data()

            # Handle gesture activation
            if len(gl.gd.actions_queue) > 0:
                action = gl.gd.actions_queue.pop()
                rc.roscm.gesture_solution_pub.publish(String(action[1]))
                if action_execution:
                    if action[1] == 'nothing_dyn':
                        print(f"===================== ACTION {action[1]} ========================")
                    else:
                        print(f"===================== ACTION {action[1]} ========================")
                        path_ = path_gen.handle_action_queue(action)
                        if path_ is not None:
                            path, waypoints = path_
                            if rc.roscm.r is not None:
                                rc.roscm.r.execute_trajectory_with_waypoints(path, waypoints)
                            if np.array(path).any():
                                pose = Pose()
                                pose.position = Point(x=path[-1][0],y=path[-1][1],z=path[-1][2])
                                pose.orientation.x = np.sqrt(2)/2
                                pose.orientation.y = np.sqrt(2)/2
                                self.goal_pose = pose
            # Handle gesture update activation
            if self.frames and action_execution:
                self.handle_action_update(rc.roscm.r)

        # Update focus target
        if self.seq % (settings.yaml_config_gestures['misc']['rate'] * 2) == 0: # every sec
            if sl.scene and len(sl.scene.object_poses) > 0:
                if rc.roscm.r is not None:
                    rc.roscm.r.add_or_edit_object(name='Focus_target', pose=sl.scene.object_poses[self.object_focus_id], timeout=0.2)


        # TODO: Possibility to print some info
        if False and self.present(): # If any hand visible
            # 2. Info + Save plot data
            print(f"fps {self.frames[-1].fps}, id {self.frames[-1].seq}")
            print(f"actions queue {[act[1] for act in gl.gd.actions_queue]}")
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
                        if rc.roscm.r is not None:
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
                            # if rc.roscm.r is not None:
                                #rc.roscm.r.add_or_edit_object(name='Focus_target', pose=sl.scene.object_poses[self.object_focus_id])
                        elif self.current_threshold_to_flip_id < -settings.yaml_config_gestures['misc']['rate']:
                            # move prev
                            self.current_threshold_to_flip_id = 0
                            self.object_focus_id -= 1
                            if self.object_focus_id == -1: self.object_focus_id = sl.scene.n-1
                            # if rc.roscm.r is not None:
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

        for h in ['r']:
            if settings.get_detection_approach(type='dynamic') == 'deterministic':
                if getattr(self.frames[-1], h).visible:
                    ## TEMP: Experimental
                    move = gl.gd.processGest_move_in_axis()
                    if move:
                        gl.gd.actions_queue.append((rc.roscm.get_clock().now().to_sec(),move,h))

    def do_live_mode(self, h='r', type='drawing', link_gesture='grab'):
        '''
        Parameters:
            rc.roscm.r (obj): coppelia/swift or other
            h (Str): read hand 'r', 'l'
            type (Str): 'simple' - position based, 'absolute', 'relavive'
            link_gesture (Str): ['<static_gesture>', 'grab'] - live mode activated when using given static gesture
        '''

        # update focused object based on what is closest
        self.object_focus_id = sl.scene.get_closest_object(self.goal_pose)

        if type == 'simple':
            self.goal_pose = tfm.transformLeapToScene(getattr(self.frames[-1],h).palm_pose(), self.ENV, self.scale, self.camera_orientation)
            if rc.roscm.r is not None:
                rc.roscm.r.go_to_pose(self.goal_pose)
            return

        relevant = getattr(gl.gd, h).static.relevant()
        now_actived_gesture = None

        if relevant: now_actived_gesture = relevant.activate_name
        a = False
        b = False
        # When condifitioned gesture as parameter is activated
        if now_actived_gesture and now_actived_gesture == link_gesture:
            a = True
        # Gesture is 'grab', it is not in list, but it is activated externally
        elif link_gesture == 'grab' and getattr(self.frames[-1], h).grab_strength > 0.8:
            a = True

        if self.live_mode == 'Separate eef rot':
            if getattr(self.frames[-1], h).pinch_strength > 0.8:
                b = True

        if type == 'absolute':
            if a:
                self.goal_pose = tfm.transformLeapToScene(getattr(self.frames[-1],h).palm_pose(), self.ENV, self.scale, self.camera_orientation)
                if rc.roscm.r is not None:
                    rc.roscm.r.go_to_pose(self.goal_pose)

        elif type == 'relative':
            # TODO:
            raise Exception("Not Implemented")
        elif type == 'drawing':
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
                    if rc.roscm.r is not None:
                        rc.roscm.r.set_gripper(eef_rot=self.eef_rot)
            else:
                self.live_mode_drawing = False
            if b:
                x,y = self.frames[-1].r.direction()[0:2]
                angle = np.arctan2(y,x)

                if not self.live_mode_drawing_rot: # init anchor
                    self.live_mode_drawing_rot = True
                    self.eef_rot_scene = deepcopy(self.eef_rot)
                    self.live_mode_drawing_eef_rot_anchor = angle

                self.eef_rot = deepcopy(self.eef_rot_scene)
                self.eef_rot += (angle - self.live_mode_drawing_eef_rot_anchor)
                if rc.roscm.r is not None:
                    rc.roscm.r.set_gripper(eef_rot=self.eef_rot)
            else:
                self.live_mode_drawing_rot = False

            if rc.roscm.r is not None:
                rc.roscm.r.go_to_pose(self.goal_pose)
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

                        if rc.roscm.r is not None:
                            rc.roscm.r.execute_trajectory_with_waypoints(None, wps)

    def grasp_on_basic_grab_gesture(self, hnds=['l']):
        for h in hnds:
            grab_strength = getattr(self.frames[-1], h).grab_strength
            if grab_strength > 0.5:
                if rc.roscm.r is not None:
                    rc.roscm.r.pick_object(sl.scene.object_names[self.object_focus_id])
            else:
                if rc.roscm.r is not None:
                    rc.roscm.r.release_object()


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
