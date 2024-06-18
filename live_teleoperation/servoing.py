




from copy import deepcopy
import time
import numpy as np
from spatialmath import UnitQuaternion
import spatialmath as sm 

from geometry_msgs.msg import Quaternion

class Servo():
    type = "drawing" # type (Str): 'simple' - position based, 'absolute', 'relavive'

    def live_handle_step(self, mod=3, scale=1.0, local_live_mode=None):
        if self.seq % mod == 0:
            if gl.gd.r_present():
                self.do_live_mode(h='r', type='drawing with collision detection', link_gesture='grab', scale=scale, local_live_mode=local_live_mode)
            else:
                self.live_mode_drawing = False
        if gl.gd.l_present():
            if self.seq % mod == 0:
                self.grasp_on_basic_grab_gesture(hnds=['l'])


    def do_live_mode(self, h='r', link_gesture='grab', scale=1.0, local_live_mode=None):
        '''
        Parameters:
            rc.roscm.r (obj): coppelia/swift or other
            h (Str): read hand 'r', 'l'
           
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
        elif link_gesture == 'grab' and getattr(gl.gd.hand_frames[-1], h).grab_strength > 0.8:
            a = True
        if Servo.type == 'absolute':
            DirectTeleoperation.absolute_teleop_step(a, self, h,scale=scale)
        elif Servo.type == 'relative':
            DirectTeleoperation.relative_teleop_step(a, self, h, scale=scale)
        elif Servo.type == 'drawing':
            DirectTeleoperation.drawing_teleop_step(a, self, h, scale=scale)
        elif Servo.type == 'drawing with collision detection':
            DirectTeleoperation.drawing_mode_with_collision_detection_step(a,self,h,scale=scale,local_live_mode=local_live_mode)
        else: raise Exception(f"Wrong parameter type ({type}) not in ['simple','absolute','relative']")


    def grasp_on_basic_grab_gesture(self, hnds=['l']):
        for h in hnds:
            grab_strength = getattr(gl.gd.hand_frames[-1], h).grab_strength
            if grab_strength > 0.5:
                if not rc.roscm.is_real and sl.scene is not None and not (sl.scene.object_names is None) and self.object_focus_id is not None:
                    rc.roscm.r.pick_object(object=sl.scene.object_names[self.object_focus_id])
                else:
                    rc.roscm.r.close_gripper()
            else:
                rc.roscm.r.release_object()



class DirectTeleoperation():
    """ TODO: Add remaining function to th
    """
    @staticmethod
    def simple_teleop_step(self, h='l'):
        self.goal_pose = tfm.transformLeapToScene(getattr(gl.gd.hand_frames[-1],h).palm_pose(), self.ENV, self.scale, self.camera_orientation)
        rc.roscm.r.go_to_pose(self.goal_pose)

    @staticmethod
    def absolute_teleop_step(a, self, h):
        if a:
            self.goal_pose = tfm.transformLeapToScene(getattr(gl.gd.hand_frames[-1],h).palm_pose(), self.ENV, self.scale, self.camera_orientation)
            rc.roscm.r.go_to_pose(self.goal_pose)

    @staticmethod
    def relative_teleop_step(a, self, h):
        raise Exception("Not Implemented")

    def drawing_teleop_step(a, self, h):
        if a:

            mouse_3d = tfm.transformLeapToScene(getattr(gl.gd.hand_frames[-1],h).palm_pose(), self.ENV, self.scale, self.camera_orientation)

            if self.live_mode == 'With eef rot':
                x,y = gl.gd.hand_frames[-1].r.direction()[0:2]
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
    def correction_by_teleop():
        print(f"Teleop started")
        while not gl.gd.present():
            time.sleep(0.1)
        while gl.gd.present():
            t = time.time()
            while time.time()-t < 5.0:
                Servo.live_handle_step(mod=1, scale=0.03, local_live_mode='no_eef_rotation')
                time.sleep(0.01)
        print(f"Teleop ended")
        return True

    @staticmethod
    def drawing_mode_with_collision_detection_step(a, self, h, scale=0.5, local_live_mode=None):
        live_mode = self.live_mode
        if local_live_mode is not None: live_mode = local_live_mode
        if a:

            mouse_3d = tfm.transformLeapToScene(getattr(gl.gd.hand_frames[-1],h).palm_pose(), self.ENV, self.scale, self.camera_orientation)

            if live_mode == 'With eef rot':
                x,y = gl.gd.hand_frames[-1].r.direction()[0:2]
                angle = np.arctan2(y,x)

            if not self.live_mode_drawing: # init anchor
                self.live_mode_drawing_anchor = mouse_3d
                self.live_mode_drawing_anchor_scene = deepcopy(self.goal_pose)
                self.live_mode_drawing = True

                if live_mode == 'With eef rot':
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
            self.goal_pose.position.x += (goal_pose[0]) * scale
            self.goal_pose.position.y += (goal_pose[1]) * scale
            self.goal_pose.position.z += (goal_pose[2]) * scale

            if live_mode == 'With eef rot':
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
    def damping_compute(position, objects, table_safety=0.03):
        '''
        Parameters:
            eef (Vector[3]): xyz of eef position
            objects (Vector[o][4]): where xyz + approx. ball collision diameter
        Return:
            damping (0 - 1): rate

        Linear damping with parameter table_safety [m]
        '''

        def t__(position):
            return np.clip(10 * (position[2]-table_safety), 0, 1)

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


        mode, magn = DirectTeleoperation.live_mode_damp_scaler(object_name, object_distance_to_bb)

        if mode == 'damping':
            goal_pose = anchor + magn * (mouse_3d - anchor)
        elif mode == 'interact':
            pass
        return goal_pose,magn

    def live_mode_damp_scaler(object_name, object_distance_to_bb):
        # different value based on object_name & type
        magn = DirectTeleoperation.sigmoid(object_distance_to_bb)
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
