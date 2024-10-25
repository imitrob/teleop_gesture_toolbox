
import argparse

from copy import deepcopy
import time
from gesture_detector.hand_processing.hand_listener import HandListener
import numpy as np
from spatialmath import UnitQuaternion
import spatialmath as sm 

from rclpy.node import Node
import rclpy
from rclpy.executors import MultiThreadedExecutor
import threading

from playsound import playsound
from geometry_msgs.msg import Point, Quaternion, Pose

# from live_teleoperation.transform import transform_leap_to_scene
from transform import transform_leap_to_scene

from robot import PandaPy
import time

class RosNode(Node):
    def __init__(self):
        super(RosNode, self).__init__("servo_node")

class Servo(PandaPy, HandListener, RosNode):
    def __init__(self,
                 teleop_hand: str = "l", 
                 aux_hand: str = "r",
                 link_gesture: str = "grab_strength",
                 scale: float = 1.0,
                 teleop_rotate_eef: bool = True,
                 ):
        """
        Panda:
            self.move_to_pose(position, orientation) # position (float[3]), orientation (float[4])
            self.grasp(width, speed, force, epsilon_inner, epsilon_outer)

        Args:
            teleop_hand (str, optional): Hand used to teleoperate. 
                Defaults to "l" left hand. "r" for right hand. "" to disable teleop.
            aux_hand (str, optional): Hand used for auxiliary action (gripper open/close).
                Defaults to "l" left hand. "r" for right hand. "" to disable aux action.
            link_gesture (str, optional): Gesture to trigger teleoperation
                Defaults to "grab_strength" - Grab gesture triggers teleoperation.
            teleop_rotate_eef (bool, optional): Reads angle of hand and rotates 7th joint.
                Defaults to True.
        """
        super(Servo, self).__init__()

        self.teleop_hand = teleop_hand 
        self.aux_hand = aux_hand
        
        self.link_gesture = link_gesture
        self.scale = scale
        self.teleop_rotate_eef = teleop_rotate_eef
        
        self.scene_anchor_save = [0.4, 0.0, 0.4, 1.0, 0.0, 0.0, 0.0] # x,y,z,qx,qy,qz,qw [m] wrt. robot base
        self.goal_pose = [0.4, 0.0, 0.4, 1.0, 0.0, 0.0, 0.0]
        self.eef_rot = 0.0
        self.trigger = False

    def is_hand_visible(self, hand):
        if (self.hand_frames and 
            self.hand_frames[-1] and
            getattr(self.hand_frames[-1],hand) and
            getattr(self.hand_frames[-1],hand).visible):
            return True
        return False

    def is_gesture_activated(self, hand, gesture):
        if self.is_hand_visible(hand):
            if getattr(getattr(self.hand_frames[-1],hand),gesture) > 0.8:
                return True
        return False

    def playontrigger(self):
        while True:
            if self.trigger and self.is_hand_visible(self.teleop_hand):
                playsound('/usr/share/sounds/Yaru/stereo/bell.oga', block=True)
            else:
                time.sleep(0.5)

    def teleoperation_step(self, trigger):
        goal_pose = transform_leap_to_scene(
            getattr(self.hand_frames[-1],self.teleop_hand).palm_position(),
            self.scale)
        
        self.move_to_pose(position=goal_pose, orientation=[1.0, 0.0, 0.0, 0.0], speed_factor=0.05)

    def step(self):
        if self.is_hand_visible(self.teleop_hand):
            trigger = self.is_gesture_activated(self.teleop_hand, self.link_gesture)
            self.teleoperation_step(trigger)
        else:
            self.is_drawing = False
        
        if self.is_hand_visible(self.aux_hand):
            grab_strength = getattr(self.hand_frames[-1], self.aux_hand).grab_strength

            # OPTION: Close gripper proportionally with grab strength:
            # self.gripper.grasp(width=(1.-grab_strength)/1, speed=0.2, force=10, epsilon_inner=0.04, epsilon_outer=0.04)
            if grab_strength > 0.8:
                if not self.gripper.read_once().is_grasped:
                    self.gripper.grasp(width=0, speed=0.2, force=10, epsilon_inner=0.04, epsilon_outer=0.04)
            elif grab_strength < 0.2:
                self.gripper.move(0.08, 0.2)

class AbsoluteTeleoperation(Servo):
    """Live mode is enabled only, when link_gesture is activated.
    """    
    def teleoperation_step(self, trigger):
        if trigger:
            goal_pose = transform_leap_to_scene(
                getattr(self.hand_frames[-1],self.teleop_hand).palm_pose(),
                self.scale)
            
            self.move_to_pose(position=goal_pose, orientation=[1.0, 0.0, 0.0, 0.0], speed_factor=0.01)

class RelativeTeleoperation(Servo):
    def teleoperation_step(self, trigger):
        raise Exception("Not Implemented")

class TeleoperationByDrawing(Servo):
    """Live mode is enabled only, when link_gesture is activated.
    """    
    def teleoperation_step(self, trigger):
        self.trigger = trigger
        if trigger:
            
            mouse3d = transform_leap_to_scene(
                getattr(self.hand_frames[-1],self.teleop_hand).palm_pose_list(),
                self.scale
            )

            if self.teleop_rotate_eef:
                x,y = self.hand_frames[-1].r.direction()[0:2]
                angle = np.arctan2(y,x)

            if not self.is_drawing: # init anchor
                self.anchor = mouse3d
                self.scene_anchor = deepcopy(self.scene_anchor_save)
                self.is_drawing = True

                if self.teleop_rotate_eef:
                    self.eef_rot_scene = deepcopy(self.eef_rot)
                    self.live_mode_drawing_eef_rot_anchor = angle

            #goal_pose = goal_pose + (mouse3d - self.anchor)
            goal_pose = deepcopy(self.scene_anchor)
            goal_pose[0] += (mouse3d[0] - self.anchor[0])
            goal_pose[1] += (mouse3d[1] - self.anchor[1])
            goal_pose[2] += (mouse3d[2] - self.anchor[2])

            if self.teleop_rotate_eef:
                self.eef_rot = deepcopy(self.eef_rot_scene)
                self.eef_rot += (angle - self.live_mode_drawing_eef_rot_anchor)

            q = UnitQuaternion([0.0,0.0,1.0,0.0])
            rot = sm.SO3(q.R) * sm.SO3.Rz(self.eef_rot)
            goal_pose[3],goal_pose[4],goal_pose[5],goal_pose[6] = UnitQuaternion(rot).vec_xyzs
            
            # Save cage
            goal_pose = np.clip(
                goal_pose,
                #        [x  , y   , z   , no limits on rotation]
                np.array([0.2, -0.4, 0.03, -10, -10, -10, -10]),
                np.array([0.6,  0.4, 0.4,   10,  10,  10,  10])
            )
            self.goal_pose = goal_pose
        else:
            self.scene_anchor_save = self.goal_pose
            self.is_drawing = False

        self.move_to_pose(position=(
            self.goal_pose[0], 
            self.goal_pose[1], 
            self.goal_pose[2]), 
            orientation=[1.0, 0.0, 0.0, 0.0], 
            speed_factor=0.01
        )



class CorrectingPositionTeleoperation(TeleoperationByDrawing):
    ''' User have limited time (duration) to correct the robot's position. '''
    def __init__(self, duration=5.0):
        super().__init__(scale=0.03, teleop_rotate_eef=False)
        self.duration = duration

    def correction_step(self):
        while not self.present():
            time.sleep(0.1)
        while self.present():
            t = time.time()
            while time.time()-t < self.duration:
                self.step(scale=0.03)
                time.sleep(0.01)

        

class TeleoperationByDrawingSomeCollisionDetection(Servo):
    """Live mode is enabled only, when link_gesture is activated.
    EXPERIMENTAL!
    """    
    def teleoperation_step(self, trigger):
        if trigger:
            mouse3d = transform_leap_to_scene(
                getattr(self.hand_frames[-1],self.teleop_hand).palm_pose_list(), 
                self.ENV, 
                self.scale, 
                self.camera_orientation
            )

            if self.teleop_rotate_eef:
                x,y = self.hand_frames[-1].r.direction()[0:2]
                angle = np.arctan2(y,x)

            if not self.is_drawing: # init anchor
                self.anchor = mouse3d
                self.scene_anchor = deepcopy(self.goal_pose)
                self.is_drawing = True

                if self.teleop_rotate_eef:
                    self.eef_rot_scene = deepcopy(self.eef_rot)
                    self.live_mode_drawing_eef_rot_anchor = angle

            #self.goal_pose = self.goal_pose + (mouse3d - self.anchor)
            self.goal_pose = deepcopy(self.scene_anchor)
            self.goal_pose.position.x += (mouse3d.position.x - self.anchor.position.x)
            self.goal_pose.position.y += (mouse3d.position.y - self.anchor.position.y)
            self.goal_pose.position.z += (mouse3d.position.z - self.anchor.position.z)

            mouse3d_list = [mouse3d.position.x- self.anchor.position.x,
            mouse3d.position.y- self.anchor.position.y, mouse3d.position.z- self.anchor.position.z]
            anchor_list = [self.scene_anchor.position.x, self.scene_anchor.position.y, self.scene_anchor.position.z]

            anchor, goal_pose = self.damping_difference(anchor=anchor_list,
                                        eef=mouse3d_list, objects=np.array([]))
            self.goal_pose = deepcopy(self.scene_anchor)
            self.goal_pose.position.x += (goal_pose[0]) * self.scale
            self.goal_pose.position.y += (goal_pose[1]) * self.scale
            self.goal_pose.position.z += (goal_pose[2]) * self.scale

            if self.teleop_rotate_eef:
                self.eef_rot = deepcopy(self.eef_rot_scene)
                self.eef_rot += (angle - self.live_mode_drawing_eef_rot_anchor)

                q = UnitQuaternion([0.0,0.0,1.0,0.0])
                rot = sm.SO3(q.R) * sm.SO3.Rz(self.eef_rot)
                qx,qy,qz,qw = UnitQuaternion(rot).vec_xyzs
                self.goal_pose.orientation = Quaternion(x=qx, y=qy, z=qz, w=qw)
        else:
            self.is_drawing = False

        self.move_to_pose(position=self.goal_pose, orientation=[1.0, 0.0, 0.0, 0.0], speed_factor=0.01)


    def compute_closest_pointing_object(self, hand_trajectory_vector, goal_pose):
        ''' TODO!
        '''
        raise NotImplementedError
        goal_pose + hand_trajectory_vector
        return object_name, object_distance_to_bb

    def damping_compute(self, position, objects, table_safety=0.03):
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

    def damping_difference(self, anchor, eef, objects):
        eef = np.array(eef)
        anchor = np.array(anchor)

        path_points = [True, True]
        for i in range(0,100,1):
            i = i/100
            v = anchor + i * (eef)
            r = self.damping_compute(v, objects)

            if r < 1.0 and path_points[0] is True:
                path_points[0] = i
            if r == 0.0 and path_points[1] is True:
                path_points[1] = i
        if path_points[0] is True: path_points[0] = 1.0
        if path_points[1] is True: path_points[1] = 1.0
        v = anchor + path_points[0] * (eef)
        v_ = anchor + path_points[1] * (eef)
        damping_factor = (self.damping_compute(v_, objects) + self.damping_compute(v, objects)) / 2
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

    def live_mode_with_damping(self, mouse3d):
        '''
        '''
        anchor = np.array([0.0,0.0,0.5])
        #mouse3d = np.array([0.0,0.0,0.4])

        goal_pose_prev = anchor + (mouse3d - anchor)



        hand_trajectory_vector = (mouse3d - anchor)/np.linalg.norm(mouse3d-anchor)

        #object_name, object_distance_to_bb = compute_closest_pointing_object(hand_trajectory_vector, goal_pose)
        object_name, object_distance_to_bb = 'box1', 0.2


        mode, magn = self.live_mode_damp_scaler(object_name, object_distance_to_bb)

        if mode == 'damping':
            goal_pose = anchor + magn * (mouse3d - anchor)
        elif mode == 'interact':
            pass
        return goal_pose,magn

    def live_mode_damp_scaler(self, object_name, object_distance_to_bb):
        # different value based on object_name & type
        magn = self.sigmoid(object_distance_to_bb)
        if False: #damping <= 0.0:
            return 'interact', magn
        else:
            return 'damping', magn

    def sigmoid(self, x, center=0.14, tau=40):
        ''' Inverted sigmoid. sigmoid(x=0)=1, sigmoid(x=center)=0.5
        '''
        return 1 / (1 + np.exp((x-center)*(-tau)))

    '''
    mouse3d = np.array([[0.0,0.0,0.4],[0.0,0.0,0.35],[0.0,0.0,0.3],[0.0,0.0,0.25],[0.0,0.0,0.2],[0.0,0.0,0.15],[0.0,0.0,0.1],[0.0,0.0,0.08], [0.0,0.0,0.06], [0.0,0.0,0.04], [0.0,0.0,0.02], [0.0,0.0,0.0]])

    magns = []
    for pmouse3d in mouse3d:
        edited,magn = live_mode_with_damping(pmouse3d)
        magns.append(magn)
    magns

    x = np.array(range(12))/10
    y = sigmoid(x)
    import matplotlib.pyplot as plt
    x
    plt.plot(mouse3d[:,2], edited[:,2])
    '''

def test_correction():
    teleop = CorrectingPositionTeleoperation()
    for i in range(5):
        print(f"correction step {i}")
        teleop.correction_step()
    print("Test done")

def main(args):
    rclpy.init()

    if args['approach'] == "CorrectingPositionTeleoperation":
        test_correction()
        return

    ''' teleop = AbsoluteTeleoperation() = eval("AbsoluteTeleoperation")() '''
    teleop = eval(args['approach'])(
        teleop_hand = args['teleop_hand'], 
        aux_hand = args['aux_hand'],
        link_gesture = args['link_gesture'],
        scale = args['scale'],
        teleop_rotate_eef = args['teleop_rotate_eef'],
    )

    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(teleop)
    
    spinning_thread = threading.Thread(target=executor.spin, args=(), daemon=True)
    spinning_thread.start()

    ctrl_thread = threading.Thread(target=teleop.ctrl_node, args=(), daemon=True)
    ctrl_thread.start()

    play_thread = threading.Thread(target=teleop.playontrigger, args=(), daemon=True)
    play_thread.start()

    while True:
        teleop.step()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="Run teleoperation node",
        description="",
        epilog="",
    )
    parser.add_argument(
        "--approach",
        default="TeleoperationByDrawing", 
        choices=[
            "Servo", # Absolute teleoperation without any (safety) constraints
            "AbsoluteTeleoperation", 
            "TeleoperationByDrawing", # (3D Mouse) Grab and Move to teleoperate
        ],
        help="Name of Servoing Class."
    )
    parser.add_argument(
        "--teleop_hand",
        default="r",
        choices=["l", "r", ""],
    )
    parser.add_argument(
        "--aux_hand",
        default="l",
        choices=["l", "r", ""],
        help="Hand that opens and closes the gripper."
    )
    parser.add_argument(
        "--link_gesture",
        default="grab_strength",
        choices=["grab_strength", # hand closed
                 "pinch_strength", # thumb and point fingers touching 
                ],
        help="The gesture activates teleoperation."
    )
    parser.add_argument(
        "--scale",
        default=1.0,
        help="Stretching hand move distance to task space distance."
    )
    parser.add_argument(
        "--teleop_rotate_eef",
        default=True,
    )

    main(vars(parser.parse_args()))
