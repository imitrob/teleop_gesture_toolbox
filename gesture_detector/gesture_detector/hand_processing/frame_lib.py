''' Frame, Hand, Finger, Bone classes definitions
Dependency:
    numpy, itertools
Independent to ROS, Leap (you won't be able to export to ROS msgs)

%timeit init empty
Frame: ~143us
Hand: ~70.5us
Figner: ~11.1us
Bone: ~2.7us
'''
import numpy as np
from itertools import combinations
# I can make independent to tf library if quaternion_from_euler function imported
try:
    import transformations
    TF_IMPORT = True
except ImportError:
    TF_IMPORT = False

try:
    from geometry_msgs.msg import Pose, PoseStamped, Point, Quaternion, Vector3
    ROS_IMPORT = True
except ImportError:
    ROS_IMPORT = False

class Frame():
    ''' Advanced variables derived from frame object
    '''
    def __init__(self, frame=None, leapgestures=None):
        if frame:
            self.import_from_leap(frame)
        else:
            # Stamp
            self.seq = 0 # ID of frame
            self.sec = 0 # seconds of frame
            self.nanosec = 0 # nanoseconds
            self.fps = 0. # Frames per second
            self.hands = 0 # Number of hands
            # Hand data
            self.l = Hand()
            self.r = Hand()

        if leapgestures:
            self.leapgestures = leapgestures
        else:
            self.leapgestures = LeapGestures()

    def which_visible(self):
        if self.l.visible and self.r.visible: return 'both'
        elif self.l.visible: return 'l'
        else: return 'r'

    def get_visible(self, pref='l'):
        if self.l.visible and self.r.visible: return getattr(self, pref)
        elif self.l.visible: return self.l
        else: return self.r

    def get_hand(self, h):
        if h == 'lr': return self.get_visible()
        elif h == 'l':
            if not self.l.visible: return None
            return self.l
        elif h == 'r':
            if not self.r.visible: return None
            return self.r
        else: raise Exception("Cannot happen")

    def present(self):
        ''' Any hand present? '''
        return (self.l.visible or self.r.visible)

    def stamp(self):
        return self.sec+self.nanosec*1e-9
    
    @property
    def secs(self):
        return self.sec
    
    @property
    def nsecs(self):
        return self.nanosec

    def import_from_json(self, seq, secs, nsecs, fps, hands, l, r, leapgestures=None, present=None):              
        self.seq = seq
        self.sec = secs
        self.nanosec = nsecs
        self.fps = fps
        self.hands = hands
        self.l = l
        self.r = r
        self.leapgestures = leapgestures
        self.present = present

    def import_from_leap(self, frame):
        self.seq = frame.id
        self.fps = frame.current_frames_per_second
        self.sec = frame.timestamp//1000000
        self.nanosec = 1000*(frame.timestamp%1000000)
        self.hands = len(frame.hands)

        self.l, self.r = Hand(), Hand()
        for hand in frame.hands:
            if hand.is_left:
                self.l = Hand(hand)
            elif hand.is_right:
                self.r = Hand(hand)

    def import_from_ros(self, msg):
        if not ROS_IMPORT: raise Exception("ROS cannot be imported")
        self.seq = msg.seq
        self.fps = msg.fps
        self.sec = msg.header.stamp.sec
        self.nanosec = msg.header.stamp.nanosec
        self.hands = msg.hands

        self.l.import_from_ros(msg.l)
        self.l.prepare_all_data()
        self.r.import_from_ros(msg.r)
        self.r.prepare_all_data()

        self.leapgestures.import_from_ros(msg.leapgestures)

    def __str__(self):
        ''' Invoke: print(Frame())
        '''
        str  = f"Frame {self.seq}: "
        str += f"fps: {self.fps}, "
        str += f"n hands: {self.hands}, "
        str += f"sec: {self.sec}, "
        str += f"nanosec: {self.nanosec}\n"

        for hand in [self.l, self.r]:
            if hand.visible:
                if hand.is_left:
                    str += f"Left hand: \n"
                elif hand.is_right:
                    str += f"Right hand: \n"
                str += f"ID: {hand.id}, "
                if hand.is_valid: str += f"valid, "
                str += f"grab_strength: {hand.grab_strength}, "
                str += f"pinch_strength: {hand.pinch_strength}, "
                str += f"confidence: {hand.confidence}, "
                str += f"palm_normal {hand.palm_normal()}, "
                str += f"direction {hand.direction()}, "
                str += f"palm_position {hand.palm_position()} \n"

                for finger in hand.fingers:
                    for bone in finger.bones:
                        basis = [bone.basis[0](), bone.basis[1](), bone.basis[2]()]
                        str += f"basis: {[item for sublist in basis for item in sublist]}, "
                        str += f"direction: {bone.direction()}, "
                        str += f"next_joint: {bone.next_joint()}, "
                        str += f"prev_joint: {bone.prev_joint()}, "
                        str += f"center: {bone.center()}, "
                        str += f"is_valid: {bone.is_valid}, "
                        str += f"length: {bone.length}, "
                        str += f"width: {bone.width} \n"

                str += f"palm_velocity: {hand.palm_velocity()}, "
                basis = [hand.basis[0](), hand.basis[1](), hand.basis[2]()]
                str += f"basis: {[item for sublist in basis for item in sublist]}, "
                str += f"palm_width: {hand.palm_width}, "
                str += f"sphere_center: {hand.sphere_center()}, "
                str += f"sphere_radius: {hand.sphere_radius}, "
                str += f"stabilized_palm_position: {hand.stabilized_palm_position()}, "
                str += f"time_visible: {hand.time_visible}, "
                str += f"wrist_position: {hand.wrist_position()} \n"

        if self.leapgestures.circle.present:
            str += f"id: {self.leapgestures.circle.id}, "
            str += f"in_progress: {self.leapgestures.circle.in_progress}, "
            str += f"clockwise: {self.leapgestures.circle.clockwise}, "
            str += f"progress: {self.leapgestures.circle.progress}, "
            str += f"angle: {self.leapgestures.circle.angle}, "
            str += f"radius: {self.leapgestures.circle.radius}, "
            str += f"state: {self.leapgestures.circle.state} \n"

        if self.leapgestures.swipe.present:
            str += f"id: {self.leapgestures.swipe.id}, "
            str += f"in_progress: {self.leapgestures.swipe.in_progress}, "
            str += f"direction: {self.leapgestures.swipe.direction}, "
            str += f"speed: {self.leapgestures.swipe.speed}, "
            str += f"state: {self.leapgestures.swipe.state} \n"

        if self.leapgestures.keytap.present:
            str += f"id: {self.leapgestures.keytap.id}, "
            str += f"in_progress: {self.leapgestures.keytap.in_progress}, "
            str += f"direction: {self.leapgestures.keytap.direction}, "
            str += f"position: {self.leapgestures.keytap.position}, "
            str += f"state: {self.leapgestures.keytap.state} \n"

        if self.leapgestures.screentap.present:
            str += f"id: {self.leapgestures.screentap.id}, "
            str += f"in_progress: {self.leapgestures.screentap.in_progress}, "
            str += f"direction: {self.leapgestures.screentap.direction}, "
            str += f"position: {self.leapgestures.screentap.position}, "
            str += f"state: {self.leapgestures.screentap.state} \n"

        str +='\n'
        return str

    def to_ros(self):
        ''' This function requires ROS and teleop_gesture_toolbox pkg
        '''
        try:
            rosm.Frame()
        except:
            try:
                import gesture_msgs.msg as rosm
            except:
                raise Exception("ROS not imported")
        # self frame_lib.Frame() -> frame gesture_msgs.msg/Frame
        frame = rosm.Frame()
        frame.fps = self.fps
        frame.hands = self.hands
        frame.header.stamp.sec = self.sec
        frame.header.stamp.nanosec = self.nanosec
        frame.seq = self.seq

        frame.leapgestures.circle_present = self.leapgestures.circle.present
        frame.leapgestures.circle_id = self.leapgestures.circle.id
        frame.leapgestures.circle_in_progress = self.leapgestures.circle.in_progress
        frame.leapgestures.circle_clockwise = self.leapgestures.circle.clockwise
        frame.leapgestures.circle_progress = self.leapgestures.circle.progress
        frame.leapgestures.circle_angle = self.leapgestures.circle.angle
        frame.leapgestures.circle_radius = self.leapgestures.circle.radius
        frame.leapgestures.circle_state = self.leapgestures.circle.state

        frame.leapgestures.swipe_present = self.leapgestures.swipe.present
        frame.leapgestures.swipe_id = self.leapgestures.swipe.id
        frame.leapgestures.swipe_in_progress = self.leapgestures.swipe.in_progress
        frame.leapgestures.swipe_direction = self.leapgestures.swipe.direction
        frame.leapgestures.swipe_speed = self.leapgestures.swipe.speed
        frame.leapgestures.swipe_state = self.leapgestures.swipe.state

        frame.leapgestures.keytap_present = self.leapgestures.keytap.present
        frame.leapgestures.keytap_id = self.leapgestures.keytap.id
        frame.leapgestures.keytap_in_progress = self.leapgestures.keytap.in_progress
        frame.leapgestures.keytap_direction = self.leapgestures.keytap.direction
        frame.leapgestures.keytap_position = self.leapgestures.keytap.position
        frame.leapgestures.keytap_state = self.leapgestures.keytap.state

        frame.leapgestures.screentap_present = self.leapgestures.screentap.present
        frame.leapgestures.screentap_id = self.leapgestures.screentap.id
        frame.leapgestures.screentap_in_progress = self.leapgestures.screentap.in_progress
        frame.leapgestures.screentap_direction = self.leapgestures.screentap.direction
        frame.leapgestures.screentap_position = self.leapgestures.screentap.position
        frame.leapgestures.screentap_state = self.leapgestures.screentap.state

        for (hand, h) in [(frame.l, self.l), (frame.r, self.r)]:
            hand.visible = h.visible
            hand.id = h.id
            hand.is_left = h.is_left
            hand.is_right = h.is_right
            hand.is_valid = h.is_valid
            hand.grab_strength = h.grab_strength
            hand.pinch_strength = h.pinch_strength
            hand.confidence = h.confidence
            hand.palm_normal = h.palm_normal()
            hand.direction = h.direction()
            hand.palm_position = h.palm_position()

            finger_bones = []
            for fn in h.fingers:
                for b in fn.bones:
                    bone = rosm.Bone()
                    basis = b.basis[0](), b.basis[1](), b.basis[2]()
                    bone.basis = [item for sublist in basis for item in sublist]
                    bone.direction = b.direction()
                    bone.next_joint = b.next_joint()
                    bone.prev_joint = b.prev_joint()
                    bone.center = b.center()
                    bone.is_valid = b.is_valid
                    bone.length = b.length
                    bone.width = b.width
                    finger_bones.append(bone)
            hand.finger_bones = finger_bones

            hand.palm_velocity = h.palm_velocity()
            basis = h.basis[0](), h.basis[1](), h.basis[2]()
            hand.basis = [item for sublist in basis for item in sublist]
            hand.palm_width = h.palm_width
            hand.sphere_center = h.sphere_center()
            hand.sphere_radius = h.sphere_radius
            hand.stabilized_palm_position = h.stabilized_palm_position()
            hand.time_visible = h.time_visible

            hand.wrist_position = h.wrist_position()
            hand.elbow_position = h.elbow_position()

            hand.arm_valid = h.arm_valid
            hand.arm_width = h.arm_width
            hand.arm_direction = h.arm_direction()
            basis = h.arm_basis[0](), h.arm_basis[1](), h.arm_basis[2]()
            hand.arm_basis = [item for sublist in basis for item in sublist]

        return frame

    def any_hand(self):
        if 'l' in self.__dict__.keys() and self.l.visible:
            return self.l
        elif 'r' in self.__dict__.keys() and self.r.visible:
            return self.r
        else:
            return None


class Hand():
    ''' Advanced variables of hand derived from hand object
    '''
    def __init__(self, hand=None):
        if hand:
            # Import Hand data from Leap Motion hand object
            self.import_from_leap(hand)
        else:
            self.visible = False
            self.id = 0
            self.is_left = False
            self.is_right = False
            self.is_valid = False
            self.grab_strength = 0.
            self.pinch_strength = 0.
            self.confidence = 0.
            self.palm_normal = Vector()
            self.direction = Vector()
            self.palm_position = Vector()
            self.fingers = [Finger(), # Thumb
                            Finger(), # Index
                            Finger(), # Middle
                            Finger(), # Ring
                            Finger()] # Pinky
            self.palm_velocity = Vector()
            self.basis = [Vector(),Vector(),Vector()]
            self.palm_width = 0.
            self.sphere_center = Vector()
            self.sphere_radius = 0.
            self.stabilized_palm_position = Vector()
            self.time_visible = 0.
            self.wrist_position = Vector()
            self.elbow_position = Vector()

            self.arm_valid = False
            self.arm_width = 0.
            self.arm_direction = Vector()
            self.arm_basis = [Vector(),Vector(),Vector()]

        # Data processed for learning
        self.wrist_angles = []
        self.bone_angles = []
        self.finger_distances = []

        self.oc_ = None
        self.oca_ = None
        self.touch12_ = None
        self.touch23_ = None
        self.touch34_ = None
        self.touch45_ = None
        self.touch13_ = None
        self.touch14_ = None
        self.touch15_ = None

    def palm_pose_list(self):
        ''' Returns [x,y,z,qx,qy,qz,qw] '''
        p = self.palm_position()
        q = self.palm_quaternion()
        return [p[0], p[1], p[2], q[0], q[1], q[2], q[3]]

    def palm_pose(self):
        if ROS_IMPORT:
            pose = Pose()
            p = self.palm_position()
            pose.position = Point(x=p[0], y=p[1], z=p[2])
            q = self.palm_quaternion()
            pose.orientation = Quaternion(x=q[0], y=q[1], z=q[2], w=q[3])
            return pose
        else: raise Exception("Cannot get palm_pose! ROS cannot be imported!")

    def palm_quaternion(self):
        return transformations.quaternion_from_euler(*self.palm_euler())

    def palm_euler(self):
        return [self.palm_normal.roll(), self.direction.pitch(), self.direction.yaw()]

    def import_from_json(self, visible, id, is_left, is_right, is_valid, grab_strength, pinch_strength, confidence, palm_normal, direction, palm_position, fingers, palm_velocity, basis, palm_width, sphere_center, sphere_radius, stabilized_palm_position, time_visible, wrist_position, elbow_position = None, arm_valid = None, arm_width = None, arm_direction = None, arm_basis = None, wrist_angles = None, bone_angles = None, finger_distances = None, finger_distances_old = None
        ):
        self.visible = visible
        self.id = id
        self.is_left = is_left
        self.is_right = is_right
        self.is_valid = is_valid
        self.grab_strength = grab_strength
        self.pinch_strength = pinch_strength
        self.confidence = confidence
        self.palm_normal = palm_normal
        self.direction = direction
        self.palm_position = palm_position
        self.fingers = fingers
        self.palm_velocity = palm_velocity
        self.basis = basis
        self.palm_width = palm_width
        self.sphere_center = sphere_center
        self.sphere_radius = sphere_radius
        self.stabilized_palm_position = stabilized_palm_position
        self.time_visible = time_visible
        self.wrist_position = wrist_position
        self.elbow_position = elbow_position
        self.arm_valid = arm_valid
        self.arm_width = arm_width
        self.arm_direction = arm_direction
        self.arm_basis = arm_basis
        self.wrist_angles = wrist_angles
        self.bone_angles = bone_angles
        self.finger_distances = finger_distances
        self.finger_distances_old = finger_distances_old

    def import_from_leap(self, hand):
        self.visible = True
        self.id = hand.id
        self.is_left = hand.is_left
        self.is_right = hand.is_right
        self.is_valid = hand.is_valid
        self.grab_strength = hand.grab_strength
        self.pinch_strength = hand.pinch_strength
        self.confidence = hand.confidence
        v = hand.palm_normal.to_float_array()
        self.palm_normal = Vector(x=v[0], y=v[1], z=v[2])
        v = hand.direction.to_float_array()
        self.direction = Vector(x=v[0], y=v[1], z=v[2])
        v = hand.palm_position.to_float_array()
        self.palm_position = Vector(x=v[0], y=v[1], z=v[2])
        self.fingers = [Finger(hand.fingers[0]),
                        Finger(hand.fingers[1]),
                        Finger(hand.fingers[2]),
                        Finger(hand.fingers[3]),
                        Finger(hand.fingers[4])]

        v = hand.palm_velocity.to_float_array()
        self.palm_velocity = Vector(x=v[0], y=v[1], z=v[2])
        v1 = hand.basis.x_basis.to_float_array()
        v2 = hand.basis.y_basis.to_float_array()
        v3 = hand.basis.z_basis.to_float_array()
        self.basis = [Vector(x=v1[0], y=v1[1], z=v1[2]),
                      Vector(x=v2[0], y=v2[1], z=v2[2]),
                      Vector(x=v3[0], y=v3[1], z=v3[2])]
        self.palm_width = hand.palm_width

        v = hand.sphere_center.to_float_array()
        self.sphere_center = Vector(x=v[0], y=v[1], z=v[2])
        self.sphere_radius = hand.sphere_radius
        v = hand.stabilized_palm_position.to_float_array()
        self.stabilized_palm_position = Vector(x=v[0], y=v[1], z=v[2])
        self.time_visible = hand.time_visible

        v = hand.wrist_position.to_float_array()
        self.wrist_position = Vector(x=v[0], y=v[1], z=v[2])
        v = hand.arm.elbow_position.to_float_array()
        self.elbow_position = Vector(x=v[0], y=v[1], z=v[2])

        self.arm_valid = hand.arm.is_valid
        self.arm_width = hand.arm.width
        v = hand.arm.direction.to_float_array()
        self.arm_direction = Vector(x=v[0], y=v[1], z=v[2])
        v1 = hand.arm.basis.x_basis.to_float_array()
        v2 = hand.arm.basis.y_basis.to_float_array()
        v3 = hand.arm.basis.z_basis.to_float_array()
        self.arm_basis = [Vector(x=v1[0], y=v1[1], z=v1[2]),
                          Vector(x=v2[0], y=v2[1], z=v2[2]),
                          Vector(x=v3[0], y=v3[1], z=v3[2])]

    def import_from_ros(self, hnd):
        self.visible = hnd.visible
        self.id = hnd.id
        self.is_left = hnd.is_left
        self.is_right = hnd.is_right
        self.is_valid = hnd.is_valid
        self.grab_strength = hnd.grab_strength
        self.pinch_strength = hnd.pinch_strength
        self.confidence = hnd.confidence
        v = hnd.palm_normal
        self.palm_normal = Vector(x=v[0], y=v[1], z=v[2])
        v = hnd.direction
        self.direction = Vector(x=v[0], y=v[1], z=v[2])
        v = hnd.palm_position
        self.palm_position = Vector(x=v[0], y=v[1], z=v[2])
        self.fingers[0].import_from_ros(hnd.finger_bones[0:4])
        self.fingers[1].import_from_ros(hnd.finger_bones[4:8])
        self.fingers[2].import_from_ros(hnd.finger_bones[8:12])
        self.fingers[3].import_from_ros(hnd.finger_bones[12:16])
        self.fingers[4].import_from_ros(hnd.finger_bones[16:20])
        v = hnd.palm_velocity
        self.palm_velocity = Vector(x=v[0], y=v[1], z=v[2])
        v1 = hnd.basis[0:3]
        v2 = hnd.basis[3:6]
        v3 = hnd.basis[6:9]
        self.basis = [Vector(x=v1[0], y=v1[1], z=v1[2]),
                      Vector(x=v2[0], y=v2[1], z=v2[2]),
                      Vector(x=v3[0], y=v3[1], z=v3[2])]
        self.palm_width = hnd.palm_width
        v = hnd.sphere_center
        self.sphere_center = Vector(x=v[0], y=v[1], z=v[2])
        self.sphere_radius = hnd.sphere_radius
        v = hnd.stabilized_palm_position
        self.stabilized_palm_position = Vector(x=v[0], y=v[1], z=v[2])
        self.time_visible = hnd.time_visible

        v = hnd.wrist_position
        self.wrist_position = Vector(x=v[0], y=v[1], z=v[2])
        v = hnd.elbow_position
        self.elbow_position = Vector(x=v[0], y=v[1], z=v[2])

        self.arm_valid = hnd.arm_valid
        self.arm_width = hnd.arm_width
        v = hnd.arm_direction
        self.arm_direction = Vector(x=v[0], y=v[1], z=v[2])
        v1 = hnd.arm_basis[0:3]
        v2 = hnd.arm_basis[3:6]
        v3 = hnd.arm_basis[6:9]
        self.arm_basis = [Vector(x=v1[0], y=v1[1], z=v1[2]),
                          Vector(x=v2[0], y=v2[1], z=v2[2]),
                          Vector(x=v3[0], y=v3[1], z=v3[2])]

    def get_angles_array(self):
        if not self.wrist_angles: self.prepare_learning_data()

        ret = self.wrist_angles[1:3]
        for finger in self.bone_angles:
            for bone in finger:
                ret.extend(bone[1:3])
        return ret
    def get_distances_array(self):
        if not finger_distances: self.prepare_learning_data()

        ret = []
        for i in self.finger_distances:
            ret.extend(i)
        return ret

    def get_palm_ros_pose(self):
        if not TF_IMPORT: raise Exception("pip install transformations==2021.6.6")
        q = transformations.quaternion_from_euler(self.palm_normal.roll(), self.direction.pitch(), self.direction.yaw())
        pose = PoseStamped()
        pose.pose.orientation = Quaternion(x=q[0], y=q[1], z=q[2], w=q[3])
        pose.pose.position = Point(x=self.palm_position[0],y=self.palm_position[1],z=self.palm_position[2])
        pose.header.stamp.sec = frame.timestamp//1000000
        pose.header.stamp.nanosec = 1000*(frame.timestamp%1000000)
        return pose

    def get_palm_euler(self):
        return [self.palm_normal.roll(), self.direction.pitch(), self.direction.yaw()]
    def get_palm_velocity(self):
        ''' Palm velocity in meters
        '''
        return self.palm_velocity/1000

    def prepare_all_data(self):
        self.prepare_open_fingers()
        self.prepare_finger_distance_combinations()
        self.prepare_learning_data()

    def point_position(self):
        return self.index_position()

    def index_position(self):
        return self.fingers[1].bones[3].next_joint()

    def index_direction(self):
        return self.fingers[1].bones[3].direction()

    def point_direction(self):
        return self.fingers[1].bones[3].direction()

    def palm_direction(self):
        return self.direction()

    def palm_thumb_direction(self):
        return np.cross(self.palm_normal(), self.palm_direction())

    def palm_thumb_angle(self, wrt='xy'):
        if wrt == 'xy':
            palm_direction = self.palm_thumb_direction()
            xy = list(palm_thumb_direction[0:2])
            xy.reverse()
            return np.rad2deg(np.arctan2(*xy))
        else: raise NotImplementedError()

    @property
    def oc(self):
        if self.oc_ is None:
            self.prepare_open_fingers()
            return self.oc_
        else:
            return self.oc_

    @property
    def oc_activates(self):
        return self.oca

    @property
    def oca(self):
        if self.oca_ is None:
            self.prepare_open_fingers()
            return self.oca_
        else:
            return self.oca_

    def get_open_fingers(self):
        ''' Stand of each finger, for deterministic gesture detection
        Returns:
            open/close fingers (Float[5]): For every finger Float value (0.,1.), 0. - finger closed, 1. - finger opened
        '''
        if not TF_IMPORT: raise Exception("pip install transformations==2021.6.6")

        oc = []
        for i in range(0,5):
            bone_1 = self.fingers[i].bones[0]
            if i == 0: bone_1 = self.fingers[i].bones[1]
            bone_4 = self.fingers[i].bones[3]
            q1 = transformations.quaternion_from_euler(0.0, np.arcsin(-bone_1.direction[1]), np.arctan2(bone_1.direction[0], bone_1.direction[2])) # roll, pitch, yaw
            q2 = transformations.quaternion_from_euler(0.0, np.arcsin(-bone_4.direction[1]), np.arctan2(bone_4.direction[0], bone_4.direction[2])) # roll, pitch, yaw
            oc.append(np.dot(q1, q2))
        return oc

    def prepare_open_fingers(self):
        oc_turn_on_thre = [0.8] * 5
        self.oc_ = self.get_open_fingers()
        self.oca_ = [self.oc_[i]>oc_turn_on_thre[i] for i in range(5)]

    def get_position_tip_of_fingers(self):
        # shape = 5 (fingers) x 3 (Cartesian position)
        position_tip_of_fingers = []
        for i in range(0,5):
            bone_1 = self.fingers[i].bones[0]
            if i == 0: bone_1 = self.fingers[i].bones[1]
            bone_4 = self.fingers[i].bones[3]
            position_tip_of_fingers.append([bone_4.next_joint()[0], bone_4.next_joint()[1], bone_4.next_joint()[2]])

        return position_tip_of_fingers

    def get_finger_distance_combinations(self):
        ''' Distance of finger tips combinations, for deterministic gesture detection
            - Compute Euclidean norm
            - Divide by 1000 is conversion (mm -> m)
        Returns:
            Finger distance combs. (Dict with Floats): Distance values in meters
        '''
        position_tip_of_fingers = self.get_position_tip_of_fingers()
        touches = {}
        touches['12'] = np.sqrt(np.sum(np.power(np.subtract(position_tip_of_fingers[1], position_tip_of_fingers[0]),2)))/1000
        touches['23'] = np.sqrt(np.sum(np.power(np.subtract(position_tip_of_fingers[2], position_tip_of_fingers[1]),2)))/1000
        touches['34'] = np.sqrt(np.sum(np.power(np.subtract(position_tip_of_fingers[3], position_tip_of_fingers[2]),2)))/1000
        touches['45'] = np.sqrt(np.sum(np.power(np.subtract(position_tip_of_fingers[4], position_tip_of_fingers[3]),2)))/1000
        touches['13'] = np.sqrt(np.sum(np.power(np.subtract(position_tip_of_fingers[2], position_tip_of_fingers[0]),2)))/1000
        touches['14'] = np.sqrt(np.sum(np.power(np.subtract(position_tip_of_fingers[3], position_tip_of_fingers[0]),2)))/1000
        touches['15'] = np.sqrt(np.sum(np.power(np.subtract(position_tip_of_fingers[4], position_tip_of_fingers[0]),2)))/1000
        return touches

    def prepare_finger_distance_combinations(self):
        touches = self.get_finger_distance_combinations()
        self.touch12_ = touches['12']
        self.touch23_ = touches['23']
        self.touch34_ = touches['34']
        self.touch45_ = touches['45']
        self.touch13_ = touches['13']
        self.touch14_ = touches['14']
        self.touch15_ = touches['15']

    @property
    def touch12(self):
        if self.touch12_ is not None:
            return self.touch12_
        else:
            self.prepare_finger_distance_combinations()
    @property
    def touch23(self):
        if self.touch23_ is not None:
            return self.touch23_
        else:
            self.prepare_finger_distance_combinations()
    @property
    def touch34(self):
        if self.touch34_ is not None:
            return self.touch34_
        else:
            self.prepare_finger_distance_combinations()
    @property
    def touch45(self):
        if self.touch45_ is not None:
            return self.touch45_
        else:
            self.prepare_finger_distance_combinations()
    @property
    def touch13(self):
        if self.touch13_ is not None:
            return self.touch13_
        else:
            self.prepare_finger_distance_combinations()
    @property
    def touch14(self):
        if self.touch14_ is not None:
            return self.touch14_
        else:
            self.prepare_finger_distance_combinations()
    @property
    def touch15(self):
        if self.touch15_ is not None:
            return self.touch15_
        else:
            self.prepare_finger_distance_combinations()

    def prepare_learning_data(self):
        ''' Data will be saved inside object
        '''
        # bone directions and angles
        hand_direction = np.array(self.direction())
        hand_angles = np.array([0., np.arcsin(-self.direction()[1]), np.arctan2(self.direction()[0], self.direction()[2])])
        v = np.array(self.palm_position()) - np.array(self.wrist_position())

        s = np.sum(v**2)
        if s != 0: wrist_direction = v / np.sqrt(s)
        else: wrist_direction = v
        wrist_angles = np.array([0., np.arcsin(-wrist_direction[1]), np.arctan2(wrist_direction[0], wrist_direction[2])])
        bone_direction, bone_angles_pre = np.zeros([5,4,3]), np.zeros([5,4,3])
        for i in range(0,5):
            for j in range(0,4):
                bone_direction[i][j] = np.array(self.fingers[i].bones[j].direction())
                bone_angles_pre[i][j] = np.array((0., np.arcsin(-self.fingers[i].bones[j].direction()[1]), np.arctan2(self.fingers[i].bones[j].direction()[0], self.fingers[i].bones[j].direction()[2])))

        # bone angles differences (shape = 2)
        self.wrist_angles = (wrist_angles - hand_angles)[1:3]
        # bone angles (shape = 5 x 4 x 2 = 40)
        # distance between finger positions (shape = comb(6,2) = 15), before 45
        len(list(combinations([1,2,3,4,5,6], 2)))
        self.bone_angles = np.zeros([5,4,2])
        for i in range(0,5):
            for j in range(0,4):
                if j == 0:
                    d1 = hand_angles
                else:
                    d1 = bone_angles_pre[i][j-1]
                d2 = bone_angles_pre[i][j]
                self.bone_angles[i][j] = (d1 - d2)[1:3]
        self.bone_angles = self.bone_angles.flatten()
        palm_position = np.array(self.palm_position())

        self.finger_distances = []
        self.finger_distances_old = []
        combs = self.get_position_tip_of_fingers()
        combs.extend([palm_position])
        for comb in combinations(combs,2):
            self.finger_distances.append(np.sqrt(np.sum(np.power(np.array(comb[0]) - np.array(comb[1]),2))))
            self.finger_distances_old.extend(np.array(comb[0]) - np.array(comb[1]))

    def get_learning_data(self, definition=1):
        ''' Return Vector of Observation Parameters
        '''
        self.prepare_learning_data()

        learning_data = list(self.wrist_angles)
        learning_data.extend(self.bone_angles)
        if definition == 0:
            learning_data.extend(self.finger_distances_old)
        elif definition == 1:
            learning_data.extend(self.finger_distances)
        else: raise Exception("Wrong type!")
        return learning_data

    def is_stop(self, threshold=0.02):
        '''
        Returns:
            stop (Bool)
        '''
        if self.visible and np.max(self.palm_velocity())/1000 < threshold:
            return True
        return False

    @property
    def stable(self):
        return self.is_stop()

    def get_learning_data_static(self, definition=0):
        return self.get_learning_data(definition=definition)

    def get_single_learning_data_dynamic(self, definition=0):
        return self.palm_position()

    ''' Deterministic static gestures recognition '''
    def gd_static_grab(self):
        return  not self.oca[0] and not self.oca[1] and not self.oca[2] and not self.oca[3] and not self.oca[4]
    def gd_static_thumbsup(self):
        return      self.oca[0] and not self.oca[1] and not self.oca[2] and not self.oca[3] and not self.oca[4]
    def gd_static_point(self):
        return                          self.oca[1] and not self.oca[2] and not self.oca[3] and not self.oca[4]
    def gd_static_two(self):
        return (not self.oca[0] and     self.oca[1] and     self.oca[2] and not self.oca[3] and not self.oca[4]) \
            or (    self.oca[0] and     self.oca[1] and not self.oca[2] and not self.oca[3] and not self.oca[4])
    def gd_static_three(self):
        return (not self.oca[0] and     self.oca[1] and     self.oca[2] and     self.oca[3] and not self.oca[4]) \
            or (    self.oca[0] and     self.oca[1] and     self.oca[2] and not self.oca[3] and not self.oca[4])
    def gd_static_four(self):
        return (not self.oca[0] and     self.oca[1] and     self.oca[2] and     self.oca[3] and     self.oca[4]) \
            or (    self.oca[0] and     self.oca[1] and     self.oca[2] and     self.oca[3] and not self.oca[4])
    def gd_static_five(self):
        return      self.oca[0] and     self.oca[1] and     self.oca[2] and     self.oca[3] and     self.oca[4]


class Vector():
    def __init__(self, x=0., y=0., z=0.):
        if isinstance(x, (tuple, list, np.ndarray)):
            self.x = x[0]
            self.y = x[1]
            self.z = x[2]
        else:
            self.x = x
            self.y = y
            self.z = z

    def __getitem__(self, key):
        if key == 0: return self.x
        elif key == 1: return self.y
        elif key == 2: return self.z
        else: raise IndexError("Set only indexes (0,1,2) as (x,y,z)")

    def __setitem__(self, key, value):
        if key == 0: self.x = value
        elif key == 1: self.y = value
        elif key == 2: self.z = value
        else: raise IndexError("Set only indexes (0,1,2) as (x,y,z)")

    def __call__(self):
        return [self.x,self.y,self.z]

    def roll(self):
        return angle_between_two_angles(self(), [0.,0.,1.])
    def pitch(self):
        return angle_between_two_angles(self(), [1.,0.,0.])
    def yaw(self):
        return angle_between_two_angles(self(), [0.,1.,0.])
        #self.pitch = np.arcsin(-self.y)
        #self.yaw = np.arctan2(self.x, self.z)

def angle_between_two_angles(vector_1, vector_2):
    unit_vector_1 = vector_1 / np.linalg.norm(vector_1)
    unit_vector_2 = vector_2 / np.linalg.norm(vector_2)
    dot_product = np.dot(unit_vector_1, unit_vector_2)
    return np.arccos(dot_product)


class Bone():
    def __init__(self, bone=None):
        if bone:
            # Import Finger data from Leap Motion finger object
            self.import_from_leap(bone)
        else:
            self.basis = [Vector(),Vector(),Vector()]
            self.direction = Vector()
            self.next_joint = Vector()
            self.prev_joint = Vector()
            self.center = Vector()
            self.is_valid = False
            self.length = 0.
            self.width = 0.
        
    def import_from_json(self, basis, direction, next_joint, prev_joint, center, is_valid, length, width):
        self.basis = basis
        self.direction = direction
        self.next_joint = next_joint
        self.prev_joint = prev_joint
        self.center = center
        self.is_valid = is_valid
        self.length = length
        self.width = width


    def import_from_leap(self, bone):
        v1 = bone.basis.x_basis.to_float_array()
        v2 = bone.basis.y_basis.to_float_array()
        v3 = bone.basis.z_basis.to_float_array()
        self.basis = [Vector(x=v1[0], y=v1[1], z=v1[2]),
                      Vector(x=v2[0], y=v2[1], z=v2[2]),
                      Vector(x=v3[0], y=v3[1], z=v3[2])]
        v = bone.direction.to_float_array()
        self.direction = Vector(x=v[0], y=v[1], z=v[2])
        v = bone.next_joint.to_float_array()
        self.next_joint = Vector(x=v[0], y=v[1], z=v[2])
        v = bone.prev_joint.to_float_array()
        self.prev_joint = Vector(x=v[0], y=v[1], z=v[2])
        v = bone.center.to_float_array()
        self.center = Vector(x=v[0], y=v[1], z=v[2])
        self.is_valid = bone.is_valid
        self.length = bone.length
        self.width = bone.width

    def import_from_ros(self, bn):
        v1 = bn.basis[0:3]
        v2 = bn.basis[3:6]
        v3 = bn.basis[6:9]
        self.basis = [Vector(x=v1[0], y=v1[1], z=v1[2]),
                      Vector(x=v2[0], y=v2[1], z=v2[2]),
                      Vector(x=v3[0], y=v3[1], z=v3[2])]
        v = bn.direction
        self.direction = Vector(x=v[0], y=v[1], z=v[2])
        v = bn.next_joint
        self.next_joint = Vector(x=v[0], y=v[1], z=v[2])
        v = bn.prev_joint
        self.prev_joint = Vector(x=v[0], y=v[1], z=v[2])
        v = bn.center
        self.center = Vector(x=v[0], y=v[1], z=v[2])
        self.is_valid = bn.is_valid
        self.length = bn.length
        self.width = bn.width

class Finger():
    def __init__(self, finger=None):
        if finger:
            # Import Finger data from Leap Motion finger object
            self.import_from_leap(finger)
        else:
            self.bones = [Bone(), # Metacarpal
                          Bone(), # Proximal
                          Bone(), # Intermediate
                          Bone()] # Distal

    def import_from_json(self, bones):
        self.bones = bones

    def import_from_leap(self, finger):
        self.bones = [Bone(finger.bone(0)),
                      Bone(finger.bone(1)),
                      Bone(finger.bone(2)),
                      Bone(finger.bone(3))]

    def import_from_ros(self, bns):
        self.bones[0].import_from_ros(bns[0])
        self.bones[1].import_from_ros(bns[1])
        self.bones[2].import_from_ros(bns[2])
        self.bones[3].import_from_ros(bns[3])


''' Leap Motion gestures definitions
'''
class LeapGestures():
    def __init__(self):
        self.circle = LeapGesturesCircle()
        self.swipe = LeapGesturesSwipe()
        self.keytap = LeapGesturesKeytap()
        self.screentap = LeapGesturesScreentap()



    def import_from_json(self, circle, swipe, keytap, screentap):
        self.circle = circle
        self.swipe = swipe
        self.keytap = keytap
        self.screentap = screentap

    def import_from_ros(self, lg):
        self.circle.present = lg.circle_present
        self.circle.id = lg.circle_id
        self.circle.in_progress = lg.circle_in_progress
        self.circle.clockwise = lg.circle_clockwise
        self.circle.progress = lg.circle_progress
        self.circle.angle = lg.circle_angle
        self.circle.radius = lg.circle_radius
        self.circle.state = lg.circle_state

        self.swipe.present = lg.swipe_present
        self.swipe.id = lg.swipe_id
        self.swipe.in_progress = lg.swipe_in_progress
        self.swipe.direction = lg.swipe_direction
        self.swipe.speed = lg.swipe_speed
        self.swipe.state = lg.swipe_state

        self.keytap.present = lg.keytap_present
        self.keytap.id = lg.keytap_id
        self.keytap.in_progress = lg.keytap_in_progress
        self.keytap.direction = lg.keytap_direction
        self.keytap.position = lg.keytap_position
        self.keytap.state = lg.keytap_state

        self.screentap.present = lg.screentap_present
        self.screentap.id = lg.screentap_id
        self.screentap.in_progress = lg.screentap_in_progress
        self.screentap.direction = lg.screentap_direction
        self.screentap.position = lg.screentap_state
        self.screentap.state = lg.screentap_position

class LeapGesturesCircle():
    def __init__(self):
        self.present = False
        self.id = 0
        self.in_progress = False
        self.clockwise = False
        self.progress = 0.
        self.angle = 0.
        self.radius = 0.
        self.state = 0

    def import_from_json(self, present, id, in_progress, clockwise, progress, angle, radius, state):
        self.present = present
        self.id = id
        self.in_progress = in_progress
        self.clockwise = clockwise
        self.progress = progress
        self.angle = angle
        self.radius = radius
        self.state = state

class LeapGesturesSwipe():
    def __init__(self):
        self.present = False
        self.id = 0
        self.in_progress = False
        self.direction = [0.,0.,0.]
        self.speed = 0.
        self.state = 0

    def import_from_json(self, present, id, in_progress, direction, speed, state):
        self.present = present
        self.id = id
        self.in_progress = in_progress
        self.direction = direction
        self.speed = speed
        self.state = state

class LeapGesturesKeytap():
    def __init__(self):
        self.present = False
        self.id = 0
        self.in_progress = False
        self.direction = [0.,0.,0.]
        self.position = [0.,0.,0.]
        self.state = 0
        self.keytap_flag = True

    def import_from_json(self, present, id, in_progress, direction, position, state, keytap_flag = None):
        self.present = present
        self.id = id
        self.in_progress = in_progress
        self.direction = direction
        self.position = position
        self.state = state
        self.keytap_flag = keytap_flag

class LeapGesturesScreentap():
    def __init__(self):
        self.present = False
        self.id = 0
        self.in_progress = False
        self.direction = [0.,0.,0.]
        self.position = [0.,0.,0.]
        self.state = 0
    
    def import_from_json(self, present, id, in_progress, direction, position, state):
        self.present = present
        self.id = id
        self.in_progress = in_progress
        self.direction = direction
        self.position = position
        self.state = state

# ----------------------------------------------

def transform_quaternion_to_direction_vector(q):
    x,y,z,w = q
    V = [0.,0.,0.]
    V[0] = 2 * (x * z - w * y)
    V[1] = 2 * (y * z + w * x)
    V[2] = 1 - 2 * (x * x + y * y)
    return V

def transform_quaternion_to_normal_vector(q):
    x,y,z,w = q
    V = [0.,0.,0.]
    V[0] = 2 * (x*y - w*z)
    V[1] = 1 - 2 * (x*x + z*z)
    V[2] = 2 * (y*z + w*x)
    return V
