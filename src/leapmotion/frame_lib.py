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
    import tf
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
            self.secs = 0 # seconds of frame
            self.nsecs = 0 # nanoseconds
            self.fps = 0. # Frames per second
            self.hands = 0 # Number of hands
            # Hand data
            self.l = Hand()
            self.r = Hand()

        if leapgestures:
            self.leapgestures = leapgestures
        else:
            self.leapgestures = LeapGestures()

    def import_from_leap(self, frame):
        self.seq = frame.id
        self.fps = frame.current_frames_per_second
        self.secs = frame.timestamp//1000000
        self.nsecs = 1000*(frame.timestamp%1000000)
        self.hands = len(frame.hands)

        self.l, self.r = Hand(), Hand()
        for hand in frame.hands:
            if hand.is_left:
                self.l = Hand(hand)
            elif hand.is_right:
                self.r = Hand(hand)

    def import_from_ros(self, msg):
        if not ROS_IMPORT: raise Exception("ROS cannot be imported")
        self.seq = msg.header.seq
        self.fps = msg.fps
        self.secs = msg.header.stamp.secs
        self.nsecs = msg.header.stamp.nsecs
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
        str += f"secs: {self.secs}, "
        str += f"nsecs: {self.nsecs}\n"

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
        ''' This function requires ROS and mirracle_gestures pkg
        '''
        try:
            rosm.Frame()
        except:
            try:
                import mirracle_gestures.msg as rosm
            except:
                raise Exception("ROS not imported")
        # self frame_lib.Frame() -> frame mirracle_gestures.msg/Frame
        frame = rosm.Frame()
        frame.fps = self.fps
        frame.hands = self.hands
        frame.header.stamp.secs = self.secs
        frame.header.stamp.nsecs = self.nsecs
        frame.header.seq = self.seq

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

            hand.finger_bones = []
            for fn in h.fingers:
                for b in fn.bones:
                    bone = Bone()

                    basis = b.basis[0](), b.basis[1](), b.basis[2]()
                    bone.basis = [item for sublist in basis for item in sublist]
                    bone.direction = b.direction()
                    bone.next_joint = b.next_joint()
                    bone.prev_joint = b.prev_joint()
                    bone.center = b.center()
                    bone.is_valid = b.is_valid
                    bone.length = b.length
                    bone.width = b.width

                    hand.finger_bones.append(bone)

            hand.palm_velocity = h.palm_velocity()
            basis = h.basis[0](), h.basis[1](), h.basis[2]()
            hand.basis = [item for sublist in basis for item in sublist]
            hand.palm_width = h.palm_width
            hand.sphere_center = h.sphere_center()
            hand.sphere_radius = h.sphere_radius
            hand.stabilized_palm_position = h.stabilized_palm_position()
            hand.time_visible = h.time_visible
            hand.wrist_position = h.wrist_position()

        return frame


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

        # Data processed for learning
        self.wrist_angles = []
        self.bone_angles = []
        self.finger_distances = []

    def palm_pose(self):
        if ROS_IMPORT:
            p = Pose()
            p.position = Point(*self.palm_position())
            p.orientation = Quaternion(*self.palm_quaternion())
            return p
        else: raise Exception("Cannot get palm_pose! ROS cannot be imported!")

    def palm_quaternion(self):
        return tf.transformations.quaternion_from_euler(*self.palm_euler())

    def palm_euler(self):
        return [self.palm_normal.roll(), self.direction.pitch(), self.direction.yaw()]

    def import_from_leap(self, hand):
        self.visible = True
        self.id = hand.id
        self.is_left = hand.is_left
        self.is_right = hand.is_right
        self.is_valid = hand.is_valid
        self.grab_strength = hand.grab_strength
        self.pinch_strength = hand.pinch_strength
        self.confidence = hand.confidence
        self.palm_normal = Vector(*hand.palm_normal.to_float_array())
        self.direction = Vector(*hand.direction.to_float_array())
        self.palm_position = Vector(*hand.palm_position.to_float_array())
        self.fingers = []
        for i in range(5):
            self.fingers.append(Finger(hand.fingers[i]))

        self.palm_velocity = Vector(*hand.palm_velocity.to_float_array())
        self.basis = [Vector(*hand.basis.x_basis.to_float_array()),
            Vector(*hand.basis.y_basis.to_float_array()),
            Vector(*hand.basis.z_basis.to_float_array())]
        self.palm_width = hand.palm_width

        self.sphere_center = Vector(*hand.sphere_center.to_float_array())
        self.sphere_radius = hand.sphere_radius
        self.stabilized_palm_position = Vector(*hand.stabilized_palm_position.to_float_array())
        self.time_visible = hand.time_visible

        self.wrist_position = Vector(*hand.wrist_position.to_float_array())
        self.elbow_position = Vector(*hand.arm.elbow_position.to_float_array())

        self.arm_valid = hand.arm.is_valid
        self.arm_width = hand.arm.width
        self.arm_direction = Vector(*hand.arm.direction.to_float_array())
        self.arm_basis = [Vector(*hand.arm.basis.x_basis.to_float_array()),
            Vector(*hand.arm.basis.y_basis.to_float_array()),
            Vector(*hand.arm.basis.z_basis.to_float_array())]

    def import_from_ros(self, hnd):
        self.visible = hnd.visible
        self.id = hnd.id
        self.is_left = hnd.is_left
        self.is_right = hnd.is_right
        self.is_valid = hnd.is_valid
        self.grab_strength = hnd.grab_strength
        self.pinch_strength = hnd.pinch_strength
        self.confidence = hnd.confidence
        self.palm_normal = Vector(hnd.palm_normal)
        self.direction = Vector(hnd.direction)
        self.palm_position = Vector(hnd.palm_position)
        self.fingers[0].import_from_ros(hnd.finger_bones[0:4])
        self.fingers[1].import_from_ros(hnd.finger_bones[4:8])
        self.fingers[2].import_from_ros(hnd.finger_bones[8:12])
        self.fingers[3].import_from_ros(hnd.finger_bones[12:16])
        self.fingers[4].import_from_ros(hnd.finger_bones[16:20])
        self.palm_velocity = Vector(hnd.palm_velocity)
        self.basis = [Vector(hnd.basis[0:3]),Vector(hnd.basis[3:6]),Vector(hnd.basis[6:9])]
        self.palm_width = hnd.palm_width
        self.sphere_center = Vector(hnd.sphere_center)
        self.sphere_radius = hnd.sphere_radius
        self.stabilized_palm_position = Vector(hnd.stabilized_palm_position)
        self.time_visible = hnd.time_visible

        self.wrist_position = Vector(hnd.wrist_position)
        self.elbow_position = Vector(hnd.elbow_position)

        self.arm_valid = hnd.arm_valid
        self.arm_width = hnd.arm_width
        self.arm_direction = Vector(hnd.arm_direction)
        self.arm_basis = [Vector(hnd.arm_basis[0:3]), Vector(hnd.arm_basis[3:6]),Vector(hnd.arm_basis[6:9])]

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
        if not TF_IMPORT: raise Exception("tf library not imported!")
        q = tf.transformations.quaternion_from_euler(self.palm_normal.roll, self.direction.pitch, self.direction.yaw)
        pose = PoseStamped()
        pose.pose.orientation = Quaternion(*q)
        pose.pose.position = Point(self.palm_position[0],self.palm_position[1],self.palm_position[2])
        pose.header.stamp.secs = frame.timestamp//1000000
        pose.header.stamp.nsecs = 1000*(frame.timestamp%1000000)
        return pose

    def get_palm_euler(self):
        return [self.palm_normal.roll, self.direction.pitch, self.direction.yaw]
    def get_palm_velocity(self):
        ''' Palm velocity in meters
        '''
        return self.palm_velocity/1000

    def prepare_all_data(self):
        self.prepare_open_fingers()
        self.prepare_finger_distance_combinations()
        self.prepare_learning_data()

    def index_position(self):
        return self.fingers[1].bones[3].next_joint()

    def get_open_fingers(self):
        ''' Stand of each finger, for deterministic gesture detection
        Returns:
            open/close fingers (Float[5]): For every finger Float value (0.,1.), 0. - finger closed, 1. - finger opened
        '''
        if not TF_IMPORT: raise Exception("tf library not imported!")

        oc = []
        for i in range(0,5):
            bone_1 = self.fingers[i].bones[0]
            if i == 0: bone_1 = self.fingers[i].bones[1]
            bone_4 = self.fingers[i].bones[3]
            q1 = tf.transformations.quaternion_from_euler(0.0, np.arcsin(-bone_1.direction[1]), np.arctan2(bone_1.direction[0], bone_1.direction[2])) # roll, pitch, yaw
            q2 = tf.transformations.quaternion_from_euler(0.0, np.arcsin(-bone_4.direction[1]), np.arctan2(bone_4.direction[0], bone_4.direction[2])) # roll, pitch, yaw
            oc.append(np.dot(q1, q2))
        return oc

    def prepare_open_fingers(self):
        self.oc = self.get_open_fingers()

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
        touches['12'] = np.sum(np.power(np.subtract(position_tip_of_fingers[1], position_tip_of_fingers[0]),2))/1000
        touches['23'] = np.sum(np.power(np.subtract(position_tip_of_fingers[2], position_tip_of_fingers[1]),2))/1000
        touches['34'] = np.sum(np.power(np.subtract(position_tip_of_fingers[3], position_tip_of_fingers[2]),2))/1000
        touches['45'] = np.sum(np.power(np.subtract(position_tip_of_fingers[4], position_tip_of_fingers[3]),2))/1000
        touches['13'] = np.sum(np.power(np.subtract(position_tip_of_fingers[2], position_tip_of_fingers[0]),2))/1000
        touches['14'] = np.sum(np.power(np.subtract(position_tip_of_fingers[3], position_tip_of_fingers[0]),2))/1000
        touches['15'] = np.sum(np.power(np.subtract(position_tip_of_fingers[4], position_tip_of_fingers[0]),2))/1000
        return touches

    def prepare_finger_distance_combinations(self):
        touches = self.get_finger_distance_combinations()
        self.touch12 = touches['12']
        self.touch23 = touches['23']
        self.touch34 = touches['34']
        self.touch45 = touches['45']
        self.touch13 = touches['13']
        self.touch14 = touches['14']
        self.touch15 = touches['15']

    def prepare_learning_data(self):
        ''' Data will be saved inside object
        '''
        # bone directions and angles
        hand_direction = np.array(self.direction())
        hand_angles = np.array([0., np.arcsin(-self.direction()[1]), np.arctan2(self.direction()[0], self.direction()[2])])
        v = np.array(self.palm_position()) - np.array(self.wrist_position())

        wrist_direction = v / np.sqrt(np.sum(v**2))
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
        try: self.finger_distances_old
        except AttributeError: self.prepare_learning_data()

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
        if self.palm_velocity[0]/1000 < threshold and self.palm_velocity[1]/1000 < threshold and self.palm_velocity[2]/1000 < threshold:
            return True
        return False

    def get_learning_data_static(self, definition=0):
        return self.get_learning_data(definition=definition)

    def get_single_learning_data_dynamic(self, definition=0):
        return self.palm_position()


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

    def import_from_leap(self, bone):
        self.basis = [Vector(*bone.basis.x_basis.to_float_array()),
        Vector(*bone.basis.y_basis.to_float_array()),
        Vector(*bone.basis.z_basis.to_float_array())]
        self.direction = Vector(*bone.direction.to_float_array())
        self.next_joint = Vector(*bone.next_joint.to_float_array())
        self.prev_joint = Vector(*bone.prev_joint.to_float_array())
        self.center = Vector(*bone.center.to_float_array())
        self.is_valid = bone.is_valid
        self.length = bone.length
        self.width = bone.width

    def import_from_ros(self, bn):
        self.basis = [Vector(bn.basis[0:3]),Vector(bn.basis[3:6]),Vector(bn.basis[6:9])]
        self.direction = Vector(bn.direction)
        self.next_joint = Vector(bn.next_joint)
        self.prev_joint = Vector(bn.prev_joint)
        self.center = Vector(bn.center)
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

    def import_from_leap(self, finger):
        self.bones = []
        for i in range(4):
            self.bones.append(Bone(finger.bone(i)))

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

class LeapGesturesSwipe():
    def __init__(self):
        self.present = False
        self.id = 0
        self.in_progress = False
        self.direction = [0.,0.,0.]
        self.speed = 0.
        self.state = 0

class LeapGesturesKeytap():
    def __init__(self):
        self.present = False
        self.id = 0
        self.in_progress = False
        self.direction = [0.,0.,0.]
        self.position = [0.,0.,0.]
        self.state = 0

class LeapGesturesScreentap():
    def __init__(self):
        self.present = False
        self.id = 0
        self.in_progress = False
        self.direction = [0.,0.,0.]
        self.position = [0.,0.,0.]
        self.state = 0
#
