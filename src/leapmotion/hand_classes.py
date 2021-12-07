'''

'''
import numpy as np
# I can make independent to tf library if quaternion_from_euler function imported
try:
    import tf
    TF_IMPORT = True
except ImportError:
    TF_IMPORT = False

class Frame():
    ''' Advanced variables derived from frame object
    '''
    def __init__(self, frame=None):

        if frame: self.import_from_leap(frame); return
        # Stamp
        self.seq = 0 # ID of frame
        self.secs = 0 # seconds of frame
        self.nsecs = 0 # nanoseconds
        self.fps = 0 # Frames per second
        self.hands = 0 # Number of hands
        # Hand data
        self.l = Hand()
        self.r = Hand()

        # Leap gestures
        self.leapgestures = LeapGestures()

    def import_from_leap(self, frame):
        self.seq = frame.id
        self.fps = frame.current_frames_per_second
        self.secs = frame.timestamp//1000000
        self.nsecs = 1000*(frame.timestamp%1000000)
        self.hands = len(frame.hands)
        for hand in frame.hands:
            if hand.is_left:
                self.l = Hand(hand)
            elif hand.is_right:
                self.r = Hand(hand)
            Hand()

class Hand():
    ''' Advanced variables of hand derived from hand object
    '''
    def __init__(self, hand=None):
        # Data processed for learning
        self.wrist_angles = []
        self.bone_angles = []
        self.finger_distances = []

        # Import Hand data from Leap Motion hand object
        if hand: self.import_from_leap(hand); return

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

    def import_from_leap(hand):
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


    def prepare_open_fingers(self):
        ''' Stand of each finger, for deterministic gesture detection
        Returns:
            open/close fingers (Float[5]): For every finger Float value (0.,1.), 0. - finger closed, 1. - finger opened
        '''
        if not TF_IMPORT: raise Exception("tf library not imported!")

        oc = []
        for i in range(0,5):
            bone_1 = self.fingers[i].bone[0]
            if i == 0: bone_1 = self.fingers[i].bone[1]
            bone_4 = self.fingers[i].bone[3]
            q1 = tf.transformations.quaternion_from_euler(0.0, asin(-bone_1.direction[1]), atan2(bone_1.direction[0], bone_1.direction[2])) # roll, pitch, yaw
            q2 = tf.transformations.quaternion_from_euler(0.0, asin(-bone_4.direction[1]), atan2(bone_4.direction[0], bone_4.direction[2])) # roll, pitch, yaw
            oc.append(np.dot(q1, q2))
        return oc


    def prepare_finger_distance_combinations(self):
        ''' Distance of finger tips combinations, for deterministic gesture detection
            - Compute Euclidean norm
            - Divide by 1000 is conversion (mm -> m)
        Returns:
            Finger distance combs. (Dict with Floats): Distance values in meters
        '''
        # shape = 5 (fingers) x 3 (Cartesian position)
        position_tip_of_fingers = []
        for i in range(0,5):
            bone_1 = self.fingers[i].bone[0]
            if i == 0: bone_1 = self.fingers[i].bone[1]
            bone_4 = self.fingers[i].bone[3]
            position_tip_of_fingers.append([bone_4.next_joint[0], bone_4.next_joint[1], bone_4.next_joint[2]])

        tch = {}
        tch['12'] = np.sum(np.power(np.subtract(position_of_fingers[1], position_of_fingers[0]),2))/1000
        tch['23'] = np.sum(np.power(np.subtract(position_of_fingers[2], position_of_fingers[1]),2))/1000
        tch['34'] = np.sum(np.power(np.subtract(position_of_fingers[3], position_of_fingers[2]),2))/1000
        tch['45'] = np.sum(np.power(np.subtract(position_of_fingers[4], position_of_fingers[3]),2))/1000
        tch['13'] = np.sum(np.power(np.subtract(position_of_fingers[2], position_of_fingers[0]),2))/1000
        tch['14'] = np.sum(np.power(np.subtract(position_of_fingers[3], position_of_fingers[0]),2))/1000
        tch['15'] = np.sum(np.power(np.subtract(position_of_fingers[4], position_of_fingers[0]),2))/1000
        return tch

    def prepare_learning_data(self):
        ''' Data will be saved inside object
        '''
        # bone directions and angles
        hand_direction = np.array(self.direction)
        hand_angles = np.array([0., asin(-self.direction[1]), atan2(self.direction[0], self.direction[2])])
        v = np.array(self.palm_position.to_float_array()) - np.array(self.wrist_position.to_float_array())

        wrist_direction = v / np.sqrt(np.sum(v**2))
        wrist_angles = np.array([0., asin(-wrist_direction[1]), atan2(wrist_direction[0], wrist_direction[2])])
        bone_direction, bone_angles_pre = np.zeros([5,4,3]), np.zeros([5,4,3])
        for i in range(0,5):
            for j in range(0,4):
                bone_direction[i][j] = np.array(self.fingers[i].bone[j].direction.to_float_array())
                bone_angles_pre[i][j] = np.array((0., np.arcsin(-self.fingers[i].bone[j].direction[1]), np.arctan2(self.fingers[i].bone[j].direction[0], self.fingers[i].bone[j].direction[2])))

        # bone angles differences
        self.wrist_angles = wrist_angles - hand_angles
        self.bone_angles = np.zeros([5,4,3])
        for i in range(0,5):
            for j in range(0,4):
                if j == 0:
                    d1 = hand_angles
                else:
                    d1 = bone_angles_pre[i][j-1]
                d2 = bone_angles_pre[i][j]
                self.bone_angles[i][j] = d1 - d2
        # distance between finger positions
        palm_position = np.array(self.palm_position.to_float_array())

        self.finger_distances = []
        combs = position_of_fingers; combs.extend([palm_position])
        for comb in combinations(combs,2):
            self.finger_distances.append(np.array(comb[0]) - np.array(comb[1]))

    def is_stop(self, threshold=0.02):
        '''
        Returns:
            stop (Bool)
        '''
        if self.palm_velocity[0]/1000 < threshold and self.palm_velocity[1]/1000 < threshold and self.palm_velocity[2]/1000 < threshold:
            return True
        return False


class Vector():
    def __init__(self, x=0., y=0., z=0.):
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
    def __getattr__(self, attribute):
        if attribute == 'pitch':
            return np.arcsin(-self.y)
        elif attribute == 'yaw':
            return np.arctan2(self.x, self.z)


class Bone():
    def __init__(self, bone=None):
        # Import Finger data from Leap Motion finger object
        if bone: self.import_from_leap(bone); return

        self.basis = [Vector(),Vector(),Vector()]
        self.direction = Vector()
        self.next_joint = Vector()
        self.prev_joint = Vector()
        self.center = Vector()
        self.is_valid = False
        self.length = 0.
        self.width = 0.

    def import_from_leap(bone):
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

class Finger():
    def __init__(self, finger=None):
        # Import Finger data from Leap Motion finger object
        if finger: self.import_from_leap(finger); return

        self.bones = [Bone(), # Metacarpal
                     Bone(), # Proximal
                     Bone(), # Intermediate
                     Bone()] # Distal

    def import_from_leap(finger):
        self.bone = []
        for i in range(4):
            self.bone.append(Bone(finger.bone(i)))

''' Leap Motion gestures definitions
'''
class LeapGestures():
    def __init__(self):
        self.circle = LeapGesturesCircle()
        self.swipe = LeapGesturesSwipe()
        self.keytap = LeapGesturesKeytap()
        self.screentap = LeapGesturesScreentap()

class LeapGesturesCircle():
    def __init__(self):
        self.toggle = False

        self.in_progress = False
        self.clockwise = False
        self.progress = 0.
        self.angle = 0.
        self.radius = 0.
        self.state = 0

class LeapGesturesSwipe():
    def __init__(self):
        self.toggle = False

        self.in_progress = False
        self.direction = [0.,0.,0.]
        self.speed = 0.
        self.state = 0

class LeapGesturesKeytap():
    def __init__(self):
        self.toggle = False

        self.in_progress = False
        self.direction = [0.,0.,0.]
        self.state = 0

class LeapGesturesScreentap():
    def __init__(self):
        self.toggle = False

        self.in_progress = False
        self.direction = [0.,0.,0.]
        self.state = 0
#
