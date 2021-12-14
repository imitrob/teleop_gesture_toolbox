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
    def __init__(self, frame=None, leapgestures=None):
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

        if frame: self.import_from_leap(frame, leapgestures)

    def import_from_leap(self, frame, leapgestures):
        self.seq = frame.id
        self.fps = frame.current_frames_per_second
        self.secs = frame.timestamp//1000000
        self.nsecs = 1000*(frame.timestamp%1000000)
        self.hands = len(frame.hands)

        self.l, self.r = None, None
        for hand in frame.hands:
            if hand.is_left:
                self.l = Hand(hand)
            elif hand.is_right:
                self.r = Hand(hand)
            Hand()

        if leapgestures: self.leapgestures = leapgestures

    def import_from_ros(self, msg):
        self.seq = msg.header.seq
        self.fps = msg.fps
        self.secs = msg.header.secs
        self.nsecs = msg.header.nsecs
        self.hands = msg.hands

        self.l.import_from_ros(msg.l)
        self.r.import_from_ros(msg.r)

        self.leapgestures.import_from_ros(msg.leapgestures)

    def __str__(self):
        ''' Invoke: print(Frame())
        '''
        str = f""
        str += f"fps {self.fps} "
        str += f"n hands {self.hands} "
        str += f"secs {self.secs} "
        str += f"nsecs {self.nsecs} "
        str += f"seq {self.seq} "

        for hand in [self.l, self.r]:
            str += f"id {hand.id} "
            str += f"is_left {hand.is_left} "
            str += f"is_right {hand.is_right} "
            str += f"is_valid {hand.is_valid} "
            str += f"grab_strength {hand.grab_strength} "
            str += f"pinch_strength {hand.pinch_strength} "
            str += f"confidence {hand.confidence} "
            str += f"palm_normal {hand.palm_normal()} "
            str += f"direction {hand.direction()} "
            str += f"palm_position {hand.palm_position()} "

            for finger in hand.fingers:
                for bone in finger.bones:
                    basis = [bone.basis[0](), bone.basis[1](), bone.basis[2]()]
                    str += f"basis {[item for sublist in basis for item in sublist]} "
                    str += f"direction {bone.direction()} "
                    str += f"next_joint {bone.next_joint()} "
                    str += f"prev_joint {bone.prev_joint()} "
                    str += f"center {bone.center()} "
                    str += f"is_valid {bone.is_valid} "
                    str += f"length {bone.length} "
                    str += f"width {bone.width} "

            str += f"palm_velocity {hand.palm_velocity()} "
            basis = [hand.basis[0](), hand.basis[1](), hand.basis[2]()]
            str += f"basis {[item for sublist in basis for item in sublist]} "
            str += f"palm_width {hand.palm_width} "
            str += f"sphere_center {hand.sphere_center()} "
            str += f"sphere_radius {hand.sphere_radius} "
            str += f"stabilized_palm_position {hand.stabilized_palm_position()} "
            str += f"time_visible {hand.time_visible} "
            str += f"wrist_position {hand.wrist_position()} "

        str += f"id {self.leapgestures.circle.id} "
        str += f"in_progress {self.leapgestures.circle.in_progress} "
        str += f"clockwise {self.leapgestures.circle.clockwise} "
        str += f"progress {self.leapgestures.circle.progress} "
        str += f"angle {self.leapgestures.circle.angle} "
        str += f"radius {self.leapgestures.circle.radius} "
        str += f"state {self.leapgestures.circle.state} "

        str += f"id {self.leapgestures.swipe.id} "
        str += f"in_progress {self.leapgestures.swipe.in_progress} "
        str += f"direction {self.leapgestures.swipe.direction} "
        str += f"speed {self.leapgestures.swipe.speed} "
        str += f"state {self.leapgestures.swipe.state} "

        str += f"id {self.leapgestures.keytap.id} "
        str += f"in_progress {self.leapgestures.keytap.in_progress} "
        str += f"direction {self.leapgestures.keytap.direction} "
        str += f"position {self.leapgestures.keytap.position} "
        str += f"state {self.leapgestures.keytap.state} "

        str += f"id {self.leapgestures.screentap.id} "
        str += f"in_progress {self.leapgestures.screentap.in_progress} "
        str += f"direction {self.leapgestures.screentap.direction} "
        str += f"position {self.leapgestures.screentap.position} "
        str += f"state {self.leapgestures.screentap.state} "

        return str

    def to_ros(self):
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

        frame.leapgestures.circle_id = self.leapgestures.circle.id
        frame.leapgestures.circle_in_progress = self.leapgestures.circle.in_progress
        frame.leapgestures.circle_clockwise = self.leapgestures.circle.clockwise
        frame.leapgestures.circle_progress = self.leapgestures.circle.progress
        frame.leapgestures.circle_angle = self.leapgestures.circle.angle
        frame.leapgestures.circle_radius = self.leapgestures.circle.radius
        frame.leapgestures.circle_state = self.leapgestures.circle.state

        frame.leapgestures.swipe_id = self.leapgestures.swipe.id
        frame.leapgestures.swipe_in_progress = self.leapgestures.swipe.in_progress
        frame.leapgestures.swipe_direction = self.leapgestures.swipe.direction
        frame.leapgestures.swipe_speed = self.leapgestures.swipe.speed
        frame.leapgestures.swipe_state = self.leapgestures.swipe.state

        frame.leapgestures.keytap_id = self.leapgestures.keytap.id
        frame.leapgestures.keytap_in_progress = self.leapgestures.keytap.in_progress
        frame.leapgestures.keytap_direction = self.leapgestures.keytap.direction
        frame.leapgestures.keytap_position = self.leapgestures.keytap.position
        frame.leapgestures.keytap_state = self.leapgestures.keytap.state

        frame.leapgestures.screentap_id = self.leapgestures.screentap.id
        frame.leapgestures.screentap_in_progress = self.leapgestures.screentap.in_progress
        frame.leapgestures.screentap_direction = self.leapgestures.screentap.direction
        frame.leapgestures.screentap_position = self.leapgestures.screentap.position
        frame.leapgestures.screentap_state = self.leapgestures.screentap.state

        for (hand, h) in [(frame.l, self.l), (frame.r, self.r)]:
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

        ## Deprecated data: For compatibility
        ## Filled with function
        self.conf = 0.0
        self.OC = [0.0] * 5
        self.TCH12, self.TCH23, self.TCH34, self.TCH45 = [0.0] * 4
        self.TCH13, self.TCH14, self.TCH15 = [0.0] * 3
        self.vel = [0.0] * 3
        self.pPose = None
        self.pRaw = [0.0] * 6 # palm pose: x, y, z, roll, pitch, yaw
        self.pNormDir = [0.0] * 6 # palm normal vector and direction vector
        self.time_last_stop = 0.0
        self.grab = 0.0
        self.pinch = 0.0
        self.wrist_hand_angles_diff = []
        self.fingers_angles_diff = []
        self.pos_diff_comb = []
        self.index_position = []

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

    def prepare_deprecated_data(self):
        self.conf = self.confidence
        self.prepare_open_fingers()
        self.prepare_finger_distance_combinations()

        self.vel = self.palm_velocity()

        self.pRaw = self.palm_position() # palm pose: x, y, z, roll, pitch, yaw
        self.pRaw.extend([self.palm_normal.roll, self.direction.pitch, self.direction.yaw])
        self.pNormDir = self.palm_normal() # palm normal vector and direction vector
        self.pNormDir.extend(self.direction())

        self.grab = self.grab_strength
        self.pinch = self.pinch_strength

        # PoseStamped
        try:
            from geometry_msgs.msg import Pose, PoseStamped, Point, Quaternion, Vector3
            import tf
            self.pPose = PoseStamped()
            self.pPose.pose.position = Point(*self.pRaw[0:3])
            self.pPose.pose.orientation = Quaternion(*tf.transformations.quaternion_from_euler(*pRaw[3:6]))
        except:
            print("Trying to prepare_deprecated_data but ROS of tf is not found!")

        self.time_last_stop = 0.0

        if not self.wrist_angles: self.prepare_learning_data()
        self.wrist_hand_angles_diff = self.wrist_angles
        self.fingers_angles_diff = self.bone_angles
        self.pos_diff_comb = finger_distances
        self.index_position = self.fingers[1].bones[3].next_joint()


    def get_open_fingers(self):
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

    def prepare_open_fingers(self):
        self.OC = self.get_open_fingers()

    def get_finger_distance_combinations(self):
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

    def prepare_finger_distance_combinations(self):
        tch = self.get_finger_distance_combinations()
        self.TCH12 = tch['12']
        self.TCH23 = tch['23']
        self.TCH34 = tch['34']
        self.TCH45 = tch['45']
        self.TCH13 = tch['13']
        self.TCH14 = tch['14']
        self.TCH15 = tch['15']

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

    def get_learning_data(self, type='all_defined'):
        ''' Return Vector of Observation Parameters
        '''
        if not self.wrist_angles: self.prepare_learning_data()

        learning_data = self.wrist_angles
        learning_data.extend(self.bone_angles)
        learning_data.extend(self.finger_distances)
        return learning_data

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
        # Import Finger data from Leap Motion finger object
        if finger: self.import_from_leap(finger); return

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
        self.circle.id = lg.circle_id
        self.circle.in_progress = lg.circle_in_progress
        self.circle.clockwise = lg.circle_clockwise
        self.circle.progress = lg.circle_progress
        self.circle.angle = lg.circle_angle
        self.circle.radius = lg.circle_radius
        self.circle.state = lg.circle_state

        self.swipe.id = lg.swipe_id
        self.swipe.in_progress = lg.swipe_in_progress
        self.swipe.direction = lg.swipe_direction
        self.swipe.speed = lg.swipe_speed
        self.swipe.state = lg.swipe_state

        self.keytap.id = lg.keytap_id
        self.keytap.in_progress = lg.keytap_in_progress
        self.keytap.direction = lg.keytap_direction
        self.keytap.position = lg.keytap_position
        self.keytap.state = lg.keytap_state

        self.screentap.id = lg.screentap_id
        self.screentap.in_progress = lg.screentap_in_progress
        self.screentap.direction = lg.screentap_direction
        self.screentap.position = lg.screentap_state
        self.screentap.state = lg.screentap_position

class LeapGesturesCircle():
    def __init__(self):
        self.id = 0
        self.in_progress = False
        self.clockwise = False
        self.progress = 0.
        self.angle = 0.
        self.radius = 0.
        self.state = 0

class LeapGesturesSwipe():
    def __init__(self):
        self.id = 0
        self.in_progress = False
        self.direction = [0.,0.,0.]
        self.speed = 0.
        self.state = 0

class LeapGesturesKeytap():
    def __init__(self):
        self.id = 0
        self.in_progress = False
        self.direction = [0.,0.,0.]
        self.position = [0.,0.,0.]
        self.state = 0

class LeapGesturesScreentap():
    def __init__(self):
        self.id = 0
        self.in_progress = False
        self.direction = [0.,0.,0.]
        self.position = [0.,0.,0.]
        self.state = 0
#
