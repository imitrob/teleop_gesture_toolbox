
from copy import deepcopy
from gesture_detector.hand_processing.frame_lib import Vector, Frame, Hand, Finger, Bone, LeapGestures
import time
import numpy as np

class FrameAdder():
    def __init__(self):
        self.seq = 0
        self.last_stamp = 0.

    def add_frame(self, hand_landmarks):
        seq = self.seq
        self.seq+=1
        stamp = time.time()
        fps = 1 / (stamp - self.last_stamp)
        self.last_stamp = stamp
        return CustomFrame(seq, stamp, fps, hand_landmarks)

class V:
    def __init__(self, x, y, z):
        self.x, self.y, self.z = x, y, z

class CustomFrame(Frame):
    def __init__(self, seq, stamp, fps, hand_landmarks):
        if hand_landmarks is not None:
            
            if "x" not in dir(hand_landmarks[0][0]):
                # converts array to object coordinates
                # landmark [0.,0.,0.], x,y,z -> V(0.,0.,0.), V.x, V.y, V.z
                hand_landmarks_labcoord = []
                for hand in hand_landmarks:
                    landmarks_labcoord = []
                    for landmark in hand:
                        landmarks_labcoord.append(V(landmark[0],landmark[1],landmark[2]))
                    hand_landmarks_labcoord.append(landmarks_labcoord)
                hand_landmarks = hand_landmarks_labcoord

            self.import_from_landmarks(seq, stamp, fps, hand_landmarks)
        else:
            # Stamp
            self.seq = 0 # ID of frame
            self.sec = 0 # seconds of frame
            self.nanosec = 0 # nanoseconds
            self.fps = 0. # Frames per second
            self.hands = 0 # Number of hands
            # Hand data
            self.l = CustomHand()
            self.r = CustomHand()
        self.leapgestures = LeapGestures()

    def import_from_landmarks(self, seq, stamp, fps, hand_landmarks):
                
        self.seq = seq
        self.fps = fps
        self.sec = int(stamp)
        self.nanosec = int(1000000000*(stamp%1))
        self.hands = len(hand_landmarks)

        self.l, self.r = CustomHand(), CustomHand()
        for hand in hand_landmarks:
            if is_left_hand(hand):
                self.l = CustomHand(True, hand)
            else:
                self.r = CustomHand(False, hand)

def is_left_hand(landmarks):
    # Thumb tip is landmark 4, Pinky tip is landmark 20

    thumb_x = landmarks[4].x
    pinky_x = landmarks[20].x
    
    # If the thumb_x is less than pinky_x, it is the left hand
    return thumb_x > pinky_x

def vector_from_landmarks(landmark1, landmark2):
    """Compute vector from two Mediapipe landmarks."""
    return np.array([landmark2.x - landmark1.x, landmark2.y - landmark1.y, landmark2.z - landmark1.z])

def palm_normal_vector(landmarks):
    """Compute palm normal vector using the wrist, index finger MCP, and pinky MCP."""
    wrist = landmarks[0]
    index_mcp = landmarks[5]
    pinky_mcp = landmarks[17]

    # Vectors in the palm
    wrist_to_index = vector_from_landmarks(wrist, index_mcp)
    wrist_to_pinky = vector_from_landmarks(wrist, pinky_mcp)

    # Compute cross product to find the palm normal
    normal_vector = np.cross(wrist_to_index, wrist_to_pinky)
    
    # Normalize the vector to unit length
    normal_vector = normal_vector / np.linalg.norm(normal_vector)
    return normal_vector

def hand_direction_vector(landmarks):
    """Compute hand direction vector from wrist to middle finger tip."""
    wrist = landmarks[0]
    middle_finger_tip = landmarks[12]
    
    direction_vector = vector_from_landmarks(wrist, middle_finger_tip)
    
    # Normalize the direction vector
    direction_vector = direction_vector / np.linalg.norm(direction_vector)
    return direction_vector

def palm_position(landmarks):
    """Approximate the palm position by averaging key palm points."""
    wrist = landmarks[0]
    thumb_cmc = landmarks[1]
    pinky_mcp = landmarks[17]
    
    # Compute the average of these key points
    palm_center = (np.array([wrist.x, wrist.y, wrist.z]) +
                   np.array([thumb_cmc.x, thumb_cmc.y, thumb_cmc.z]) +
                   np.array([pinky_mcp.x, pinky_mcp.y, pinky_mcp.z])) / 3.0
    return palm_center

def normalize_vector(vector):
    """Normalize the given vector to unit length."""
    return vector / np.linalg.norm(vector)

def hand_basis_matrix(landmarks):
    """Compute the hand basis (3x3 matrix) using palm normal, hand direction, and a third orthogonal vector."""
    # Palm normal vector
    wrist = landmarks[0]
    index_mcp = landmarks[5]
    pinky_mcp = landmarks[17]
    
    wrist_to_index = vector_from_landmarks(wrist, index_mcp)
    wrist_to_pinky = vector_from_landmarks(wrist, pinky_mcp)
    
    # Palm normal as cross product
    palm_normal = np.cross(wrist_to_index, wrist_to_pinky)
    palm_normal = normalize_vector(palm_normal)

    # Hand direction vector (from wrist to middle finger tip)
    middle_finger_tip = landmarks[12]
    hand_direction = vector_from_landmarks(wrist, middle_finger_tip)
    hand_direction = normalize_vector(hand_direction)
    
    # Third orthogonal vector as the cross product of normal and direction
    third_vector = np.cross(palm_normal, hand_direction)
    third_vector = normalize_vector(third_vector)
    
    # Hand basis matrix (columns are hand direction, third vector, and palm normal)
    basis_matrix = np.array([hand_direction, third_vector, palm_normal]).T
    
    return basis_matrix

def palm_width(landmarks):
    """Compute the palm width as the distance between the index MCP and pinky MCP."""
    index_mcp = landmarks[5]
    pinky_mcp = landmarks[17]
    
    # Compute distance between index MCP and pinky MCP
    width = np.linalg.norm(vector_from_landmarks(index_mcp, pinky_mcp))
    return width

def compute_distance(point1, point2):
    """Compute Euclidean distance between two 3D points."""
    return np.linalg.norm(point1 - point2)

def average_landmarks(landmarks, indices):
    """Compute the average of a list of landmark points given their indices."""
    points = np.array([[landmarks[i].x, landmarks[i].y, landmarks[i].z] for i in indices])
    return np.mean(points, axis=0)

def hand_sphere(landmarks):
    """Compute the sphere that the hand is holding, returning the center and radius."""
    # Use key points: thumb tip, index tip, middle tip, pinky tip, wrist
    key_indices = [4, 8, 12, 16, 20]  # Thumb tip, index tip, middle tip, ring tip, pinky tip
    palm_indices = [0]  # Optionally include the wrist to add stability
    
    # Compute the center of the sphere as the average of these key points
    sphere_center = average_landmarks(landmarks, key_indices + palm_indices)
    
    # Compute distances from the center to each key point (for radius estimation)
    distances = [
        compute_distance(sphere_center, np.array([landmarks[i].x, landmarks[i].y, landmarks[i].z]))
        for i in key_indices
    ]
    
    # Approximate radius as the average distance to the key points
    sphere_radius = np.mean(distances)
    
    return sphere_center, sphere_radius

def compute_distance(point1, point2):
    """Compute Euclidean distance between two 3D points."""
    return np.linalg.norm(point1 - point2)

def average_landmarks(landmarks, indices):
    """Compute the average of a list of landmark points given their indices."""
    points = np.array([[landmarks[i].x, landmarks[i].y, landmarks[i].z] for i in indices])
    return np.mean(points, axis=0)

def grab_strength(landmarks):
    """Compute grab strength (0 to 1) based on finger positions relative to the palm."""
    # Key landmarks: thumb tip (4), index tip (8), middle tip (12), ring tip (16), pinky tip (20), wrist (0)
    finger_tip_indices = [4, 8, 12, 16, 20]
    
    # Compute palm center (could be the wrist or average of several palm points)
    palm_center = np.array([landmarks[0].x, landmarks[0].y, landmarks[0].z])  # Use wrist as palm center
    
    # Compute distances from each fingertip to the palm center
    finger_distances_to_palm = [
        compute_distance(palm_center, np.array([landmarks[i].x, landmarks[i].y, landmarks[i].z]))
        for i in finger_tip_indices
    ]
    
    # Normalize distances (shorter distance means more curled)
    max_possible_distance = max(finger_distances_to_palm)  # Maximum when hand is fully open
    normalized_distances = [1 - (d / max_possible_distance) for d in finger_distances_to_palm]  # Closer is stronger grab
    
    # Combine the normalized distances to estimate the grab strength
    grab_strength_value = np.mean(normalized_distances)  # Average normalized distance as grab strength

    # Optional: Consider additional factors like finger spread
    # Example: Thumb to index distance (spread distance)
    thumb_to_index_distance = compute_distance(
        np.array([landmarks[4].x, landmarks[4].y, landmarks[4].z]),  # Thumb tip
        np.array([landmarks[8].x, landmarks[8].y, landmarks[8].z])   # Index tip
    )
    normalized_thumb_index_distance = 1 - (thumb_to_index_distance / max_possible_distance)  # Smaller spread = stronger grab
    
    # Adjust grab strength based on thumb-to-index distance
    grab_strength_value = (grab_strength_value + normalized_thumb_index_distance) / 2
    
    # Ensure grab strength is between 0 and 1
    grab_strength_value = np.clip(grab_strength_value, 0, 1)
    
    return grab_strength_value

class CustomHand(Hand):
    ''' Advanced variables of hand derived from hand object
    '''
    def __init__(self, is_left=None, hand=None):
        if hand is not None:
            # Import Hand data from Leap Motion hand object
            self.import_from_landmarks(is_left, hand)
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
            self.fingers = [CustomFinger(), # Thumb
                            CustomFinger(), # Index
                            CustomFinger(), # Middle
                            CustomFinger(), # Ring
                            CustomFinger()] # Pinky
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

    def wrist_metacarpal_bone_start(self, wrist, metacarpal_end, 
                                    distance=0.02, # [m] distance from wrist to metacarpal start
        ):
        v = np.array([metacarpal_end.x - wrist.x, metacarpal_end.y - wrist.y, metacarpal_end.z - wrist.z])
        v_cut = distance * 1000 * v / np.linalg.norm(v) 
        return V(x = v_cut[0] + wrist.x, y = v_cut[1] + wrist.y, z = v_cut[2] + wrist.z)

    def import_from_landmarks(self, is_left, hand):
        self.visible = True
        self.id = 0
        self.is_left = is_left
        self.is_right = not is_left
        self.is_valid = True # If hand detected, is valid
        self.grab_strength = grab_strength(hand)
        self.pinch_strength = 0.0
        self.confidence = 1.0 # ?
        v = palm_normal_vector(hand)
        self.palm_normal = Vector(x=v[0], y=v[1], z=v[2])
        v = hand_direction_vector(hand)
        self.direction = Vector(x=v[0], y=v[1], z=v[2])
        v = palm_position(hand)
        self.palm_position = Vector(x=v[0], y=v[1], z=v[2])
            

        self.fingers = [CustomFinger([self.wrist_metacarpal_bone_start(hand[0], hand[1]), *hand[1:5]]),
                        CustomFinger([self.wrist_metacarpal_bone_start(hand[0], hand[5]), *hand[5:9]]),
                        CustomFinger([self.wrist_metacarpal_bone_start(hand[0], hand[9]), *hand[9:13]]),
                        CustomFinger([self.wrist_metacarpal_bone_start(hand[0], hand[13]), *hand[13:17]]),
                        CustomFinger([self.wrist_metacarpal_bone_start(hand[0], hand[17]), *hand[17:21]])]

        v = [0., 0., 0.] # TODO
        self.palm_velocity = Vector(x=v[0], y=v[1], z=v[2])
        basis = hand_basis_matrix(hand)
        self.basis = [Vector(x=basis[0][0], y=basis[0][1], z=basis[0][2]),
                      Vector(x=basis[1][0], y=basis[1][1], z=basis[1][2]),
                      Vector(x=basis[2][0], y=basis[2][1], z=basis[2][2])]
        self.palm_width = palm_width(hand)

        sphere_center, sphere_radius = hand_sphere(hand)
        v = sphere_center
        self.sphere_center = Vector(x=v[0], y=v[1], z=v[2])
        self.sphere_radius = sphere_radius
        v = deepcopy(self.palm_position) # TODO: make stabilized
        self.stabilized_palm_position = Vector(x=v[0], y=v[1], z=v[2])
        self.time_visible = 0.0 # TODO

        v = [hand[0].x, hand[0].y, hand[0].z] # wrist
        self.wrist_position = Vector(x=v[0], y=v[1], z=v[2])
        v = [0.,0.,0.] # TODO: current mediapipe doesn't have elbow
        self.elbow_position = Vector(x=v[0], y=v[1], z=v[2])

        self.arm_valid = False # No data about arm
        self.arm_width = 0.0
        v = [0.,0.,0.]
        self.arm_direction = Vector(x=v[0], y=v[1], z=v[2])
        self.arm_basis = [Vector(x=1.0, y=0.0, z=0.0),
                          Vector(x=0.0, y=1.0, z=0.0),
                          Vector(x=0.0, y=0.0, z=1.0)]

class CustomFinger(Finger):
    def __init__(self, finger=None):
        if finger:
            # Import Finger data from Leap Motion finger object
            self.import_from_landmarks(finger)
        else:
            self.bones = [CustomBone(), # Metacarpal
                          CustomBone(), # Proximal
                          CustomBone(), # Intermediate
                          CustomBone()] # Distal


    def import_from_landmarks(self, finger):
        self.bones = [CustomBone(finger[0:2]),
                      CustomBone(finger[1:3]),
                      CustomBone(finger[2:4]),
                      CustomBone(finger[3:5])]


class CustomBone(Bone):
    def __init__(self, bone=None):
        if bone:
            # Import Finger data from Leap Motion finger object
            self.import_from_landmarks(bone)
        else:
            self.basis = [Vector(),Vector(),Vector()]
            self.direction = Vector()
            self.next_joint = Vector()
            self.prev_joint = Vector()
            self.center = Vector()
            self.is_valid = False
            self.length = 0.
            self.width = 0.
        
    def import_from_landmarks(self, bone):
        landmark1 = bone[0]
        landmark2 = bone[1]
        landmark1 = np.array([landmark1.x,landmark1.y,landmark1.z])
        landmark2 = np.array([landmark2.x,landmark2.y,landmark2.z])
        self.basis = [Vector(x=0.0, y=0.0, z=0.0),
                      Vector(x=0.0, y=0.0, z=0.0),
                      Vector(x=0.0, y=0.0, z=0.0)]
        v = landmark2 - landmark1
        self.direction = Vector(x=v[0], y=v[1], z=v[2])
        v = landmark2
        self.next_joint = Vector(x=v[0], y=v[1], z=v[2])
        v = landmark1
        self.prev_joint = Vector(x=v[0], y=v[1], z=v[2])
        v = (landmark2 - landmark1) / 2 + landmark1
        self.center = Vector(x=v[0], y=v[1], z=v[2])
        self.is_valid = True
        self.length = np.linalg.norm(landmark2 - landmark1)
        self.width = 0.0

