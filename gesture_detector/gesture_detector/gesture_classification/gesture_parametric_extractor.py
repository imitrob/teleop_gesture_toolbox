
import numpy as np
from spatialmath.base import r2q
from quaternion_algebra.slerp import intrinsic

def get_speed(hand_frames, hand_tag, max_search = 200, velocity_thr = 1):
    velocities = []
    
    i = 1
    while True:
        try:
            v = np.linalg.norm(np.array(getattr(hand_frames[-i], hand_tag).palm_velocity()))
        except IndexError:
            break
        velocities.append(v)
        i+=1
        if v < velocity_thr or i > max_search:
            break
        
    return np.array(velocities).mean()
    
def crossed_distance(hand_frames, hand_tag, max_search = 200, velocity_thr = 1):
    
    i = 1
    position = getattr(hand_frames[-i], hand_tag).palm_position()
    max_dist = 0.0

    while True:
        try:
            v = np.linalg.norm(getattr(hand_frames[-i], hand_tag).palm_velocity())
        except IndexError:
            break
        position_2 = getattr(hand_frames[-i], hand_tag).palm_position()
        dist = np.linalg.norm(np.array(position) - np.array(position_2))
        if dist > max_dist:
            max_dist = dist

        i+=1
        if v < velocity_thr or i > max_search:
            break
    
    return max_dist


def to_q(basis):
    return np.quaternion(*r2q(np.array([b() for b in basis])))


def get_rotation(hand_frames, hand_tag, max_search = 200, velocity_thr = 1):
    i = 1
    q = to_q(getattr(hand_frames[-i], hand_tag).basis)
    max_rot = 0.0

    while True:
        try:
            v = np.linalg.norm(getattr(hand_frames[-i], hand_tag).palm_velocity())
        except IndexError:
            break
        q2 = to_q(getattr(hand_frames[-i], hand_tag).basis)
        rot = intrinsic(q, q2)
        if rot > max_rot:
            max_rot = rot

        i+=1
        if v < velocity_thr or i > max_search:
            break
    
    return max_rot
