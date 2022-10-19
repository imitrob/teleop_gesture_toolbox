from os_and_utils.transformations import Transformations as tfm
import numpy as np

def is_hand_inside_ball(frame_hand):
    ''' Hand palm is in ball with diameter=0.1m, ball is in center of base_position
    '''
    # HARDCODED: leap
    base_position = [0.,0.,0.2]

    palm_position_scene = tfm.transformLeapToBase(frame_hand.palm_position(), out='position')

    distance = np.linalg.norm(np.array(base_position) - np.array(palm_position_scene))

    if distance < 0.1:
        return True
    else:
        return False
