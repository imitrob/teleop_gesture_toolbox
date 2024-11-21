

import numpy as np

transform_leap_to_base_corner_config_translation = [1.07, 0.4, 0.01]

def transform_leap_to_base(pose):
    '''
    t (list[3]): Measured
    '''
    t = transform_leap_to_base_corner_config_translation

    assert isinstance(pose, (list,np.ndarray, tuple))
    assert len(pose) == 3

    x,y,z=pose
    return np.array([z/1000+t[0], x/1000+t[1], y/1000+t[2]])

def transform_leap_to_leapworld(pose):
    if len(pose) == 3:
        x,y,z = pose
        return np.array([-x/1000, z/1000, y/1000])
    else:
        x,y,z,qx,qy,qz,qw = pose
        return np.array([-x/1000, z/1000, y/1000, qx,qy,qz,qw])