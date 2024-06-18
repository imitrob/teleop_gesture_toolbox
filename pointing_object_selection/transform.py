

import numpy as np


transformLeapToBase__CornerConfig_translation = [1.07, 0.4, 0.01]
@staticmethod
def transformLeapToBase__CornerConfig(pose):
    '''
    t (list[3]): Measured
    '''
    t = transformLeapToBase__CornerConfig_translation

    assert isinstance(pose, (list,np.ndarray, tuple))
    assert len(pose) == 3

    x,y,z=pose
    return np.array([z/1000+t[0], x/1000+t[1], y/1000+t[2]])
