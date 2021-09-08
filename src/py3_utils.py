#!/usr/bin/env python3.7
''' Utilities functions
'''
import numpy as np
def quaternion_multiply(quaternion1, quaternion0):
    ''' Better to list that here, than from importing it externally via pkg tf
    Parameters:
        quaternion1, quaternion0 (Float[4]), Format (x,y,z,w)
    Outputs:
        quaternion output (Float[4]), Format (x,y,z,w)
    '''
    x0, y0, z0, w0 = quaternion0
    x1, y1, z1, w1 = quaternion1
    return np.array([x1 * w0 + y1 * z0 - z1 * y0 + w1 * x0,
                     -x1 * z0 + y1 * w0 + z1 * x0 + w1 * y0,
                     x1 * y0 - y1 * x0 + z1 * w0 + w1 * z0,
                     -x1 * x0 - y1 * y0 - z1 * z0 + w1 * w0], dtype=np.float64)
