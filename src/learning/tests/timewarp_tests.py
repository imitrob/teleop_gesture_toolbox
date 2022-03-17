from copy import deepcopy
from fastdtw import fastdtw
import numpy as np

swipe_up = np.array([[0.0,0.0,-0.2],[0.0,0.0,-0.1],[0.0,0.0,0.0],[0.0,0.0,0.1],[0.0,0.0,0.2]])
swipe_down = np.array([[0.0,0.0,0.2],[0.0,0.0,0.1],[0.0,0.0,0.0],[0.0,0.0,-0.1],[0.0,0.0,-0.2]])
swipe_left = np.array([[0.2,0.0,0.0],[0.1,0.0,0.0],[0.0,0.0,0.0],[-0.1,0.0,0.0],[-0.2,0.0,0.0]])
swipe_front_right = np.array([[-0.2,0.0,0.0],[-0.1,0.0,0.0],[0.0,0.0,0.0],[0.1,0.0,0.0],[0.2,0.0,0.0]])

swipe_left_1 = np.array([[1.2,0.0,0.0],[1.1,0.0,0.0],[1.0,0.0,0.0],[0.9,0.0,0.0],[0.8,0.0,0.0]])
swipe_left_2 = np.array([[0.8,0.0,0.0],[0.4,0.0,0.0],[0.0,0.0,0.0],[-0.4,0.0,0.0],[-0.8,0.0,0.0]])

def normalize(data_composition):
    data_composition_ = []
    data_composition0 = deepcopy(data_composition[len(data_composition)//2])
    for n in range(len(data_composition)):
        data_composition_.append(np.subtract(data_composition[n], data_composition0))
    return np.array(data_composition_)

def normalize2(data_composition):
    Xpalm = np.array(data_composition)
    Xpalm_ = []
    for p in Xpalm:
        p_ = []
        p0 = deepcopy(p[len(p)//2])
        for n in range(0, len(p)):
            p_.append(np.subtract(p[n], p0))
        Xpalm_.append(p_)

    return np.array(Xpalm_)


dists = np.array([swipe_up, swipe_down, swipe_left, swipe_front_right,swipe_left_1, swipe_left_2])
norm_dists = normalize2(dists)

for dist2 in norm_dists:
    dist, _ = fastdtw(swipe_left, dist2)
    print(dist)

dist, _ = fastdtw(swipe_left, swipe_left_1)

''' Idea 1
Scale upper bound of gesture

'''

def scale_limit(data, limit_distance = 0.2):
    data_ = []
    for path in data:
        path_ = []
        path = np.swapaxes(path, 0, 1)
        for dim in range(3):
            _1d = path[dim]
            if (_1d.max() - _1d.min()) > limit_distance:
                path_.append(_1d/(_1d.max() - _1d.min())*limit_distance)
            else:
                path_.append(_1d)
        path_ = np.swapaxes(path_, 0, 1)
        data_.append(path_)
    return data_


scale_limit(dists)













#
