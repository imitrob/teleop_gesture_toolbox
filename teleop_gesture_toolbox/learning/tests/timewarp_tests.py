'''
Some ideas about timewarp improvement

!!!
Needs polishing to new versions
!!!
'''

from copy import deepcopy
from fastdtw import fastdtw
import numpy as np
import sys; sys.path.append("../..")
import os_and_utils.settings as settings; settings.init()
from os_and_utils.visualizer_lib import VisualizerLib, ScenePlot

points = 15
swipe_up = 0.001*np.ones((points,6)); swipe_up[:,2] = np.linspace(-0.2,0.2,points)
swipe_down = 0.001*np.ones((points,6)); swipe_down[:,2] = np.linspace(0.2,-0.2,points)
swipe_left = 0.001*np.ones((points,6)); swipe_left[:,0] = np.linspace(0.2,-0.2,points)
swipe_front_right = 0.001*np.ones((points,6)); swipe_front_right[:,0] = np.linspace(-0.2,0.2,points)

''' Transposed by 1.0 meter '''
swipe_left_1 = 0.001*np.ones((points,6)); swipe_left_1[:,0] = np.linspace(0.5,-0.5,points);
swipe_left_1[:,1] = np.linspace(0.02,-0.02,points)
swipe_left_1[:,2] = np.linspace(-0.02,0.02,points)
swipe_left_1[:,2]=swipe_left_1[:,2]+1.0

''' Long movement '''
swipe_left_2 = 0.1*np.ones((points,6)); swipe_left_2[:,0] = np.sin(np.linspace(1.4,-1.4,points))
swipe_left_2[:,1] = np.cos(np.linspace(1.4,-1.4,points))
swipe_left_2[:,2] = np.linspace(-1.0,1.0,points)
swipe_left_2[:,3] = [0.3,0.25,0.2,0.15,0.1,0.05,0.05,0.05,0.05,0.05,0.05,0.1,0.15,0.2,0.25]
swipe_left_2[:,4] = np.array([0.3,0.25,0.2,0.15,0.1,0.05,0.05,0.05,0.05,0.05,0.05,0.1,0.15,0.2,0.25])/100
swipe_left_2[:,5] = np.array([0.3,0.25,0.2,0.15,0.1,0.05,0.05,0.05,0.05,0.05,0.05,0.1,0.15,0.2,0.25])/100
swipe_left_2[:,5]
''' '''
swipe_left_3 = 0.1*np.ones((points,6)); swipe_left_3[:,0] = np.linspace(0.8,-0.8,points)
swipe_left_3[-1] = [-3, -2, -2, 1,1,1]
swipe_left_3[-2] = [-3, -2, -2, 1,1,1]
swipe_left_3[-3] = [-5, -1, -2, 1,1,1]
swipe_left_3[-4] = [-3, -1, -0.3, 1,1,1]
swipe_left_3[-5] = [-2, -0.0, -0.2, 1,1,1]

swipe_left_3[3] = [2, 0.0, 0.2, 1,1,1]
swipe_left_3[2] = [3, 1, 0.3, 1,1,1]
swipe_left_3[1] = [5, 1, 2, 1,1,1]
swipe_left_3[0] = [5, 1, 2, 1,1,1]


dists = np.array([swipe_up, swipe_down, swipe_left, swipe_front_right,swipe_left_1, swipe_left_2, swipe_left_3])
dists = np.array([swipe_left_2])
dists.shape

ScenePlot.my_plot([], [], legend=['Half circle', 'Swipe left (Transposed)', 'Swipe left (Scaled)', 'Swipe left (Diverse start/end)', 'c','d','e','f','h','i','j','k','l','m','n'], leap=False, series_marking='', promp_paths_variances = dists)

exit()
def normalize_middle_single(path):
    ''' Normalize path based on middle path point
    Parameters:
        path (Float[pts x dim]): 2D array
    '''
    path_new = []
    p_mid = deepcopy(path[len(path)//2])
    for n in range(len(path)):
        path_new.append(np.subtract(path[n], p_mid))
    return np.array(path_new)

def normalize_middle(data):
    ''' Normalize paths based on middle path point
    Parameters:
        data (Float[n_paths x pts x dim]): 3D array
    '''
    data_new = []
    for path in np.array(data):
        data_new.append(normalize_middle_single(path))

    return np.array(data_new)

for dist2 in dists:
    dist, _ = fastdtw(swipe_left, dist2)
    print(dist)

norm_dists = normalize2(dists)

for dist2 in norm_dists:
    dist, _ = fastdtw(swipe_left, dist2)
    print(dist)


''' Idea 1
Scale upper bound of gesture
'''

def scale_limit(data, limit_distance = 0.2):
    ''' Scale independently each dimension when crosses limit_distance
    Parameters:
        data (Float [n_paths x pts x dim]): 3D array
    '''
    data_new = []
    for path in data:
        path_new = []
        path = np.swapaxes(path, 0, 1)
        for dim in range(len(path)):
            _1d = path[dim]
            if (_1d.max() - _1d.min()) > limit_distance:
                path_new.append(_1d/(_1d.max() - _1d.min()) * limit_distance)
            else:
                path_new.append(_1d)
        path_new = np.swapaxes(path_new, 0, 1)
        data_new.append(path_new)
    return np.array(data_new)


norm_scaled_dists = scale_limit(norm_dists)

for dist2 in norm_scaled_dists:
    dist, _ = fastdtw(swipe_left, dist2)
    print(dist)


''' Idea:
segment paths to distinguish based on variance

swipe_up, swipe_down, swipe_left, swipe_front_right,swipe_left_1, swipe_left_2
'''

for dist in norm_scaled_dists:
    dist, connections = fastdtw(swipe_left, dist)
    print(dist, connections)

''' Inputs '''
assert len(norm_scaled_dists[0]) == len(norm_scaled_dists[1]) == len(norm_scaled_dists[2]) == len(norm_scaled_dists[3]) == len(norm_scaled_dists[4]), "Not same lengths"

variances_repre = np.ones(np.array(norm_scaled_dists).shape[0:2])
variances_repre[:,0] = 1000
variances_repre[:,1] = 1000
variances_repre[:,-1] = 1000
variances_repre[:,-2] = 1000

viz = VisualizerLib()
viz.visualize_new_fig('Variance', dim=2, move_figure=False)
viz.visualize_2d(list(zip(list(range(len(variances_repre[0]))), variances_repre[0])), xlabel='Path points [-]', ylabel='Variance [-]')
viz.show()

splits = 4
points = len(swipe_left)
points_per_splits = points//splits

swipe_left_ = norm_scaled_dists[2]
for path, path_variance in zip(norm_scaled_dists, variances_repre):
    path_dists = timewarp_variance_single(swipe_left_, path, path_variance)
    print(np.round(np.sum(path_dists),2))

path_test = swipe_left_
path_repre = norm_scaled_dists[0]
path_variance = variances_repre[0]

def timewarp_variance_single(path_test, path_repre, path_variance):
    path_dists = []
    # points to connect
    connect_points = None
    for i in range(splits):
        ''' Get slice limits '''
        limits = slice(i*points_per_splits,(i+1)*points_per_splits)
        ''' Last iteration, slice remaining points '''
        if i == splits-1: limits = slice(i*points_per_splits, None)

        ''' slice paths & variances '''
        slice_test = path_test[limits]
        slice_repre = path_repre[limits]
        slice_repre_vars = path_variance[limits]

        if connect_points is not None: slice_test = np.insert(slice_test, 0, connect_points, axis=0)

        dist, connections = fastdtw(slice_test, slice_repre)

        connect_points = slice_test[min(connections[-1]):len(slice_test)-1]

        ''' Weighed sum '''
        weighted_dist = dist * 1/np.mean(slice_repre_vars)
        path_dists.append(weighted_dist)

    return path_dists


viz = VisualizerLib()
viz.visualize_new_fig(title="Improvement", dim=2, move_figure=False)
for i in range(len(plot_data)):
    viz.visualize_2d(data=list(zip(list(range(len(plot_data[i]))), plot_data[i])), xlabel='methods', ylabel='DTW distance')
viz.show()
plot_data = np.array([
[3.2, 3.2, 0.0, 3.2, 15.0, 4.114285714285714, 35.542857142857144],
[3.2, 3.2 ,0.0, 3.2, 5.828670879282072e-16, 4.114285714285714, 35.542857142857144],
[2.4000000000000004, 2.4000000000000004, 0.5428571428571428, 2.4000000000000004, 0.5428571428571428, 0.5428571428571428, 1.7445714285714289],
[0.32, 0.32, 0.0, 0.32, 0.0, 0.0, 0.13]
])

np.round(1/plot_data,2)



#
