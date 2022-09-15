#!/usr/bin/env python3.8
'''
1. Reads data from ROS msgs
2. Loads data from saved files
3. Can compare data via plot
'''
from importlib import reload
import numpy as np
import sys; sys.path.append('../..')
import settings; settings.init()
import gestures_lib as gl; gl.init()

### REads data from input observaitons for detections
import rospy

global received_gestures_data
received_gestures_data = []

def handle_observations_msg(msg):
    global received_gestures_data
    received_gestures_data.append((np.array(msg.observations.data).reshape(5,3),{}))

from mirracle_gestures.msg import DetectionObservations
rospy.init_node("just_plotter", anonymous=True)
rospy.Subscriber('/mirracle_gestures/dynamic_detection_observations', DetectionObservations, handle_observations_msg)


### reads the load_dynamic data or something
#import os_and_utils.loading; reload(os_and_utils.loading)
from os_and_utils.loading import DatasetLoader

X, Y = DatasetLoader({'n':5, 'scene_frame':1, 'normalize':1}).load_dynamic(settings.paths.learn_path, gl.gd.l.dynamic.info.names, new=True)
X, Y = DatasetLoader({'n':5, 'scene_frame':1, 'normalize':1}).load_dynamic(settings.paths.learn_path, gl.gd.l.dynamic.info.names)

### plot both
from os_and_utils.visualizer_lib import VisualizerLib, ScenePlot
import learning.timewarp_lib
fdtw = learning.timewarp_lib.fastdtw_()

import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import time as t
import argparse
from os_and_utils import scenes as sl; sl.init()

fdtw.X_ProMP = np.array([[[ 9.34916240e-03, -3.22637994e-04,  5.70740635e-02],
        [ 3.20322835e-03, -5.31848569e-04,  1.16234503e-02],
        [-4.18444467e-05, -1.86018529e-05, -2.03782993e-04],
        [-1.84678973e-03,  4.73201842e-04, -5.73070619e-03],
        [-3.11730254e-03,  1.55623579e-03, -1.00000000e-02]],
        [[-5.39258518e-02, -1.39937855e-02, -1.31020465e-02],
        [-2.17303043e-02, -7.32983055e-03, -2.70709678e-03],
        [-2.56488927e-05, -7.86159794e-05,  1.78127883e-04],
        [ 8.29435393e-03,  4.52524623e-03,  1.90829016e-03],
        [ 1.43810477e-02,  1.06273822e-02,  3.62373658e-03]],
        [[ 2.81896267e-02,  7.54311043e-03,  8.81534705e-03],
        [ 1.01424542e-02,  1.77965117e-03,  5.33457027e-03],
        [-1.13541018e-04, -1.15389080e-04,  1.18152111e-04],
        [-6.42634857e-03, -1.40794439e-03, -1.28982971e-03],
        [-1.13009125e-02, -2.32172429e-03, -2.57586930e-03]],
        [[-1.41718675e-02,  4.01831684e-04, -7.76268953e-02],
        [-6.47842571e-03,  1.59140210e-03, -3.77313649e-02],
        [ 6.90216858e-05, -4.80521879e-05, -7.26920098e-04],
        [ 3.23778311e-03, -3.04369494e-04,  6.87671961e-03],
        [ 4.83667339e-03, -1.21014664e-03,  1.76684737e-02]],
        [[ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00],
        [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00],
        [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00],
        [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00],
        [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00]]])

Paths_Waypoints = [(fdtw.X_ProMP[i], {}) for i in range(5)]
ScenePlot.my_plot([], Paths_Waypoints, boundbox=False, leap=False)
my_plot([], Paths_Waypoints, boundbox=False, leap=False, filename='modelling_swipes', size=(8,8), legend=gl.gd.Gs_dynamic)




received_gestures_data[-1:]
id = 0
gl.gd.dynamic_info().names[id]
if True:
    fdtw.X_ProMP2 = np.array([[[ 9.34916240e-03, -3.22637994e-04,  5.70740635e-02],
        [ 3.20322835e-03, -5.31848569e-04,  1.16234503e-02],
        [-4.18444467e-05, -1.86018529e-05, -2.03782993e-04],
        [-1.84678973e-03,  4.73201842e-04, -5.73070619e-03],
        [-3.11730254e-03,  1.55623579e-03, -5.00000000e-02]],
        [[-5.39258518e-02, -1.39937855e-02, -1.31020465e-02],
        [-2.17303043e-02, -7.32983055e-03, -2.70709678e-03],
        [-2.56488927e-05, -7.86159794e-05,  1.78127883e-04],
        [ 8.29435393e-03,  4.52524623e-03,  1.90829016e-03],
        [ 1.43810477e-02,  1.06273822e-02,  3.62373658e-03]],
        [[ 2.81896267e-02,  7.54311043e-03,  8.81534705e-03],
        [ 1.01424542e-02,  1.77965117e-03,  5.33457027e-03],
        [-1.13541018e-04, -1.15389080e-04,  1.18152111e-04],
        [-6.42634857e-03, -1.40794439e-03, -1.28982971e-03],
        [-1.13009125e-02, -2.32172429e-03, -2.57586930e-03]],
        [[-1.41718675e-02,  4.01831684e-04, -7.76268953e-02],
        [-6.47842571e-03,  1.59140210e-03, -3.77313649e-02],
        [ 6.90216858e-05, -4.80521879e-05, -7.26920098e-04],
        [ 3.23778311e-03, -3.04369494e-04,  6.87671961e-03],
        [ 4.83667339e-03, -1.21014664e-03,  5.76684737e-02]],
        [[ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00],
        [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00],
        [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00],
        [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00],
        [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00]]])
if True:
    fdtw.X_ProMP3 = np.array([[[ 0.0, 0.0,  5.70740635e-02],
        [ 0.0,  0.0,  1.16234503e-02],
        [ 0.0,  0.0, -2.03782993e-04],
        [ 0.0,  0.0, -5.73070619e-03],
        [ 0.0,  0.0, -5.00000000e-02]],
        [[-5.39258518e-02, -0.0, -0.0],
        [-2.17303043e-02, -0.0, -0.0],
        [-2.56488927e-05, -0.0,  0.0],
        [ 8.29435393e-03,  0.0,  0.0],
        [ 1.43810477e-02,  0.0,  0.0]],
        [[ 2.81896267e-02,  0.0,  0.0],
        [ 1.01424542e-02,  0.0,  0.0],
        [-1.13541018e-04, -0.0,  0.0],
        [-6.42634857e-03, -0.0, -0.0],
        [-1.13009125e-02, -0.0, -0.0]],
        [[-0.0,  0.0, -7.76268953e-02],
        [-0.0,  0.0, -3.77313649e-02],
        [ 0.0, -0.0, -7.26920098e-04],
        [ 0.0, -0.0,  6.87671961e-03],
        [ 0.0, -0.0,  5.76684737e-02]],
        [[ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00],
        [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00],
        [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00],
        [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00],
        [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00]]])

ScenePlot.my_plot([], [(fdtw.X_ProMP2[0], {}), (fdtw.X_ProMP2[1], {}), (fdtw.X_ProMP2[2], {}), (fdtw.X_ProMP2[3], {}), (fdtw.X_ProMP2[4], {})], boundbox=False, leap=False)
ScenePlot.my_plot([], [(fdtw.X_ProMP3[0], {}), (fdtw.X_ProMP3[1], {}), (fdtw.X_ProMP3[2], {}), (fdtw.X_ProMP3[3], {}), (fdtw.X_ProMP3[4], {})], boundbox=False, leap=False)
id = 3
gl.gd.dynamic_info().names[id]
ScenePlot.my_plot(X[Y==id][0:3], [(fdtw.X_ProMP[id], {})], boundbox=True, leap=False)

ScenePlot.my_plot([], received_gestures_data[-3:-2], boundbox=True, leap=False)




id = 0
gl.gd.dynamic_info().names[id]
ScenePlot.my_plot([fdtw.X_ProMP[id]], received_gestures_data[-3:-2], boundbox=True, leap=False)
id = 3
gl.gd.dynamic_info().names[id]
ScenePlot.my_plot([fdtw.X_ProMP[id]], received_gestures_data[-3:-2], boundbox=True, leap=False)

fdtw.sample(x=x[0][0].flatten(), y=y, print_out=True, format='inverse_array')


fdtw.sample(x=fdtw.X_ProMP[3].flatten(), y=y, print_out=True, format='inverse_array')

counts = fdtw.counts

from scipy.spatial.distance import euclidean
from fastdtw import fastdtw
dist = np.zeros([len(counts),])
for i in range(0,len(counts)):
    dist[i], _ = fastdtw(x[0][0].flatten(), fdtw.X_ProMP[i], radius=10, dist=euclidean)


dist = np.zeros([len(counts),])
for i in range(0,len(counts)):
    dist[i], _ = fastdtw(fdtw.X_ProMP[3], fdtw.X_ProMP[i], radius=1, dist=euclidean)
dist

fdtw.X_ProMP[0]
fdtw.X_ProMP[3]
received_gestures_data[-3:-2]

x[0][0][:,1] = np.array([0.,0.,0.,0.,0.])
x[0][0][:,0]
x[0][0]

X_test = np.array([[[ 0.0, 0.0,  0.05], # 'swipe_down'
    [ 0.0,  0.0,  0.025],
    [ 0.0,  0.0, 0.0],
    [ 0.0,  0.0, -0.025],
    [ 0.0,  0.0, -0.05]],
    [[-0.05, -0.0, -0.0], # 'swipe_front_right'
    [-0.025, -0.0, -0.0],
    [0.0, -0.0,  0.0],
    [0.025,  0.0,  0.0],
    [0.05,  0.0,  0.0]],
    [[ 0.05,  0.0,  0.0], # 'swipe_left'
    [ 0.025,  0.0,  0.0],
    [0.0, -0.0,  0.0],
    [-0.025, -0.0, -0.0],
    [-0.05, -0.0, -0.0]],
    [[-0.0,  0.0, -0.05], # 'swipe_up'
    [-0.0,  0.0, -0.025],
    [ 0.0, -0.0, 0.0],
    [ 0.0, -0.0,  0.025],
    [ 0.0, -0.0,  0.05]],
    [[ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00], # 'nothing_dyn'
    [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00],
    [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00],
    [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00],
    [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00]]])



tt =[array([ 0.05807601, -0.01161252,  0.01275706]), array([ 0.067695  , -0.01230547,  0.00964423]), array([0., 0., 0.]), array([-0.18943798, -0.0330545 , -0.0083631 ]), array([-0.22814462, -0.07072939, -0.03345309])]




tt = np.array(tt)
tt
dist = np.zeros([len(counts),])
for i in range(0,len(counts)):
    dist[i], _ = fastdtw(tt, X_test[i], radius=1, dist=euclidean)
dist
gl.gd.Gs_dynamic[np.argmin(dist)]


X_test
fdtw.X_ProMP


 X_test



'''OPOSITE SIDE
'''
X_opositeside = [[[ 0. ,    0.    , 0.05 ],
                  [ 0. ,    0.    , 0.025],
                  [ 0. ,    0.    , 0.   ],
                  [ 0. ,    0.   , -0.025],
                  [ 0. ,    0.   , -0.05 ]],

                 [[-0.05  ,-0.   , -0.   ],
                  [-0.025 ,-0.   , -0.   ],
                  [ 0.    ,-0.   ,  0.   ],
                  [ 0.025 , 0.   ,  0.   ],
                  [ 0.05  , 0.   ,  0.   ]],

                 [[ 0.05  , 0.    , 0.   ],
                  [ 0.025 , 0.   ,  0.   ],
                  [ 0.    ,-0.   ,  0.   ],
                  [-0.025 ,-0.   , -0.   ],
                  [-0.05  ,-0.   , -0.   ]],

                 [[-0.    , 0.   , -0.05 ],
                  [-0.    , 0.   , -0.025],
                  [ 0.    ,-0.   ,  0.   ],
                  [ 0.    ,-0.   ,  0.025],
                  [ 0.    ,-0.   ,  0.05 ]],

                 [[ 0.    , 0.   ,  0.   ],
                  [ 0.    , 0.   ,  0.   ],
                  [ 0.    , 0.   ,  0.   ],
                  [ 0.    , 0.   ,  0.   ],
                  [ 0.    , 0.   ,  0.   ]]]
X_opositeside = np.array(X_opositeside)
X_opositeside.shape
tt_opo = [0.05807600784301757, -0.011612522125244138, 0.012757064819335945, 0.06769500350952148, -0.01230546951293945, 0.009644226074218748, 0.0, 0.0, 0.0, -0.18943798446655274, -0.03305449676513672, -0.008363098144531261, -0.22814461898803712, -0.07072938919067383, -0.03345309448242187]
tt

dist = np.zeros([len(counts),])
for i in range(0,len(counts)):
    dist[i], _ = fastdtw(tt_opo, X_opositeside[i], radius=1, dist=euclidean)
dist
gl.gd.Gs_dynamic[np.argmin(dist)]

len(tt_opo)

X_opositeside[i].shape


''' Fix
'''
tt_opo_fixed = np.array(tt_opo).reshape(5,3)


dist = np.zeros([len(counts),])
for i in range(0,len(counts)):
    dist[i], _ = fastdtw(tt_opo_fixed, X_opositeside[i], radius=1, dist=euclidean)
dist
gl.gd.Gs_dynamic[np.argmin(dist)]

len(tt_opo)






#
