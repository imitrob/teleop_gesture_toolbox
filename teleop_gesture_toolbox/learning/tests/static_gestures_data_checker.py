'''
Plot saved data of 'opened_hand','closed_hand' gestures
'''
import sys; sys.path.append("../..")
from os_and_utils import settings; settings.init()
sys.path.append(settings.paths.teleop_gesture_toolbox_path+'/leapmotion')
import numpy as np
from os_and_utils import visualizer_lib as vis

frames = np.load(settings.paths.learn_path+'/opened_hand/0.npy', allow_pickle=True)
hand_line_figs = [vis.HandPlot.generate_hand_lines(frame, 'l', alpha=0.3) for frame in frames]
vis.HandPlot.hand_plot(hand_line_figs)


frames = np.load(settings.paths.learn_path+'/closed_hand/1.npy', allow_pickle=True)
hand_line_figs = [vis.HandPlot.generate_hand_lines(frame, 'l', alpha=0.3) for frame in frames]
vis.HandPlot.hand_plot(hand_line_figs)
