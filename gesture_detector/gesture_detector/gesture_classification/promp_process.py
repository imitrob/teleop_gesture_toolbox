
from promps import promp_paraschos as promp_approach

import gesture_detector
from gesture_detector.utils.loading import DatasetLoader
import numpy as np

def do_promp():
    dataloader_args = {'normalize':1, 'n':5, 'scene_frame':1, 'inverse':1}
    X, Y = DatasetLoader(dataloader_args).load_dynamic(gesture_detector.gesture_data_path, GS_DYNAMIC)

    self.X_ProMP = promp_approach.construct_promp_trajectories(self.X, self.Y, condition=False)


def save_mocked_model():
    
    X_ProMP = np.array([[[ 0.0, 0.0,  0.05], # 'swipe_down'
        [ 0.0,  0.0,  0.025],
        [ 0.0,  0.0, 0.0],
        [ 0.0,  0.0, -0.025],
        [ 0.0,  0.0, -0.05]],
        [[-0.03, -0.0, -0.0], # 'swipe_front_right'
        [-0.015, -0.0, -0.0],
        [0.0, -0.0,  0.0],
        [0.015,  0.0,  0.0],
        [0.03,  0.0,  0.0]],
        [[ 0.03,  0.0,  0.0], # 'swipe_left'
        [ 0.015,  0.0,  0.0],
        [0.0, -0.0,  0.0],
        [-0.015, -0.0, -0.0],
        [-0.03, -0.0, -0.0]],
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
    
    counts = [1,1,1,1,1]
    
    np.savez("network99_dynamic_mocked.npz", X_ProMP=X_ProMP, X=X_ProMP, Y=[0,1,2,3,4])

save_mocked_model()