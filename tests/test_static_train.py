import sys, os
import numpy as np

try:
    from teleop_gesture_toolbox.learning.pymc_train import Experiments
except:
    Experiments = None # pytest discover everytime

# from teleop_gesture_toolbox.os_and_utils.utils import ordered_load, GlobalPaths, load_params
# from teleop_gesture_toolbox.os_and_utils.parse_yaml import ParseYAML
# PATHS = GlobalPaths(change_working_directory=True)
# from teleop_gesture_toolbox.os_and_utils.loading import HandDataLoader, DatasetLoader


def test_static_data_load():
    X, y = DatasetLoader({'input_definition_version':1, 'interpolate':1}).load_static(PATHS.learn_path, ['grab', 'pinch', 'point', 'two', 'three', 'four', 'five', 'thumbsup'], new=True)

def test_train_network():
    assert Experiments is not None
    args = {
    'experiment': "trainWithParameters",
    'inference_type': 'ADVI', 
    'layers': 2, 
    'input_definition_version': 1, 
    'split': 0.3, 
    'take_every': 10, 
    'iter': 1, 
    'n_hidden': 25, 
    'seed': 1564, 
    'gesture_type': "static", 
    'model_filename': 'network_just_for_testing', 
    'full_dataload': False, 
    'engine': "PyMC", 
    'save': True, 
    'test': True
    } 
    
    experiment = getattr(Experiments(), args['experiment'])
    experiment(args=args)