
import sys, os, argparse, time
import numpy as np
import matplotlib.pyplot as plt
from os_and_utils import settings

sys.path.append('hand_processing') # .../src/hand_processing
#from loading import HandDataLoader, DatasetLoader
from os_and_utils.loading import HandDataLoader, DatasetLoader

import gesture_classification.gestures_lib as gl
import os_and_utils.scenes as sl
import os_and_utils.move_lib as ml
import os_and_utils.ros_communication_main as rc

class PathGenDummy():
    def __init__(self):

        approach = 'TBD'
        # Motion Primitives
        self.Ms = gl.gd.l.mp.info.names
        #print("MP gestures", self.Ms)

        # Load Motion Primitive paths
        #self.X, self.Y, self.robot_promps = DatasetLoader(['interpolate', 'discards']).load_mp(settings.paths.learn_path, self.Ms, approach)

    def handle_action_queue(self, action):
        return None
        '''
        action_stamp, action_name, action_hand = action

        vars = gl.gd.var_generate(action_hand, action_stamp)

        path = self.generate_path(action_name, vars=vars, tmp_action_stamp=action_stamp)
        return path
        '''

    def generate_path(self, id, vars={}, tmp_action_stamp=None):
        ''' Main function
        Parameters:
            id (str): gesture ID (string)
            X (ndarray[rec x t x 3 (xyz)]): The training data
            vars (GestureDataHand or Hand at all?): nested variables
        Returns:
            trajectory ([n][x,y,z]): n ~ 100 (?) It is ProMP output path (When static MP -> returns None)
            waypoints (dict{} of Waypoint() instances), where index is time (0-1) of trajectory execution
        '''
        return path_
