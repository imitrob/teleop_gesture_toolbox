#!/usr/bin/env python3
import os, sys
import pickle
import numpy as np
from copy import deepcopy

from sklearn.model_selection import train_test_split
from scipy.spatial.distance import euclidean
from fastdtw import fastdtw
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt

import yaml

sys.path.append('../os_and_utils')
from utils import ordered_load, GlobalPaths
from loading import HandDataLoader, DatasetLoader


def load_gestures_config(ws_folder):
    # Set of training gestures is given by 'gesture_recording.yaml' file
    with open(os.path.expanduser("~/"+ws_folder+"/src/mirracle_gestures/include/custom_settings/gesture_recording.yaml"), 'r') as stream:
        gestures_data_loaded = ordered_load(stream, yaml.SafeLoader)

    Gs = []
    keys = gestures_data_loaded.keys()
    Gs_set = gestures_data_loaded['using_set']
    configRecognition = gestures_data_loaded['Recognition']
    del gestures_data_loaded['using_set']; del gestures_data_loaded['Recording']; del gestures_data_loaded['configGestures']; del gestures_data_loaded['Recognition']

    # Check if yaml file is setup properly
    try:
        gestures_data_loaded[Gs_set]
    except:
        raise Exception("Error in gesture_recording.yaml, using_set variable, does not point to any available set below!")
    try:
        gestures_data_loaded[Gs_set].keys()
    except:
        raise Exception("Error in gesture_recording.yaml, used gesture set does not have any item!")
    # Setup gesture list
    Gs = gestures_data_loaded[Gs_set].keys()

    if configRecognition['args']:
        args = configRecognition['args']
    else:
        args = {"all_defined":1, "middle":1, 's':1}

    return Gs, args


class NNWrapper():
    ''' Object that holds information about neural network
        Methods for load and save the network are static
            - Use: NNWrapper.save_network()
    '''
    def __init__(self, X_train=None, approx=None, neural_network=None, Gs=[], args={}, accuracy=-1):
        # Set of Gestures in current network
        self.Gs = Gs

        # Training arguments and import configurations
        self.args = args
        # Accuracy on test data of loaded network <0.,1.>
        self.accuracy = accuracy

        # Training data X
        self.X_train = X_train
        # NN data
        self.approx, self.neural_network = approx, neural_network

    @staticmethod
    def save_network(X_train, approx, neural_network, network_path, name=None, Gs=[], args={}, accuracy=-1):
        '''
        Parameters:
            X_train (ndarray): Your X training data
            approx, neural_network (PyMC3): Neural network data for sampling
            network_path (Str): Path to network folder (e.g. '/home/<user>/<ws>/src/mirracle_gestures/include/data/Trained_network/')
            name (Str): Output network name
                - name not specified -> will save as network0.pkl, network1.pkl, network2.pkl, ...
                - name specified -> save as name
            Gs (Str[]): List of used gesture names
            args (Str{}): List of arguments of NN and training
            accuracy (Float <0.,1.>): Accuracy on test data
        '''
        print("Saving network")
        if name == None:
            n_network = ""
            for i in range(0,200):
                if not os.path.isfile(network_path+"network"+str(i)+".pkl"):
                    n_network = str(i)
                    break
            name = "network"+str(n_network)
        else:
            if not os.path.isfile(network_path+name+".pkl"):
                print("Error: file "+name+" exists, network is not saved!")

        wrapper = NNWrapper(X_train, approx, neural_network, Gs=Gs, args=args, accuracy=accuracy)
        with open(network_path+name+'.pkl', 'wb') as output:
            pickle.dump(wrapper, output, pickle.HIGHEST_PROTOCOL)

        print("Network: network"+n_network+".pkl saved")

    @staticmethod
    def load_network(network_path, name=None):
        '''
        Parameters:
            network_path (Str): Path to network folder (e.g. '/home/<user>/<ws>/src/mirracle_gestures/include/data/Trained_network/')
            name (Str): Network name to load
        Returns:
            wrapper (NetworkWrapper())
        '''
        wrapper = NNWrapper()
        with open(network_path+name, 'rb') as input:
            wrapper = pickle.load(input, encoding="latin1")

        return wrapper

#
