#!/usr/bin/env python3
import os, sys
import pickle
import numpy as np
from copy import deepcopy
import theano
from sklearn.preprocessing import scale
from sklearn.model_selection import train_test_split
from scipy.spatial.distance import euclidean
from fastdtw import fastdtw
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
floatX = theano.config.floatX
import yaml

sys.path.append('../os_and_utils')
from utils import ordered_load, GlobalPaths

def import_data(learn_path=None, args={}, Gs=None, dataset_files=[], import_method='numpy'):
    ''' Prepares X, Y datasets from files format
        - Loads data from learn_path, gesture recordings are saved here
            - Every gesture has its own directory
            - Every gesture recording has its own file
            Pros: Easy manipulation with recording (deletion, move)
            Cons: Slower load time
    Parameters:
        learn_path (Str): Path to learn directory (gesture recordings)
        args (Str{}): Flags for import
            - 's' - Cuts recording to have only the last 1 second of recording
            - 'normalize' - Palm trajectory will start at zero (x,y,z=0,0,0 at time t=0)
            - 'user_defined' - hand data are shortened
            - 'all_defined' - all hand data are used (use as default)

            - 'absolute' - use absolute coordinates as variables (was not useful, will delete)
            - 'absolute+finger' - use abs.coord+finger abs.coord. as var. (was not useful, will delete)

            - 'interpolate' - X data will be interpolated to have static length of 100 samples
            For static gestures, only one sample is needed, but there is possibility to use more samples through recording
            - 'middle' - Uses only one sample, the one in the middle
            - 'average' - Average data throughout recording and use also one sample
            - 'as_dimesion' - Use X as dimension (was not useful, will delete)
            - 'take_every_10' - Use every tenth record sample to the dataset
            - 'take_every' - Use every record sample to the dataset

        Gs (Str[]): Set of gesture names, given names corresponds to gesture folders (from which to load gestures)
        dataset_files (Str[]): Array of filenames of dataset, which will be discarted
    Returns:
        X (ndarray): X dataset
        Y (ndarray): Y dataset
        X_palm (ndarray): Palm trajectory positions
        DX_palm (ndarray): Palm trajectory velocities
        dataset_files: Array of filenames of dataset which were read from disc
    '''
    if not learn_path:
        paths = GlobalPaths()
        learn_path = paths.learn_path

    X = []
    Y = []
    HandData, HandDataFlags = HandDataLoader().load_directory(learn_path, Gs)


    Xpalm, DXpalm = DatasetLoader.get_dynamic(data)

    X_ = []
    Y_ = []
    if 'user_defined' in args:
        for n, X_n in enumerate(X):
            row = []
            for m, X_nt in enumerate(X_n):
                row.append([X_nt.r.OC[0], X_nt.r.OC[1], X_nt.r.OC[2], X_nt.r.OC[3], X_nt.r.OC[4],
                X_nt.r.TCH12, X_nt.r.TCH23, X_nt.r.TCH34, X_nt.r.TCH45, X_nt.r.TCH13, X_nt.r.TCH14, X_nt.r.TCH15])
            X_.append(row)
    # This is the default option
    else: #elif 'all_defined' in args:
        for n, X_n in enumerate(X):
            row = []
            for m, X_nt in enumerate(X_n):
                X_nt_ = []
                X_nt_.extend(X_nt.r.wrist_hand_angles_diff[1:3])
                X_nt_.extend(ext_fingers_angles_diff(X_nt.r.fingers_angles_diff))
                X_nt_.extend(ext_pos_diff_comb(X_nt.r.pos_diff_comb))
                if 'absolute' in args:
                    X_nt_.extend(X_nt.r.pRaw)
                    if 'absolute+finger' in args:
                        X_nt_.extend(X_nt.r.index_position)
                if len(X_nt_) != 0:
                    row.append(np.array(X_nt_))
            X_.append(row)
            Y_.append(Y[n])
        print("Defined dataset type: all_defined")

    X = X_
    Y = Y_

    len(Y)
    if 'interpolate' in args:
        X_interpolated = []
        for n,sample in enumerate(X):
            X_interpolated_sample = []
            for dim in range(0,len(sample[0])):
                f2 = interp1d(np.linspace(0,1, num=len(np.array(sample)[:,dim])), np.array(sample)[:,dim], kind='cubic')
                X_interpolated_sample.append(f2(np.linspace(0,1, num=101)))
            X_interpolated.append(np.array(X_interpolated_sample).T)
        X=np.array(X_interpolated)
        print("Data interpolated")


    X_ = []
    Y_ = []
    for n, X_n in enumerate(X):
        if 'middle' in args:
            X_.append(deepcopy(X_n[0]))
            Y_.append(deepcopy(Y[n]))
        elif 'average'in args: X_.append(deepcopy(avg_dataframe(X_n)))
        elif 'as_dimesion' in args: X_.append(deepcopy(X_n))

        # This is the default option
        elif 'take_every' in args:
            for i in range(0, len(X_n), int(args['take_every'])):
                X_.append(deepcopy(X_n[i]))
                Y_.append(deepcopy(Y[n]))
            #Y = Y_
        else: # 'take_every' == 10
            for i in range(0, len(X_n), 10):
                X_.append(deepcopy(X_n[i]))
                Y_.append(deepcopy(Y[n]))
            #Y = Y_
    X = X_
    Y = Y_
    X = np.array(X)
    Y = np.array(Y)
    Xpalm.shape
    Y.shape
    Y

    X = scale(X)
    X = X.astype(floatX)
    Y = Y.astype(floatX)
    X = np.array(X)
    Y = np.array(Y)

    data = {
    'static': {'X': X, 'Y': Y},
    'dynamic': {'Xpalm': Xpalm, 'DXpalm': DXpalm, 'Y': Ydyn},
    'info': {'dataset_files': dataset_files}
    }
    return data


## Average dataframe
def avg_dataframe(data_n):
    data_avg = np.zeros([len(data_n[0])])
    for data_nt in data_n:
        data_avg += data_nt
    data_avg *= (1/len(data_n))
    return data_avg


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
