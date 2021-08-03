#!/usr/bin/env python3.7
from os.path import isfile
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

if __name__=="__main__":
    import sys
    import os
    print(sys.version_info)

    PATH = os.path.dirname(os.path.realpath(__file__))
    sys.path.insert(1, PATH)
    sys.path.insert(1, os.path.abspath(os.path.join(PATH, '..')))

    import settings
    settings.init()

    Gs_names = ['Grab', 'Pinch', 'Point', 'Respectful', 'Spock', 'Rock', 'Victory']
    #, 'Italian', 'Rotate', 'Swipe_Up', 'Pin', 'Touch', 'Swipe_Left', 'Swipe_Down', 'Swipe_Right']
    Gs_names = ['Rotate', 'Swipe_Up', 'Pin', 'Touch', 'Swipe_Left', 'Swipe_Down', 'Swipe_Right']
    Gs_names = ['Rotate', 'Swipe_Up', 'Pin', 'Touch']
    args = []
    args.append("all_defined")
    args.append("1s")
    args.append("take_every_10")
    args.append("normalize")


## Extraction functions
def ext_fingers_angles_diff(fingers_angles_diff):
    ret = []
    for i in fingers_angles_diff:
        for j in i:
            ret.append(j[1])
            ret.append(j[2])
    return ret

def ext_pos_diff_comb(pos_diff_comb):
    ret = []
    for i in pos_diff_comb:
        ret.extend(i)
    return ret

def import_data(settings, args, Gs_names=['Rotate', 'Swipe_Up', 'Pin', 'Touch']):
    ''' Prepares X, Y datasets from files format
    '''
    args
    X = []
    Y = []
    ## Load data from file
    for n, G in enumerate(Gs_names):
        i = 0
        while isfile(settings.LEARN_PATH+G+"/"+str(i)+".pkl"):
            with open(settings.LEARN_PATH+G+"/"+str(i)+".pkl", 'rb') as input:
                X.append(pickle.load(input, encoding="latin1"))
                Y.append(n)
            i += 1
    data = deepcopy(X)
    if X == []: raise Exception('No data')

    X_ = []
    Y_ = []
    ## Pick only last 1 second of recording
    if '1s' in args:
        for n,rec in enumerate(X):
            ## Find how long one second is:
            i = 0
            while i < 300:
                i+=1
                if abs((rec[-1].r.pPose.header.stamp-rec[-i].r.pPose.header.stamp).to_sec()) > 1:
                    break
            ## Pick i elements
            if i < 10:
                continue
            c = False
            for r in rec:
                if r.r.wrist_hand_angles_diff[1:3] == []:
                    c = True
            if c: continue
            rec_ = []
            for j in range(-i, 0):
                rec_.append(deepcopy(rec[j]))
            X_.append(rec_)
            Y_.append(Y[n])
        X = X_
        Y = Y_
        print("1s")

    def elemOverN(X, n):
        c=0
        for i in X:
            if len(i) > n:
                c+=1
        return c
    elemOverN(X, 10)

    ## Pick: samples x time x palm_position

    Xpalm = []
    for sample in X:
        row = []
        for t in sample:
            if t.r.index_position == []:
                t.r.index_position = [0.,0.,0.]

            l2 = np.linalg.norm(np.array(t.r.pRaw[0:3]) - np.array(t.r.index_position))
            row.append([*t.r.pRaw[0:3], l2])#,*t.r.index_position])
        Xpalm.append(row)



    if 'normalize' in args:
        Xpalm = np.array(Xpalm)
        Xpalm_ = []
        for p in Xpalm:
            p_ = []
            p0 = deepcopy(p[0])
            for n in range(0, len(p)):
                p_.append(np.subtract(p[n], p0))
            Xpalm_.append(p_)

        Xpalm = Xpalm_



    ## Interpolate palm_positions, to 100 time samples
    Xpalm_interpolated = []
    for n,sample in enumerate(Xpalm):
        Xpalm_interpolated_sample = []
        for dim in range(0,3):
            f2 = interp1d(np.linspace(0,1, num=len(np.array(sample)[:,dim])), np.array(sample)[:,dim], kind='cubic')
            Xpalm_interpolated_sample.append(f2(np.linspace(0,1, num=101)))
        Xpalm_interpolated.append(np.array(Xpalm_interpolated_sample).T)
    Xpalm_interpolated=np.array(Xpalm_interpolated)



    ## Create derivation to palm_positions
    dx = 1/100
    DXpalm_interpolated = []
    for sample in Xpalm_interpolated:
        DXpalm_interpolated_sample = []
        sampleT = sample.T
        for dim in range(0,3):
            DXpalm_interpolated_sample.append(np.diff(sampleT[dim]))
        DXpalm_interpolated.append(np.array(DXpalm_interpolated_sample).T)
    DXpalm_interpolated = np.array(DXpalm_interpolated)

    Xpalm = np.array(Xpalm_interpolated)
    DXpalm = np.array(DXpalm_interpolated)

    Xpalm_ = []
    np.array(Xpalm).shape
    Xpalm = np.swapaxes(Xpalm, 1, 2)
    for n,dim1 in enumerate(Xpalm):
        for m,dim2 in enumerate(dim1):
            Xpalm[n,m] = (dim2 - np.min(dim2)) / (np.max(dim2) - np.min(dim2))
    Xpalm = np.swapaxes(Xpalm, 1, 2)
    Xpalm

    # backup

    y_original = deepcopy(Y)

    X_ = []
    Y_ = []
    if 'user_defined' in args:
        for n, X_n in enumerate(X):
            row = []
            for m, X_nt in enumerate(X_n):
                row.append([X_nt.r.OC[0], X_nt.r.OC[1], X_nt.r.OC[2], X_nt.r.OC[3], X_nt.r.OC[4],
                X_nt.r.TCH12, X_nt.r.TCH23, X_nt.r.TCH34, X_nt.r.TCH45, X_nt.r.TCH13, X_nt.r.TCH14, X_nt.r.TCH15])
            X_.append(row)
    elif 'all_defined' in args:
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
        print("all_defined")
    X = X_
    Y = Y_

    if 'interpolate' in args:
        X_interpolated = []
        for n,sample in enumerate(X):
            X_interpolated_sample = []
            for dim in range(0,len(sample[0])):
                f2 = interp1d(np.linspace(0,1, num=len(np.array(sample)[:,dim])), np.array(sample)[:,dim], kind='cubic')
                X_interpolated_sample.append(f2(np.linspace(0,1, num=101)))
            X_interpolated.append(np.array(X_interpolated_sample).T)
        X=np.array(X_interpolated)


    X_ = []
    Y_ = []
    for n, X_n in enumerate(X):
        if 'middle' in args:
            X_.append(deepcopy(X_n[0]))
            Y_.append(deepcopy(Y[n]))
        elif 'average'in args: X_.append(deepcopy(avg_dataframe(X_n)))
        elif 'as_dimesion' in args: X_.append(deepcopy(X_n))
        elif 'take_every_10' in args:
            for i in range(0, len(X_n), 10):
                X_.append(deepcopy(X_n[i]))
                Y_.append(deepcopy(Y[n]))
            #Y = Y_
        elif 'take_every' in args:
            for X_nt in X_n:
                X_.append(deepcopy(X_nt))
                Y_.append(deepcopy(Y[n]))
        else: raise Exception("Bad option")
    X = X_
    Y = Y_
    X = np.array(X)
    Y = np.array(Y)
    X.shape
    Y.shape
    Y

    return X, Y, data, Xpalm, DXpalm


## Average dataframe
def avg_dataframe(data_n):
    data_avg = np.zeros([len(data_n[0])])
    for data_nt in data_n:
        data_avg += data_nt
    data_avg *= (1/len(data_n))
    return data_avg


## Saving network
class Zabal():
    def __init__(self, _sample_proba=None, X_train=None, approx=None, neural_network=None):
        self._sample_proba = _sample_proba
        self.X_train = X_train
        self.approx = approx
        self.neural_network = neural_network


def save_network(settings,_sample_proba, X_train, approx, neural_network):
    print("saving network")
    n_network = ""
    for i in range(0,200):
        if not isfile(settings.NETWORK_PATH+"/network"+str(i)+".pkl"):
            n_network = str(i)
            break

    zabal = Zabal(_sample_proba, X_train, approx, neural_network)
    with open(settings.NETWORK_PATH+"/network"+str(n_network)+'.pkl', 'wb') as output:
        pickle.dump(zabal, output, pickle.HIGHEST_PROTOCOL)

    print("Network: network"+n_network+".pkl saved")

def load_network(settings, name='network1.pkl'):
    zabal = Zabal()
    with open(settings.NETWORK_PATH+"/"+name, 'rb') as input:
        zabal = pickle.load(input, encoding="latin1")

    return zabal._sample_proba, zabal.X_train, zabal.approx, zabal.neural_network


#
