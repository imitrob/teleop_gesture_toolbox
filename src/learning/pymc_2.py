#!/usr/bin/env python3.7

import sys
import os
print(sys.version_info)
from os.path import expanduser

THIS_FILE_PATH = os.path.dirname(os.path.realpath(__file__))
THIS_FILE_TMP = os.path.abspath(os.path.join(THIS_FILE_PATH, '..', '..'))
WS_FOLDER = THIS_FILE_TMP.split('/')[-1]

sys.path.insert(1, expanduser("~/"+WS_FOLDER+"/src/mirracle_gestures/src/learning"))
sys.path.insert(1, expanduser("~/"+WS_FOLDER+"/src/mirracle_gestures/src"))

import settings
settings.init()
from import_data import *

from warnings import filterwarnings

import matplotlib.pyplot as plt
import numpy as np
import pymc3 as pm
import pandas as pd
import seaborn as sns
import sklearn
import theano
import theano.tensor as T
import random

from sklearn import svm, datasets
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import scale
floatX = theano.config.floatX
from sklearn.metrics import confusion_matrix
import csv
from sklearn.metrics import plot_confusion_matrix
import confusion_matrix_pretty_print
from fastdtw import fastdtw
import promp_lib2
from statistics import mode, StatisticsError
from itertools import combinations, permutations

sys.path.insert(1, os.path.expanduser("~/promps_python"))
from promp.discrete_promp import DiscretePROMP
from promp.linear_sys_dyn import LinearSysDyn
from promp.promp_ctrl import PROMPCtrl
from numpy import diff
from numpy import linalg as LA


rc = {'xtick.labelsize': 20, 'ytick.labelsize': 20, 'axes.labelsize': 20, 'font.size': 20,
      'legend.fontsize': 12.0, 'axes.titlesize': 10, "figure.figsize": [12, 6]}
sns.set(rc = rc)
sns.set_style("white")

#%config InlineBackend.figure_format = 'retina'
filterwarnings("ignore")

Gs = ['Rotate', 'Swipe_Up', 'Pin', 'Touch', 'Swipe_Left', 'Swipe_Down', 'Swipe_Right']
Gs = ['Rotate', 'Swipe_Up', 'Pin', 'Touch']
Gs = ['Swipe_Left', 'Swipe_Right']
Gs = ['Swipe_Left', 'Swipe_Right', 'Swipe_Down', 'Swipe_Up']
Gs = ['Grab', 'Pinch', 'Point', 'Respectful', 'Spock', 'Rock', 'Victory']
args = ["all_defined", "middle", "palm", '1s']#,'interpolate','normalize', 'diag_sigma_W']
X, Y, data, Xpalm, DXpalm = import_data(settings, args, Gs_names=Gs)

X = scale(X)
X = X.astype(floatX)
Y = Y.astype(floatX)
X = np.array(X)
Y = np.array(Y)
print(X.shape)
print(Y.shape)
print(Xpalm.shape)
assert len(X) == len(Y) == len(Xpalm), "Lengths not match"
len(Y[Y==0])
len(Y[Y==1])
len(Y[Y==2])

if False:
    X,Y = promp_lib2.get_weights(Xpalm,DXpalm,Y)
    X.shape

def time_warp_launch(args=[]):
    assert X.ndim==2, "X var is 3D"
    g = list(dict.fromkeys(Y))
    g
    counts = [list(Y).count(g_) for g_ in g]
    counts
    sum(counts)

    paths = Xpalm

    if True:
        if 'eacheach' in args:
            t=time.time()
            # samples x samples
            results = np.zeros([sum(counts),])
            for j in range(0,sum(counts)):
                dist = np.zeros([sum(counts),])
                for i in range(0,sum(counts)):
                    dist[i], _ = fastdtw(paths[j], paths[i])

                mean = np.zeros([len(counts),])
                for i in range(0,len(counts)):
                    mean[i] = np.mean(dist[sum(counts[0:i]):sum(counts[0:i])+counts[i]])
                results[j] = np.argmin(mean)
            print(abs(t-time.time()))
        elif 'random' in args:
            # samples x random
            t = time.time()
            results = np.zeros([sum(counts),])
            for j in range(0,sum(counts)):
                dist = np.zeros([len(counts),])
                for i in range(0,len(counts)):
                    rand_ind = random.randint(0,counts[i]-1)
                    dist[i], _ = fastdtw(paths[j], paths[sum(counts[0:i])], dist=euclidean)

                results[j] = np.argmin(dist)
            print(abs(time.time()-t))

        elif 'promp' in args:
            t = time.time()
            paths = promp_lib2.construct_promp_trajectories(Xpalm, DXpalm, Y)
            paths.shape
            results = np.zeros([sum(counts),])
            for j in range(0,sum(counts)):
                dist = np.zeros([len(counts),])
                for i in range(0,len(counts)):
                    dist[i], _ = fastdtw(Xpalm[j], paths[i], radius=1, dist=euclidean)
                results[j] = np.argmin(dist)
            print(abs(time.time()-t))


        elif 'euclidean' in args:
            paths = promp_lib2.construct_promp_trajectories(Xpalm, DXpalm, Y)
            paths.shape
            results = np.zeros([sum(counts),])
            for j in range(0,sum(counts)):
                dist = np.zeros([len(counts),])
                for i in range(0,len(counts)):
                    dist[i], _ = fastdtw(Xpalm[j], paths[i], dist=euclidean)
                results[j] = np.argmin(dist)

        elif 'crossvalidation' in args:
            # samples x random
            results = np.zeros([sum(counts),])
            for j in range(0,sum(counts)):
                dist = np.zeros([len(counts),])
                for i in range(0,len(counts)):
                    rand_ind = random.randint(0,counts[i]-1)
                    index1 = sum(counts[0:i])
                    index2 = sum(counts[0:i])+rand_ind
                    dist[i], _ = fastdtw(paths[j], paths[index1:index2], dist=euclidean)

                results[j] = np.argmin(dist)

    name = 'tmp'
    confusion_matrix_pretty_print.plot_confusion_matrix_from_data(Y, results, Gs,
      annot=True, cmap = 'Oranges', fmt='1.8f', fz=10, lw=25, cbar=False, figsize=[6,6], show_null_values=2, pred_val_axis='y', name=name)

    print("Accuracy = {}%".format((Y == results).mean() * 100))
#time_warp_launch(args=args)




def promp_launch(args=args):
    assert X.ndim==2, "X var is 3D"
    g = list(dict.fromkeys(Y))
    counts = [list(Y).count(g_) for g_ in g]

    paths = Xpalm[:,:,:]

    if True:
        # compute weights
        weights = []
        for i in range(0,len(g)):
            row = []
            for j in range(0,paths.shape[2]):
                d_promp = DiscretePROMP(data=paths[Y==i,:,j])
                d_promp.train()
                d_promp.set_start(data[0][0])
                d_promp.set_goal(data[0][-1])
                if '_mean_W' in args:
                    row.extend(np.array(d_promp._mean_W))
                elif '_sigma_W' in args:
                    row.extend(np.array(d_promp._sigma_W).flatten())
                elif 'diag_sigma_W' in args:
                    row.extend(np.diag(d_promp._sigma_W))
                elif '_W' in args:
                    row.extend(np.array(d_promp._W))
                else: raise Exception("Wrong args")
            weights.append(row)

        # compute weights test
        weights_test = []
        for i in range(0,len(paths)):
            row = []
            for j in range(0, paths.shape[2]):
                d_promp = DiscretePROMP(data=np.vstack([paths[i:i+1,:,j],paths[i:i+1,:,j]]))
                d_promp.train()
                d_promp.set_start(data[0][0])
                d_promp.set_goal(data[0][-1])
                if '_mean_W' in args:
                    row.extend(np.array(d_promp._mean_W))
                elif '_sigma_W' in args:
                    row.extend(np.array(d_promp._sigma_W).flatten())
                elif 'diag_sigma_W' in args:
                    row.extend(np.diag(d_promp._sigma_W))
                elif '_W' in args:
                    row.extend(np.array(d_promp._W))
                else: raise Exception("Wrong args")
            weights_test.append(row)
        weights_test = np.array(weights_test)
        weights = np.array(weights)

        weights.shape
        paths[Y==1].shape
        weights_test.shape
        weights.shape
        len(weights)
        # compare weights with weights_test
        len(weights_test)

        results = []
        for k in range(0,len(weights_test)):
            varwinning = []
            for j in range(0,len(weights_test[0])):
                comparearray = []
                for i in range(0, len(g)):
                    comparearray.append(abs(weights[i,j] - weights_test[k,j]))
                varwinning.append(comparearray)
            dists = []


            varwinning = np.array(varwinning).T
            for row in varwinning:
                vars = (len(varwinning[0])**(1/2))
                dist = LA.norm(np.reshape(np.array(row), (-1,int(vars))), 'fro')
                dists.append(dist)
            results.append(np.argmin(dists))
        results = np.array(results)
        results.shape
        if False:
            d_promp = DiscretePROMP(data=paths[Y==0,:,0])
            d_promp.train()

            for traj in paths[Y==0,:,0]:
                plt.figure("ProMP-Pos")
                plt.plot(traj, 'k', alpha=0.2)


            pos_1, vel_1, acc_1 = d_promp.generate_trajectory(phase_speed=1.,  randomness=1e-1)

            plt.figure("ProMP-Pos")
            plt.xlabel("time - normalized [%]")
            plt.ylabel("x position")
            plt.plot(pos_1, 'r', lw=5)
            plt.savefig(os.path.expanduser("~/Pictures/plot1.png"), format='png')

        name = 'tmp'
        confusion_matrix_pretty_print.plot_confusion_matrix_from_data(Y, results, Gs,
          annot=True, cmap = 'Oranges', fmt='1.8f', fz=12, lw=1.5, cbar=False, figsize=[5,5], show_null_values=2, pred_val_axis='y', name=name)

          print("Accuracy = {}%".format((Y == results).mean() * 100))






promp_launch(args=args)

X.shape
Y.shape
assert not np.any(np.isnan(X))
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3)


for i in range(2,3):
    fig, ax = plt.subplots()
    ax.scatter(X[Y == 0, 0], X[Y == 0, i], label="Class 0")
    ax.scatter(X[Y == 1, 0], X[Y == 1, i], color="r", label="Class "+str(i))
    sns.despine()
    ax.legend()
    ax.set(xlabel="X", ylabel="Y")

def construct_nn_2l(ann_input, ann_output, n_hidden = [10], out_n=7):
    ''' Model description
        -> n_hidden - number of hidden layers
        -> number of layers
        -> type of layer distribution (normal, bernoulli, uniform, poisson)
        -> distribution parameters (mu, sigma)
        -> activation function (tanh, sigmoid)

    '''

    # Initialize random weights between each layer
    init_1 = np.random.randn(X.shape[1], n_hidden[0]).astype(floatX)
    #init_2 = np.random.randn(n_hidden[0], n_hidden[1]).astype(floatX)
    #init_3 = np.random.randn(n_hidden[1], n_hidden[2]).astype(floatX)
    init_out = np.random.randn(n_hidden[0], out_n).astype(floatX)

    with pm.Model() as neural_network:
        # Trick: Turn inputs and outputs into shared variables using the data container pm.Data
        # It's still the same thing, but we can later change the values of the shared variable
        # (to switch in the test-data later) and pymc3 will just use the new data.
        # Kind-of like a pointer we can redirect.
        # For more info, see: http://deeplearning.net/software/theano/library/compile/shared.html
        ann_input = pm.Data("ann_input", X_train)
        ann_output = pm.Data("ann_output", Y_train)

        # Weights from input to hidden layer
        weights_in_1 = pm.Normal("w_in_1", 0, sigma=1, shape=(X.shape[1], n_hidden[0]), testval=init_1)

        # Weights from 1st to 2nd layer
        #weights_1_2 = pm.Normal("w_1_2", 0, sigma=1, shape=(n_hidden[0], n_hidden[1]), testval=init_2)

        # Weights from 1st to 2nd layer
        #weights_2_3 = pm.Normal("w_2_3", 0, sigma=1, shape=(n_hidden[1], n_hidden[2]), testval=init_3)

        # Weights from hidden layer to output
        weights_3_out = pm.Normal("w_2_out", 0, sigma=1, shape=(n_hidden[0], out_n), testval=init_out)

        # Build neural-network using tanh activation function
        act_1 = pm.math.tanh(pm.math.dot(ann_input, weights_in_1))
        #act_2 = pm.math.tanh(pm.math.dot(act_1, weights_1_2))
        #act_3 = pm.math.tanh(pm.math.dot(act_2, weights_2_3))
        act_out = pm.math.sigmoid(pm.math.dot(act_1, weights_3_out))

        out = pm.Categorical(
            "out",
            act_out,
            observed=ann_output,
            total_size=Y_train.shape[0],  # IMPORTANT for minibatches
        )

    return neural_network

def construct_nn_3l(ann_input, ann_output, n_hidden = [10,10], out_n=7):
    init_1 = np.random.randn(X.shape[1], n_hidden[0]).astype(floatX)
    init_2 = np.random.randn(n_hidden[0], n_hidden[1]).astype(floatX)
    init_out = np.random.randn(n_hidden[1], out_n).astype(floatX)
    with pm.Model() as neural_network:
        ann_input = pm.Data("ann_input", X_train)
        ann_output = pm.Data("ann_output", Y_train)
        weights_in_1 = pm.Normal("w_in_1", 0, sigma=1, shape=(X.shape[1], n_hidden[0]), testval=init_1)
        weights_1_2 = pm.Normal("w_1_2", 0, sigma=1, shape=(n_hidden[0], n_hidden[1]), testval=init_2)
        weights_3_out = pm.Normal("w_2_out", 0, sigma=1, shape=(n_hidden[1], out_n), testval=init_out)
        act_1 = pm.math.tanh(pm.math.dot(ann_input, weights_in_1))
        act_2 = pm.math.tanh(pm.math.dot(act_1, weights_1_2))
        act_out = pm.math.sigmoid(pm.math.dot(act_2, weights_3_out))
        out = pm.Categorical(
            "out",
            act_out,
            observed=ann_output,
            total_size=Y_train.shape[0],  # IMPORTANT for minibatches
        )
    return neural_network

if True:
    n_hidden = [50]
    name = 'conf_pretty_2l_n_'+str(n_hidden)+"_alldata"
    neural_network = construct_nn_2l(X_train, Y_train, n_hidden=n_hidden)
    neural_network
    pm.set_tt_rng(theano.sandbox.rng_mrg.MRG_RandomStream(101)) # Set random seeds


with neural_network:
    '''
        -> inference type (ADVI)
        -> n - number of iterations
    '''
    inference = pm.ADVI()
    approx = pm.fit(n=50000, method=inference)

if False:
    _, _, approx, neural_network = load_network(settings, name='network6.pkl')


if True:
    plt.figure(figsize=(12,6))
    plt.plot(-inference.hist, label="new ADVI", alpha=0.3)
    plt.plot(approx.hist, label="old ADVI", alpha=0.3)
    plt.legend()
    plt.ylabel("ELBO")
    plt.xlabel("iteration");

if True:
    import sys
    import os
    print(sys.version_info)
    from os.path import expanduser
    sys.path.insert(1, expanduser("~/"+WS_FOLDER+"/src/mirracle_gestures/src/learning"))
    sys.path.insert(1, expanduser("~/"+WS_FOLDER+"/src/mirracle_gestures/src"))

    import settings
    settings.init()
    from import_data import *

    from warnings import filterwarnings

    import matplotlib.pyplot as plt
    import numpy as np
    import pymc3 as pm
    import pandas as pd
    import seaborn as sns
    import sklearn
    import theano
    import theano.tensor as T
    import random

    from sklearn import svm, datasets
    from sklearn.datasets import make_moons
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import scale
    floatX = theano.config.floatX
    from sklearn.metrics import confusion_matrix
    import csv
    from sklearn.metrics import plot_confusion_matrix
    import confusion_matrix_pretty_print
    from fastdtw import fastdtw
    import promp_lib2
    from statistics import mode, StatisticsError
    from itertools import combinations, permutations

    sys.path.insert(1, os.path.expanduser("~/promps_python"))
    from promp.discrete_promp import DiscretePROMP
    from promp.linear_sys_dyn import LinearSysDyn
    from promp.promp_ctrl import PROMPCtrl
    from numpy import diff
    from numpy import linalg as LA

    person = 'person8'
    Gs = ['Grab', 'Pinch', 'Point', 'Respectful', 'Spock', 'Rock', 'Victory']
    args = ["all_defined", "middle", "palm"]#, '1s'] #,'interpolate','normalize', 'diag_sigma_W']
    settings.LEARN_PATH = os.path.expanduser("~/"+WS_FOLDER+"/src/mirracle_gestures/include/data/")+person+"/"
    X, Y, data, Xpalm, DXpalm = import_data(settings, args, Gs_names=Gs)
    X = scale(X)
    X = X.astype(floatX)
    Y = Y.astype(floatX)
    X_test = np.array(X)
    Y_test = np.array(Y)

    x = T.matrix("X")
    n = T.iscalar("n")
    _, X_train, approx, neural_network = load_network(settings, name='network5.pkl')
    x.tag.test_value = np.empty_like(X_train[:10])
    n.tag.test_value = 100
    _sample_proba = approx.sample_node(
        neural_network.out.distribution.p, size=n, more_replacements={neural_network["ann_input"]: x}
    )
    sample_proba = theano.function([x, n], _sample_proba)

    pred = sample_proba(X_test,1000).mean(0)
    y_pred = np.argmax(pred, axis=1)
    Y_test = np.array(Y_test).astype(int)
    y_pred = np.array(y_pred).astype(int)
    print("Accuracy = {}%".format((Y_test == y_pred).mean() * 100))

    name = person
    confusion_matrix_pretty_print.plot_confusion_matrix_from_data(Y_test, y_pred, Gs,
      annot=True, cmap = 'Oranges', fmt='1.8f', fz=12, lw=1.5, cbar=False, figsize=[9,9], show_null_values=2, pred_val_axis='y', name=name)








    ###
    ##
    ##
    ##










    pred_t = sample_proba(X_train,10000).mean(0)
    y_pred_t = np.argmax(pred_t, axis=1)
    print("Accuracy on train = {}%".format((Y_train == y_pred_t).mean() * 100))

    if ((Y_test == y_pred).mean() * 100) > 85:
        save_network(settings,_sample_proba, X_train, approx, neural_network)

    name = 'network5'
    confusion_matrix_pretty_print.plot_confusion_matrix_from_data(Y_test, y_pred, Gs,
      annot=True, cmap = 'Oranges', fmt='1.8f', fz=12, lw=1.5, cbar=False, figsize=[9,9], show_null_values=2, pred_val_axis='y', name=name)


'''
obs -> number of observations for each recording = 12
n_hidden -> number of hidden inner nodes = 10, 20, 50, 100
GEST -> number of recognizing gestures = 7

Data:
7 types of gestures, 41 sample recordings of each gesture, 12 observations per recording
input dataset X -> shape(7*41, 12)
              y -> shape(41,)

Model 3 layers:
w_in_1 ∼ Normal, shape(obs, n_hidden)
w_1_out ∼ Normal, shape(n_hidden, GEST)
out ∼ Categorical

w_in_1 ∼ Normal(mu=0.0, sigma=1.0)
w_1_out ∼ Normal(mu=0.0, sigma=1.0)

Accuracy:
n_hidden = 10, 78% test data, 79% train data
n_hidden = 20, 65% test data, 66% train data
n_hidden = 50, 82% test data, 80% train data
n_hidden =100, 84% test data, 81% train data
n_hidden =200, 77% test data, 85% train data (~5min. learn)

Model 4 layers:
w_in_1 ∼ Normal, shape(obs, n_hidden)
w_1_2 ∼ Normal, shape(n_hidden, n_hidden)
w_2_out ∼ Normal, shape(n_hidden, GEST)
out ∼ Categorical

w_in_1 ∼ Normal(mu=0.0, sigma=1.0)
w_1_2 ∼ Normal(mu=0.0, sigma=1.0)
w_2_out ∼ Normal(mu=0.0, sigma=1.0)


n_hidden =  5, 57% test data, 56% train data
n_hidden = 10, 79% test data, 75% train data
n_hidden = 20, 76% test data, 78% train data
n_hidden = 50, 76% test data, 78% train data
n_hidden = 100, 80% test data, 82% train data (~3min. learn)

Bigger dataset:
7 types of gestures, 1230 sample recordings of each gesture, 12 observations per recording
input dataset X -> shape(7*1230, 12)
              y -> shape(41,)

Accuracy using model 3 layers:

n_hidden = 10, ~60% test data, ~60% train data
n_hidden = 50,  81% test data,  82% train data (~10min. learn) (Fig1. Confusion table)
n_hidden =100,  78% test data,  77% train data (~40min. learn)

Removing Pinch gesture -> 6 types of gestures:
n_hidden = 50, 89.7% test data, 90.2% train data (~10min. learn) (conf_pretty_3l_n_[50]_6gests.png)

Using 'all defined' dataset. Data:
7 types of gestures, 1230 sample recordings of each gesture, 87 observations per recording
input dataset X -> shape(7*1230, 87)
              y -> shape(7*1230,)

Accuracy using model 3 layers:

n_hidden = 20, 87% test data, 87.2% train data, (~12min. learn)
n_hidden = 50, 96% test data, 96% train data, 50000 iter. (~40min. learn)
n_hidden = 80, 86% test data, 88% train data (~45min. learn)

'''
pred_t.shape


# Create bechmark functions
trace = approx.sample(draws=5000)
def production_step1():
    pm.set_data(new_data={"ann_input": X_test, "ann_output": Y_test}, model=neural_network)
    ppc = pm.sample_posterior_predictive(
        trace, samples=500, progressbar=False, model=neural_network
    )
    ppc['out'].shape




pred = sample_proba(X_test, 500).mean(0)
pred = np.argmax(pred, axis=1)
if True:
    fig, ax = plt.subplots()
    ax.scatter(X_test[pred == 0, 0], X_test[pred == 0, 1])
    ax.scatter(X_test[pred == 1, 0], X_test[pred == 1, 1], color="r")
    sns.despine()
    ax.set(title="Predicted labels in testing set", xlabel="X", ylabel="Y");

print("Accuracy = {}%".format((Y_test == pred).mean() * 100))


grid = pm.floatX(np.mgrid[-3:3:100j, -3:3:100j])
grid_2d = grid.reshape(2, -1).T
dummy_out = np.ones(grid.shape[1], dtype=np.int8)
grid_2d.shape
grid_12d = []
for i in range(len(grid_2d)):
    grid_12d.append([grid_2d[i,0], grid_2d[i,1], 0, 0,0, 0, 0, 0, 0, 0, 0, 0])

ppc = sample_proba(grid_12d, 500)

if True:
    cmap = sns.diverging_palette(250, 12, s=85, l=25, as_cmap=True)
    fig, ax = plt.subplots(figsize=(16, 9))
    contour = ax.contourf(grid[0], grid[1], ppc.mean(axis=0).reshape(100, 100), cmap=cmap)
    ax.scatter(X_test[pred == 0, 0], X_test[pred == 0, 1])
    ax.scatter(X_test[pred == 1, 0], X_test[pred == 1, 1], color="r")
    cbar = plt.colorbar(contour, ax=ax)
    _ = ax.set(xlim=(-3, 3), ylim=(-3, 3), xlabel="X", ylabel="Y")
    cbar.ax.set_ylabel("Posterior predictive mean probability of class label = 0");
    plt.show()



if True:
    cmap = sns.cubehelix_palette(light=1, as_cmap=True)
    fig, ax = plt.subplots(figsize=(16, 9))
    contour = ax.contourf(grid[0], grid[1], ppc.std(axis=0).reshape(100, 100), cmap=cmap)
    ax.scatter(X_test[pred == 0, 0], X_test[pred == 0, 1])
    ax.scatter(X_test[pred == 1, 0], X_test[pred == 1, 1], color="r")
    cbar = plt.colorbar(contour, ax=ax)
    _ = ax.set(xlim=(-3, 3), ylim=(-3, 3), xlabel="X", ylabel="Y")
    cbar.ax.set_ylabel("Uncertainty (posterior predictive standard deviation)");
    plt.show()

minibatch_x = pm.Minibatch(X_train, batch_size=50)
minibatch_y = pm.Minibatch(Y_train, batch_size=50)
neural_network_minibatch = construct_nn(minibatch_x, minibatch_y)
with neural_network_minibatch:
    approx = pm.fit(40000, method=pm.ADVI())


if True:
    plt.plot(inference.hist)
    plt.ylabel("ELBO")
    plt.xlabel("iteration");
    plt.show()



pm.traceplot(trace);


## Confusion matrix with seaborn and pandas
#cm = confusion_matrix(Y_test, y_pred)
#df_cm = pd.DataFrame(cm, index = Gs,
#                  columns = Gs)
#plt.figure(figsize = (10,7))
#plt.savefig(settings.PLOTS_PATH+name+'.eps', format='eps')
#sns.heatmap(df_cm, annot=True)

# copied here, because leapmotionlistener operates on python 2 and there is need to use the class here
class GestureDetection():
    @staticmethod
    def all(f):
        if settings.frames_adv:
            GestureDetection.processTch(f=f)
            GestureDetection.processOc(f=f)

            GestureDetection.processPose_grab(f=f)
            GestureDetection.processPose_pinch(f=f)
            GestureDetection.processPose_pointing(f=f)
            GestureDetection.processPose_respectful(f=f)
            GestureDetection.processPose_spock(f=f)
            GestureDetection.processPose_rock(f=f)
            GestureDetection.processPose_victory(f=f)
            #GestureDetection.processPose_italian()

            #GestureDetection.processGest_move_in_axis()
            #GestureDetection.processGest_rotation_in_axis()

            #GestureDetection.processComb_goToConfig()

    @staticmethod
    def processTch(f):
        fa = settings.frames_adv[f]
        if fa.r.visible:
            if fa.r.conf > settings.gd.r.MIN_CONFIDENCE:
                settings.gd.r.conf = True
            else:
                settings.gd.r.conf = False

            if fa.r.TCH12 > settings.gd.r.TCH_TURN_ON_DIST[0] and settings.gd.r.conf:
                settings.gd.r.tch12 = False
            elif fa.r.TCH12 < settings.gd.r.TCH_TURN_OFF_DIST[0]:
                settings.gd.r.tch12 = True
            if fa.r.TCH23 > settings.gd.r.TCH_TURN_ON_DIST[1] and settings.gd.r.conf:
                settings.gd.r.tch23 = False
            elif fa.r.TCH23 < settings.gd.r.TCH_TURN_OFF_DIST[1]:
                settings.gd.r.tch23 = True
            if fa.r.TCH34 > settings.gd.r.TCH_TURN_ON_DIST[2] and settings.gd.r.conf:
                settings.gd.r.tch34 = False
            elif fa.r.TCH34 < settings.gd.r.TCH_TURN_OFF_DIST[2]:
                settings.gd.r.tch34 = True
            if fa.r.TCH45 > settings.gd.r.TCH_TURN_ON_DIST[3] and settings.gd.r.conf:
                settings.gd.r.tch45 = False
            elif fa.r.TCH45 < settings.gd.r.TCH_TURN_OFF_DIST[3]:
                settings.gd.r.tch45 = True

            if fa.r.TCH13 > settings.gd.r.TCH_TURN_ON_DIST[4] and settings.gd.r.conf:
                settings.gd.r.tch13 = False
            elif fa.r.TCH13 < settings.gd.r.TCH_TURN_OFF_DIST[4]:
                settings.gd.r.tch13 = True
            if fa.r.TCH14 > settings.gd.r.TCH_TURN_ON_DIST[5] and settings.gd.r.conf:
                settings.gd.r.tch14 = False
            elif fa.r.TCH14 < settings.gd.r.TCH_TURN_OFF_DIST[5]:
                settings.gd.r.tch14 = True
            if fa.r.TCH15 > settings.gd.r.TCH_TURN_ON_DIST[6] and settings.gd.r.conf:
                settings.gd.r.tch15 = False
            elif fa.r.TCH15 < settings.gd.r.TCH_TURN_OFF_DIST[6]:
                settings.gd.r.tch15 = True

    @staticmethod
    def processOc(f):
        fa = settings.frames_adv[f]
        if fa.r.visible:
            gd = settings.gd.r
            if fa.r.conf > gd.MIN_CONFIDENCE:
                gd.conf = True
            else:
                gd.conf = False

            for i in range(0,5):
                if fa.r.OC[i] > gd.OC_TURN_ON_THRE[i] and gd.conf:
                    gd.oc[i] = True
                elif fa.r.OC[i] < gd.OC_TURN_OFF_THRE[i]:
                    gd.oc[i] = False

    @staticmethod
    def processPose_grab(f):
        fa = settings.frames_adv[f]
        if fa.l.visible:
            gd = settings.gd.l
            g = gd.poses[gd.POSES["grab"]]
            g.prob = fa.l.grab
            # gesture toggle processing
            if fa.l.grab > g.TURN_ON_THRE:
                g.toggle = True
            elif fa.l.grab < g.TURN_OFF_THRE:
                g.toggle = False
        if fa.r.visible:
            gd = settings.gd.r
            g = gd.poses[gd.POSES["grab"]]
            g.prob = fa.r.grab
            # gesture toggle processing
            if fa.r.grab > g.TURN_ON_THRE and gd.conf:
                g.toggle = True
            elif fa.r.grab < g.TURN_OFF_THRE:
                g.toggle = False

    @staticmethod
    def processPose_pinch(f):
        fa = settings.frames_adv[f]
        if fa.r.visible:
            gd = settings.gd.r
            g = gd.poses[gd.POSES["pinch"]]
            g.prob = fa.r.pinch
            if fa.r.pinch > g.TURN_ON_THRE and gd.conf:
                g.toggle = True
                g.time_visible += 0.01
            elif fa.r.pinch < g.TURN_OFF_THRE:
                g.toggle = False
                g.time_visible = 0.0

    @staticmethod
    def processPose_pointing(f):
        ''' tch, oc functions need to be called before to get fingers O/C
        '''
        fa = settings.frames_adv[f]
        if fa.r.visible:
            gd = settings.gd.r
            g = gd.poses[gd.POSES["pointing"]]
            if gd.oc[1] is True and gd.oc[2] is False and gd.oc[3] is False and gd.oc[4] is False:
                g.toggle = True
                g.time_visible += 0.01
            elif gd.oc[1] is False or gd.oc[3] is True or gd.oc[4] is True:
                g.toggle = False
                g.time_visible = 0.0

    @staticmethod
    def processPose_respectful(f):
        ''' tch, oc functions need to be called before to get fingers O/C
        '''
        fa = settings.frames_adv[f]
        if fa.r.visible:
            gd = settings.gd.r
            g = gd.poses[gd.POSES["respectful"]]
            if gd.oc[0] is False and gd.oc[1] is True and gd.oc[2] is True and gd.oc[3] is True and gd.oc[4] is False:
                g.toggle = True
                g.time_visible = 1
            elif gd.oc[0] is True or gd.oc[1] is False or gd.oc[2] is False or gd.oc[3] is False or gd.oc[4] is True:
                g.toggle = False

    @staticmethod
    def processPose_spock(f):
        ''' tch, oc functions need to be called before to get fingers O/C
        '''
        fa = settings.frames_adv[f]
        if fa.r.visible:
            gd = settings.gd.r
            g = gd.poses[gd.POSES["spock"]]
            if gd.oc[1] is True and gd.oc[2] is True and gd.oc[3] is True and gd.oc[4] is True and gd.tch23 is True and gd.tch34 is False and gd.tch45 is True:
                g.toggle = True
                g.time_visible = 1
            elif gd.oc[1] is False or gd.oc[2] is False or gd.oc[3] is False or gd.oc[4] is False or gd.tch23 is False or gd.tch34 is True or gd.tch45 is False:
                g.toggle = False

    @staticmethod
    def processPose_rock(f):
        ''' tch, oc functions need to be called before to get fingers O/C
        '''
        fa = settings.frames_adv[f]
        if fa.r.visible:
            gd = settings.gd.r
            g = gd.poses[gd.POSES["rock"]]
            if gd.oc[1] is True and gd.oc[4] is True and gd.oc[2] is False and gd.oc[3] is False:
                g.toggle = True
                g.time_visible = 1
            elif gd.oc[1] is False or gd.oc[2] is True or gd.oc[3] is True or gd.oc[4] is False:
                g.toggle = False

    @staticmethod
    def processPose_victory(f):
        ''' tch, oc functions need to be called before to get fingers O/C
        '''
        fa = settings.frames_adv[f]
        if fa.r.visible:
            gd = settings.gd.r
            g = gd.poses[gd.POSES["victory"]]
            if gd.oc[1] is True and gd.oc[2] is True and gd.oc[3] is False and gd.oc[4] is False and gd.oc[0] is False:
                g.toggle = True
                g.time_visible = 1
            elif gd.oc[1] is False or gd.oc[2] is False or gd.oc[3] is True or gd.oc[4] is True or gd.oc[0] is True:
                g.toggle = False

    @staticmethod
    def processPose_italian():
        ''' tch, oc functions need to be called before to get fingers O/C
        '''
        fa = settings.frames_adv[-1]
        if fa.r.visible:
            gd = settings.gd.r
            g = gd.poses[gd.POSES["italian"]]
            if gd.tch12 is True and gd.tch23 is True and gd.tch34 is True and gd.tch45 is True:
                g.toggle = True
                g.time_visible = 1
            elif gd.tch12 is False or gd.tch23 is False or gd.tch34 is False or gd.tch45 is False:
                g.toggle = False

    @staticmethod
    def processComb_goToConfig():
        ''' tch, oc functions need to be called before to get fingers O/C
        '''
        fa = settings.frames_adv[-1]
        g = settings.gd.r.gests[settings.gd.r.GESTS["move_in_axis"]]
        g_time = settings.gd.r.poses[settings.gd.r.POSES["pointing"]].time_visible
        if g_time > 2:
            if g.toggle[0] and g.move[0]:
                settings.WindowState = 1
            if g.toggle[0] and not g.move[0]:
                settings.WindowState = 0


    @staticmethod
    def processGest_move_in_axis():

        fa = settings.frames_adv[-1]
        if fa.r.visible:
            gd = settings.gd.r
            g = gd.gests[gd.GESTS["move_in_axis"]]
            g_tmp = deepcopy(g.toggle)
            if abs(fa.r.vel[0]) > g.MIN_THRE and fa.r.vel[1] < g.MAX_THRE and fa.r.vel[2] < g.MAX_THRE:
                g.toggle[0] = True
                g.time_visible = 1
                g.move[0] = True if fa.r.vel[0] > g.MIN_THRE else False

            else:
                g.toggle[0] = False
            if abs(fa.r.vel[1]) > g.MIN_THRE and fa.r.vel[0] < g.MAX_THRE and fa.r.vel[2] < g.MAX_THRE:
                g.time_visible = 1
                g.toggle[1] = True
                g.move[1] = True if fa.r.vel[1] > g.MIN_THRE else False

            else:
                g.toggle[1] = False
            if abs(fa.r.vel[2]) > g.MIN_THRE and fa.r.vel[0] < g.MAX_THRE and fa.r.vel[1] < g.MAX_THRE:
                g.time_visible = 1
                g.toggle[2] = True
                g.move[2] = True if fa.r.vel[2] > g.MIN_THRE else False

            else:
                g.toggle[2] = False


    @staticmethod
    def processGest_rotation_in_axis():
        '''
        '''
        fa = settings.frames_adv[-1]
        if fa.r.visible:
            euler = fa.r.pRaw[3:6]
            gd = settings.gd.r
            g = gd.gests[gd.GESTS["rotation_in_axis"]]
            g_tmp = deepcopy(g.toggle)
            if (euler[0] > g.MAX_THRE[0] or euler[0] < g.MIN_THRE[0]):
                g.time_visible = 1
                g.toggle[0] = True
                g.move[0] = True if euler[0] > g.MAX_THRE[0] else False
                if g_tmp[0] == False:
                    settings.mo.gestureGoalPoseRotUpdate(0, g.move[0])
            else:
                g.toggle[0] = False
            if (euler[1] > g.MAX_THRE[1] or euler[1] < g.MIN_THRE[1]):
                g.toggle[1] = True
                g.time_visible = 1
                g.move[1] = True if euler[1] > g.MAX_THRE[1] else False
                if g_tmp[1] == False:
                    settings.mo.gestureGoalPoseRotUpdate(1, g.move[1])
            else:
                g.toggle[1] = False
            if (euler[1] > g.MAX_THRE[2] or euler[1] < g.MIN_THRE[2]):
                g.toggle[2] = True
                g.time_visible = 1
                g.move[2] = True if euler[2] > g.MAX_THRE[2] else False
                if g_tmp[2] == False:
                    settings.mo.gestureGoalPoseRotUpdate(2, g.move[2])
            else:
                g.toggle[2] = False




def evaluate_deterministic_gestures():
    ## load the files
    Gs = ['Rotate', 'Swipe_Up','Pin', 'Touch']
    X = []
    Y = []
    ## Load data from file
    settings.LEARN_PATH = os.path.expanduser("~/"+WS_FOLDER+"/src/mirracle_gestures/include/data/person8/")
    Gs = ['Grab', 'Pinch', 'Point', 'Respectful', 'Spock', 'Rock', 'Victory']
    for n, G in enumerate(Gs):
        i = 0
        while isfile(settings.LEARN_PATH+G+"/"+str(i)+".pkl"):
            with open(settings.LEARN_PATH+G+"/"+str(i)+".pkl", 'rb') as input:
                X.append(pickle.load(input, encoding="latin1"))
                Y.append(n)
            i += 1
    if X == []: raise Exception('No data')
    ## --->>> X is array of processed frames


    def firstTrueValueIndex(pose_values):

        for n,i in enumerate(pose_values):
            if i:
                return n
        return False

    ## set the processed frame
    DETECTPOSESORGESTURES = True
    Y_pred = []
    len(X)
    len(Y)

    for i,x in enumerate(X):
        #x = X[0]
        settings.frames_adv = x
        ## generate gestures
        toggles = [0.] * 7
        for j in range(0,300):
            GestureDetection.all(f=j)
            ## take values from gest
            g = []
            if DETECTPOSESORGESTURES:
                g = settings.gd.r.poses
                for k in range(0,7):
                    if g[k].toggle:
                        toggles[k] = g[k].toggle
            else:
                g = settings.gd.r.gests

                toggles = [g[settings.gd.r.GESTS["circ"]].toggle, g[settings.gd.r.GESTS["swipe"]].toggle, g[settings.gd.r.GESTS["pin"]].toggle, g[settings.gd.r.GESTS["touch"]].toggle]
                #g[settings.gd.r.GESTS["move_in_axis"]].toggle

                #pose_values_ = [the_ress[0].toggle, the_ress[2].toggle, the_ress[3].toggle, False, False, False, False]
                #sw = the_ress[4]
                #if sw.toggle[0]:
                #    if sw.move[0] == True:
                #        pose_values_[4] = True#swipe right
                #    else:
                #        pose_values_[3] = True#swipe left
                #if sw.toggle[1]:
                #    if sw.move[1] == True:
                #        pose_values_[5] = True#swipe up
                #    else:
                #        pose_values_[6] = True#swipe down
                #pose_values = pose_values_

        #print(int(Y[i]))
        #print(toggles, Y[i], i)
        if toggles[int(Y[i])] == True:
            Y_pred.append(Y[i])

        else:
            Y_pred.append(firstTrueValueIndex(pose_values))

    len(Y_pred)
    len(Y)

    Y = np.array(Y).astype(int)
    Y_pred = np.array(Y_pred).astype(int)
    if True in Y:
        print("sda")

    Y_pred = []
    Y = []
    name = 'tmp'
    confusion_matrix_pretty_print.plot_confusion_matrix_from_data(Y, Y_pred, Gs,
      annot=True, cmap = 'Oranges', fmt='1.8f', fz=10, lw=25, cbar=False, figsize=[9,9], show_null_values=2, pred_val_axis='y', name=name)

    print("Accuracy = {}%".format((Y == Y_pred).mean() * 100))
