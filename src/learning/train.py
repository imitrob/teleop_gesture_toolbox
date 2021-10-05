#!/usr/bin/env python3
'''
This file is constructed in a way to be executable by:
    - script
    - IPython notebook

TODO:
    - ProMP Python to the README
    -
'''
# (Under this if statement):
#   - Imports
#   - Paths insert (able to import python scripts)
#   - Get gesture names list from YAML
#   - Get learn path, networks path
if True:
    import sys
    import os
    assert sys.version_info[0]==3 and sys.version_info[1]>6, "Wrong Python version (min.v.req. 3.7), it is: "+sys.version
    from os.path import expanduser

    # Get abspath and import another files in repository
    def isnotebook():
        try:
            shell = get_ipython().__class__.__name__
            if shell == 'ZMQInteractiveShell':
                return True   # Jupyter notebook or qtconsole
            elif shell == 'TerminalInteractiveShell':
                return False  # Terminal running IPython
            else:
                return False  # Other type (?)
        except NameError:
            return False      # Probably standard Python interpreter
    if isnotebook():
        WS_FOLDER = os.getcwd().split('/')[-1]
        "Notebook user, check, if this is your workspace folder"
        WS_FOLDER
        sys.path.insert(1, expanduser("~/"+WS_FOLDER+"/src/mirracle_gestures/src/learning"))
        sys.path.insert(1, expanduser("~/"+WS_FOLDER+"/src/mirracle_gestures/src"))
    if not isnotebook():
        THIS_FILE_PATH = os.path.dirname(os.path.realpath(__file__))
        THIS_FILE_TMP = os.path.abspath(os.path.join(THIS_FILE_PATH, '..', '..'))
        WS_FOLDER = THIS_FILE_TMP.split('/')[-1]

        sys.path.insert(1, expanduser("~/"+WS_FOLDER+"/src/mirracle_gestures/src/learning"))
        sys.path.insert(1, expanduser("~/"+WS_FOLDER+"/src/mirracle_gestures/src"))

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

    from timewarp_lib import *

    if os.path.isdir(expanduser("~/promps_python")):
        PROMP_ON = True
        sys.path.insert(1, os.path.expanduser("~/promps_python"))
        from promp.discrete_promp import DiscretePROMP
        from promp.linear_sys_dyn import LinearSysDyn
        from promp.promp_ctrl import PROMPCtrl
        from numpy import diff
        from numpy import linalg as LA
    if not os.path.isdir(expanduser("~/promps_python")):
        PROMP_ON = False

    rc = {'xtick.labelsize': 20, 'ytick.labelsize': 20, 'axes.labelsize': 20, 'font.size': 20,
          'legend.fontsize': 12.0, 'axes.titlesize': 10, "figure.figsize": [12, 6]}
    sns.set(rc = rc)
    sns.set_style("white")

    #%config InlineBackend.figure_format = 'retina'
    filterwarnings("ignore")

    # Set of training gestures is given by 'gesture_recording.yaml' file
    import yaml
    from collections import OrderedDict
    # function for loading dict from file ordered
    def ordered_load(stream, Loader=yaml.SafeLoader, object_pairs_hook=OrderedDict):
        class OrderedLoader(Loader):
            pass
        def construct_mapping(loader, node):
            loader.flatten_mapping(node)
            return object_pairs_hook(loader.construct_pairs(node))
        OrderedLoader.add_constructor(
            yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
            construct_mapping)
        return yaml.load(stream, OrderedLoader)

    with open(expanduser("~/"+WS_FOLDER+"/src/mirracle_gestures/include/custom_settings/gesture_recording.yaml"), 'r') as stream:
        gestures_data_loaded = ordered_load(stream, yaml.SafeLoader)

    Gs = []
    Gs_static = []
    Gs_dynamic = []
    for g in gestures_data_loaded['staticGestures'].keys():
        go = gestures_data_loaded['staticGestures'][g]
        try:
            if go['train']:
                Gs.append(g)
                Gs_static.append(g)
        except KeyError:
            pass
    for g in gestures_data_loaded['dynamicGestures'].keys():
        go = gestures_data_loaded['dynamicGestures'][g]
        try:
            if go['train']:
                Gs.append(g)
                Gs_dynamic.append(g)
        except KeyError:
            pass

    print("Gestures for training are: ", Gs)


    if gestures_data_loaded['Recognition']['args']:
        args = gestures_data_loaded['Recognition']['args']
    else:
        args = ["all_defined", "middle", "palm", '1s']#,'interpolate','normalize', 'diag_sigma_W']

    args.append('palm')
    args.append('1s')
    print("Arguments for training are: ", args)

    learn_path = expanduser('~/'+WS_FOLDER+'/src/mirracle_gestures/include/data/learning/')
    network_path = expanduser('~/'+WS_FOLDER+'/src/mirracle_gestures/include/data/Trained_network/')

    print("Learn path is: ", learn_path)
    print("Network path is: ", network_path)

    # Check and Import of ROS message types to Jupyter notebook
    try:
        import geometry_msgs
    except ModuleNotFoundError:
        sys.path.insert(1, expanduser("/opt/ros/melodic/lib/python2.7/dist-packages"))



## Import all data from learning folder
if True:
    data = import_data(learn_path, args, Gs=Gs_static)

    X = data['static']['X']
    Y = data['static']['Y']
    Xpalm = data['dynamic']['Xpalm']
    DXpalm = data['dynamic']['DXpalm']
    Ydyn = data['dynamic']['Ydyn']
    data = data['all_data']

    args
    print(X.shape)
    print(Y.shape)
    print(Xpalm.shape)
    assert len(X) == len(Y), "Lengths not match"
    #assert len(X) == len(Xpalm), "Lengths not match"
    len(Y[Y==0])
    len(Y[Y==1])
    len(Y[Y==2])


## Dynamic gestures via promp (not useful)
if False:
    X,Y = promp_lib2.get_weights(Xpalm,DXpalm,Y)
    X.shape

    promp_launch(args=args)

## Dynamic gestures via timewarping
args.append('eacheach')
time_warp_launch(Xpalm, Y, args=args)

## Split dataset
if True:
    X.shape
    Y.shape
    assert not np.any(np.isnan(X))
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3)

## Plot some class from dataset
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

## Train with given NN
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

## Load Network
if False:
    nn = NNWrapper.load_network(network_path, name='network0.pkl')
    X_train = nn.X_train
    approx = nn.approx
    neural_network = nn.neural_network
    Gs = nn.Gs

if True:
    plt.figure(figsize=(12,6))
    plt.plot(-inference.hist, label="new ADVI", alpha=0.3)
    plt.plot(approx.hist, label="old ADVI", alpha=0.3)
    plt.legend()
    plt.ylabel("ELBO")
    plt.xlabel("iteration");

## Evaluate network
if True:
    x = T.matrix("X")
    n = T.iscalar("n")
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

## Evaluate on train data
if True:
    pred_t = sample_proba(X_train,1000).mean(0)
    y_pred_t = np.argmax(pred_t, axis=1)
    X_train.shape
    Y_train.shape
    y_pred_t.shape
    print("Accuracy on train = {}%".format((Y_train == y_pred_t).mean() * 100))

    if ((Y_test == y_pred).mean() * 100) > 85:
        save_network(X_train, approx, neural_network, network_path, name=None, Gs=Gs, args=args, accuracy=((Y_train == y_pred_t).mean() * 100))

    name = 'network5'
    confusion_matrix_pretty_print.plot_confusion_matrix_from_data(Y_test, y_pred, Gs,
      annot=True, cmap = 'Oranges', fmt='1.8f', fz=12, lw=1.5, cbar=False, figsize=[9,9], show_null_values=2, pred_val_axis='y', name=name)


## Save the network
if True:
    NNWrapper.save_network(X_train, approx, neural_network, network_path=network_path, name=None, Gs=Gs, args=args, accuracy=0.9881)
    confusion_matrix_pretty_print.plot_confusion_matrix_from_data(Y_test, y_pred, Gs,
      annot=True, cmap = 'Oranges', fmt='1.8f', fz=12, lw=1.5, cbar=False, figsize=[9,9], show_null_values=2, pred_val_axis='y', name='asd')


'''
Journal:

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
#plt.savefig(plots_path+name+'.eps', format='eps')
#sns.heatmap(df_cm, annot=True)
