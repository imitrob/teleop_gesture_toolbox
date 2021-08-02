import sys
import os
from os.path import expanduser

THIS_FILE_PATH = os.path.dirname(os.path.realpath(__file__))
THIS_FILE_TMP = os.path.abspath(os.path.join(THIS_FILE_PATH, '..', '..'))
WS_FOLDER = THIS_FILE_TMP.split('/')[-1]

sys.path.insert(1, expanduser("~/"+WS_FOLDER+"/src/mirracle_gestures/src/learning"))
sys.path.insert(1, expanduser("~/"+WS_FOLDER+"/src/mirracle_gestures/src"))

import settings
settings.init()
from import_data import *

import numpy as np
import theano
floatX = theano.config.floatX
import pymc3 as pm
import theano.tensor as T
from sklearn.preprocessing import normalize
from os.path import isfile
import pickle
from copy import deepcopy
from sklearn.preprocessing import scale
import matplotlib.pyplot as plt
from pymc3.variational.callbacks import CheckParametersConvergence
import seaborn as sns
import theano



def build_network(X_train, Y_train):
    # We're initialising the weights here (the parameters we need to optimise)
    # Note that the shape of the distribution should match the dimension of the layer.
    # So, first distribution should go from X.shape[1] = 9 to 8
    initial_weights_1 = np.random.randn(X.shape[1],8).astype(floatX)
    initial_weights_2 = np.random.randn(8, 7).astype(floatX)
    initial_weights_p = np.random.randn(7).astype(floatX)

    # Initialising a model
    with pm.Model() as neural_network:
        # Denoting input data
        features = pm.Data('ann_input', X_train)
        output = pm.Data('ann_output', Y_train)

        # Denoting targets

        # We're now taking the set of parameters and assigning a prior distribution to them.
        # The pm.Normal assigns a Normal distribution with mean 0 and standard deviation 1
        prior_1 = pm.Normal('w_in_1', 0 ,  #mean
                  sigma=1, # standard deviation
                  shape=(X.shape[1], 8), #shape of set of parameters
                  testval=initial_weights_1) #initialised parameters
        prior_1

        prior_2 = pm.Normal('w_1_2', 0, sigma=1, shape=(8, 7),
                            testval=initial_weights_2)
        prior_2
        prior_perceptron = pm.Normal('w_3_out', 0, sigma=1,
                            shape=(7,), testval=initial_weights_p)
        prior_perceptron
        # Now, we'll assign the functional form of each layer
        # tanh for the first three and sigmoid for the perceptron
        layer_1 = pm.math.tanh(pm.math.dot(features, prior_1))
        layer_1
        layer_2 = pm.math.tanh(pm.math.dot(layer_1, prior_2))
        perceptron = pm.math.sigmoid( pm.math.dot(layer_2,
                                      prior_perceptron))
        layer_2
        # A bernoulli distribution as the likelihood helps model the 0,1 target data as pass/fails
        likelihood = pm.Bernoulli('out', output, observed=output,
                        total_size=Y_train.shape[0])
        likelihood

    return neural_network

## Start

X,Y = import_data(settings)
X.shape
Y.shape

if True:
    fig, ax = plt.subplots()
    ax.scatter(X[Y == 0, 0], X[Y == 0, 1], label="Class 0")
    ax.scatter(X[Y == 1, 0], X[Y == 1, 1], color="r", label="Class 1")
    sns.despine()
    ax.legend()
    ax.set(xlabel="X", ylabel="Y", title="Toy binary classification data set")

X_train, X_test, Y_train, Y_test = prepare_data(X,Y)

ann_input = theano.shared(X_train)
ann_output = theano.shared(Y)
neural_network = build_network(X_train, Y_train)
neural_network

pm.set_tt_rng(theano.sandbox.rng_mrg.MRG_RandomStream(101))

with neural_network:
    advi = pm.ADVI()
    tracker = pm.callbacks.Tracker(
        mean=advi.approx.mean.eval,  # callable that returns mean
        std=advi.approx.std.eval,  # callable that returns std
    )
    approx = pm.fit(n=200000, method=advi) #, callbacks=[tracker], obj_optimizer=pm.sgd(learning_rate=1e-3), obj_n_mc=10000, total_grad_norm_constraint=10.)
    # n is the number of iterations for ADVI
    # method is where we denote the ADVI() from PyMC3

if True:
    plt.figure(figsize=(12,6))
    #plt.plot(-inference.hist, label='new ADVI', alpha=.3)
    plt.plot(approx.hist, label='old ADVI', alpha=.3)
    plt.legend()
    plt.ylabel('ELBO')
    plt.xlabel('iteration');
    plt.plot(approx.hist)


neural_network.check_test_point()





#
