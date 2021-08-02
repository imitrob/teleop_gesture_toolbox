''' Constructing multidimensional
'''

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

from warnings import filterwarnings

import matplotlib.pyplot as plt
import numpy as np
import pymc3 as pm
import seaborn as sns
import sklearn
import theano
import theano.tensor as T

from sklearn import datasets
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import scale

rc = {'xtick.labelsize': 20, 'ytick.labelsize': 20, 'axes.labelsize': 20, 'font.size': 20,
      'legend.fontsize': 12.0, 'axes.titlesize': 10, "figure.figsize": [12, 6]}
sns.set(rc = rc)

%config InlineBackend.figure_format = 'retina'
floatX = theano.config.floatX
filterwarnings("ignore")
sns.set_style("white")

X, Y = import_data(settings, option='all_in')

X_train, X_test, Y_train, Y_test = prepare_data(X,Y)

X.shape
Y.shape

for i in range(1,6):
    fig, ax = plt.subplots()
    ax.scatter(X[Y == 0, 0], X[Y == 0, i], label="Class 0")
    ax.scatter(X[Y == 1, 0], X[Y == 1, i], color="r", label="Class "+str(i))
    sns.despine()
    ax.legend()
    ax.set(xlabel="X", ylabel="Y", title="Classification data set between two classes");

def construct_nn(ann_input, ann_output):
    ''' Model description
        -> n_hidden - number of hidden layers
        -> number of layers
        -> type of layer distribution (normal, bernoulli, uniform, poisson)
        -> distribution parameters (mu, sigma)
        -> activation function (tanh, sigmoid)

    '''
    n_hidden = 5

    # Initialize random weights between each layer
    init_0 = np.random.randn(X.shape[1], 100).astype(floatX)
    init_1 = np.random.randn(100, n_hidden).astype(floatX)
    init_2 = np.random.randn(n_hidden, n_hidden).astype(floatX)
    init_out = np.random.randn(n_hidden).astype(floatX)

    with pm.Model() as neural_network:
        # Trick: Turn inputs and outputs into shared variables using the data container pm.Data
        # It's still the same thing, but we can later change the values of the shared variable
        # (to switch in the test-data later) and pymc3 will just use the new data.
        # Kind-of like a pointer we can redirect.
        # For more info, see: http://deeplearning.net/software/theano/library/compile/shared.html
        ann_input = pm.Data("ann_input", X_train)
        ann_output = pm.Data("ann_output", Y_train)

        # Weights from input to hidden layer
        weights_in_0 = pm.Normal("w_in_0", 0, sigma=1, shape=(X.shape[1], 100), testval=init_0)
        #
        weights_in_1 = pm.Normal("w_in_1", 0, sigma=1, shape=(100, n_hidden), testval=init_1)
        # Weights from 1st to 2nd layer
        weights_1_2 = pm.Normal("w_1_2", 0, sigma=1, shape=(n_hidden, n_hidden), testval=init_2)
        # Weights from hidden layer to output
        weights_2_out = pm.Normal("w_2_out", 0, sigma=1, shape=(n_hidden,), testval=init_out)

        # Build neural-network using tanh activation function
        act_0 = pm.math.tanh(pm.math.dot(ann_input, weights_in_0))
        act_1 = pm.math.tanh(pm.math.dot(act_0, weights_in_1))
        act_2 = pm.math.tanh(pm.math.dot(act_1, weights_1_2))
        act_out = pm.math.sigmoid(pm.math.dot(act_2, weights_2_out))

        # Binary classification -> Bernoulli likelihood
        out = pm.Bernoulli(
            "out",
            act_out,
            observed=ann_output,
            total_size=Y_train.shape[0],  # IMPORTANT for minibatches
        )
    return neural_network


neural_network = construct_nn(X_train, Y_train)
neural_network


# Set random streams
pm.set_tt_rng(theano.sandbox.rng_mrg.MRG_RandomStream(101))


%time

with neural_network:
    '''
        -> inference type (ADVI)
        -> n - number of iterations
    '''
    inference = pm.ADVI()
    approx = pm.fit(n=30000, method=inference)


if True:
    plt.plot(-inference.hist, label="new ADVI", alpha=0.3)
    plt.plot(approx.hist, label="old ADVI", alpha=0.3)
    plt.legend()
    plt.ylabel("ELBO")
    plt.xlabel("iteration");


trace = approx.sample(draws=5000)

# We can get predicted probability from model
neural_network.out.distribution.p

# create symbolic input
x = T.matrix("X")
# symbolic number of samples is supported, we build vectorized posterior on the fly
n = T.iscalar("n")
# Do not forget test_values or set theano.config.compute_test_value = 'off'
x.tag.test_value = np.empty_like(X_train[:10])
n.tag.test_value = 100
_sample_proba = approx.sample_node(
    neural_network.out.distribution.p, size=n, more_replacements={neural_network["ann_input"]: x}
)
# It is time to compile the function
# No updates are needed for Approximation random generator
# Efficient vectorized form of sampling is used
sample_proba = theano.function([x, n], _sample_proba)


# Create bechmark functions
def production_step1():
    pm.set_data(new_data={"ann_input": X_test, "ann_output": Y_test}, model=neural_network)
    ppc = pm.sample_posterior_predictive(
        trace, samples=500, progressbar=False, model=neural_network
    )

    # Use probability of > 0.5 to assume prediction of class 1
    pred = ppc["out"].mean(axis=0) > 0.5


def production_step2():
    sample_proba(X_test, 500).mean(0) > 0.5



%timeit production_step1()
%timeit production_step2()

X_test.shape

# TODO: Connect to the program with learned network
pred = sample_proba(X_test, 500).mean(0) > 0.5

pred

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
    grid_12d.append([grid_2d[i,0], grid_2d[i,1], 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])

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
