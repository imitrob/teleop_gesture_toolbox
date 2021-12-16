#!/usr/bin/env python3
'''
- Recommended to save alongside with mirracle_gestures package (ROS dependent)
    > Expects source files (mirracle_gestures) in ~/<your workspace>/src/mirracle_gestures/src and ~/<your workspace>/src/mirracle_gestures/src/learning
    > Expects dataset recordings saved in ~/<your workspace>/src/mirracle_gestures/include/data/learning/
        - Download dataset from: https://drive.google.com/drive/u/0/folders/1lasIj7vPenx_ZkMyofvmro6-xtzYdyVm
    > (optional) Expects saved neural_networks in ~/<your workspace>/src/mirracle_gestures/include/data/Trained_network/
- Can be run with python3 train.py

'''

if True:
    import sys
    import os
    assert sys.version_info[0]==3 and sys.version_info[1]>6, "Wrong Python version (min.v.req. 3.7), it is: "+sys.version
    from os.path import expanduser
    def isnotebook():
        try:
            shell = get_ipython().__class__.__name__
            if shell == 'ZMQInteractiveShell': return True
            else: return False
        except NameError: return False
    if isnotebook():
        paths.ws_folder = os.getcwd().split('/')[-5]
        sys.path.append('..')
    if not isnotebook():
        THIS_FILE_PATH = os.path.dirname(os.path.realpath(__file__))
        THIS_FILE_TMP = os.path.abspath(os.path.join(THIS_FILE_PATH, '..', '..', '..', '..'))
        paths.ws_folder = THIS_FILE_TMP.split('/')[-1]

        sys.path.insert(1, expanduser("~/"+paths.ws_folder+"/src/mirracle_gestures/src/learning"))
        sys.path.insert(1, expanduser("~/"+paths.ws_folder+"/src/mirracle_gestures/src"))
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
    import time
    import argparse

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
    from scipy import stats

    from timewarp_lib import *

    # Check and Import of ROS message types to Jupyter notebook
    try:
        import geometry_msgs
    except ModuleNotFoundError:
        sys.path.insert(1, expanduser("/opt/ros/melodic/lib/python2.7/dist-packages"))

    rc = {'xtick.labelsize': 20, 'ytick.labelsize': 20, 'axes.labelsize': 20, 'font.size': 20,
          'legend.fontsize': 12.0, 'axes.titlesize': 10, "figure.figsize": [12, 6]}
    sns.set(rc = rc)
    sns.set_style("white")

    #%config InlineBackend.figure_format = 'retina'
    filterwarnings("ignore")

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


    #%matplotlib inline
    import arviz as az
    import matplotlib as mpl
    import theano.tensor as tt

    from pymc3 import Model, Normal, Slice, sample
    from pymc3.distributions import Interpolated
    from scipy import stats

    plt.style.use("seaborn-darkgrid")
    print(f"Running on PyMC3 v{pm.__version__}")

    # Initialize random number generator
    np.random.seed(0)

    learn_path = expanduser('~/'+paths.ws_folder+'/src/mirracle_gestures/include/data/learning/')
    network_path = expanduser('~/'+paths.ws_folder+'/src/mirracle_gestures/include/data/Trained_network/')

    print("Saved gestures are in folder: ", learn_path)
    print("Saved networks are in folder (for loading): ", network_path)


class PyMC3Train():

    def __init__(self, Gs=[], args={}):
        self.Gs, self.args = load_gestures_config(paths.ws_folder)
        global learn_path, network_path
        self.learn_path = learn_path
        self.network_path = network_path
        if Gs: self.Gs = Gs
        if args: self.args = args
        self.approx = None
        print("Gestures for training are: ", self.Gs)
        print("Arguments for training are: ", self.args)

    def online_thread(self):
        '''
        '''
        self.new_data_arrived = False
        self.Gs = ['open', 'close']
        self.args = {'all_defined':True, 'split':0.6, 'take_every':3, 's':1, 'iter':[2500,1000], 'n_hidden':[50], 'samples':2000}

        self.dataset_files = []
        self.import_records(dataset_files = self.dataset_files)
        self.split()
        self.train()
        self.evaluate()

        while True:
            print("LOOP STARTED")

            # read the new variables
            self.import_records(dataset_files = self.dataset_files)

            if self.new_data_arrived:
                # train the network
                self.train_update(self.X, self.Y)

                # save the weights
                self.approx

                # test on test_data
                self.evaluate()

            else:
                input()

    def import_records(self, dataset_files=[]):
        ''' Import static gestures, Import all data from learning folder
        Object (self) parameters:
            Gs_static (List): Static gesture names
            args (Str{}): Various arguments
        Object (self) returns:
            X,Y (ndarray): Static gestures dataset
            Xpalm (ndarray): Palm trajectories positions
            DYpalm (ndarray): Palm trajectories velocities
            Ydyn (1darray): (Y solutions flags for palm trajectories)
        '''
        # Takes about 50sec.
        data = import_data(self.learn_path, self.args, Gs=self.Gs, dataset_files=dataset_files)
        self.new_data_arrived = True
        if not data:
            self.new_data_arrived = False
            print("No new data!")
            return
        self.X = data['static']['X']
        self.Y = data['static']['Y']
        self.Xpalm = data['dynamic']['Xpalm']
        self.DXpalm = data['dynamic']['DXpalm']
        self.Ydyn = data['dynamic']['Y']
        self.dataset_files = data['info']['dataset_files']

        print("Gestures imported: ", self.Gs)
        print("Args used: ", self.args)
        #print("X shape: ", self.X.shape)
        #print("Y shape: ", self.Y.shape)
        #print("Xpalm shape", self.Xpalm.shape)
        assert len(self.X) == len(self.Y), "Lengths not match"
        #assert len(self.Xpalm) == len(self.Ydyn), "Lengths not match"
        print("Gesture 0 recordings: ", len(self.Y[self.Y==0]))

    def split(self):
        ''' Splits the data
        Object (self) parameters:
            X, Y (ndarray): Dataset
            args (Str{}):
                - test_size
        Object (self) returns:
            X_train, X_test, Y_train, Y_test (ndarray): Split dataset
        '''
        test_size = 0.3
        if 'split' in self.args: test_size = self.args['split']
        print("Split test_size: ", test_size)

        assert not np.any(np.isnan(self.X))
        self.X_train, self.X_test, self.Y_train, self.Y_test = train_test_split(self.X, self.Y, test_size=test_size)#, stratify=self.Y)

    def plot_dataset(self):
        ''' Plot some class from dataset
        Object (self) parameters:
            X, Y (ndarray): Dataset
        '''
        for i in range(2,3):
            fig, ax = plt.subplots()
            ax.scatter(self.X[self.Y == 0, 0], self.X[self.Y == 0, i], label="class 0")
            ax.scatter(self.X[self.Y == 1, 0], self.X[self.Y == 1, i], color="r", label="class "+str(i))
            sns.despine()
            ax.legend()
            ax.set(xlabel="X", ylabel="Y")


    def construct_nn_2l(self, X, Y, n_hidden = [10], out_n=7, update=False):
        ''' Model description
            -> n_hidden - number of hidden layers
            -> number of layers
            -> type of layer distribution (normal, bernoulli, uniform, poisson)
            -> distribution parameters (mu, sigma)
            -> activation function (tanh, sigmoid)
        '''
        # Initialize random weights between each layer
        if update:
            mus = self.approx.mean.eval()
            sigmas = self.approx.std.eval()
            mus1 = mus[0:4350]
            mus2 = mus[4350:]
            mus1=mus1.T.reshape(87,50)
            mus2=mus2.T.reshape(50,7)
            sigmas1 = sigmas[0:4350]
            sigmas2 = sigmas[4350:]
            sigmas1 = sigmas1.T.reshape(87,50)
            sigmas2 = sigmas2.T.reshape(50,7)

            mus1.astype(floatX)
            mus2.astype(floatX)
            sigmas1.astype(floatX)
            sigmas2.astype(floatX)
        else:
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
            ann_input = pm.Data("ann_input", X)
            ann_output = pm.Data("ann_output", Y)

            if update:
                # Weights from input to hidden layer
                weights_in_1 = pm.Normal("w_in_1", mus1, sigma=sigmas1, shape=(self.X.shape[1], n_hidden[0]), testval=mus1)

                # Weights from 1st to 2nd layer
                #weights_1_2 = pm.Normal("w_1_2", 0, sigma=1, shape=(n_hidden[0], n_hidden[1]), testval=init_2)

                # Weights from 1st to 2nd layer
                #weights_2_3 = pm.Normal("w_2_3", 0, sigma=1, shape=(n_hidden[1], n_hidden[2]), testval=init_3)

                # Weights from hidden layer to output
                weights_3_out = pm.Normal("w_2_out", mus2, sigma=sigmas2, shape=(n_hidden[0], out_n), testval=mus2)
            else:
                # Weights from input to hidden layer
                weights_in_1 = pm.Normal("w_in_1", 0, sigma=1, shape=(self.X.shape[1], n_hidden[0]), testval=init_1)

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
                total_size=Y.shape[0],  # IMPORTANT for minibatches
            )

        nn_string = str(neural_network)
        nn_string = nn_string.split('\n')
        print(nn_string[0]+", len ("+str(self.X_train.shape[1])+") -> ("+str(n_hidden[0])+")")
        print(nn_string[1]+", len ("+str(n_hidden[0])+") -> ("+str(out_n)+")")
        print(nn_string[2]+", len ("+str(out_n)+") -> (1)")

        return neural_network

    def construct_nn_3l(self, ann_input, ann_output, n_hidden = [10,10], out_n=7):
        init_1 = np.random.randn(self.X.shape[1], n_hidden[0]).astype(floatX)
        init_2 = np.random.randn(n_hidden[0], n_hidden[1]).astype(floatX)
        init_out = np.random.randn(n_hidden[1], out_n).astype(floatX)
        with pm.Model() as neural_network:
            ann_input = pm.Data("ann_input", self.X_train)
            ann_output = pm.Data("ann_output", self.Y_train)
            weights_in_1 = pm.Normal("w_in_1", 0, sigma=1, shape=(self.X.shape[1], n_hidden[0]), testval=init_1)
            weights_1_2 = pm.Normal("w_1_2", 0, sigma=1, shape=(n_hidden[0], n_hidden[1]), testval=init_2)
            weights_3_out = pm.Normal("w_2_out", 0, sigma=1, shape=(n_hidden[1], out_n), testval=init_out)
            act_1 = pm.math.tanh(pm.math.dot(ann_input, weights_in_1))
            act_2 = pm.math.tanh(pm.math.dot(act_1, weights_1_2))
            act_out = pm.math.sigmoid(pm.math.dot(act_2, weights_3_out))
            out = pm.Categorical(
                "out",
                act_out,
                observed=ann_output,
                total_size=self.Y_train.shape[0],  # IMPORTANT for minibatches
            )
        return neural_network

    def train(self):
        ''' Train with given NN
        Object (self) parameters:
            X_train, Y_train (ndarray): Dataset
            args (Str{}):
                - n_hidden (Int[]): layers size hidden inside NN
                - iter (Int[]): Learning number of iterations, [initial,updates]
        Object (self) returns:
            neural_network (PyMC3 NN): Model
            approx (PyMC3 obj <mean field group>): Approximation
        '''
        iter, n_hidden, inference_type = [50000], [50], 'ADVI' # DEFAULT values
        if 'iter' in self.args: iter = self.args['iter']
        if 'n_hidden' in self.args: n_hidden = self.args['n_hidden']
        if 'inference_type' in self.args: inference_type = self.args['inference_type']

        print("Iterations: ", iter[0], " number of hidden layers: ", n_hidden)
        g = list(dict.fromkeys(self.Y_train))

        self.neural_network = self.construct_nn_2l(self.X_train, self.Y_train, n_hidden=n_hidden)
        pm.set_tt_rng(theano.sandbox.rng_mrg.MRG_RandomStream(101)) # Set random seeds

        with self.neural_network:
            if inference_type == 'FullRankADVI':
                self.inference = pm.FullRankADVI()
            else: # default option 'ADVI'
                self.inference = pm.ADVI()
            tstart = time.time()

            self.approx = pm.fit(n=iter[0], method=self.inference)#, callbacks=self.clbcks)
            print("Train time [s]: ", time.time()-tstart)

    def train_update(self, X_update, Y_update):
        ''' Train with given NN update
        Object (self) parameters:
            X_train, Y_train (ndarray): Dataset
            args (Str{}):
                - n_hidden (Int[]): layers size hidden inside NN
                - iter (Int[]): Learning number of iterations, [initial,updates]
        Object (self) returns:
            neural_network (PyMC3 NN): Model
            approx (PyMC3 obj <mean field group>): Approximation
        '''
        iter, n_hidden, inference_type = [50000], [50], 'ADVI' # DEFAULT values
        if 'iter' in self.args: iter = self.args['iter']
        if 'n_hidden' in self.args: n_hidden = self.args['n_hidden']
        if 'inference_type' in self.args: inference_type = self.args['inference_type']

        X_update = np.array(X_update)
        Y_update = np.array(Y_update)

        #print("Iterations: ", iter[1], " number of hidden layers: ", n_hidden)
        #g = list(dict.fromkeys(Y_update))
        #print("X train shape: ", X_update.shape, "Y train shape: ", Y_update.shape, " X test shape: ", self.X_test.shape, " Y test shape: ", self.Y_test.shape, " counts: ", [list(np.array(Y_update,int)).count(g_) for g_ in g])

        self.neural_network = self.construct_nn_2l(X_update, Y_update, n_hidden=n_hidden, update=True)
        pm.set_tt_rng(theano.sandbox.rng_mrg.MRG_RandomStream(101)) # Set random seeds

        with self.neural_network:
            if inference_type == 'FullRankADVI':
                self.inference = pm.FullRankADVI()
            else: # default option 'ADVI'
                self.inference = pm.ADVI()
            tstart = time.time()
            self.approx = pm.fit(n=iter[1], method=self.inference, local_rv=self.approx.params)#, callbacks=[self.clbcks])
            print("Train time [s]: ", time.time()-tstart)

    def clbcks(approx, losses, i):
        if i % 500 == 499:
            print("@@@@ clbcks @@@@")
            self.approx = approx
            self.evaluate()
            print("@@@@@@@@@@@@@@@@")

    def load(self, name='network0.pkl'):
        ''' Load Network
        Object (self) returns:
            neural_network (PyMC3 NN model)
            approx (PyMC3 <mean field>): Approximation
            X_train (ndarray): Network training dataset
        '''
        nn = NNWrapper.load_network(self.network_path, name=name)
        self.X_train = nn.X_train
        self.approx = nn.approx
        self.neural_network = nn.neural_network
        self.Gs = nn.Gs
        self.args = nn.args


    def plot_train(self):
        '''
        Object (self) parameters:
            inference (PyMC3 ADVI): Inference type
            approx (PyMC3 obj): Approximation
        '''
        plt.figure(figsize=(12,6))
        #plt.plot(-self.inference.hist, label="new ADVI", alpha=0.3)
        plt.plot(self.approx.hist, label="old ADVI", alpha=0.3)
        plt.legend()
        plt.ylabel("ELBO")
        plt.xlabel("iteration")

    def evaluate(self, cutTestTo=0):
        ''' Evaluate network
        Object (self) parameters:
            X_train, Y_train (ndarray): Train datatset
            approx (PyMC3 obj): Approximation
            neural_network (PyMC3 obj): Model
            X_test, Y_test (ndarray): Test dataset
            args (Str{})
        '''
        X_test = deepcopy(self.X_test)
        Y_test = deepcopy(self.Y_test)
        if cutTestTo:
            X_test = X_test[Y_test<cutTestTo]
            Y_test = Y_test[Y_test<cutTestTo]

        samples = 10000
        if 'samples' in self.args: samples = self.args['samples']

        x = T.matrix("X")
        n = T.iscalar("n")
        x.tag.test_value = np.empty_like(self.X_train[:10])
        # IMPROVEMENT? (independent to X_train)
        #x.tag.test_value = np.zeros([10,87])
        n.tag.test_value = 100
        _sample_proba = self.approx.sample_node(
            self.neural_network.out.distribution.p, size=n, more_replacements={self.neural_network["ann_input"]: x}
        )
        sample_proba = theano.function([x, n], _sample_proba)

        pred = sample_proba(X_test,samples).mean(0)
        y_pred = np.argmax(pred, axis=1)
        Y_test = np.array(Y_test).astype(int)
        y_pred = np.array(y_pred).astype(int)
        print("Accuracy = {}%".format((Y_test == y_pred).mean() * 100))

        pred_t = sample_proba(self.X_train,samples).mean(0)
        y_pred_t = np.argmax(pred_t, axis=1)

        #print("y pred t shape: ", y_pred_t.shape)
        print("Accuracy on train = {}%".format((self.Y_train == y_pred_t).mean() * 100))

        #confusion_matrix_pretty_print.plot_confusion_matrix_from_data(Y_test, y_pred, self.Gs, annot=True, cmap = 'Oranges', fmt='1.8f', fz=12, lw=1.5, cbar=False, figsize=[9,9], show_null_values=2, pred_val_axis='y', name="")
        return (Y_test == y_pred).mean() * 100, (self.Y_train == y_pred_t).mean() * 100

    def save(self, name=None, accuracy=-1.):
        ''' Save the network
        Parameters:
            name (Str): NoneType for default
            accuracy (Float): Accuracy (on test data) saved to file model
        Object (self) parameters:
            X_train (ndarray): Dataset
            approx (PyMC3 obj): Approximation
            neural_network (PyMC3 obj): Model
            network_path (String): Path to network folder, where the network will be saved
            Gs (String[]): Array of gestures defined in model & approximation
            args (String{}): Arguments for saving
        '''
        NNWrapper.save_network(self.X_train, self.approx, self.neural_network, network_path=self.network_path, name=name, Gs=self.Gs, args=self.args, accuracy=accuracy)
        confusion_matrix_pretty_print.plot_confusion_matrix_from_data(self.Y_test, self.y_pred, self.Gs,
          annot=True, cmap = 'Oranges', fmt='1.8f', fz=12, lw=1.5, cbar=False, figsize=[9,9], show_null_values=2, pred_val_axis='y', name=str(name))

    def from_posterior(self, param, samples, k=100):
        smin, smax = np.min(samples), np.max(samples)
        width = smax - smin
        x = np.linspace(smin, smax, k)
        y = stats.gaussian_kde(samples)(x)

        # what was never sampled should have a small probability but not 0,
        # so we'll extend the domain and use linear approximation of density on it
        x = np.concatenate([[x[0] - 3 * width], x, [x[-1] + 3 * width]])
        y = np.concatenate([[0], y, [0]])
        return pm.Interpolated(param, x, y)
        # ?
        # Y1 = np.random.normal(loc=57, scale=5.42, size=100)

    def extract_approx(self):
        ''' Approximation -> params weights/values
        Object (self) parameters:
            approx (PyMC3 obj): Approximation
        Object (self) returns:
            μ, ρ (ndarray): Mean field values
        '''
        print("approx param 0 name: ", self.approx.params[0])
        print("approx param 1 name: ", self.approx.params[1])

        self.μ = approx.bij.rmap(self.approx.params[0].eval())
        print("w_in_1 μ shape: ", self.μ['w_in_1'].shape)
        print("w_2_out μ shape: ", self.μ['w_2_out'].shape)
        self.ρ = approx.bij.rmap(approx.params[1].eval())
        print("w_in_1 ρ shape: ", self.ρ['w_in_1'].shape)
        print("w_2_out ρ shape: ", self.ρ['w_2_out'].shape)


    def benchmark(self):
        ''' Create bechmark functions, BETA
        '''
        trace = self.approx.sample(draws=5000)
        def production_step1():
            pm.set_data(new_data={"ann_input": self.X_test, "ann_output": self.Y_test}, model=self.neural_network)
            ppc = pm.sample_posterior_predictive(
                trace, samples=500, progressbar=False, model=self.neural_network
            )
            ppc['out'].shape


        pred = sample_proba(self.X_test, 500).mean(0)
        pred = np.argmax(pred, axis=1)

        fig, ax = plt.subplots()
        ax.scatter(self.X_test[self.pred == 0, 0], self.X_test[self.pred == 0, 1])
        ax.scatter(self.X_test[self.pred == 1, 0], self.X_test[self.pred == 1, 1], color="r")
        sns.despine()
        ax.set(title="Predicted labels in testing set", xlabel="X", ylabel="Y");

        print("Accuracy = {}%".format((self.Y_test == pred).mean() * 100))

        ## THERE IS PROBLEM about what vars will I compare and plot
        grid = pm.floatX(np.mgrid[-3:3:100j, -3:3:100j])
        grid_2d = grid.reshape(2, -1).T
        dummy_out = np.ones(grid.shape[1], dtype=np.int8)
        grid_2d.shape
        grid_12d = []
        for i in range(len(grid_2d)):
            grid_12d.append([grid_2d[i,0], grid_2d[i,1], 0, 0,0, 0, 0, 0, 0, 0, 0, 0])

        ppc = sample_proba(grid_12d, 500)

        cmap = sns.diverging_palette(250, 12, s=85, l=25, as_cmap=True)
        fig, ax = plt.subplots(figsize=(16, 9))
        contour = ax.contourf(grid[0], grid[1], ppc.mean(axis=0).reshape(100, 100), cmap=cmap)
        ax.scatter(X_test[pred == 0, 0], X_test[pred == 0, 1])
        ax.scatter(X_test[pred == 1, 0], X_test[pred == 1, 1], color="r")
        cbar = plt.colorbar(contour, ax=ax)
        _ = ax.set(xlim=(-3, 3), ylim=(-3, 3), xlabel="X", ylabel="Y")
        cbar.ax.set_ylabel("Posterior predictive mean probability of class label = 0");
        plt.show()

        cmap = sns.cubehelix_palette(light=1, as_cmap=True)
        fig, ax = plt.subplots(figsize=(16, 9))
        contour = ax.contourf(grid[0], grid[1], ppc.std(axis=0).reshape(100, 100), cmap=cmap)
        ax.scatter(X_test[pred == 0, 0], X_test[pred == 0, 1])
        ax.scatter(X_test[pred == 1, 0], X_test[pred == 1, 1], color="r")
        cbar = plt.colorbar(contour, ax=ax)
        _ = ax.set(xlim=(-3, 3), ylim=(-3, 3), xlabel="X", ylabel="Y")
        cbar.ax.set_ylabel("Uncertainty (posterior predictive standard deviation)");
        plt.show()


    def train_minibatch(self):
        ''' Train minibatches
        Global parameters:
            X_train, Y_train (ndarray): Train dataset
        '''
        iter, n_hidden, inference_type = [50000], [50], 'ADVI' # DEFAULT values
        if 'iter' in self.args: iter = self.args['iter']
        if 'n_hidden' in self.args: n_hidden = self.args['n_hidden']
        if 'inference_type' in self.args: inference_type = self.args['inference_type']
        # batch_size=50, time->20s
        #
        minibatch_x = pm.Minibatch(self.X_train, batch_size=50)
        minibatch_y = pm.Minibatch(self.Y_train, batch_size=50)
        self.neural_network = self.construct_nn_2l(minibatch_x, minibatch_y,n_hidden=n_hidden)
        tstart = time.time()

        with self.neural_network:
            self.inference = pm.ADVI()
            self.approx = pm.fit(iter[0], method=inference_type)
        print("Train time [s]: ", time.time()-tstart)

        plt.plot(self.approx.hist)
        plt.ylabel("ELBO")
        plt.xlabel("iteration");
        plt.show()

    def train_dynamic_promp(self):
        ''' Dynamic gestures via promp (not useful)
        '''

        self.X, self.Y = promp_lib2.get_weights(self.Xpalm,self.DXpalm,self.Ydyn)
        print("X shape", X.shape)

        promp_launch(args=self.args)

    def train_dynamic_timewarping(self):
        ''' Dynamic gestures via timewarping
        '''
        self.args['eacheach'] = True
        time_warp_launch(self.Xpalm, self.Y, args=self.args)


    ''' ============ continuous learning section =============
        ====================================================== '''
    def split_k_parts(self, split=[0.3, 0.1]):
        ''' Split dataset into three parts
        Parameters:
            split (Float[]): Portions of dataset divided into:
                - split[0] -> X_test, Y_test
                - split[1] -> X_train, Y_train
                - split[2:] -> X_update[], Y_update[]
        Object (self) parameters:
            X,Y (ndarray): Dataset
        Object (self) returns:
            X_train, Y_train, X_test, Y_test, X_update[], X_update[] (ndarray): Splitted dataset based on proportions
        '''
        print("X shape", self.X.shape)
        print("Y shape", self.Y.shape)
        assert not np.any(np.isnan(self.X))
        # Split to test/train data
        self.X_train, self.X_test, self.Y_train, self.Y_test = train_test_split(self.X, self.Y, test_size=split.pop(0))
        self.X_update, self.Y_update = [[]] * len(split), [[]] * len(split)
        for i in range(0, len(split)):
            if i == 0:
                self.X_train, self.X_update[0], self.Y_train, self.Y_update[0] = train_test_split(self.X_train, self.Y_train, test_size=split[i])
                continue
            self.X_update[i], self.X_update[i+1], self.Y_update[i], self.Y_update[i+1] = train_test_split(self.X_update[i], self.Y_update[i], test_size=split[i])

    def split_4_parts(self):
        ''' Split dataset into proportions: 0.3 test part + 0.2 train data + 4*0.125 updates data
        Object (self) parameters:
            X,Y (ndarray): Dataset
        Object (self) returns:
            X_train, Y_train, X_test, Y_test, X_update, Y_update (ndarray): Splitted dataset
        '''
        assert not np.any(np.isnan(self.X))
        self.X_train, self.X_test, self.Y_train, self.Y_test = train_test_split(self.X, self.Y, test_size=0.3)#, stratify=self.Y)
        self.X_update, self.Y_update = [[]] * 4, [[]] * 4

        self.X_train, self.X_tmp, self.Y_train, self.Y_tmp = train_test_split(self.X_train, self.Y_train, test_size=0.715)#, stratify=self.Y_train)

        self.X_update[0], self.X_update[1], self.Y_update[0], self.Y_update[1] = train_test_split(self.X_tmp, self.Y_tmp, test_size=0.5)#, stratify=self.Y_train)

        self.X_update[2], self.X_update[3], self.Y_update[2], self.Y_update[3] = train_test_split(self.X_update[0], self.Y_update[0], test_size=0.5)#, stratify=self.Y4[0])
        self.X_update[0], self.X_update[1], self.Y_update[0], self.Y_update[1] = train_test_split(self.X_update[1], self.Y_update[1], test_size=0.5)#, stratify=self.Y4[1])

    def split_k_parts_mini(self, N_RECORDINGS_FOR_EACH_LEARNING=[1,2,5,10,20,50]):
        '''
        Object (self) parameters:
            X,Y (ndarray): Dataset
        Object (self) returns:
            X_train, Y_train, X_test, Y_test, X_train_update, Y_train_update (ndarray): Splitted dataset
        '''
        assert not np.any(np.isnan(self.X))
        self.X_train, self.X_test, self.Y_train, self.Y_test = train_test_split(self.X, self.Y, test_size=0.3, stratify=self.Y)
        self.X_update, self.Y_update = [[]] * len(N_RECORDINGS_FOR_EACH_LEARNING), [[]] * len(N_RECORDINGS_FOR_EACH_LEARNING)

        for n,e in enumerate(N_RECORDINGS_FOR_EACH_LEARNING):
            self.X_update[n] = []
            sum = 0
            for i in range(0,7):
                 self.X_update[n].append(self.X_train[self.Y_train==i][sum:sum+e])
            self.X_update[n] = np.vstack(self.X_update[n])
            self.Y_update[n] = np.repeat([0,1,2,3,4,5,6], e)
            sum += e
        self.X_train = self.X_update.pop(0)
        self.Y_train = self.Y_update.pop(0)

    def split_43_parts_eachgesture(self):
        '''
        Object (self) parameters:
            X,Y (ndarray): Dataset
        Object (self) returns:
            X_train, Y_train, X_test, Y_test, X_train_update, Y_train_update (ndarray): Splitted dataset
        '''
        assert not np.any(np.isnan(self.X))
        self.X_train, self.X_test, self.Y_train, self.Y_test = train_test_split(self.X, self.Y, test_size=0.3, stratify=self.Y)

        self.X_update, self.Y_update = [[],[]], [[],[]]
        for n in range(0,len(self.X_train)):

            if self.Y_train[n] == 0:
                self.X_update[0].append(deepcopy(self.X_train[n]))
                self.Y_update[0].append(deepcopy(self.Y_train[n]))
            elif self.Y_train[n] == 1:
                self.X_update[0].append(deepcopy(self.X_train[n]))
                self.Y_update[0].append(deepcopy(self.Y_train[n]))
            elif self.Y_train[n] == 2:
                self.X_update[0].append(deepcopy(self.X_train[n]))
                self.Y_update[0].append(deepcopy(self.Y_train[n]))
            elif self.Y_train[n] == 3:
                self.X_update[0].append(deepcopy(self.X_train[n]))
                self.Y_update[0].append(deepcopy(self.Y_train[n]))
            elif self.Y_train[n] == 4:
                self.X_update[1].append(deepcopy(self.X_train[n]))
                self.Y_update[1].append(deepcopy(self.Y_train[n]))
            elif self.Y_train[n] == 5:
                self.X_update[1].append(deepcopy(self.X_train[n]))
                self.Y_update[1].append(deepcopy(self.Y_train[n]))
            elif self.Y_train[n] == 6:
                self.X_update[1].append(deepcopy(self.X_train[n]))
                self.Y_update[1].append(deepcopy(self.Y_train[n]))
            else: raise Exception("Something is wrong")

        self.X_train = self.X_update.pop(0)
        self.Y_train = self.Y_update.pop(0)
        self.X_train = np.array(self.X_train)
        self.Y_train = np.array(self.Y_train)

    def split_322_parts_eachgesture(self):
        '''
        Object (self) parameters:
            X,Y (ndarray): Dataset
        Object (self) returns:
            X_train, Y_train, X_test, Y_test, X_train_update, Y_train_update (ndarray): Splitted dataset
        '''
        assert not np.any(np.isnan(self.X))
        self.X_train, self.X_test, self.Y_train, self.Y_test = train_test_split(self.X, self.Y, test_size=0.3, stratify=self.Y)

        self.X_update, self.Y_update = [[],[],[]], [[],[],[]]
        for n in range(0,len(self.X_train)):

            if self.Y_train[n] == 0:
                self.X_update[0].append(deepcopy(self.X_train[n]))
                self.Y_update[0].append(deepcopy(self.Y_train[n]))
            elif self.Y_train[n] == 1:
                self.X_update[0].append(deepcopy(self.X_train[n]))
                self.Y_update[0].append(deepcopy(self.Y_train[n]))
            elif self.Y_train[n] == 2:
                self.X_update[0].append(deepcopy(self.X_train[n]))
                self.Y_update[0].append(deepcopy(self.Y_train[n]))
            elif self.Y_train[n] == 3:
                self.X_update[1].append(deepcopy(self.X_train[n]))
                self.Y_update[1].append(deepcopy(self.Y_train[n]))
            elif self.Y_train[n] == 4:
                self.X_update[1].append(deepcopy(self.X_train[n]))
                self.Y_update[1].append(deepcopy(self.Y_train[n]))
            elif self.Y_train[n] == 5:
                self.X_update[2].append(deepcopy(self.X_train[n]))
                self.Y_update[2].append(deepcopy(self.Y_train[n]))
            elif self.Y_train[n] == 6:
                self.X_update[2].append(deepcopy(self.X_train[n]))
                self.Y_update[2].append(deepcopy(self.Y_train[n]))
            else: raise Exception("Something is wrong")

        self.X_train = self.X_update.pop(0)
        self.Y_train = self.Y_update.pop(0)
        self.X_train = np.array(self.X_train)
        self.Y_train = np.array(self.Y_train)

    def prepare_minibatch(self, n_samples=10):
        ''' Dataset -> distributions -> New dataset sampled from these distributions
            Why? Makes less recordings, used for minibatch training
        Parameters:
            n_samples (Int): Number of sampled recordings
        Global parameters:
            X, Y (ndarray): Dataset
        Global returns:
            X, Y (ndarray): New dataset
        '''
        g = list(dict.fromkeys(self.Y))
        counts = [list(self.Y).count(g_) for g_ in g]

        X_new = []
        for n in range(0,len(counts)):
            x_g_samples = self.X[self.Y==n,:].T
            row = []
            for x_sample in x_g_samples:
                mu = x_sample.mean()
                sigma = x_sample.std()

                nn = np.random.normal(mu, sigma, n_samples)
                row.append(nn)
            X_new.extend(np.array(row).T)
        X_new = np.array(X_new)

        Y_new = np.repeat([g],(n_samples))

        self.X = X_new
        self.Y = Y_new

    '''
    X = np.empty([0,87])
    a = np.empty([50,87])
    X
    b = np.vstack([b,a])
    b.shape'''
    ''' debug

    X = np.array([ [1,2], [3,4], [5,6], [6,7] ])
    Y = np.array([0,0,1,1])
    X_update = np.array([ [10,11], [11,12] ])
    Y_update = np.array([3,3])
    prepare_batch(0, X, Y, X_update, Y_update, batch_size=3) '''
    def prepare_batch(self, X, Y, X_update, Y_update, batch_size=50):
        ''' Scales down the number of samples to more relevant ones, with size as batch_size
        '''
        # join datasets
        X = np.vstack([X, X_update])
        Y = np.hstack([Y, Y_update])

        X_new = np.empty([0,len(X[0])])
        # roztridit samply na jednotliva gesta
        IDs = list(dict.fromkeys(Y)) # [0,1,2,..,<number of gestures>]
        for key in IDs:
            Xkey = X[Y==key]

            # ceil to batch_size samples
            Xkey = self.shrink_to_more_relevant(Xkey, batch_size)
            X_new = np.vstack([X_new, Xkey]) # or np.hstack ?

        Y_new = np.repeat([IDs],(batch_size))
        return X_new, Y_new

    #debug
    #Xkey = np.array([[1,2],[2,3],[3,4],[4,5],[5,6],[6,7]])
    #shrink_to_more_relevant(0, Xkey, 5)
    def shrink_to_more_relevant(self, Xkey, batch_size):
        '''
        '''
        Xkey = Xkey.T # shape [rec x 87] -> [87 x rec]
        X_new = np.empty([0,batch_size])
        for feature in Xkey: # for every feature (87)
            samples = np.random.normal(feature.mean(), feature.std(), batch_size)
            #samples2 = np.random.normal(feature.mean(), feature.std(), batch_size*10)
            #samples = []
            #for i in range(0,len(samples2),10):
            #    samples.append(np.mean(samples2[i:i+10]))
            X_new = np.vstack([X_new, np.array(samples)])
        return X_new.T


    def continuous_updates(self):
        ''' NOT WORKING
        '''
        # True parameter values
        alpha_true = 5
        beta0_true = 7
        beta1_true = 13

        # Size of dataset
        size = 100

        # Predictor variable
        X1 = np.random.randn(size)
        X2 = np.random.randn(size) * 0.2

        # Simulate outcome variable
        Y = alpha_true + beta0_true * X1 + beta1_true * X2 + np.random.randn(size)

        basic_model = Model()

        with basic_model:

            # Priors for unknown model parameters
            alpha = Normal("alpha", mu=0, sigma=1)
            beta0 = Normal("beta0", mu=12, sigma=1)
            beta1 = Normal("beta1", mu=18, sigma=1)

            # Expected value of outcome
            mu = alpha + beta0 * X1 + beta1 * X2

            # Likelihood (sampling distribution) of observations
            Y_obs = Normal("Y_obs", mu=mu, sigma=1, observed=Y)

            # draw 1000 posterior samples
            trace = sample(1000)

        az.plot_trace(trace);




        traces = [trace]
        trace


        for _ in range(10):

            # generate more data
            X1 = np.random.randn(size)
            X2 = np.random.randn(size) * 0.2
            Y = alpha_true + beta0_true * X1 + beta1_true * X2 + np.random.randn(size)

            model = Model()
            with model:
                # Priors are posteriors from previous iteration
                alpha = from_posterior("alpha", trace["alpha"])
                beta0 = from_posterior("beta0", trace["beta0"])
                beta1 = from_posterior("beta1", trace["beta1"])

                # Expected value of outcome
                mu = alpha + beta0 * X1 + beta1 * X2

                # Likelihood (sampling distribution) of observations
                Y_obs = Normal("Y_obs", mu=mu, sigma=1, observed=Y)

                # draw 10000 posterior samples
                trace = sample(1000)
                traces.append(trace)



        print("Posterior distributions after " + str(len(traces)) + " iterations.")
        cmap = mpl.cm.autumn
        for param in ["alpha", "beta0", "beta1"]:
            plt.figure(figsize=(8, 2))
            for update_i, trace in enumerate(traces):
                samples = trace[param]
                smin, smax = np.min(samples), np.max(samples)
                x = np.linspace(smin, smax, 100)
                y = stats.gaussian_kde(samples)(x)
                plt.plot(x, y, color=cmap(1 - update_i / len(traces)))
            plt.axvline({"alpha": alpha_true, "beta0": beta0_true, "beta1": beta1_true}[param], c="k")
            plt.ylabel("Frequency")
            plt.title(param)

        plt.tight_layout();


class Experiments():
    def seed_wrapper(self, fun=None, SEEDS=[93457, 12345, 45677, 82909, 75433]):
        print("------------")
        print("Seed Wrapper")
        print("------------")
        # Random seeds, specified for next repeatability
        self.accuracies, self.accuracies_train = [], []
        for n,seed in enumerate(SEEDS):
            print(str(n+1)+". seed: "+str(seed))
            accuracies_row, accuracies_train_row = [], []
            np.random.seed(seed)
            accuracies_row, accuracies_train_row = fun()
            self.accuracies.append(accuracies_row)
            self.accuracies_train.append(accuracies_train_row)
        print("Accuracies test: ", self.accuracies)
        print("Accuracies train: ", self.accuracies_train)

    def loadAndEvaluate(self):
        ''' Loads network file and evaluate it
        '''
        accuracies, accuracies_train = [], []
        print("Load and evaluate")
        print("---")
        self.train = PyMC3Train()
        self.train.import_records()
        self.train.split()
        self.train.load()
        self.train.import_records()
        self.train.split()

        accuracy, accuracy_train = self.train.evaluate()
        accuracies.append(accuracy)
        accuracies_train.append(accuracy_train)

        print("accuracies.append(", accuracies, ")")
        print("accuracies_train.append(", accuracies_train, ")")
        return accuracy, accuracy_train

    def trainWithParameters(self):
        ''' Train/evaluate, use for single training
        '''
        Gs = ['grab', 'pinch', 'point', 'respectful', 'spock', 'rock', 'victory']
        args = {'all_defined':True, 'split':0.3, 'take_every':10, 'iter':[2000], 'n_hidden':[50] }

        accuracies, accuracies_train = [], []
        print("Training With Parameters")
        print("---")
        self.train = PyMC3Train(Gs, args)

        self.train.import_records()
        self.train.split()

        self.train.train()
        accuracy, accuracy_train = self.train.evaluate()

        self.train.plot_train()
        print("accuracies.append(", accuracies, ")")
        print("accuracies_train.append(", accuracies_train, ")")
        return accuracies, accuracies_train

    def train90plus10_priors(self):
        ''' Train on 90% data then update on 10% new data, method update with priors
        '''
        Gs = ['grab', 'pinch', 'point', 'respectful', 'spock', 'rock', 'victory']
        args = {'all_defined':True, 'split':0.25, 'take_every':10, 's':1, 'iter':[2500,1000], 'n_hidden':[50], 'samples':2000}

        accuracies, accuracies_train = [], []
        print("Train on 90% data then update on 10% new data, method update with priors")
        print("---")
        self.train = PyMC3Train(Gs, args)
        self.train.import_records()
        self.train.split_k_parts()

        self.train.train()
        accuracy, accuracy_train = self.train.evaluate()
        accuracies.append(accuracy)
        accuracies_train.append(accuracy_train)

        # it needs Y_train same length, 5800 before, 1100 in update
        self.train.train_update(self.train.X_update, self.train.Y_update)
        accuracy, accuracy_train = self.train.evaluate()
        accuracies.append(accuracy)
        accuracies_train.append(accuracy_train)

        print("accuracies.append(", accuracies, ")")
        print("accuracies_train.append(", accuracies_train, ")")
        return accuracies, accuracies_train

    def train43(self):
        ''' Train first on 4 gestures, then update on 3 new gestures dataset, method independent dataset generalization
        '''
        Gs = ['grab', 'pinch', 'point', 'respectful', 'spock', 'rock', 'victory']
        args = {'all_defined':True, 'split':0.25, 'take_every':10, 's':1, 'iter':[1000,1000], 'n_hidden':[50], 'samples':2000}

        accuracies, accuracies_train = [], []
        print("Train first on 4 gestures, then update on 3 new gestures dataset, method independent dataset generalization")
        print("---")
        self.train = PyMC3Train(Gs, args)
        self.train.import_records()
        self.train.split_43_parts_eachgesture()

        self.train.train()
        accuracy, accuracy_train = self.train.evaluate(cutTestTo=4)
        accuracies.append(accuracy)
        accuracies_train.append(accuracy_train)

        self.train.X_train, self.train.Y_train = self.train.prepare_batch(self.train.X_train, self.train.Y_train, self.train.X_update[0], self.train.Y_update[0], batch_size=50)
        self.train.train()
        accuracy, accuracy_train = self.train.evaluate()
        accuracies.append(accuracy)
        accuracies_train.append(accuracy_train)

        print("accuracies.append(", accuracies, ")")
        print("accuracies_train.append(", accuracies_train, ")")
        return accuracies, accuracies_train

    def train43update(self):
        ''' Train first on 4 gestures, then update on 3 new gestures dataset, update method and dataset generalization
        '''
        Gs = ['grab', 'pinch', 'point', 'respectful', 'spock', 'rock', 'victory']
        args = {'all_defined':True, 'split':0.25, 'take_every':10, 's':1, 'iter':[2500,2500], 'n_hidden':[50], 'samples':2000}

        accuracies, accuracies_train = [], []
        print("Train first on 4 gestures, then update 3 new gestures dataset, update method and dataset generalization")
        print("---")
        self.train = PyMC3Train(Gs, args)
        self.train.import_records()
        self.train.split_43_parts_eachgesture()

        self.train.train()
        accuracy, accuracy_train = self.train.evaluate(cutTestTo=4)
        accuracies.append(accuracy)
        accuracies_train.append(accuracy_train)

        self.train.train_update(self.train.X_update[0], self.train.Y_update[0])
        accuracy, accuracy_train = self.train.evaluate()
        accuracies.append(accuracy)
        accuracies_train.append(accuracy_train)

        print("accuracies.append(", accuracies, ")")
        print("accuracies_train.append(", accuracies_train, ")")
        return accuracies, accuracies_train

    def train322(self):
        ''' Train first on 3 gestures, then update on 2+2 new gesture dataset, method independent dataset generalization
        '''
        Gs = ['grab', 'pinch', 'point', 'respectful', 'spock', 'rock', 'victory']
        args = {'all_defined':True, 'split':0.25, 'take_every':10, 's':1, 'iter':[1000,1000,1000], 'n_hidden':[50], 'samples':2000}

        accuracies, accuracies_train = [], []
        print("Train first on 3 gestures, then update on 2+2 new gesture dataset, method independent dataset generalization")
        print("---")
        self.train = PyMC3Train(Gs, args)
        self.train.import_records()
        self.train.split_322_parts_eachgesture()

        self.train.train()
        accuracy, accuracy_train = self.train.evaluate(cutTestTo=3)
        accuracies.append(accuracy)
        accuracies_train.append(accuracy_train)

        self.train.X_train, self.train.Y_train = self.train.prepare_batch(self.train.X_train, self.train.Y_train, self.train.X_update[0], self.train.Y_update[0], batch_size=50)
        self.train.train()
        accuracy, accuracy_train = self.train.evaluate(cutTestTo=5)
        accuracies.append(accuracy)
        accuracies_train.append(accuracy_train)

        self.train.X_train, self.train.Y_train = self.train.prepare_batch(self.train.X_train, self.train.Y_train, self.train.X_update[1], self.train.Y_update[1], batch_size=50)
        self.train.train()
        accuracy, accuracy_train = self.train.evaluate()
        accuracies.append(accuracy)
        accuracies_train.append(accuracy_train)

        print("accuracies.append(", accuracies, ")")
        print("accuracies_train.append(", accuracies_train, ")")
        return accuracies, accuracies_train

    def trainDataProportion(self, PROPORTION_TRAIN_DATA=[0.9,0.7,0.5,0.3,0.1]):
        ''' Grid search for parameter: proportion of train data (still tested with 30% test data each time), method independent chunks of train
        '''
        Gs = ['grab', 'pinch', 'point', 'respectful', 'spock', 'rock', 'victory']
        args = {'all_defined':True, 'take_every':10, 's':1, 'interpolate':True, 'iter':[3000]}

        accuracies, accuracies_train = [], []
        print("Grid search for parameter: proportion of train data (still tested with 30% test data each time), method independent chunks of train")
        print("---")
        for i in PROPORTION_TRAIN_DATA:
            args['split'] = [0.3, i]

            self.train = PyMC3Train(Gs, args)
            self.train.import_records()
            self.train.split_k_parts(args['split'])
            self.train.train()
            accuracy, accuracy_train = self.train.evaluate()
            accuracies.append(accuracy)
            accuracies_train.append(accuracy_train)
            print("---")
        print("accuracies.append(", accuracies, ")")
        print("accuracies_train.append(", accuracies_train, ")")
        return accuracies, accuracies_train

    def train_iterations(self, ITERATIONS=[1000, 3000, 4000, 5000, 7000, 9000]):
        ''' Grid search for parameter: iterations, method independent chunks of train
        '''
        Gs = ['grab', 'pinch', 'point', 'respectful', 'spock', 'rock', 'victory']
        args = {'all_defined':True, 'take_every':3, 's':1, 'interpolate':True, 'split':0.25, 'samples':10000 }

        accuracies, accuracies_train = [], []
        print("Grid search with parameter: iterations, method independent chunks of train")
        print("---")
        for i in ITERATIONS:
            args['iter'] = [i]

            self.train = PyMC3Train(Gs, args)
            self.train.import_records()
            self.train.split_k_parts()
            self.train.train()
            accuracy, accuracy_train = self.train.evaluate()
            accuracies.append(accuracy)
            accuracies_train.append(accuracy_train)
        print("accuracies.append(", accuracies , ")")
        print("accuracies_train.append(", accuracies_train, ")")
        return accuracies, accuracies_train

    def trainAddingkupdates(self, N_RECORDINGS_FOR_EACH_LEARNING=[1,2,5,10,20,50]):
        ''' Grid search with parameter: n recordings for each learning (MAIN_ARG_N_RECORDINGS_FOR_EACH_LEARNING), method update with priors
        '''
        Gs = ['grab', 'pinch', 'point', 'respectful', 'spock', 'rock', 'victory']
        args = { 'all_defined':True, 'take_every':10, 's':1, 'interpolate':True, 'iter':[3000,3000], 'split':0.3, 'n_hidden':[50] }

        accuracies, accuracies_train = [], []
        print("Grid search with parameter: n recordings for each learning (MAIN_ARG_N_RECORDINGS_FOR_EACH_LEARNING), method update with priors")
        print("---")
        self.train = PyMC3Train(Gs, args)
        self.train.import_records()
        self.train.split_k_parts_mini(N_RECORDINGS_FOR_EACH_LEARNING=MAIN_ARG_N_RECORDINGS_FOR_EACH_LEARNING)
        self.train.train()
        accuracy, accuracy_train = self.train.evaluate()
        accuracies.append(accuracy)
        accuracies_train.append(accuracy_train)
        print("---")
        for i in range(0,5):
            self.train.train_update(self.train.X_update[i], self.train.Y_update[i])
            accuracy, accuracy_train = self.train.evaluate()
            accuracies.append(accuracy)
            accuracies_train.append(accuracy_train)
            print("---")
        print("accuracies.append(", accuracies, ")")
        print("accuracies_train.append(", accuracies_train, ")")
        return accuracies, accuracies_train

    def trainAdding4updates(self):
        ''' Train first on 25% train data and then three times update on 25% train data (tested on previously splited 30% test data), method update with priors
        '''
        Gs = ['grab', 'pinch', 'point', 'respectful', 'spock', 'rock', 'victory']
        args = {'all_defined':True, 'take_every':10, 's':1, 'interpolate':True, 'iter':[3000,3000], 'split':0.3, 'n_hidden':[50] }

        accuracies, accuracies_train = [], []
        print("Train first on 25% train data and then three times update on 25% train data (tested on previously splited 30% test data) with priors")
        print("---")
        self.train = PyMC3Train(Gs, args)
        self.train.import_records()
        self.train.split_4_parts()
        self.train.train()
        accuracy, accuracy_train = self.train.evaluate()
        accuracies.append(accuracy)
        accuracies_train.append(accuracy_train)
        print("---")
        for i in range(0,4):
            self.train.train_update(self.train.X_update[i], self.train.Y_update[i])
            accuracy, accuracy_train = self.train.evaluate()
            accuracies.append(accuracy)
            accuracies_train.append(accuracy_train)
            print("---")
        print("accuracies.append(", accuracies, ")")
        print("accuracies_train.append(", accuracies_train, ")")
        return accuracies, accuracies_train

    # NOT WORKING RIGHT NOW
    def train90plus10_minibatch(self):
        ''' Train first on 90% data and then update on 10% (tested on previously splited 30% test data), method update with minibatch
        '''
        Gs = ['grab', 'pinch', 'point', 'respectful', 'spock', 'rock', 'victory']
        args = { 'all_defined':True, 'take_every':10, 's':1, 'iter':[2000,2000], 'n_hidden':[50], 'samples':1000}

        accuracies, accuracies_train = [], []
        print("Train first on 90% data and then update on 10% (tested on previously splited 30% test data), method update with minibatch")
        print("---")
        self.train = PyMC3Train(Gs, args)
        self.train.import_records()
        self.train.split_k_parts()
        self.train.prepare_minibatch()

        self.train.train_minibatch()

        accuracy, accuracy_train = self.train.evaluate()
        accuracies.append(accuracy)
        accuracies_train.append(accuracy_train)

        self.train.train_update(train.X_update[0], train.Y_update[0])
        accuracy, accuracy_train = self.train.evaluate()
        accuracies.append(accuracy)
        accuracies_train.append(accuracy_train)
        print("accuracies.append(", accuracies, ")")
        print("accuracies_train.append(", accuracies_train, ")")
        return accuracies, accuracies_train

    def trainMinibatches(self):
        ''' MiniBatches
        State: Not working, Need to update
        '''
        Gs = ['grab', 'pinch', 'point', 'respectful', 'spock', 'rock', 'victory']
        args = { 'all_defined':True, 'take_every':10 }

        accuracies, accuracies_train = [], []
        print("Training 17.5% + 3 x 17.5% with priors")
        print("---")
        self.train = PyMC3Train(Gs, args)
        self.train.import_records()
        self.train.split_4_parts()
        for i in range(0,4):
            self.train.train_minibatch()
            self.train.evaluate()

        print("accuracies.append(", accuracies, ")")
        print("accuracies_train.append(", accuracies_train, ")")
        return accuracies, accuracies_train


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description='')
    parser.add_argument('--experiment', default="train_with_parameters", type=str, help='(default=%(default)s)', required=True)
    parser.add_argument('--seed_wrapper', default=False, type=bool, help='(default=%(default)s)')
    args=parser.parse_args()

    e = Experiments()

    if args.experiment == 'load_and_evaluate':
        experiment = e.loadAndEvaluate
    elif args.experiment == 'train_with_parameters':
        experiment = e.trainWithParameters
    elif args.experiment == 'train90plus10_priors':
        experiment = e.train90plus10_priors
    elif args.experiment == 'train90plus10_minibatch':
        experiment = e.train90plus10_minibatch
    elif args.experiment == 'train43':
        experiment = e.train43
    elif args.experiment == 'train43update':
        experiment = e.train43update
    elif args.experiment == 'train322':
        experiment = e.train322
    elif args.experiment == 'train322update':
        experiment = e.train322update
    elif args.experiment == 'train_data_proportion':
        experiment = e.trainDataProportion
    elif args.experiment == 'train_iterations':
        experiment = e.train_iterations
    elif args.experiment == 'train_Addingkupdates':
        experiment = e.trainAddingkupdates
    elif args.experiment == 'trainAdding4updates':
        experiment = e.trainAdding4updates
    else: raise Exception("You have chosen wrong experiment")

    if args.seed_wrapper:
        e.seed_wrapper(experiment)
    else:
        experiment()








#
