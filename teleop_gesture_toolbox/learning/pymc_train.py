#!/usr/bin/env python
'''
- Recommended to use alongside with teleop_gesture_toolbox package
    > Expects source files (teleop_gesture_toolbox) in ~/<your workspace>/src/teleop_gesture_toolbox/teleop_gesture_toolbox and ~/<your workspace>/src/teleop_gesture_toolbox/teleop_gesture_toolbox/learning
    > Expects dataset recordings saved in ~/<your workspace>/src/teleop_gesture_toolbox/include/data/learning/
        - Download dataset from: https://drive.google.com/drive/u/0/folders/1lasIj7vPenx_ZkMyofvmro6-xtzYdyVm
    > (optional) Expects saved neural_networks in ~/<your workspace>/src/teleop_gesture_toolbox/include/data/trained_networks/
'''

import sys, os, time, argparse
from teleop_gesture_toolbox.os_and_utils.utils import ordered_load, GlobalPaths, load_params
from teleop_gesture_toolbox.os_and_utils.parse_yaml import ParseYAML
PATHS = GlobalPaths(change_working_directory=True)
from warnings import filterwarnings; filterwarnings("ignore")
from copy import deepcopy
import matplotlib.pyplot as plt
import numpy as np
import pymc as pm
import seaborn as sns
# from aesara.tensor.random.utils import RandomStream
from pytensor.tensor.random.utils import RandomStream
# import aesara as pt
import pytensor as pt

from sklearn.model_selection import train_test_split
floatX = pt.config.floatX
from sklearn.metrics import confusion_matrix
import teleop_gesture_toolbox.os_and_utils.confusion_matrix_pretty_print as confusion_matrix_pretty_printfastdtw
from scipy import stats

rc = {'xtick.labelsize': 20, 'ytick.labelsize': 20, 'axes.labelsize': 20, 'font.size': 20,
        'legend.fontsize': 12.0, 'axes.titlesize': 10, "figure.figsize": [12, 6]}
sns.set(rc = rc)
sns.set_style("white")

from teleop_gesture_toolbox.os_and_utils.loading import HandDataLoader, DatasetLoader
from teleop_gesture_toolbox.os_and_utils.nnwrapper import NNWrapper

#%matplotlib inline
import arviz as az
import matplotlib as mpl

from pymc import Model, Normal, Slice, sample
from pymc.distributions import Interpolated
from scipy import stats

plt.style.use("dark_background")
print(f"Running on PyMC v{pm.__version__}")

# Initialize random number generator
np.random.seed(0)

import gesture_classification.gestures_lib as gl
from teleop_gesture_toolbox.os_and_utils import settings
settings.init()
gl.init()

class PyMC3Train():

    def __init__(self, args={}, type='static'):

        if type == 'static':
            self.Gs = gl.gd.Gs_static
        elif type == 'dynamic':
            self.Gs = gl.gd.Gs_dynamic
        assert len(self.Gs) > 1

        self.learn_path = PATHS.learn_path
        self.network_path = PATHS.network_path
        self.args = args
        self.approx = None
        print("Gestures for training are: ", self.Gs)
        print("Arguments for training are: ", self.args)

    def import_records(self, dataset_files=[], type_='static'):
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
        if type_ == 'static':
            self.X, self.Y = DatasetLoader({'input_definition_version':1, 'interpolate':1}).load_static(PATHS.learn_path, self.Gs, new=self.args['full_dataload'])
        elif type_ == 'dynamic':
            dataloader_args = {'interpolate':1, 'discards':1, 'normalize':1, 'normalize_dim':1, 'n':0}
            self.X, self.Y = DatasetLoader(dataloader_args).load_dynamic(PATHS.learn_path, self.Gs)
        print(f"X shape={np.array(self.X).shape}")

        #HandData, HandDataFlags = HandDataLoader().load_directory_update(PATHS.learn_path, self.Gs)
        #self.X, self.Y = DatasetLoader().get_static(HandData, HandDataFlags)

        #self.Xpalm, self.Ydyn = DatasetLoader().get_dynamic(HandData, HandDataFlags)

        self.new_data_arrived = True
        if len(self.Y) == 0:
            self.new_data_arrived = False
            print("No new data!")
            return

        # import matplotlib.pyplot as plt
        # plt.hist(self.X.flatten(), bins=100)
        # plt.show()

        #print("Gestures imported: ", self.Gs)
        #print("Args used: ", self.args)
        #print("X shape: ", self.X.shape)
        #print("Y shape: ", self.Y.shape)
        #print("Xpalm shape", self.Xpalm.shape)
        assert len(self.X) == len(self.Y), "Lengths not match"
        g = list(dict.fromkeys(self.Y))
        counts = [list(self.Y).count(g_) for g_ in g]
        print(f"counts = {counts}")
        self.args['counts'] = counts
        #assert len(self.Xpalm) == len(self.Ydyn), "Lengths not match"
        #print("Gesture 0 recordings: ", len(self.Y[self.Y==0]))

    def split(self):
        ''' Splits the data
        Object (self) parameters:
            X, Y (ndarray): Dataset
            args (Str{}):
                - test_size
        Object (self) returns:
            X_train, X_test, Y_train, Y_test (ndarray): Split dataset
        '''
        assert not np.any(np.isnan(self.X))
        self.X_train, self.X_test, self.Y_train, self.Y_test = train_test_split(self.X, self.Y, test_size=self.args['split'])

    def plot_dataset(self):
        ''' Plot some class from dataset
        Object (self) parameters:
            X, Y (ndarray): Dataset
        '''
        for i in range(2,3):
            fig, ax = plt.subplots()
            ax.scatter(self.X[self.Y == 0, 0], self.X[self.Y == 0, i], label="class 0")
            ax.scatter(self.X[self.Y == 1, 0], self.X[self.Y == 1, i], color="r", label="class "+str(i))
            #sns.despine()
            ax.legend()
            ax.set(xlabel="X", ylabel="Y")


    def construct_nn_2l(self, X, Y, n_hidden, out_n=8, update=False, rng=None):
        ''' Model description
            -> n_hidden - number of hidden layers
            -> number of layers
            -> type of layer distribution (normal, bernoulli, uniform, poisson)
            -> distribution parameters (mu, sigma)
            -> activation function (tanh, sigmoid)
        Parameters:
            update (Bool): If true, weights are updated from self.approx
        '''
        init_1 = rng.standard_normal(size=(X.shape[1], n_hidden)).astype(floatX)
        init_out = rng.standard_normal(size=(n_hidden, out_n)).astype(floatX)

        coords = {
            "hidden_layer_1": np.arange(n_hidden), # 20
            "train_cols": np.arange(X.shape[1]), # 57
            "out_n": np.arange(out_n), # 8
            "obs_id": np.arange(X.shape[0]), # ~17000 for training?
        }

        with pm.Model(coords=coords) as neural_network:
            ann_input = pm.Data("ann_input", X, mutable=True, dims=("obs_id", "train_cols"))
            ann_output = pm.Data("ann_output", Y, mutable=True, dims="obs_id")

            # Weights from input to hidden layer
            weights_in_1 = pm.Normal("w_in_1", 0, sigma=1, initval=init_1, dims=("train_cols", "hidden_layer_1")) # 57 -> 20

            # Weights from hidden layer to output
            weights_3_out = pm.Normal("w_2_out", 0, sigma=1, initval=init_out, dims=("hidden_layer_1", "out_n")) # 20 -> 8

            # Build neural-network using tanh activation function
            act_1 = pm.math.tanh(pm.math.dot(ann_input, weights_in_1))
            #act_2 = pm.math.tanh(pm.math.dot(act_1, weights_1_2))
            #act_3 = pm.math.tanh(pm.math.dot(act_2, weights_2_3))
            
            act_out = pm.math.softmax(pm.math.dot(act_1, weights_3_out), axis=-1)
            
            out = pm.Categorical(
                "out",
                p=act_out,
                observed=ann_output,
                total_size=Y.shape[0],  # IMPORTANT for minibatches
                dims="obs_id"
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

        print("Iterations: ", self.args['iter'], " number of hidden layers: ", self.args['n_hidden'])
        g = list(dict.fromkeys(self.Y_train))

        RANDOM_SEED = 1564
        rng = np.random.default_rng(RANDOM_SEED)
        az.style.use("arviz-darkgrid")

        self.neural_network = self.construct_nn_2l(self.X_train, self.Y_train, out_n=len(g),n_hidden=self.args['n_hidden'], rng=rng)

        # pm.model_to_graphviz(self.neural_network)
        # with self.neural_network:
        #     self.neural_network.debug()
        #     pm.find_MAP()

        with self.neural_network:
            # if self.args['inference_type'] == 'FullRankADVI':
            #     self.inference = pm.FullRankADVI()
            # else: # default option 'ADVI'
            #     self.inference = pm.ADVI()
            tstart = time.time()

            self.approx = pm.fit(n=self.args['iter']) #, method=self.inference)#, callbacks=self.clbcks)
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
        # if cutTestTo:
        #     X_test = X_test[Y_test<cutTestTo]
        #     Y_test = Y_test[Y_test<cutTestTo]

        # x = pt.tensor.matrix("X")
        # n = pt.tensor.iscalar("n")
        # x.tag.test_value = np.empty_like(self.X_train[:10])
        # IMPROVEMENT? (independent to X_train)
        #x.tag.test_value = np.zeros([10,87])
        # n.tag.test_value = 100

        trace = self.approx.sample(draws=100)
        with self.neural_network:
            pm.set_data(new_data={'ann_input': self.X_train})
            ppc = pm.sample_posterior_predictive(trace)
            trace.extend(ppc)
        pred = ppc.posterior_predictive['out']
        # _sample_proba = self.approx.sample_node(
        #     self.neural_network.out.distribution.p, size=n, more_replacements={self.neural_network["ann_input"]: x}
        # )
        # sample_proba = theano.function([x, n], _sample_proba)
        #pred = sample_proba(X_test,self.args['samples']).mean(0)
        shape = np.array(pred).shape
        
        np.save("/home/petr/Downloads/pred.npy", pred)
        Y_train = np.array(self.Y_train).astype(int)
        np.save("/home/petr/Downloads/Y_train.npy", Y_train)

        y_pred = np.max(np.array(pred)[0], axis=0)
        y_pred = np.array(y_pred).astype(int)

        acc = (Y_train == y_pred).mean() * 100
        print(f"Accuracy = {acc}%")
        return acc


        pred_t = sample_proba(self.X_train,samples).mean(0)
        y_pred_t = np.argmax(pred_t, axis=1)
        #this wil be reowrked I tihnk about 
        #print("y pred t shape: ", y_pred_t.shape)
        print("Accuracy on train = {}%".format((self.Y_train == y_pred_t).mean() * 100))

        #confusion_matrix_pretty_print.plot_confusion_matrix_from_data(self.Y_test, self.y_pred, self.Gs,
        #annot=True, cmap = 'Oranges', fmt='1.8f', fz=12, lw=1.5, cbar=False, figsize=[9,9], show_null_values=2, pred_val_axis='y', name=str(name))

        confusion_matrix_pretty_print.plot_confusion_matrix_from_data(Y_test, y_pred, self.Gs, annot=True, cmap = 'Oranges', fmt='1.8f', fz=12, lw=1.5, cbar=False, figsize=[9,9], show_null_values=2, pred_val_axis='y', name="")
        return (Y_test == y_pred).mean() * 100, 0 #(self.Y_train == y_pred_t).mean() * 100


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
        #X_train, approx, neural_network, network_path, name=None, Gs=[], type='', engine='', args={}, accuracy=-1, record_keys=[], filenames=[]

        NNWrapper.save_network(self.X_train, self.approx, self.neural_network,
        network_path=self.network_path, name=name,
        Gs=self.Gs, type='static', engine='PyMC3',
        args=self.args, accuracy=accuracy, filenames=gl.gd.l.static.info.filenames, record_keys=gl.gd.l.static.info.record_keys)

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



class Experiments():

    ''' THERE IS BETTER WAY TO DO THIS
    def seed_wrapper(self, fun=None, args=None, SEEDS=[93457, 12345, 45677, 82909, 75433]):
        print("------------")
        print("Seed Wrapper")
        print("------------")
        # Random seeds, specified for next repeatability
        self.accuracies, self.accuracies_train = [], []
        for n,seed in enumerate(SEEDS):
            print(str(n+1)+". seed: "+str(seed))
            accuracies_row, accuracies_train_row = [], []
            np.random.seed(seed)
            if args:
                accuracies_row, accuracies_train_row = fun(args=args)
            else:
                accuracies_row, accuracies_train_row = fun()
            self.accuracies.append(accuracies_row)
            self.accuracies_train.append(accuracies_train_row)
        print("Accuracies test: ", self.accuracies)
        print("Accuracies train: ", self.accuracies_train)
    '''
    
    def loadAndEvaluate(self, args):
        ''' Loads network file and evaluate it
        '''
        print("Load and evaluate\n---")
        self.train = PyMC3Train(args=args)
        self.train.import_records()
        self.train.split()
        self.train.load(name=self.args['model_filename'])
        self.train.import_records()
        self.train.split()

        return self.train.evaluate()
        
    def trainWithParameters(self, args):
        ''' Train/evaluate + Save
        '''
        print("Training With Parameters and Save\n---")
        self.train = PyMC3Train(args=args)

        self.train.import_records()
        print("Split test_size: ", self.train.args['split'])
        self.train.split()

        self.train.train()
        accuracy, accuracy_train = self.train.evaluate()

        if args['save']:
            self.train.save(name=self.train.args['model_filename'], accuracy=accuracy)

        return accuracy, accuracy_train



if __name__ == '__main__':
    parser=argparse.ArgumentParser(description='')
    parser.add_argument('--experiment', default="trainWithParameters", type=str, help='(default=%(default))', choices=['loadAndEvaluate', 'trainWithParameters'])
    # parser.add_argument('--seed_wrapper', default=False, type=bool, help='(default=%(default))')

    parser.add_argument('--model_filename', default='new_network_y24', type=str, help='(default=%(default))')
    parser.add_argument('--inference_type', default='ADVI', type=str, help='(default=%(default))')
    parser.add_argument('--input_definition_version', default=1, type=int, help='(default=%(default))')
    parser.add_argument('--split', default=0.3, type=float, help='(default=%(default))')
    parser.add_argument('--take_every', default=10, type=int, help='(default=%(default))')
    parser.add_argument('--iter', default=70000, type=int, help='(default=%(default))')
    parser.add_argument('--n_hidden', default=25, type=int, help='(default=%(default))')
    
    
    
    parser.add_argument('--full_dataload', default=False, type=bool, help='(default=%(default))')


    parser.add_argument('--save', default=False, type=bool, help='(default=%(default))')
    args=parser.parse_args().__dict__

    experiment = getattr(Experiments(), args['experiment'])
    # if args['seed_wrapper']:
    #     e.seed_wrapper(experiment, args=args)
    # else:
    experiment(args=args)

