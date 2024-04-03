#!/usr/bin/env python
'''
- Recommended to use alongside with teleop_gesture_toolbox package
    > Expects source files (teleop_gesture_toolbox) in ~/<your workspace>/src/teleop_gesture_toolbox/teleop_gesture_toolbox and ~/<your workspace>/src/teleop_gesture_toolbox/teleop_gesture_toolbox/learning
    > Expects dataset recordings saved in ~/<your workspace>/src/teleop_gesture_toolbox/include/data/learning/
        - Download dataset from: https://drive.google.com/drive/u/0/folders/1lasIj7vPenx_ZkMyofvmro6-xtzYdyVm
    > (optional) Expects saved neural_networks in ~/<your workspace>/src/teleop_gesture_toolbox/include/data/trained_networks/
'''

import sys, os, time, argparse
from typing import Dict, Optional, Union
from teleop_gesture_toolbox.os_and_utils.utils import ordered_load, GlobalPaths, load_params
from teleop_gesture_toolbox.os_and_utils.parse_yaml import ParseYAML
PATHS = GlobalPaths(change_working_directory=True)
from warnings import filterwarnings; filterwarnings("ignore")
from copy import deepcopy
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymc as pm
import seaborn as sns
from pytensor.tensor.random.utils import RandomStream
import pytensor as pt

from sklearn.model_selection import train_test_split
floatX = pt.config.floatX
from sklearn.metrics import confusion_matrix
import teleop_gesture_toolbox.os_and_utils.confusion_matrix_pretty_print as confusion_matrix_pretty_print
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
gl.init(load_trained=False)

az.style.use("arviz-darkgrid")

class PyMCModel():
        # Give the model a name
    _model_type = "VIClassifier"

    # And a version
    version = "0.1"

    def __init__(self, args):
        self.model_config = self.get_default_model_config(args)

    # self.model
    def build_model(self, X: pd.DataFrame, y: pd.Series, **kwargs):
        """
        build_model creates the PyMC model

        Parameters:
        model_config: dictionary
            it is a dictionary with all the parameters that we need in our model example:  a_loc, a_scale, b_loc
        X : pd.DataFrame
            The input data that is going to be used in the model. This should be a DataFrame
            containing the features (predictors) for the model. For efficiency reasons, it should
            only contain the necessary data columns, not the entire available dataset, as this
            will be encoded into the data used to recreate the model.

        y : pd.Series
            The target data for the model. This should be a Series representing the output
            or dependent variable for the model.
        
        rng: Prior values random generator
        
        Parameters:
            update (Bool): If true, weights are updated from self.approx

        kwargs : dict
            Additional keyword arguments that may be used for model configuration.
        """
        assert self.model_config['layers'] == 2
        assert self.model_config['inference_type'] == 'ADVI'
        assert self.model_config['engine'] == 'PyMC'
        

        # Check the type of X and y and adjust access accordingly
        X = X #["input"].values
        y = y.values if isinstance(y, pd.Series) else y

        n_hidden = self.model_config['n_hidden']
        
        out_n = len(list(dict.fromkeys(y)))
        rng = np.random.default_rng(self.model_config['seed'])

        init_1 = rng.standard_normal(size=(X.shape[1], n_hidden)).astype(floatX)
        init_out = rng.standard_normal(size=(n_hidden, out_n)).astype(floatX)

        coords = {
            "hidden_layer_1": np.arange(n_hidden), # 20
            "train_cols": np.arange(X.shape[1]), # 57
            "out_n": np.arange(out_n), # 8
            "obs_id": np.arange(X.shape[0]), # ~17000 for training?
        }

        with pm.Model(coords=coords) as self.model:
            ann_input = pm.Data("ann_input", X, mutable=True, dims=("obs_id", "train_cols"))
            ann_output = pm.Data("ann_output", y, mutable=True, dims="obs_id")

            # Weights from input to hidden layer
            weights_in_1 = pm.Normal("w_in_1", 0, sigma=1, initval=init_1, dims=("train_cols", "hidden_layer_1")) # 57 -> 20

            # Weights from hidden layer to output
            weights_3_out = pm.Normal("w_2_out", 0, sigma=1, initval=init_out, dims=("hidden_layer_1", "out_n")) # 20 -> 8

            # Build neural-network using tanh activation function
            act_1 = pm.math.tanh(pm.math.dot(ann_input, weights_in_1))
            
            act_out = pm.math.softmax(pm.math.dot(act_1, weights_3_out), axis=-1)
            
            out = pm.Categorical(
                "out",
                p=act_out,
                observed=ann_output,
                total_size=y.shape[0],  # IMPORTANT for minibatches
                dims="obs_id"
            )


    @staticmethod
    def get_default_model_config(args) -> Dict:

        if args['gesture_type'] == 'static':
            Gs = gl.gd.Gs_static
        elif args['gesture_type'] == 'dynamic':
            Gs = gl.gd.Gs_dynamic
        assert len(Gs) > 1

        model_config = args
        model_config["Gs"] = Gs
        model_config["learn_path"] = PATHS.learn_path
        model_config["network_path"] = PATHS.network_path
        
        return model_config

    @property
    def output_var(self):
        return "y"

    @property
    def _serializable_model_config(self) -> Dict[str, Union[int, float, Dict]]:
        """
        _serializable_model_config is a property that returns a dictionary with all the model parameters that we want to save.
        as some of the data structures are not json serializable, we need to convert them to json serializable objects.
        Some models will need them, others can just define them to return the model_config.
        """
        return self.model_config

    def _save_input_params(self, idata) -> None:
        """
        Saves any additional model parameters (other than the dataset) to the idata object.

        These parameters are stored within `idata.attrs` using keys that correspond to the parameter names.
        If you don't need to store any extra parameters, you can leave this method unimplemented.

        Example:
            For saving customer IDs provided as an 'customer_ids' input to the model:
            self.customer_ids = customer_ids.values #this line is done outside of the function, preferably at the initialization of the model object.
            idata.attrs["customer_ids"] = json.dumps(self.customer_ids.tolist())  # Convert numpy array to a JSON-serializable list.
        """
        pass

        pass

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
        if self.model_config['gesture_type'] == 'static':
            X, y = DatasetLoader({'input_definition_version':1, 'interpolate':1, 'take_every': self.model_config['take_every']}).load_static(PATHS.learn_path, self.model_config['Gs'], new=self.model_config['full_dataload'])
        elif self.model_config['gesture_type'] == 'dynamic':
            dataloader_args = {'interpolate':1, 'discards':1, 'normalize':1, 'normalize_dim':1, 'n':0}
            X, y = DatasetLoader(dataloader_args).load_dynamic(PATHS.learn_path, self.model_config['Gs'])
        print(f"X shape={np.array(X).shape}")

        #HandData, HandDataFlags = HandDataLoader().load_directory_update(PATHS.learn_path, self.Gs)
        #self.X, self.Y = DatasetLoader().get_static(HandData, HandDataFlags)

        #self.Xpalm, self.Ydyn = DatasetLoader().get_dynamic(HandData, HandDataFlags)

        self.new_data_arrived = True
        if len(y) == 0:
            self.new_data_arrived = False
            print("No new data!")
            return

        assert len(X) == len(y), "Lengths not match"
        g = list(dict.fromkeys(y))
        counts = [list(y).count(g_) for g_ in g]
        print(f"counts = {counts}")
        self.model_config['counts'] = counts
        return X, y

    def split(self, X, y):
        ''' Splits the data
        '''
        assert not np.any(np.isnan(X))
        return train_test_split(X, y, test_size=self.model_config['split'])

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

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> az.InferenceData:
        """
        """
        self.build_model(X, y)

        with self.model:
            tstart = time.time()
            self.idata = pm.fit(n=self.model_config['iter'])
            print("Train time [s]: ", time.time()-tstart)

        return self.idata

    def evaluate(self, fitted_model, X_test, y_test, X_train, y_train):
        ''' Evaluate network
        Object (self) parameters:
            fitted_model (PyMC3 obj): Approximation
            X_train, Y_train (ndarray): Train datatset
            model (PyMC3 obj): Model
            X_test, Y_test (ndarray): Test dataset
            args (Str{})
        '''
        X_test = deepcopy(X_test)
        Y_test = deepcopy(y_test)

        # test on X_train
        trace = fitted_model.sample(draws=100)
        with self.model:
            pm.set_data(new_data={'ann_input': X_train})
            ppc = pm.sample_posterior_predictive(trace)
            trace.extend(ppc)
        pred = ppc.posterior_predictive['out']
        
        y_train = np.array(y_train).astype(int)
        y_pred = np.max(np.array(pred)[0], axis=0)
        y_pred = np.array(y_pred).astype(int)

        acc = (y_train == y_pred).mean() * 100
        print(f"Train accuracy = {acc}%")
        self.model_config['acc_train'] = acc

        confusion_matrix_pretty_print.plot_confusion_matrix_from_data(y_train, y_pred, self.model_config['Gs'], annot=True, cmap = 'Oranges', fmt='1.8f', fz=12, lw=1.5, cbar=False, figsize=[9,9], show_null_values=2, pred_val_axis='y', name=self.model_config['model_filename'])

        # test on X_test
        trace = fitted_model.sample(draws=100)
        with self.model:
            ## TEMPORARY HACK
            X_train[:len(X_test)] = X_test
            pm.set_data(new_data={'ann_input': X_train})
            ppc = pm.sample_posterior_predictive(trace)
            trace.extend(ppc)
        pred = ppc.posterior_predictive['out']
        
        y_test = np.array(y_test).astype(int)
        y_pred = np.max(np.array(pred)[0,:,0:len(X_test)], axis=0)
        y_pred = np.array(y_pred).astype(int)

        acc = (y_test == y_pred).mean() * 100
        print(f"Test accuracy = {acc}%")
        self.model_config['acc_test'] = acc

        confusion_matrix_pretty_print.plot_confusion_matrix_from_data(Y_test, y_pred, self.model_config['Gs'], annot=True, cmap = 'Oranges', fmt='1.8f', fz=12, lw=1.5, cbar=False, figsize=[9,9], show_null_values=2, pred_val_axis='y', name=self.model_config['model_filename'])

        return acc


    def save(self, fitted_model, X_train, y_train, X_test, y_test):
        ''' Save the network
        '''
        mu =    np.array(fitted_model.mean.eval())
        sigma = np.array(fitted_model.std.eval())
        assert (sigma>=0).all(), f"Negative sigma {sigma}"
        
        assert len(mu) == (X_train.shape[1] * self.model_config['n_hidden'] + self.model_config['n_hidden'] * len(list(dict.fromkeys(y_train))))

        # save
        name = self.model_config['model_filename']
        n_network = ""
        if name == None:
            for i in range(0,200):
                if not os.path.isfile(self.network_path+"network"+str(i)+".pkl"):
                    n_network = str(i)
                    break
            name = f"network{n_network}"
        else:
            if os.path.isfile(self.model_config['network_path']+name+".npy"):
                print(f"File {name} exists, network is rewritten!")

        saved_tuple = np.array((X_train, y_train, X_test, y_test, 
                mu, sigma, self.model_config), dtype=object)
        np.save(self.model_config['network_path']+name, 
                saved_tuple)

        print(f"Network: {name}.npy saved")

    def load(self):
        ''' Load Network
        '''
        name = self.model_config['model_filename']
        network_path = self.model_config['network_path']
        
        X_train, y_train, X_test, y_test, mu, sigma, model_config = np.load(network_path+name+".npy", allow_pickle=True)

        out_n = len(list(dict.fromkeys(y_train)))
        layer1_bound = X_train.shape[1] * model_config['n_hidden']

        mus1 = mu[0:layer1_bound]
        mus2 = mu[layer1_bound:]

        mus1=mus1.T.reshape(X_train.shape[1],model_config['n_hidden'])
        mus2=mus2.T.reshape(model_config['n_hidden'],out_n)

        sigmas1 = sigma[0:layer1_bound]
        sigmas2 = sigma[layer1_bound:]
        sigmas1 = sigmas1.T.reshape(X_train.shape[1],model_config['n_hidden'])
        sigmas2 = sigmas2.T.reshape(model_config['n_hidden'],out_n)

        mus1.astype(floatX)
        mus2.astype(floatX)
        sigmas1.astype(floatX)
        sigmas2.astype(floatX)
        assert (sigmas1>=0).all(), f"Negative sigma {sigmas1}"
        assert (sigmas2>=0).all(), f"Negative sigma {sigmas2}"


        assert self.model_config['layers'] == 2
        assert self.model_config['inference_type'] == 'ADVI'
        assert self.model_config['engine'] == 'PyMC'
        

        # Check the type of X and y and adjust access accordingly
        X = X_train #["input"].values
        y = y_train

        n_hidden = self.model_config['n_hidden']
        
        out_n = len(list(dict.fromkeys(y)))
        rng = np.random.default_rng(self.model_config['seed'])

        init_1 = rng.standard_normal(size=(X.shape[1], n_hidden)).astype(floatX)
        init_out = rng.standard_normal(size=(n_hidden, out_n)).astype(floatX)

        coords = {
            "hidden_layer_1": np.arange(n_hidden), # 20
            "train_cols": np.arange(X.shape[1]), # 57
            "out_n": np.arange(out_n), # 8
            "obs_id": np.arange(X.shape[0]), # ~17000 for training?
        }

        with pm.Model(coords=coords) as self.model:
            ann_input = pm.Data("ann_input", X, mutable=True, dims=("obs_id", "train_cols"))
            ann_output = pm.Data("ann_output", y, mutable=True, dims="obs_id")

            weights_in_1 = pm.Normal("w_in_1", mus1, sigma=sigmas1, shape=(X.shape[1], model_config['n_hidden']), initval=mus1)
            weights_3_out = pm.Normal("w_2_out", mus2, sigma=sigmas2, shape=(model_config['n_hidden'], out_n), initval=mus2)

            # Build neural-network using tanh activation function
            act_1 = pm.math.tanh(pm.math.dot(ann_input, weights_in_1))
            
            act_out = pm.math.softmax(pm.math.dot(act_1, weights_3_out), axis=-1)
            
            out = pm.Categorical(
                "out",
                p=act_out,
                observed=ann_output,
                total_size=y.shape[0],  # IMPORTANT for minibatches
                dims="obs_id"
            )

            fitted_model = pm.fit(n=1)
            
        return fitted_model, X_train, y_train, X_test, y_test




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
        self.pymcmodel = PyMCModel(args)
        fitted_model, X_train, y_train, X_test, y_test = self.pymcmodel.load()
        acc = self.pymcmodel.evaluate(fitted_model, X_test, y_test, X_train, y_train)
        
    def trainWithParameters(self, args):
        ''' Train/evaluate + Save
        '''
        print("Training With Parameters and Save\n---")
        self.pymcmodel = PyMCModel(args)

        X, y = self.pymcmodel.import_records()
        X_train, X_test, y_train, y_test = self.pymcmodel.split(X, y)

        fitted_model = self.pymcmodel.fit(X_train, y_train)
        acc = self.pymcmodel.evaluate(fitted_model, X_test, y_test, X_train, y_train)


        if args['save']:
            self.pymcmodel.save(fitted_model, X_train, y_train, X_test, y_test)

        self.pymcmodel = None
        self.pymcmodel = PyMCModel(args)

        if args['test']:
            fitted_model, X_train, y_train, X_test, y_test = self.pymcmodel.load()
            acc = self.pymcmodel.evaluate(fitted_model, X_test, y_test, X_train, y_train)
            



if __name__ == '__main__':
    parser=argparse.ArgumentParser(description='')
    parser.add_argument('--experiment', default="trainWithParameters", type=str, help='(default=%(default))', choices=['loadAndEvaluate', 'trainWithParameters'])
    # parser.add_argument('--seed_wrapper', default=False, type=bool, help='(default=%(default))')

    parser.add_argument('--inference_type', default='ADVI', type=str, help='(default=%(default))')
    parser.add_argument('--layers', default=2, type=int, help='(default=%(default))')
    parser.add_argument('--input_definition_version', default=1, type=int, help='(default=%(default))')
    parser.add_argument('--split', default=0.3, type=float, help='(default=%(default))')
    parser.add_argument('--take_every', default=4, type=int, help='(default=%(default))')
    parser.add_argument('--iter', default=10000, type=int, help='(default=%(default))')
    parser.add_argument('--n_hidden', default=30, type=int, help='(default=%(default))')
    
    parser.add_argument('--seed', default=93457, type=int, help='(default=%(default))')
    parser.add_argument('--gesture_type', default="static", type=str, help='(default=%(default))')
    
    parser.add_argument('--model_filename', default='new_network_y24', type=str, help='(default=%(default))')
    parser.add_argument('--full_dataload', default=True, type=bool, help='(default=%(default))')

    parser.add_argument('--engine', default="PyMC", type=str, help='(default=%(default))')

    parser.add_argument('--save', default=True, type=bool, help='(default=%(default))')
    parser.add_argument('--test', default=True, type=bool, help='(default=%(default))')
    args=parser.parse_args().__dict__

    experiment = getattr(Experiments(), args['experiment'])
    # if args['seed_wrapper']:
    #     e.seed_wrapper(experiment, args=args)
    # else:
    experiment(args=args)

