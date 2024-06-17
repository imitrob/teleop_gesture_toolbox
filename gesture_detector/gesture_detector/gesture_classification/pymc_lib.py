#!/usr/bin/env python
import json
import os, time, argparse
from typing import Dict, Optional
from warnings import filterwarnings; filterwarnings("ignore")
from copy import deepcopy
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymc as pm
from pymc.variational import ADVI
from pymc import model_to_graphviz
import seaborn as sns
import pytensor as pt

from sklearn.model_selection import train_test_split
floatX = pt.config.floatX
from sklearn.metrics import confusion_matrix
from collections import Counter
import arviz as az
from collections import Counter

# Initialize random number generator
np.random.seed(0)

import gesture_detector
from gesture_detector.utils.loading import DatasetLoader

az.style.use("arviz-darkgrid")
plt.style.use("dark_background")
print(f"Running on PyMC v{pm.__version__}")
rc = {'xtick.labelsize': 20, 'ytick.labelsize': 20, 'axes.labelsize': 20, 'font.size': 20,
        'legend.fontsize': 12.0, 'axes.titlesize': 10, "figure.figsize": [12, 6]}
sns.set(rc = rc)
sns.set_style("white")

class PyMCModel():
    def __init__(self, model_config):
        self.model_config = model_config

    # self.model
    def build_model(self, X: pd.DataFrame, y: pd.Series, **kwargs):
        """
        build_model creates the PyMC model
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

        init_1 = np.random.randn(X.shape[1], n_hidden).astype(floatX) #rng.standard_normal(size=(X.shape[1], n_hidden)).astype(floatX)
        # init_2 = rng.standard_normal(size=(n_hidden, n_hidden)).astype(floatX)
        init_out = np.random.randn(n_hidden, out_n).astype(floatX) #rng.standard_normal(size=(n_hidden, out_n)).astype(floatX)

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
            weights_1 = pm.Normal("w_1", 0, sigma=1, initval=init_1, dims=("train_cols", "hidden_layer_1"))
            #weights_2 = pm.Normal("w_2", 0, sigma=1, initval=init_2, dims=("hidden_layer_1", "hidden_layer_1"))
            weights_3 = pm.Normal("w_3", 0, sigma=1, initval=init_out, dims=("hidden_layer_1", "out_n"))

            # Build neural-network using tanh activation function
            act_1 = pm.math.tanh(pm.math.dot(ann_input, weights_1))
            #act_2 = pm.math.tanh(pm.math.dot(act_1, weights_2))
            act_out = pm.math.softmax(pm.math.dot(act_1, weights_3), axis=1)
            
            out = pm.Categorical(
                "out",
                p=act_out,
                observed=ann_output,
                total_size=y.shape[0],  # IMPORTANT for minibatches
                dims="obs_id"
            )

    @property
    def output_var(self):
        return "y"

    def import_records(self, dataset_files=[]):
        ''' Import static gestures, Import all data from learning folder
        '''
        if self.model_config['gesture_type'] == 'static':
            X, y = DatasetLoader({'input_definition_version':1, 'interpolate':1, 'take_every': self.model_config['take_every']}).load_static(gesture_detector.gesture_data_path, self.model_config['Gs'], new=self.model_config['full_dataload'])
        elif self.model_config['gesture_type'] == 'dynamic':
            dataloader_args = {'interpolate':1, 'discards':1, 'normalize':1, 'normalize_dim':1, 'n':0}
            X, y = DatasetLoader(dataloader_args).load_dynamic(gesture_detector.gesture_data_path, self.model_config['Gs'])
        print(f"X shape={np.array(X).shape}")

        assert not np.any(np.isnan(X))

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
        '''
        X_test = deepcopy(X_test)
        y_test = deepcopy(y_test)

        # test on X_train
        def eval(X_true, y_true, save_name, draws=1000):
            built_model = self.build_model_from_values(X_true, y_true, fitted_model.mean.eval(), fitted_model.std.eval(), self.model_config)

            trace = built_model.sample(draws=draws)
            with self.model:
                ppc = pm.sample_posterior_predictive(trace)
                trace.extend(ppc)
            pred = np.array(ppc.posterior_predictive['out'])[0].T
            
            y_true = np.array(y_true).astype(int)
            
            y_pred = []
            for row in pred:
                y_pred.append(Counter(row).most_common()[0][0])
            y_pred = np.array(y_pred).astype(int)

            acc = (y_true == y_pred).mean() * 100
            print(f"Train accuracy = {acc}%")
            self.model_config[save_name] = acc

            # confusion_matrix_pretty_print.plot_confusion_matrix_from_data(y_true, y_pred, self.model_config['Gs'], annot=True, cmap = 'Oranges', fmt='1.8f', fz=12, lw=1.5, cbar=False, figsize=[9,9], show_null_values=2, pred_val_axis='y', name=f"{save_name}_{self.model_config['model_filename']}")

        eval(X_train, y_train, 'acc_train', draws=300)
        eval(X_test, y_test, 'acc_test', draws=300)


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
                if not os.path.isfile(gesture_detector.saved_models_path+"network"+str(i)+".pkl"):
                    n_network = str(i)
                    break
            name = f"network{n_network}"
        else:
            if os.path.isfile(gesture_detector.saved_models_path+name+".npy"):
                print(f"File {name} exists, network is rewritten!")

        with open(gesture_detector.saved_models_path+name+".json", 'w') as f:
            json.dump(self.model_config, f, indent=4)

        np.savez(gesture_detector.saved_models_path+name, 
            X_train=X_train, y_train=y_train, X_test=X_test, 
            y_test=y_test, mu=mu, sigma=sigma,
        )

        print(f"Network: {name}.npy saved")

    def build_model_from_values(self, X_train, y_train, mu, sigma, model_config, out_n=None):

        if out_n is None:
            out_n = len(list(dict.fromkeys(y_train)))
        
        layer1_bound = X_train.shape[1] * model_config['n_hidden']
        n_hidden = self.model_config['n_hidden']
        
        mus1 = mu[0:layer1_bound]
        mus2 = mu[layer1_bound:]
        mus1=mus1.T.reshape(X_train.shape[1],model_config['n_hidden'])
        mus2=mus2.T.reshape(model_config['n_hidden'],out_n)
        mus1.astype(floatX)
        mus2.astype(floatX)

        sigmas1 = sigma[0:layer1_bound]
        sigmas2 = sigma[layer1_bound:]
        sigmas1 = sigmas1.T.reshape(X_train.shape[1],model_config['n_hidden'])
        sigmas2 = sigmas2.T.reshape(model_config['n_hidden'],out_n)
        sigmas1.astype(floatX)
        sigmas2.astype(floatX)

        coords = {
            "hidden_layer_1": np.arange(n_hidden),
            "train_cols": np.arange(X_train.shape[1]),
            "out_n": np.arange(out_n),
            "obs_id": np.arange(X_train.shape[0]),
        }
        
        with pm.Model(coords=coords) as self.model:
            ann_input = pm.Data("ann_input", X_train, mutable=True, dims=("obs_id", "train_cols"))
            ann_output = pm.Data("ann_output", y_train, mutable=True, dims="obs_id")
            
            weights_in_1 = pm.Normal("w_in_1", mus1, sigma=sigmas1, shape=(X_train.shape[1], model_config['n_hidden']), initval=mus1)
            weights_3_out = pm.Normal("w_2_out", mus2, sigma=sigmas2, shape=(model_config['n_hidden'], out_n), initval=mus2)
            
            # Build neural-network using tanh activation function
            act_1 = pm.math.tanh(pm.math.dot(ann_input, weights_in_1))
            
            act_out = pm.math.softmax(pm.math.dot(act_1, weights_3_out), axis=-1)
            
            out = pm.Categorical(
                "out",
                p=act_out,
                observed=ann_output,
                total_size=y_train.shape[0],  # IMPORTANT for minibatches
                dims="obs_id"
            )

            loaded_model = pm.fit(n=0, method=ADVI())
        return loaded_model
    

    def load(self):
        ''' Load Network
        '''
        name = self.model_config['model_filename']

        with open(gesture_detector.saved_models_path+name+".json", 'r') as f:
            model_config = json.load(f)
        nn = np.load(gesture_detector.saved_models_path+name+".npz")

        loaded_model = self.build_model_from_values(nn['X_train'], nn['y_train'], nn['mu'], nn['sigma'], model_config)

        return loaded_model, nn['X_train'], nn['y_train'], nn['X_test'], nn['y_test']

    def sample(self, built_model, draws=100):

        t1 = time.perf_counter()

        trace = built_model.sample(draws=draws)
        with self.model:
            ppc = pm.sample_posterior_predictive(trace)
            trace.extend(ppc)
        pred = np.array(ppc.posterior_predictive['out'])[0].T

        y_pred = Counter(pred[0]).most_common()[0][0]
        print(f"tt {time.perf_counter()-t1}")
        return y_pred        


class PyMC_Sample():
    def __init__(self, draws=100):
        self.pymcmodel = None
        self.built_model = None
        self.draws = draws

    def sample(self, data):
        trace = self.built_model.sample(draws=self.draws)
        with self.pymcmodel.model:
            ppc = pm.sample_posterior_predictive(trace, progressbar=False)
            trace.extend(ppc)
        pred = np.array(ppc.posterior_predictive['out'])[0].T

        

        y_pred = Counter(pred[0]).most_common()[0][0]

        probs = self.pred_labels_to_probs(pred)

        return y_pred, probs

    def pred_labels_to_probs(self, labels):
        
        labels = np.array(labels)
        
        # Get unique labels and their counts
        unique_labels, counts = np.unique(labels, return_counts=True)

        # Initialize the array for all label counts
        all_counts = np.zeros(self.out_n)

        # Assign the counts to the corresponding indices
        all_counts[unique_labels] = counts

        # Calculate probabilities
        probabilities = all_counts / all_counts.sum()

        return probabilities


    def init(self, nn, model_config):
        X_train = nn['X_train']
        y_train = nn['y_train']
        X_test = nn['X_test']
        y_test = nn['y_test']
        mu = nn['mu']
        sigma = nn['sigma']

        self.pymcmodel = PyMCModel(model_config)
        fitted_model, X_train, y_train, X_test, y_test = self.pymcmodel.load()
        X_true = X_test[0:1]
        y_true = y_test[0:1]
        out_n = len(list(dict.fromkeys(y_train)))
        self.out_n = out_n
        self.built_model = self.pymcmodel.build_model_from_values(X_true, y_true, fitted_model.mean.eval(), fitted_model.std.eval(), self.pymcmodel.model_config, out_n=out_n)
   

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
        X_train, X_test, y_train, y_test = train_test_split(X, y, 
            test_size=self.pymcmodel.model_config['split'])
        
        fitted_model = self.pymcmodel.fit(X_train, y_train)
        acc = self.pymcmodel.evaluate(fitted_model, X_test, y_test, X_train, y_train)


        if args['save']:
            self.pymcmodel.save(fitted_model, X_train, y_train, X_test, y_test)

        self.pymcmodel = None
        self.pymcmodel = PyMCModel(args)

        if args['test']:
            fitted_model, X_train, y_train, X_test, y_test = self.pymcmodel.load()
            acc = self.pymcmodel.evaluate(fitted_model, X_test, y_test, X_train, y_train)
            
    def sampleThread(self, args):
        args = _args()
        self.pymcmodel = PyMCModel(args)
        fitted_model, X_train, y_train, X_test, y_test = self.pymcmodel.load()
        X_true = X_test[0:1]
        y_true = y_test[0:1]
        out_n = len(list(dict.fromkeys(y_train)))
        built_model = self.pymcmodel.build_model_from_values(X_true, y_true, fitted_model.mean.eval(), fitted_model.std.eval(), self.pymcmodel.model_config, out_n=out_n)

        while True:
            self.pymcmodel.sample(built_model)

    
def _args():
    """ Optimal config for 8 gestures, each 30 recordings (~2000 samples) per gestures
        - 20 n_hidden nodes
        - 70000 iterations
        - 0.3 split: 30% for testing
        - 4 take_every, take every 4th sample from hand records
        - True full_dataload: Load all hand data from record folder
        -> Should reach up to 99% accuracy
    """

    parser=argparse.ArgumentParser(description='')
    parser.add_argument('--experiment', default="sampleThread", type=str, help='(default=%(default))', choices=['loadAndEvaluate', 'trainWithParameters', 'sampleThread'])

    # const
    parser.add_argument('--inference_type', default='ADVI', type=str, help='(default=%(default))')
    parser.add_argument('--layers', default=2, type=int, help='(default=%(default))')
    parser.add_argument('--gesture_type', default="static", type=str, help='(default=%(default))')
    parser.add_argument('--engine', default="PyMC", type=str, help='(default=%(default))')
    parser.add_argument('--seed_wrapper', default=False, type=bool, help='(default=%(default))')

    # data load
    parser.add_argument('--full_dataload', default=False, type=bool, help='(default=%(default))')
    parser.add_argument('--input_definition_version', default=1, type=int, help='(default=%(default))')
    parser.add_argument('--take_every', default=4, type=int, help='(default=%(default))')
    parser.add_argument('--split', default=0.3, type=float, help='(default=%(default))')
    
    # model
    parser.add_argument('--n_hidden', default=20, type=int, help='(default=%(default))')
    # fit
    parser.add_argument('--iter', default=70000, type=int, help='(default=%(default))')
    parser.add_argument('--seed', default=93457, type=int, help='(default=%(default))')
    
    
    parser.add_argument('--model_filename', default='network99', type=str, help='(default=%(default))')
    parser.add_argument('--save', default=False, type=bool, help='(default=%(default))')
    parser.add_argument('--test', default=False, type=bool, help='(default=%(default))')

    return parser.parse_args().__dict__

if __name__ == '__main__':
    args = _args()
    experiment = getattr(Experiments(), args['experiment'])
    # if args['seed_wrapper']:
    #     e.seed_wrapper(experiment, args=args)
    # else:
    experiment(args=args)
