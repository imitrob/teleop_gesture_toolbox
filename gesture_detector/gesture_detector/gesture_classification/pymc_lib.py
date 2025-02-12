#!/usr/bin/env python
import os, time, argparse, json
from typing import Optional
from warnings import filterwarnings; filterwarnings("ignore")
from copy import deepcopy
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymc as pm
from pymc.variational import ADVI
import seaborn as sns
import pytensor as pt

from sklearn.model_selection import train_test_split
floatX = pt.config.floatX
from collections import Counter
import arviz as az
from gesture_detector.utils.pretty_confusion_matrix import pp_matrix_from_data

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
import logging
logging.getLogger("pymc").setLevel(logging.ERROR)

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

    def import_records(self, gesture_names):
        ''' Import static gestures, Import all data from learning folder
        '''
        print(f"Loading dataset from: {gesture_detector.gesture_data_path}")
        if self.model_config['gesture_type'] == 'static':
            X, y = DatasetLoader({'input_definition_version':1, 'take_every': self.model_config['take_every']}).load_static(gesture_detector.gesture_data_path, gesture_names)
        elif self.model_config['gesture_type'] == 'dynamic':
            dataloader_args = {'interpolate':1, 'discards':1, 'normalize':1, 'normalize_dim':1, 'n':0}
            X, y = DatasetLoader(dataloader_args).load_dynamic(gesture_detector.gesture_data_path, gesture_names)
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
            print(f"Accurracy {save_name} = {acc}%")
            self.model_config[save_name] = acc

            pp_matrix_from_data(y_true, y_pred, self.model_config['gestures'], annot=True, cmap = 'Oranges', cbar=False, figsize=(9,9))
            return acc

        acc = eval(X_train, y_train, 'acc_train', draws=300)
        acc = eval(X_test, y_test, 'acc_test', draws=300)
        return acc

    def save(self, fitted_model, X_train, y_train, X_test, y_test):
        ''' Save the network
        '''
        mu =    np.array(fitted_model.mean.eval())
        sigma = np.array(fitted_model.std.eval())
        assert (sigma>=0).all(), f"Negative sigma {sigma}"
        
        assert len(mu) == (X_train.shape[1] * self.model_config['n_hidden'] + self.model_config['n_hidden'] * len(list(dict.fromkeys(y_train))))

        # save
        name = self.model_config['model_name']
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

    def build_model_from_values(self, X_train, y_train, mu, sigma, model_config, out_n=None, init_appr=True):

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
            if init_appr:
                loaded_model = pm.fit(n=0, method=ADVI(), progressbar=False)
            else:
                loaded_model = None
        return loaded_model
    

    def load(self):
        ''' Load Network
        '''
        name = self.model_config['model_name']

        with open(gesture_detector.saved_models_path+name+".json", 'r') as f:
            model_config = json.load(f)
        nn = np.load(gesture_detector.saved_models_path+name+".npz")

        loaded_model = self.build_model_from_values(nn['X_train'], nn['y_train'], nn['mu'], nn['sigma'], model_config)

        return loaded_model, nn['X_train'], nn['y_train'], nn['X_test'], nn['y_test']

    def sample(self, built_model, draws=100):
        trace = built_model.sample(draws=draws)
        with self.model:
            ppc = pm.sample_posterior_predictive(trace)
            trace.extend(ppc)
        pred = np.array(ppc.posterior_predictive['out'])[0].T

        y_pred = Counter(pred[0]).most_common()[0][0]
        return y_pred        


class PyMC_Sample():
    def __init__(self, draws=100):
        self.pymcmodel = None
        self.built_model = None
        self.draws = draws

    def sample(self, data):
        data = np.array([data])
        assert data.ndim == 2

        self.pymcmodel.build_model_from_values(self.X_true, self.y_true_1, self.fitted_model.mean.eval(), self.fitted_model.std.eval(), self.pymcmodel.model_config, out_n=self.out_n, init_appr=False)


        with self.pymcmodel.model:
            self.pymcmodel.model.set_data("ann_input", data)
            
        trace = self.built_model.sample(draws=self.draws)
        with self.pymcmodel.model:
            ppc = pm.sample_posterior_predictive(trace, progressbar=False)
            trace.extend(ppc)
        pred = np.array(ppc.posterior_predictive['out'])[0].T
        
        y_pred = []
        for row in pred:
            y_pred.append(Counter(row).most_common()[0][0])

        y_pred = np.array(y_pred).astype(int)

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
        self.fitted_model, X_train, y_train, X_test, y_test = self.pymcmodel.load()
        self.X_true = X_test[0:1]
        self.y_true_1 = y_test[0:1]
        out_n = len(list(dict.fromkeys(y_train)))
        self.out_n = out_n
        self.built_model = self.pymcmodel.build_model_from_values(self.X_true, self.y_true_1, self.fitted_model.mean.eval(), self.fitted_model.std.eval(), self.pymcmodel.model_config, out_n=self.out_n)
   

class Experiments():
    def load_and_evaluate(self, args):
        ''' Loads network file and evaluate it
        '''
        print("Load and evaluate\n---")
        self.pymcmodel = PyMCModel(args)
        fitted_model, X_train, y_train, X_test, y_test = self.pymcmodel.load()
        acc = self.pymcmodel.evaluate(fitted_model, X_test, y_test, X_train, y_train)
        return acc

    def train_custom(self, args):
        ''' Train/evaluate + Save
        '''
        print("Training With Parameters and Save\n---")
        self.pymcmodel = PyMCModel(args)

        X, y = self.pymcmodel.import_records(gesture_names=args['gestures'])
        X_train, X_test, y_train, y_test = train_test_split(X, y, 
            test_size=self.pymcmodel.model_config['split'])
        
        fitted_model = self.pymcmodel.fit(X_train, y_train)
        acc = self.pymcmodel.evaluate(fitted_model, X_test, y_test, X_train, y_train)


        if args['save']:
            self.pymcmodel.save(fitted_model, X_train, y_train, X_test, y_test)

        self.pymcmodel = None
        self.pymcmodel = PyMCModel(args)

        fitted_model, X_train, y_train, X_test, y_test = self.pymcmodel.load()
        acc = self.pymcmodel.evaluate(fitted_model, X_test, y_test, X_train, y_train)
        return acc

    def sample_node(self, args):
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
    parser=argparse.ArgumentParser(description='')
    parser.add_argument('--experiment', default="train_custom", type=str, help='(default=%(default))', choices=['load_and_evaluate', 'train_custom', 'sample_node'])

    # List of gestures to train
    parser.add_argument('--gestures', nargs='+', default=['grab','pinch','point','two','three','four','five','thumbsup'], help='List of gestures')
    parser.add_argument('--model_name', default='common_gestures', type=str, help='(default=%(default))')
    parser.add_argument('--save', default=True, type=bool, help='(default=%(default))')

    # model
    parser.add_argument('--n_hidden', default=20, type=int, help='(default=%(default))')
    parser.add_argument('--split', default=0.3, type=float, help='(default=%(default))')
    parser.add_argument('--take_every', default=4, type=int, help='Frame dropping policy')
    # fit
    parser.add_argument('--iter', default=70000, type=int, help='(default=%(default))')
    parser.add_argument('--seed', default=93457, type=int, help='(default=%(default))')

    # Const
    parser.add_argument('--inference_type', default='ADVI', type=str, help='(default=%(default))')
    parser.add_argument('--layers', default=2, type=int, help='(default=%(default))')
    parser.add_argument('--gesture_type', default="static", type=str, help='(default=%(default))')
    parser.add_argument('--engine', default="PyMC", type=str, help='(default=%(default))')
    parser.add_argument('--input_definition_version', default=1, type=int, help='Train hand features')


    return parser.parse_args().__dict__

if __name__ == '__main__':
    args = _args()
    experiment = getattr(Experiments(), args['experiment'])

    experiment(args=args)
