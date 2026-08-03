#!/usr/bin/env python
''' Torch replacement of the former PyMC backend.
    Same model: 2-layer tanh Bayesian NN, mean-field variational inference
    (ADVI equivalent, Bayes-by-backprop), posterior = independent Normal(mu, sigma).
    Saved format is unchanged (.json config + .npz with X/y splits and flat
    mu/sigma = [W1.ravel(), W2.ravel()]), so old PyMC-trained models load as-is.
'''
import os, time, argparse, json
import numpy as np
import torch
import torch.nn.functional as F

import gesture_detector

np.random.seed(0)
torch.manual_seed(0)


class TorchModel():
    def __init__(self, model_config):
        self.model_config = model_config
        self.mu = None     # flat posterior means, np.float64 (n_weights,)
        self.sigma = None  # flat posterior stds, np.float64 (n_weights,)
        self.out_n = None

    def _unflatten(self, flat, n_in, out_n):
        ''' Flat vector -> (W1 (n_in, n_hidden), W2 (n_hidden, out_n)), same
            ordering the PyMC version used. '''
        n_hidden = self.model_config['n_hidden']
        bound = n_in * n_hidden
        t = torch.as_tensor(np.asarray(flat), dtype=torch.float32)
        return t[:bound].reshape(n_in, n_hidden), t[bound:].reshape(n_hidden, out_n)

    def import_records(self, gesture_names):
        ''' Import static gestures, Import all data from learning folder
        '''
        from gesture_detector.utils.loading import DatasetLoader
        print(f"Loading dataset from: {gesture_detector.gesture_data_path}")
        if self.model_config['gesture_type'] == 'static':
            X, y = DatasetLoader({'input_definition_version':1, 'take_every': self.model_config['take_every']}).load_static(gesture_detector.gesture_data_path, gesture_names)
        elif self.model_config['gesture_type'] == 'dynamic':
            dataloader_args = {'interpolate':1, 'discards':1, 'normalize':1, 'normalize_dim':1, 'n':0}
            X, y = DatasetLoader(dataloader_args).load_dynamic(gesture_detector.gesture_data_path, gesture_names)
        print(f"X shape={np.array(X).shape}")

        assert not np.any(np.isnan(X))
        assert len(X) == len(y), "Lengths not match"
        if len(y) == 0:
            print("No new data!")
            return

        g = list(dict.fromkeys(y))
        counts = [list(y).count(g_) for g_ in g]
        print(f"counts = {counts}")
        self.model_config['counts'] = counts
        return X, y

    def fit(self, X, y):
        ''' Mean-field VI: maximize ELBO = E_q[log p(y|X,w)] - KL(q(w) || N(0,1)),
            one reparametrized weight sample per step.
        '''
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        X = torch.as_tensor(np.asarray(X), dtype=torch.float32, device=device)
        y = torch.as_tensor(np.asarray(y), dtype=torch.long, device=device)
        self.out_n = len(torch.unique(y))
        n_hidden = self.model_config['n_hidden']
        bound = X.shape[1] * n_hidden
        n_weights = bound + n_hidden * self.out_n

        torch.manual_seed(self.model_config['seed'])
        mu = torch.randn(n_weights, device=device, requires_grad=True)
        log_sigma = torch.full((n_weights,), -3.0, device=device, requires_grad=True)

        opt = torch.optim.Adam([mu, log_sigma], lr=1e-2)
        tstart = time.time()
        for i in range(self.model_config['iter']):
            opt.zero_grad()
            sigma = log_sigma.exp()
            w = mu + sigma * torch.randn_like(mu)
            logits = torch.tanh(X @ w[:bound].reshape(X.shape[1], n_hidden)) @ w[bound:].reshape(n_hidden, self.out_n)
            nll = F.cross_entropy(logits, y, reduction='sum')
            kl = 0.5 * (sigma**2 + mu**2 - 1.0 - 2.0*log_sigma).sum()
            (nll + kl).backward()
            opt.step()
        print("Train time [s]: ", time.time()-tstart)

        self.mu = mu.detach().cpu().numpy().astype(np.float64)
        self.sigma = log_sigma.exp().detach().cpu().numpy().astype(np.float64)
        return self

    def sample_predict(self, X, draws):
        ''' Posterior predictive: sample weights `draws` times, sample one
            categorical outcome per draw. Returns labels (n_obs, draws). '''
        X = torch.as_tensor(np.asarray(X), dtype=torch.float32)
        W1m, W2m = self._unflatten(self.mu, X.shape[1], self.out_n)
        W1s, W2s = self._unflatten(self.sigma, X.shape[1], self.out_n)
        with torch.no_grad():
            W1 = W1m + W1s * torch.randn(draws, *W1m.shape)
            W2 = W2m + W2s * torch.randn(draws, *W2m.shape)
            logits = torch.tanh(X @ W1) @ W2                      # (draws, n_obs, out_n)
            outcomes = torch.distributions.Categorical(logits=logits).sample()
        return outcomes.numpy().T                                 # (n_obs, draws)

    def evaluate(self, X_test, y_test, X_train, y_train):
        ''' Evaluate network
        '''
        self.out_n = len(list(dict.fromkeys(y_train)))

        def eval(X_true, y_true, save_name, draws=300):
            pred = self.sample_predict(X_true, draws)
            y_pred = np.array([np.bincount(row, minlength=self.out_n).argmax() for row in pred])
            y_true = np.array(y_true).astype(int)

            acc = (y_true == y_pred).mean() * 100
            print(f"Accurracy {save_name} = {acc}%")
            self.model_config[save_name] = acc

            from gesture_detector.utils.pretty_confusion_matrix import pp_matrix_from_data
            pp_matrix_from_data(y_true, y_pred, self.model_config['gestures'], annot=True, cmap = 'Oranges', cbar=False, figsize=(9,9))
            return acc

        eval(X_train, y_train, 'acc_train')
        return eval(X_test, y_test, 'acc_test')

    def save(self, X_train, y_train, X_test, y_test):
        ''' Save the network
        '''
        assert (self.sigma >= 0).all(), f"Negative sigma {self.sigma}"
        assert len(self.mu) == (X_train.shape[1] * self.model_config['n_hidden'] + self.model_config['n_hidden'] * len(list(dict.fromkeys(y_train))))

        name = self.model_config['model_name']
        if name is None:
            for i in range(0,200):
                if not os.path.isfile(gesture_detector.saved_models_path+"network"+str(i)+".npz"):
                    name = f"network{i}"
                    break
        elif os.path.isfile(gesture_detector.saved_models_path+name+".npz"):
            print(f"File {name} exists, network is rewritten!")

        with open(gesture_detector.saved_models_path+name+".json", 'w') as f:
            json.dump(self.model_config, f, indent=4)

        np.savez(gesture_detector.saved_models_path+name,
            X_train=X_train, y_train=y_train, X_test=X_test,
            y_test=y_test, mu=self.mu, sigma=self.sigma,
        )
        print(f"Network: {name}.npz saved")

    def load(self):
        ''' Load Network
        '''
        name = self.model_config['model_name']
        with open(gesture_detector.saved_models_path+name+".json", 'r') as f:
            self.model_config = json.load(f)
        nn = np.load(gesture_detector.saved_models_path+name+".npz")

        self.mu = nn['mu']
        self.sigma = nn['sigma']
        self.out_n = len(list(dict.fromkeys(nn['y_train'])))
        return nn['X_train'], nn['y_train'], nn['X_test'], nn['y_test']


class Torch_Sample():
    ''' Drop-in for the former PyMC_Sample: init(nn, model_config), then
        sample(x) -> (predicted label, class probabilities). '''
    def __init__(self, draws=100):
        self.model = None
        self.draws = draws

    def init(self, nn, model_config):
        self.model = TorchModel(model_config)
        self.model.mu = nn['mu']
        self.model.sigma = nn['sigma']
        self.model.out_n = len(list(dict.fromkeys(nn['y_train'])))
        self.out_n = self.model.out_n

    def sample(self, data):
        data = np.atleast_2d(np.array(data))
        assert data.ndim == 2

        pred = self.model.sample_predict(data, self.draws)   # (1, draws)
        probs = self.pred_labels_to_probs(pred[0])
        return int(probs.argmax()), probs

    def pred_labels_to_probs(self, labels):
        counts = np.bincount(np.asarray(labels, dtype=int), minlength=self.out_n)
        return counts / counts.sum()


class Experiments():
    def load_and_evaluate(self, args):
        ''' Loads network file and evaluate it
        '''
        print("Load and evaluate\n---")
        self.model = TorchModel(args)
        X_train, y_train, X_test, y_test = self.model.load()
        return self.model.evaluate(X_test, y_test, X_train, y_train)

    def train_custom(self, args):
        ''' Train/evaluate + Save
        '''
        from sklearn.model_selection import train_test_split
        print("Training With Parameters and Save\n---")
        self.model = TorchModel(args)

        X, y = self.model.import_records(gesture_names=args['gestures'])
        X_train, X_test, y_train, y_test = train_test_split(X, y,
            test_size=self.model.model_config['split'])

        self.model.fit(X_train, y_train)
        acc = self.model.evaluate(X_test, y_test, X_train, y_train)

        if args['save']:
            self.model.save(X_train, y_train, X_test, y_test)
            return self.load_and_evaluate(args)
        return acc


def _args():
    parser=argparse.ArgumentParser(description='')
    parser.add_argument('--experiment', default="train_custom", type=str, help='(default=%(default))', choices=['load_and_evaluate', 'train_custom'])

    # List of gestures to train
    parser.add_argument('--gestures', nargs='+', default=['grab','pinch','point','two','three','four','five','thumbsup'], help='List of gestures')
    parser.add_argument('--model_name', default='common_gestures', type=str, help='(default=%(default))')
    parser.add_argument('--save', default=True, type=bool, help='(default=%(default))')

    # model
    parser.add_argument('--n_hidden', default=20, type=int, help='(default=%(default))')
    parser.add_argument('--split', default=0.3, type=float, help='(default=%(default))')
    parser.add_argument('--take_every', default=4, type=int, help='Frame dropping policy')
    # fit
    parser.add_argument('--iter', default=5000, type=int, help='(default=%(default))')
    parser.add_argument('--seed', default=93457, type=int, help='(default=%(default))')

    # Const
    parser.add_argument('--inference_type', default='ADVI', type=str, help='(default=%(default))')
    parser.add_argument('--layers', default=2, type=int, help='(default=%(default))')
    parser.add_argument('--gesture_type', default="static", type=str, help='(default=%(default))')
    parser.add_argument('--engine', default="Torch", type=str, help='(default=%(default))')
    parser.add_argument('--input_definition_version', default=1, type=int, help='Train hand features')

    return parser.parse_args().__dict__

if __name__ == '__main__':
    args = _args()
    experiment = getattr(Experiments(), args['experiment'])

    experiment(args=args)
