#!/usr/bin/env python3
''' Run main_sample_thread.py with: gesture_detection_approach: pymc3
                                    in include/custom_settings/main_config.yaml
'''
import numpy as np
import pytensor as pt
from teleop_gesture_toolbox.learning.pymc_train import PyMCModel

class PyMC_Sample():
    def __init__(self, draws=100):
        self.pymcmodel = None
        self.built_model = None
        self.draws = draws

    def sample(self, data):
        trace = built_model.sample(draws=draws)
        with self.model:
            ppc = pm.sample_posterior_predictive(trace)
            trace.extend(ppc)
        pred = np.array(ppc.posterior_predictive['out'])[0].T

        y_pred = Counter(pred[0]).most_common()[0][0]
        return y_pred  


    def init(self, nn):
        X_train, y_train, X_test, y_test, mu, sigma, model_config = nn
        self.pymcmodel = PyMCModel(args=model_config)
        fitted_model, X_train, y_train, X_test, y_test = self.pymcmodel.load()
        X_true = X_test[0:1]
        y_true = y_test[0:1]
        out_n = len(list(dict.fromkeys(y_train)))
        self.built_model = self.pymcmodel.build_model_from_values(X_true, y_true, fitted_model.mean.eval(), fitted_model.std.eval(), self.pymcmodel.model_config, out_n=out_n)





if __name__ == '__main__':
    import teleop_gesture_toolbox.os_and_utils.settings as settings; settings.init()

    network='network99.pkl'
    pymc_sampler = PyMC_Sample()
    if network not in os.listdir(settings.paths.network_path):
        raise Exception(f"network {network} not found in folder {settings.paths.network_path}")
    
    X_train, y_train, X_test, y_test, mu, sigma, model_config = np.load(network_path+name+".npy", allow_pickle=True)
    # nn = NNWrapper.load_network(settings.paths.network_path, name=network)
    
    # Gs = nn.Gs
    # filenames = nn.filenames
    # record_keys = [str(i) for i in nn.record_keys]
    # args = nn.args
    # type = nn.type

    nn = pymc_sampler.init(nn)

    ''' Update '''
    sys.path.append(f"hand_processing")
    from os_and_utils import visualizer_lib as vis
    frames = np.load(settings.paths.learn_path+'/closed_hand/1.npy', allow_pickle=True)
    hand_line_figs = [vis.HandPlot.generate_hand_lines(frame, 'l', alpha=0.3) for frame in frames]

    data = np.array([0.06358485779154385, -0.15050339775998145, -0.22214693578367792, 2.868813702202628, 0.09517636474821028, 0.8165705052108913, 0.15058080289077483, -0.3327055107505606, 0.19957592131630633, -1.3544711145853408, -0.31356914103819583, 3.3265682136814276, 1.3378504480284665, 0.3009527012544441, -0.5657777620983043, -3.432787973998601, -0.5377633198294556, 0.050924886506170886, -0.3000064873040276, 3.18944005358219, 1.2742267754250138, 0.7883472993762892, -0.5816858549579922, -4.093264877585241, -0.5057476568840614, 0.16258783574451474, -0.2801703970706638, 3.039251411900846, 1.176345477625465, 0.8931816602771204, -0.4957298965394693, 2.039563627612269, -0.5003515309816441, -6.06494882248769, -0.2999152171820148, 2.894129690850832, 1.0964997109652377, 1.131581456299915, -0.5593977937646291, 1.6156277935175338, -0.4452050833276003, 0.2834757969583701, 28.02719057591241, 36.97758428419364, 44.905674553083244, 49.116098237015315, 53.49713435802105, 9.495764391453497, 19.974466899718255, 28.39720228278824, 47.38622422359043, 12.684653976994538, 24.175882774913863, 50.29311618545047, 13.651614689482463, 47.63181684696371, 37.944478413384985])
    data = np.array(frames[-1].l.get_learning_data())


    pred = pymc_sampler.sample(data)
