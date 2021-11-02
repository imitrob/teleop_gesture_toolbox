class NetworkWrapper():
    def __init__(self, _sample_proba=None, X_train=None, approx=None, neural_network=None, Gs=[], observation_type="", time_series_operation="", position=""):
        ## Gestures and config
        self.gesture_names = Gs

        self.observation_type = observation_type #all_defined
        self.time_series_operation = time_series_operation #take_every_10
        self.position = position

        ## NN data
        self._sample_proba = _sample_proba
        self.X_train = X_train
        self.approx = approx
        self.neural_network = neural_network



def save_network(settings,_sample_proba, X_train, approx, neural_network, Gs=[], observation_type='', time_series_operation='', position=''):
    print("saving network")
    n_network = ""
    for i in range(0,200):
        if not isfile(settings.NETWORK_PATH+"/network"+str(i)+".pkl"):
            n_network = str(i)
            break

    wrapper = NetworkWrapper(_sample_proba, X_train, approx, neural_network,Gs=Gs, observation_type=observation_type, time_series_operation=time_series_operation, position=position)
    with open(settings.NETWORK_PATH+"/network"+str(n_network)+'.pkl', 'wb') as output:
        pickle.dump(wrapper, output, pickle.HIGHEST_PROTOCOL)

    print("Network: network"+n_network+".pkl saved")

def load_network(settings, name='network0.pkl'):
    wrapper = NetworkWrapper()
    with open(settings.NETWORK_PATH+"/"+name, 'rb') as input:
        wrapper = pickle.load(input, encoding="latin1")

    return wrapper
