import os, pickle

class NNWrapper():
    ''' Object that holds information about neural network
        Methods for load and save the network are static
            - Use: NNWrapper.save_network()
    '''
    def __init__(self, X_train=None, approx=None, neural_network=None, Gs=[], type='', engine='', args={}, accuracy=-1, record_keys=[], filenames=[]):
        # Set of Gestures in current network
        self.Gs = Gs
        self.record_keys = record_keys
        self.filenames = filenames

        self.type = type
        self.engine = engine

        # Training arguments and import configurations
        self.args = args
        # Accuracy on test data of loaded network <0.,1.>
        self.accuracy = accuracy

        # Training data X, (for backup reasons)
        self.X_train = X_train
        # NN data
        self.approx, self.neural_network = approx, neural_network

    @staticmethod
    def save_network(X_train, approx, neural_network, network_path, name=None, Gs=[], type='', engine='', args={}, accuracy=-1, record_keys=[], filenames=[]):
        '''
        Parameters:
            X_train (ndarray): Your X training data
            approx, neural_network (PyMC3): Neural network data for sampling
            network_path (Str): Path to network folder (e.g. '/home/<user>/<ws>/src/teleop_gesture_toolbox/include/data/trained_networks/')
            name (Str): Output network name
                - name not specified -> will save as network0.pkl, network1.pkl, network2.pkl, ...
                - name specified -> save as name
            Gs (Str[]): List of used gesture names
            args (Str{}): List of arguments of NN and training
            accuracy (Float <0.,1.>): Accuracy on test data
        '''
        print("Saving network")
        n_network = ""
        if name == None:
            for i in range(0,200):
                if not os.path.isfile(network_path+"network"+str(i)+".pkl"):
                    n_network = str(i)
                    break
            name = "network"+str(n_network)
        else:
            # Get rid of extension if included
            if name[-4:] == '.pkl':
                name = name[:-4]
            if os.path.isfile(network_path+name+".pkl"):
                print("File "+name+" exists, network is rewritten!")

        wrapper = NNWrapper(X_train, approx, neural_network, Gs=Gs, type=type, engine=engine, args=args, accuracy=accuracy, record_keys=record_keys, filenames=filenames)
        with open(network_path+name+'.pkl', 'wb') as output:
            pickle.dump(wrapper, output, pickle.HIGHEST_PROTOCOL)

        print(f"Network: {name}.pkl saved")

    @staticmethod
    def load_network(network_path, name=None):
        '''
        Parameters:
            network_path (Str): Path to network folder (e.g. '/home/<user>/<ws>/src/teleop_gesture_toolbox/include/data/trained_networks/')
            name (Str): Network name to load
        Returns:
            wrapper (NetworkWrapper())
        '''
        wrapper = NNWrapper()
        with open(network_path+name, 'rb') as input:
            wrapper = pickle.load(input, encoding="latin1")

        return wrapper


if __name__ == '__main__':
    import sys; sys.path.append('..')
    from os_and_utils import settings; settings.init()
    network_name = 'PyMC3-main-set-3.pkl'
    network_name = 'network0.pkl'
    nw = NNWrapper.load_network(settings.paths.network_path, network_name)
    nw.accuracy
    nw.args
    nw.type
    nw.record_keys
    network_name
    nw.X_train[0]

    NNWrapper.save_network(nw.X_train, nw.approx, nw.neural_network, settings.paths.network_path, name=network_name, Gs=nw.Gs, type=nw.type, engine=nw.engine, args=nw.args, accuracy=nw.accuracy, record_keys=nw.record_keys, filenames=nw.filenames)




#
