import os
import pickle


import os_and_utils.import_data as import_data
import os_and_utils.import_data

class NNWrapper():
    ''' Object that holds information about neural network
        Methods for load and save the network are static
            - Use: NNWrapper.save_network()
    '''
    def __init__(self, X_train=None, approx=None, neural_network=None, Gs=[], args={}, accuracy=-1):
        # Set of Gestures in current network
        self.Gs = Gs

        # Training arguments and import configurations
        self.args = args
        # Accuracy on test data of loaded network <0.,1.>
        self.accuracy = accuracy

        # Training data X
        self.X_train = X_train
        # NN data
        self.approx, self.neural_network = approx, neural_network

    @staticmethod
    def save_network(X_train, approx, neural_network, network_path, name=None, Gs=[], args={}, accuracy=-1):
        '''
        Parameters:
            X_train (ndarray): Your X training data
            approx, neural_network (PyMC3): Neural network data for sampling
            network_path (Str): Path to network folder (e.g. '/home/<user>/<ws>/src/mirracle_gestures/include/data/Trained_network/')
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
            if not os.path.isfile(network_path+name+".pkl"):
                print("Error: file "+name+" exists, network is not saved!")

        wrapper = NNWrapper(X_train, approx, neural_network, Gs=Gs, args=args, accuracy=accuracy)
        with open(network_path+name+'.pkl', 'wb') as output:
            pickle.dump(wrapper, output, pickle.HIGHEST_PROTOCOL)

        print("Network: network"+n_network+".pkl saved")

    @staticmethod
    def load_network(network_path, name=None):
        '''
        Parameters:
            network_path (Str): Path to network folder (e.g. '/home/<user>/<ws>/src/mirracle_gestures/include/data/Trained_network/')
            name (Str): Network name to load
        Returns:
            wrapper (NetworkWrapper())
        '''
        wrapper = NNWrapper()
        import os_and_utils.import_data as import_data
        with open(network_path+name, 'rb') as input:
            wrapper = pickle.load(input, encoding="latin1")

        return wrapper

if False:
    nw = NNWrapper.load_network('/home/pierro/my_ws/src/mirracle_gestures/include/data/Trained_network/', 'network0.pkl')
    NNWrapper.save_network(nw.X_train, nw.approx, nw.neural_network, '/home/pierro/my_ws/src/mirracle_gestures/include/data/Trained_network/', name='network0', Gs=nw.Gs, args=nw.args, accuracy=nw.accuracy)










#
