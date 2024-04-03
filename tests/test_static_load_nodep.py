import sys, os
import numpy as np

def test_load_saved_network():
    network_path = os.path.expanduser("~/"+os.getcwd().split('/')[3]+'/src/teleop_gesture_toolbox/include/data/trained_networks/')
    network = 'new_network_y24.npy'

    data = np.load(network_path+network, allow_pickle=True)

    print("Network with name: ", network, " has accuracy: ", data[6]['acc_train'])
    print("Set of gestures are: ", ', '.join(data[6]['Gs']))
    print("---------------------------------------------")
    print("Import/train arguments (flags) are: ", data[6])

if __name__ == '__main__':
    test_load_saved_network()