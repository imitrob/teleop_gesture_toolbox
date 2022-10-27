import sys, os
sys.path.append("../os_and_utils")
sys.path.append("..")
network_path = os.path.expanduser("~/"+os.getcwd().split('/')[3]+'/src/teleop_gesture_toolbox/include/data/trained_networks/')
from nnwrapper import NNWrapper

network = input("Enter network file to read info: (with .pkl extension) (located in `include/data/trained_networks/`)")
nn = NNWrapper.load_network(network_path, name=network)

print("Network with name: ", network, " has accuracy: ", nn.accuracy)
print("Set of gestures are: ", ', '.join(nn.Gs))
print("Import/train arguments (flags) are: ", ', '.join(nn.args))
