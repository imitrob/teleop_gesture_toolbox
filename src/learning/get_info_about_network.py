import sys
import os
from os.path import expanduser

from import_data import *

PATH = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(1, PATH)
sys.path.insert(1, os.path.abspath(os.path.join(PATH, '..')))
network_path = expanduser("~/"+PATH.split('/')[3]+'/src/teleop_gesture_toolbox/include/data/Trained_network/')

network = input("Enter network file to read info: ")
nn = NNWrapper.load_network(network_path, name=network)
args = nn.args
Gs = nn.Gs
accuracy = nn.accuracy
print("Network with name: ", network, " has accuracy: ", accuracy)
print("Set of gestures are: ", ', '.join(Gs))
print("Import/train arguments (flags) are: ", ', '.join(args))
