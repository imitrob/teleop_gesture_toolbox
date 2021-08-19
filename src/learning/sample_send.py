#!/usr/bin/env python3.7
''' Executes PyMC sample with pre-learned model
    -> Receives configuration from main.py through /pymcin topic
    -> Publishes output through /pymcout topic
    -> Prelearned neural network in include/data/learned_networks
'''
import sys
import os
PATH = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(1, PATH)
sys.path.insert(1, os.path.abspath(os.path.join(PATH, '..')))

import settings
settings.init(minimal=True)
from import_data import *

from warnings import filterwarnings

import matplotlib.pyplot as plt
import numpy as np
import pymc3 as pm
import seaborn as sns
import sklearn
import theano
import theano.tensor as T

from sklearn import datasets
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import scale
floatX = theano.config.floatX

from std_msgs.msg import Int8, Float64MultiArray
import rospy

def callback(data):
    ''' When received configuration, generates/sends output
    '''
    out = pymc(data)
    pub.publish(out)

def pymc(data):
    ''' Sample from learned network
    '''
    vars = data.data



    pred = sample_proba([vars],100).mean(0)
    ret = np.argmax(pred[0])
    return ret

if True:
    rospy.init_node('pymcpublisher', anonymous=True)

    _sample_proba, X_train, approx, neural_network = load_network(settings)
    x = T.matrix("X")
    n = T.iscalar("n")
    x.tag.test_value = np.empty_like(X_train[:10])
    n.tag.test_value = 100
    _sample_proba = approx.sample_node(
        neural_network.out.distribution.p, size=n, more_replacements={neural_network["ann_input"]: x}
    )
    sample_proba = theano.function([x, n], _sample_proba)

    pub = rospy.Publisher('/pymcout', Int8, queue_size=5)
    sub = rospy.Subscriber('/pymcin', Float64MultiArray, callback)

    rospy.spin()





#
