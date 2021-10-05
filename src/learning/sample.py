#!/usr/bin/env python3
''' Executes PyMC sampling with pre-learned model
    -> Receives configuration from main.py through /mirracle_gestures/pymcin topic
    -> Publishes output through /mirracle_gestures/pymcout topic
    -> Prelearned neural network in include/data/learned_networks
    -> NETWORK file can be changed through /mirracle_gestures/change_network service
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

from mirracle_gestures.srv import ChangeNetwork, ChangeNetworkResponse
from std_msgs.msg import Int8, Float64MultiArray
import rospy
import time

import threading
from threading import Thread

class ProbabilisticGestureComputationSamplePublisher():
    def __init__(self):
        self.sample_proba = None
        self._sample_proba = None
        self.X_train = None
        self.approx = None
        self.neural_network = None

        self.Gs = None
        self.args = []

        self.sem = threading.Semaphore()

        rospy.init_node('pymcpublisher', anonymous=True)
        rospy.Service('/mirracle_gestures/change_network', ChangeNetwork, self.change_network_callback)
        self.init_pymc(settings.GESTURE_NETWORK_FILE)
        self.pub = rospy.Publisher('/mirracle_gestures/pymcout', Int8, queue_size=5)
        self.sub = rospy.Subscriber('/mirracle_gestures/pymcin', Float64MultiArray, self.callback)


    def callback(self, data):
        ''' When received configuration, generates/sends output
        '''
        out = self.pymc(data)
        self.pub.publish(out)

    def change_network_callback(self, msg):
        ''' Receives service callback of network change
        '''
        self.init_pymc(msg.data)

        msg = ChangeNetworkResponse()
        msg.Gs = self.Gs
        msg.args = self.args
        msg.success = True
        return msg

    def pymc(self, data):
        ''' Sample from learned network
        '''
        vars = data.data

        self.sem.acquire()
        pred = self.sample_proba([vars],100).mean(0)
        self.sem.release()
        ret = np.argmax(pred[0])
        return ret

    def init_pymc(self, network):
        if network in os.listdir(settings.NETWORK_PATH):
            nn = NNWrapper.load_network(settings.NETWORK_PATH, name=network)
            self.X_train = nn.X_train
            self.approx = nn.approx
            self.neural_network = nn.neural_network
            self.Gs = nn.Gs
            self.args = nn.args
            x = T.matrix("X")
            n = T.iscalar("n")
            x.tag.test_value = np.empty_like(self.X_train[:10])
            n.tag.test_value = 100
            self._sample_proba = self.approx.sample_node(
                self.neural_network.out.distribution.p, size=n, more_replacements={self.neural_network["ann_input"]: x}
            )
            self.sem.acquire()
            self.sample_proba = theano.function([x, n], self._sample_proba)
            self.sem.release()

            rospy.loginfo("[Sample thread] network is: "+network)
        else:
            rospy.logerr("[Sample thread] file not found")




ProbabilisticGestureComputationSamplePublisher()
rospy.spin()


#
