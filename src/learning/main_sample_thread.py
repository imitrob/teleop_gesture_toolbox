#!/usr/bin/env python3
''' Executes PyMC sampling with pre-learned model
    -> Receives observations (input) through /mirracle_gestures/static_detection_observations topic
    -> Publishes solutions (output) through /mirracle_gestures/static_detection_observations topic
    -> Prelearned neural networks are saved in include/data/learned_networks
    -> Network file can be changed through /mirracle_gestures/change_network service
'''
import sys, os
PATH = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(1, PATH)
sys.path.insert(1, os.path.abspath(os.path.join(PATH, '..')))

import settings
settings.init()

import numpy as np
import rospy

from mirracle_gestures.msg import DetectionSolution, DetectionObservations
from mirracle_gestures.srv import ChangeNetwork, ChangeNetworkResponse
from std_msgs.msg import Int8, Float64MultiArray

import threading

from os_and_utils.nnwrapper import NNWrapper

class ClassificationSampler():
    def __init__(self, type='static'):
        self.Gs = None
        self.args = []

        self.sem = threading.Semaphore()

        self.detection_approach = settings.configGestures['static_detection_approach']
        if self.detection_approach in ['PyMC3', 'PyMC', 'pymc3', 'pymc']:
            self.detection_approach = 'PyMC3'
            from learning.pymc3_sample import PyMC3_Sample as sample_approach
        elif self.detection_approach in ['PyTorch', 'Torch', 'pytorch', 'torch']:
            self.detection_approach = 'PyTorch'
            from learning.pytorch_sample import PyTorch_Sample as sample_approach
        self.sample_approach = sample_approach()

        rospy.init_node('classification_sampler', anonymous=True)
        rospy.Service('/mirracle_gestures/change_network', ChangeNetwork, self.change_network_callback)
        self.init(settings.configGestures['network_file'])
        self.pub = rospy.Publisher(f'/mirracle_gestures/{type}_detection_solutions', DetectionSolution, queue_size=5)
        rospy.Subscriber(f'/mirracle_gestures/{type}_detection_observations', DetectionObservations, self.callback)
        rospy.spin()

    def callback(self, data):
        ''' When received configuration, generates/sends output
            - header output the header of detection, therefore it just copies it
        '''
        pred = self.pymc(data.observations.data)
        id = np.argmax(pred[0])

        sol = DetectionSolution()
        sol.id = id
        sol.probabilities.data = pred[0]
        sol.header = data.header
        sol.sensor_seq = data.sensor_seq
        sol.approach = self.detection_approach
        self.pub.publish(sol)

    def change_network_callback(self, msg):
        ''' Receives service callback of network change
        '''
        self.init(msg.data)

        msg = ChangeNetworkResponse()
        msg.Gs = self.Gs
        msg.args = self.args
        msg.success = True
        return msg

    def pymc(self, data):
        ''' Sample from learned network
        '''
        self.sem.acquire()
        pred = self.sample_approach.sample(data)
        self.sem.release()
        return pred

    def init(self, network):
        if network in os.listdir(settings.paths.network_path):
            nn = NNWrapper.load_network(settings.paths.network_path, name=network)

            self.Gs = nn.Gs
            self.args = nn.args

            self.sem.acquire()
            self.nn = self.sample_approach.init(nn)
            self.sem.release()

            rospy.loginfo(f"[Sample thread] network is: {network}")
        else:
            rospy.logerr("[Sample thread] file not found")


if __name__ == '__main__':
    type = rospy.get_param('/mirracle_config/type', 'static')
    if type == 'static' and settings.configGestures['static_detection_approach']:
        ClassificationSampler(type=type)

    if type == 'dynamic' and settings.configGestures['dynamic_detection_approach']:
        ClassificationSampler(type=type)





#
