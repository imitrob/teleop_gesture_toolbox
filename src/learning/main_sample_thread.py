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

        self.detection_approach = settings.get_detection_approach(type)
        if self.detection_approach in ['PyMC3', 'PyMC', 'pymc3', 'pymc']:
            self.detection_approach = 'PyMC3'
            from learning.pymc3_sample import PyMC3_Sample as sample_approach
        elif self.detection_approach in ['PyTorch', 'Torch', 'pytorch', 'torch']:
            self.detection_approach = 'PyTorch'
            from learning.pytorch_sample import PyTorch_Sample as sample_approach
        elif self.detection_approach in ['DTW', 'dtw', 'fastdtw', 'fastDTW', 'dynamictimewarp']:
            self.detection_approach = 'DTW'
            from learning.timewarp_lib import fastdtw_ as sample_approach
        else: raise Exception("No detection approach found!")
        self.sample_approach = sample_approach()

        rospy.init_node('classification_sampler', anonymous=True)
        rospy.Service(f'/mirracle_gestures/change_{type}_network', ChangeNetwork, self.change_network_callback)

        if settings.get_network_file(type) == None:
            return
        self.init(settings.get_network_file(type))
        self.pub = rospy.Publisher(f'/mirracle_gestures/{type}_detection_solutions', DetectionSolution, queue_size=5)
        rospy.Subscriber(f'/mirracle_gestures/{type}_detection_observations', DetectionObservations, self.callback)
        rospy.spin()
        self.type = type

    def callback(self, data):
        ''' When received configuration, generates/sends output
            - header output the header of detection, therefore it just copies it
        '''
        self.sem.acquire()
        pred = self.sample_approach.sample(data.observations)
        self.sem.release()
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
        msg.type = self.type
        msg.filenames = self.filenames
        msg.record_keys = self.record_keys
        msg.args = self.args
        msg.success = True
        return msg

    def init(self, network):
        if network in os.listdir(settings.paths.network_path):
            nn = NNWrapper.load_network(settings.paths.network_path, name=network)

            self.Gs = nn.Gs
            self.filenames = nn.filenames
            self.record_keys = [str(i) for i in nn.record_keys]
            self.args = nn.args
            self.type = nn.type

            self.sem.acquire()
            self.nn = self.sample_approach.init(nn)
            self.sem.release()

            rospy.loginfo(f"[Sample thread] network is: {network}")
        elif os.path.isdir(settings.paths.UCB_path+'checkpoints/'+network):
            self.sem.acquire()
            self.nn = self.sample_approach.init(network)
            self.sem.release()
        else:
            rospy.logerr(f"[Sample thread] file ({network}) not found")


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] in ['static', 'dynamic']:
        type = sys.argv[1]
    else:
        type = rospy.get_param('/mirracle_config/type', 'static')


    if type == 'static' and settings.get_detection_approach(type='static'):
        print(f"Launching {type} sampling thread!")
        ClassificationSampler(type=type)

    if type == 'dynamic' and settings.get_detection_approach(type='dynamic') and settings.get_network_file(type='dynamic')!=None:
        print(f"Launching {type} sampling thread!")
        ClassificationSampler(type=type)





#
