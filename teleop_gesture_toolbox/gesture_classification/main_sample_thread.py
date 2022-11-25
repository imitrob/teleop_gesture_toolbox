#!/usr/bin/env python3
''' Executes PyMC sampling with pre-learned model
    -> Receives observations (input) through /teleop_gesture_toolbox/static_detection_observations topic
    -> Publishes solutions (output) through /teleop_gesture_toolbox/static_detection_observations topic
    -> Prelearned neural networks are saved in include/data/learned_networks
    -> Network file can be changed through /teleop_gesture_toolbox/change_network service
'''
import sys, os
PATH = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(1, PATH)
sys.path.insert(1, os.path.abspath(os.path.join(PATH, '..')))
# for ros2 run, which has current working directory in ~/<colcon_ws>
sys.path.append("install/teleop_gesture_toolbox/lib/python3.9/site-packages/teleop_gesture_toolbox")

import os_and_utils.settings as settings; settings.init()

import numpy as np
import rclpy
from rclpy.node import Node
import time

from teleop_gesture_toolbox.msg import DetectionSolution, DetectionObservations
from teleop_gesture_toolbox.srv import ChangeNetwork
from std_msgs.msg import Int8, Float64MultiArray

import threading

from os_and_utils.nnwrapper import NNWrapper

class ClassificationSampler(Node):
    def __init__(self, type='static'):
        super().__init__(f"classifier_name_{type}")

        self.Gs = None
        self.args = []

        self.sem = threading.Semaphore()

        self.detection_approach = settings.get_detection_approach(type)
        if self.detection_approach in ['PyMC3', 'PyMC', 'pymc3', 'pymc']:
            self.detection_approach = 'PyMC3'
            from gesture_classification.pymc3_sample import PyMC3_Sample as sample_approach
        elif self.detection_approach in ['PyTorch', 'Torch', 'pytorch', 'torch']:
            self.detection_approach = 'PyTorch'
            from gesture_classification.pytorch_sample import PyTorch_Sample as sample_approach
        elif self.detection_approach in ['DTW', 'dtw', 'fastdtw', 'fastDTW', 'dynamictimewarp']:
            self.detection_approach = 'DTW'
            from gesture_classification.timewarp_lib import fastdtw_ as sample_approach
        else: raise Exception(f"No detection approach found! {type} detection is now offline, check configs.")
        self.sample_approach = sample_approach()

        self.change_type_network_srv = self.create_service(ChangeNetwork, f'/teleop_gesture_toolbox/change_{type}_network', self.change_network_callback)

        if settings.get_network_file(type) == None:
            return
        self.init(settings.get_network_file(type))

        self.pub = self.create_publisher(DetectionSolution, f'/teleop_gesture_toolbox/{type}_detection_solutions', 10)

        self.create_subscription(DetectionObservations, f'/teleop_gesture_toolbox/{type}_detection_observations', self.callback, 10)
        self.type = type

        self.seq = 0

    def callback(self, data):
        ''' When received configuration, generates/sends output
            - header output the header of detection, therefore it just copies it
        '''
        t1 = time.perf_counter()
        self.sem.acquire()
        pred = self.sample_approach.sample(data.observations)
        self.sem.release()

        id = np.argmax(pred[0])
        pred = pred[0]

        sol = DetectionSolution()
        sol.id = int(id)
        sol.probabilities.data = list(np.array(pred, dtype=float))
        sol.header = data.header
        sol.seq = self.seq
        sol.sensor_seq = data.sensor_seq
        sol.approach = self.detection_approach
        self.pub.publish(sol)

        self.seq += 1

    def change_network_callback(self, msg):
        ''' Receives service callback of network change
        '''
        self.init(msg.data)

        msg.Gs = self.Gs
        msg.type = self.type
        msg.filenames = self.filenames
        msg.record_keys = self.record_keys
        msg.args = self.args
        msg.success = True
        return msg

    def init(self, network):
        ''' network parameter is classic file name for PyMC3 '''
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

            self.get_logger().info(f"[Sample thread] network is: {network}")
            ''' network parameter is folder name for PyTorch network folder '''
        elif os.path.isdir(settings.paths.UCB_path+'checkpoints/'+network):
            self.sem.acquire()
            self.nn = self.sample_approach.init(network)
            self.sem.release()
            ''' DTW has as netowork parameter a method name'''
        elif self.detection_approach == 'DTW':
            self.sem.acquire()
            self.nn = self.sample_approach.init(network)
            self.sem.release()
        else:
            print(f"[Sample thread] file ({network}) not found")

class TmpRosNode(Node):
    def __init__(self):
        super().__init__("tmp_reading_parameters_node")

def main():
    if len(sys.argv) > 1 and sys.argv[1] in ['static', 'dynamic']:
        type = sys.argv[1]
    else:
        rclpy.init()
        rosnode = TmpRosNode()
        if rosnode.has_parameter('/project_config/type'):
            type = rosnode.get_parameter('/project_config/type')
        else:
            type = 'static'
        rosnode.destroy_node()
        rclpy.shutdown()

    if type == 'static' and settings.get_detection_approach(type='static'):
        print(f"Launching {type} sampling thread!")
        rclpy.init()
        rosnode = ClassificationSampler(type=type)
        rclpy.spin(rosnode)

    if type == 'dynamic' and settings.get_detection_approach(type='dynamic') and settings.get_network_file(type='dynamic')!=None:
        print(f"Launching {type} sampling thread!")
        rclpy.init()
        rosnode = ClassificationSampler(type=type)
        rclpy.spin(rosnode)

if __name__ == '__main__':
    main()

#
