#!/usr/bin/env python3
''' Executes sampling with pre-learned model
    -> Receives observations (input) through /teleop_gesture_toolbox/static_detection_observations topic
    -> Publishes solutions (output) through /teleop_gesture_toolbox/static_detection_observations topic
    -> Prelearned neural networks are saved in saved_models folder
'''
import sys, os, json, time, threading
import numpy as np
import rclpy
from rclpy.node import Node

from gesture_msgs.msg import DetectionSolution, DetectionObservations
from gesture_msgs.srv import GetModelConfig
import gesture_detector
from gesture_detector.gesture_classification.pymc_lib import PyMC_Sample
from gesture_detector.gesture_classification.timewarp_lib import fastdtw_

class ClassificationSampler(Node):
    def __init__(self, network_name: str):
        with open(gesture_detector.saved_models_path+network_name+".json", 'r') as f:
            model_config = json.load(f)
        model = np.load(gesture_detector.saved_models_path+network_name+".npz")

        self.type = model_config['gesture_type']
        super().__init__(f"classifier_name_{self.type}")

        self.gestures = model_config['gestures']
        self.args = model_config

        self.sem = threading.Semaphore()

        self.detection_approach = model_config['engine']
        if self.detection_approach == 'PyMC':
            self.sample_approach = PyMC_Sample()
        elif self.detection_approach == 'DTW':
            self.sample_approach = fastdtw_()
        else: raise Exception(f"No detection approach found! {self.type} detection is now offline, check configs.")

        self.sem.acquire()
        self.model = self.sample_approach.init(model, model_config)
        self.sem.release()

        self.pub = self.create_publisher(DetectionSolution, f'/teleop_gesture_toolbox/{self.type}_detection_solutions', 10)
        self.create_subscription(DetectionObservations, f'/teleop_gesture_toolbox/{self.type}_detection_observations', self.callback, 10)

        self.seq = 0
        self.get_logger().info(f"[Sample thread] network is: {network_name}")

        self.srv = self.create_service(GetModelConfig, f"/teleop_gesture_toolbox/{self.type}_detection_info", self.send_sampler_config)
        

    def callback(self, data):
        ''' When received configuration, generates/sends output
            - header output the header of detection, therefore it just copies it
        '''
        t1 = time.perf_counter()
        self.sem.acquire()

        # prepare input data 
        x = np.array(data.observations.data).squeeze()
        assert x.ndim == 1

        pred, probs = self.sample_approach.sample(x)
        self.sem.release()

        sol = DetectionSolution()

        sol.id = int(pred)
        sol.probabilities.data = list(np.array(probs, dtype=float))

        sol.header = data.header
        sol.seq = self.seq
        sol.sensor_seq = data.sensor_seq
        sol.approach = self.detection_approach
        self.pub.publish(sol)

        self.seq += 1

    def send_sampler_config(self, request, response):
        response.gestures = list(self.gestures)
        return response

def run_static():
    print(f"Launching static sampling thread!")
    rclpy.init()
    rosnode = ClassificationSampler(network_name='common_gestures')
    rclpy.spin(rosnode)

def run_dynamic():
    print(f"Launching dynamic sampling thread!")
    rclpy.init()
    rosnode = ClassificationSampler(network_name='DTW99')
    rclpy.spin(rosnode)


def run_from_rosparam():
    rclpy.init()
    node = Node("tmp_node")
    node.declare_parameter('model', '')
    network_name = node.get_parameter('model').get_parameter_value().string_value
    node.destroy_node()

    rosnode = ClassificationSampler(network_name)
    rclpy.spin(rosnode)


def main():
    if len(sys.argv) > 1 and sys.argv[1] in ['static', 'dynamic']:
        type = sys.argv[1]

    if type == 'static':
        run_static()

    if type == 'dynamic':
        run_dynamic()

if __name__ == '__main__':
    main()
