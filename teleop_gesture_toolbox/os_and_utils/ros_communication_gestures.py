import rclpy
from rclpy.node import Node
from os_and_utils import settings
import numpy as np
from copy import deepcopy
import os_and_utils.move_lib as ml
from os_and_utils.parse_yaml import ParseYAML
import gesture_classification.gestures_lib as gl
from os_and_utils.transformations import Transformations as tfm

from std_msgs.msg import Float64MultiArray, MultiArrayDimension, String
from hand_processing.frame_lib import Frame
import teleop_gesture_toolbox.msg as rosm
from teleop_gesture_toolbox.msg import DetectionSolution, DetectionObservations

class ROSComm(Node):
    def __init__(self):
        super().__init__('ros_comm_gestures')
        self.examplesub = self.create_subscription(rosm.Frame, '/hand_frame', ROSComm.hand_frame_callback, 10)

        if settings.launch_gesture_detection:
            self.create_subscription(DetectionSolution, '/teleop_gesture_toolbox/static_detection_solutions', self.save_static_detection_solutions_callback, 10)
            self.static_detection_observations_pub = self.create_publisher(DetectionObservations,'/teleop_gesture_toolbox/static_detection_observations', 5)

            self.create_subscription(DetectionSolution, '/teleop_gesture_toolbox/dynamic_detection_solutions', self.save_dynamic_detection_solutions_callback, 10)
            self.dynamic_detection_observations_pub = self.create_publisher(DetectionObservations, '/teleop_gesture_toolbox/dynamic_detection_observations', 5)

        self.hand_mode = settings.get_hand_mode()

        self.gesture_solution_pub = self.create_publisher(String, '/teleop_gesture_toolbox/filtered_gestures', 5)

    def get_time(self):
        return self.get_clock().now().nanoseconds/1e9

    def gesture_solution_publish(self, msg):
        return self.gesture_solution_pub.publish(msg)

    @staticmethod
    def hand_frame_callback(data):
        ''' Hand data received by ROS msg is saved
        '''
        #print("new data")
        f = Frame()
        f.import_from_ros(data)
        ml.md.frames.append(f)

    @staticmethod
    def save_static_detection_solutions_callback(data):
        gl.gd.new_record(data, type='static')

    @staticmethod
    def save_dynamic_detection_solutions_callback(data):
        gl.gd.new_record(data, type='dynamic')

    def send_g_data(self, dynamic_detection_window=1.5):
        ''' Sends appropriate gesture data as ROS msg
            Launched node for static/dynamic detection.
        '''
        if len(ml.md.frames) == 0: return
        hand_mode = self.hand_mode

        msg = DetectionObservations()
        msg.observations = Float64MultiArray()
        msg.sensor_seq = ml.md.frames[-1].seq
        msg.header.stamp.sec = ml.md.frames[-1].sec
        msg.header.stamp.nanosec = ml.md.frames[-1].nanosec

        mad1 = MultiArrayDimension()
        mad1.label = 'time'
        mad2 = MultiArrayDimension()
        mad2.label = 'xyz'

        for key in hand_mode.keys():
            args = gl.gd.static_network_info

            send_static_g_data_bool = hand_mode is not None and 'static' in hand_mode[key] and gl.gd.static_network_info is not None
            if send_static_g_data_bool:
                if key == 'l':
                    if ml.md.l_present():
                        msg.observations.data = ml.md.frames[-1].l.get_learning_data_static(definition=args['input_definition_version'])
                        msg.header.frame_id = 'l'
                        self.static_detection_observations_pub.publish(msg)
                elif key == 'r':
                    if ml.md.r_present():
                        msg.observations.data = ml.md.frames[-1].r.get_learning_data_static(definition=args['input_definition_version'])
                        msg.header.frame_id = 'r'
                        self.static_detection_observations_pub.publish(msg)


            time_samples = settings.yaml_config_gestures['misc_network_args']['time_samples']
            send_dynamic_g_data_bool = hand_mode is not None and 'dynamic' in hand_mode[key] and len(ml.md.frames) > time_samples and gl.gd.dynamic_network_info is not None
            if send_dynamic_g_data_bool:
                args = gl.gd.dynamic_network_info
                if getattr(ml.md, key+'_present')():# and getattr(ml.md.frames[-1], key).grab_strength < 0.5:
                    try:
                        '''  '''
                        n = 1
                        visibles = []
                        while True:
                            ttt = ml.md.frames[-1].stamp() - ml.md.frames[-n].stamp()
                            visibles.append( ml.md.frames[-n].visible )
                            if ttt > 1.5: break
                            n += 1
                        if not np.array(visibles).all():
                            return

                        ''' Creates timestamp indexes starting with [-1, -x, ...] '''
                        time_samples_series = [-1]
                        time_samples_series.extend((n * np.array(range(-1, -time_samples, -1))  / time_samples).astype(int))
                        time_samples_series.sort()

                        ''' Compose data '''
                        data_composition = []
                        for time_sample in time_samples_series:
                            data_composition.append(getattr(ml.md.frames[time_sample], key).get_single_learning_data_dynamic(definition=args['input_definition_version']))

                        ''' Transform to Leap frame id '''
                        data_composition_ = []
                        for point in data_composition:
                            data_composition_.append(tfm.transformLeapToBase(point, out='position'))
                        data_composition = data_composition_

                        ''' Check if the length of composed data is aorund 1sec '''
                        ttt = ml.md.frames[-1].stamp() - ml.md.frames[int(time_samples_series[0])].stamp()
                        if not (0.7 <= ttt <= 2.0):
                            print(f"WARNING: data frame composed is {ttt} long")
                        ''' Subtract middle path point from all path points '''

                        #if 'normalize' in args:
                        data_composition_ = []
                        data_composition0 = deepcopy(data_composition[len(data_composition)//2])
                        for n in range(0, len(data_composition)):
                            data_composition_.append(np.subtract(data_composition[n], data_composition0))
                        data_composition = data_composition_

                        ''' Fill the ROS msg '''
                        data_composition = np.array(data_composition, dtype=float)

                        mad1.size = data_composition.shape[0]
                        mad2.size = data_composition.shape[1]
                        data_composition = list(data_composition.flatten())
                        msg.observations.data = data_composition
                        msg.observations.layout.dim = [mad1, mad2]
                        msg.header.frame_id = key
                        self.dynamic_detection_observations_pub.publish(msg)
                    except IndexError:
                        pass

    def detection_thread(self, freq=1., args={}):
        if not ROS: raise Exception("ROS cannot be imported!")

        rclpy.init()

        settings.gesture_detection_on = True
        settings.launch_gesture_detection = True

        roscm = ROSComm()

        configGestures = ParseYAML.load_gesture_config_file(settings.paths.custom_settings_yaml)
        hand_mode_set = configGestures['using_hand_mode_set']
        hand_mode = dict(configGestures['hand_mode_sets'][hand_mode_set])

        rate = roscm.create_rate(freq)

        while rclpy.ok():
            if ml.md.frames:
                send_g_data(roscm, hand_mode, args)

            rate.sleep()

def init():
    global roscm
    rclpy.init()
    roscm = ROSComm()
