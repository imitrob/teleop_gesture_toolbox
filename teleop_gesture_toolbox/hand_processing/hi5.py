#!/usr/bin/env python
import sys, os; sys.path.append("..")
import numpy as np
if os.getcwd()[-4:] == '.ros': # if running from roslaunch
    import rospkg; rospack = rospkg.RosPack(); rospack.list()
    sys.path.append(rospack.get_path('teleop_gesture_toolbox')+'/src/')
# for ros2 run, which has current working directory in ~/<colcon_ws>
sys.path.append("../python3.9/site-packages/teleop_gesture_toolbox")
import time, argparse, inspect

from os_and_utils.saving import Recording
from os_and_utils.utils import ros_enabled, GlobalPaths
import frame_lib
import ctypes

import rclpy
from rclpy.node import Node
import teleop_gesture_toolbox.msg as rosm
import teleop_gesture_toolbox.srv as ross
from geometry_msgs.msg import PoseStamped
from crow_msgs.msg import GlovesPosition
#ross.SaveHandRecord, ross.SaveHandRecordResponseS
#from sensor_msgs.msg import Image

LEFT_TRACKER_TOPIC = '/htc/tracker_LHR_5E1AEA9F'
RIGHT_TRACKER_TOPIC = '/htc/tracker_LHR_CF29F0F4'
HI5_GLOVES_TOPIC = '/htc/gloves'

class Hi5PublisherNode(Node):
    def __init__(self):
        super().__init__('hi5_publisher')
        self.frame_publisher = self.create_publisher(rosm.Frame, '/hand_frame', 5)
        #self.srv = self.create_service(ross.SaveHandRecord, 'save_hand_record', self.save_hand_record_callback)
        self.hi5_left_tracker_sub = self.create_subscription(PoseStamped, LEFT_TRACKER_TOPIC, self.receive_left_tracker, 5)
        self.hi5_right_tracker_sub = self.create_subscription(PoseStamped, RIGHT_TRACKER_TOPIC, self.receive_right_tracker, 5)
        self.hi5_gloves_subscriber_sub = self.create_subscription(GlovesPosition, HI5_GLOVES_TOPIC, self.receive_hi5_and_send_handframe_data, 5)

        self.left_tracker = PoseStamped()
        self.right_tracker = PoseStamped()
        self.id = 0


    #def save_hand_record_callback(self, request, response):
    #    self.record.recording_requests.append([request.directory, request.save_method, request.recording_length])
    #    response.success = True
    #    return response

    def receive_left_tracker(self, msg):
        #print("leftric")
        self.left_tracker = msg

    def receive_right_tracker(self, msg):
        #print("rightric")
        self.right_tracker = msg

    def receive_hi5_and_send_handframe_data(self, msg):

        f = frame_lib.Frame((msg, self.left_tracker, self.right_tracker, self.id), None, import_device_if_available='hi5')
        self.id += 1
        if f is None:
            return

        fros = f.to_ros()
        self.frame_publisher.publish(fros)
        print("sent")
        #self.record.auto_handle(f)
        if self.print_on:
            #print(f.r.get_learning_data(definition=1))
            print(f)

def spin_default(listener, record_settings):
    time.sleep(0.5)

def spin_record_with_enter(listener, record_settings):
    input()
    listener.record.recording_requests.append(record_settings)

def main():
    parser=argparse.ArgumentParser(description='')
    parser.add_argument('--print', default=False, type=bool, help='(default=%(default)s)')
    #parser.add_argument('--record_with_enter', default=False, type=bool, help='(default=%(default)s)', choices=[True, False])
    #parser.add_argument('--recording_length', default=1., type=float, help='(default=%(default)s)')
    #parser.add_argument('--directory', default='', type=str, help='(default=%(default)s)')
    #parser.add_argument('--save_method', default='numpy', type=str, help='(default=%(default)s)')
    # TODO:
    #parser.add_argument("--send_images", default=False, type=bool)

    try:
        args=parser.parse_args()
        print_on = args.print
        #record_with_enter = args.record_with_enter
        #recording_length = args.recording_length
        #directory = args.directory
        #save_method = args.save_method
        #send_images = args.send_images
    except:
        print("[Leap] parse_args failed -> parse_known_args")
        args=parser.parse_known_args()
        print_on = False
        #record_with_enter = False
        #recording_length = 1.0
        #directory = ''
        #save_method = 'numpy'
        #send_images = False

    #if directory == '':
    #    #directory = os.path.abspath(os.path.join(os.path.dirname(inspect.getabsfile(inspect.currentframe())), '..', '..'))+'/include/data/learning/'
    #    directory = GlobalPaths().learn_path

    print(f"[Hi5] Printing: {print_on}")#, record_with_enter: {record_with_enter}, recording_length {recording_length}, directory {directory}, save_method {save_method}")

    #record_settings = [directory, save_method, recording_length]
    record_settings = None

    rclpy.init(args=None)

    # Create a sample listener and controller
    listener = Hi5PublisherNode()
    #listener.send_images = send_images
    listener.print_on = print_on

    # Have the sample listener receive events from the controller

    #if record_with_enter:
    #    spin = spin_record_with_enter
    #else:
    #if True:
    #    spin = spin_default
    #while rclpy.ok():
        #spin(listener, record_settings)
    rclpy.spin(listener)

if __name__ == "__main__":
    main()
    print("[leap] Exit")
