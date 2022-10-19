#!/usr/bin/env python
import sys, os; sys.path.append("..")

print(os.getcwd())
import numpy as np
np.save("/home/petr/Downloads/out.txt", str(os.getcwd()))
if os.getcwd()[-4:] == '.ros': # if running from roslaunch
    import rospkg; rospack = rospkg.RosPack(); rospack.list()
    sys.path.append(rospack.get_path('teleop_gesture_toolbox')+'/src/')
# for ros2 run, which has current working directory in ~/<colcon_ws>
sys.path.append("../python3.9/site-packages/teleop_gesture_toolbox")
import Leap
import time, argparse, inspect

from os_and_utils.saving import Recording
from os_and_utils.utils import ros_enabled, GlobalPaths
import frame_lib

if ros_enabled():
    import rclpy
    from rclpy.node import Node
    import teleop_gesture_toolbox.msg as rosm
    import teleop_gesture_toolbox.srv as ross
    #ross.SaveHandRecord, ross.SaveHandRecordResponseS
    print("ROS Enabled!")
else:
    print("ROS Disabled!")

class LeapPublisherNode(Node):
    def __init__(self):
        super().__init__('leap_publisher')
        self.frame_publisher = self.create_publisher(rosm.Frame, '/hand_frame', 5)
        self.srv = self.create_service(ross.SaveHandRecord, 'save_hand_record', self.save_hand_record_callback)

    def save_hand_record_callback(self, request, response):
        self.record.recording_requests.append([request.directory, request.save_method, request.recording_length])
        response.success = True
        return response

class SampleListener(Leap.Listener):
    def on_init(self, controller):
        self.record = Recording()
        if ros_enabled():
            self.rosnode = LeapPublisherNode()
        print("[leap] Initialized")

    def on_connect(self, controller):
        print("[leap] Connected")
        # Enable gestures
        controller.enable_gesture(Leap.Gesture.TYPE_CIRCLE)
        controller.enable_gesture(Leap.Gesture.TYPE_KEY_TAP)
        controller.enable_gesture(Leap.Gesture.TYPE_SCREEN_TAP)
        controller.enable_gesture(Leap.Gesture.TYPE_SWIPE)

    def on_disconnect(self, controller):
        # Note: not dispatched when running in a debugger.
        print("[leap] Disconnected")

    def on_exit(self, controller):
        print("[leap] Exited")

    def on_frame(self, controller):
        ''' ~700us -> 1400Hz
        '''
        frame = controller.frame() # Current Frame
        frame1 = controller.frame(1) # Last frame
        leapgestures = LeapMotionGestures.extract(frame.gestures(), frame1)
        f = frame_lib.Frame(frame, leapgestures)

        if ros_enabled():
            fros = f.to_ros()
            self.rosnode.frame_publisher.publish(fros)

        self.record.auto_handle(f)
        if self.print_on:
            #print(f.r.get_learning_data(definition=1))
            print(f)



class LeapMotionGestures():
    @staticmethod
    def extract(gestures, frame1):
        ''' Leap Motion gestures extraction
        '''
        leap_gestures = frame_lib.LeapGestures()

        for gesture in gestures:
            if gesture.type == Leap.Gesture.TYPE_CIRCLE:
                leap_gestures.circle.present = True
                circle = Leap.CircleGesture(gesture)
                leap_gestures.circle.id = circle.id
                leap_gestures.circle.state = circle.state
                # Determine clock direction using the angle between the pointable and the circle normal
                if circle.pointable.direction.angle_to(circle.normal) <= Leap.PI/2:
                    leap_gestures.circle.clockwise = True
                else:
                    leap_gestures.circle.clockwise = False

                # Calculate the angle swept since the last frame
                swept_angle = 0
                if circle.state == Leap.Gesture.STATE_STOP:
                    leap_gestures.circle.in_progress = False
                    leap_gestures.circle.progress = circle.progress
                    previous_update = Leap.CircleGesture(frame1.gesture(circle.id))
                    swept_angle =  (circle.progress - previous_update.progress) * 2 * Leap.PI
                    leap_gestures.circle.angle = swept_angle * Leap.RAD_TO_DEG
                    leap_gestures.circle.radius = circle.radius
                elif circle.state != Leap.Gesture.STATE_START:
                    leap_gestures.circle.in_progress = True
                    previous_update = Leap.CircleGesture(frame1.gesture(circle.id))
                    swept_angle =  (circle.progress - previous_update.progress) * 2 * Leap.PI
                    leap_gestures.circle.progress = circle.progress
                    leap_gestures.circle.angle = swept_angle * Leap.RAD_TO_DEG
                    leap_gestures.circle.radius = circle.radius


            if gesture.type == Leap.Gesture.TYPE_SWIPE:
                leap_gestures.swipe.present = True
                swipe = Leap.SwipeGesture(gesture)
                leap_gestures.swipe.id = swipe.id
                leap_gestures.swipe.state = swipe.state
                if gesture.state != Leap.Gesture.STATE_START:
                    leap_gestures.swipe.in_progress = True
                if gesture.state == Leap.Gesture.STATE_STOP:
                    leap_gestures.swipe.in_progress = False
                    leap_gestures.swipe.direction = [swipe.direction[0], swipe.direction[1], swipe.direction[2]]
                    leap_gestures.swipe.speed = swipe.speed

            if gesture.type == Leap.Gesture.TYPE_KEY_TAP:
                leap_gestures.keytap.present = True
                keytap = Leap.KeyTapGesture(gesture)
                leap_gestures.keytap.id = keytap.id
                leap_gestures.keytap.state = keytap.state
                if gesture.state != Leap.Gesture.STATE_START:
                    leap_gestures.keytap.in_progress = True
                if gesture.state == Leap.Gesture.STATE_STOP:
                    leap_gestures.keytap.in_progress = False
                    leap_gestures.keytap.direction = [keytap.direction[0], keytap.direction[1], keytap.direction[2]]

            if gesture.type == Leap.Gesture.TYPE_SCREEN_TAP:
                leap_gestures.screentap.present = True
                screentap = Leap.ScreenTapGesture(gesture)
                leap_gestures.screentap.id = screentap.id
                leap_gestures.screentap.state = screentap.state
                if gesture.state != Leap.Gesture.STATE_START:
                    leap_gestures.screentap.in_progress = True
                if gesture.state == Leap.Gesture.STATE_STOP:
                    leap_gestures.screentap.in_progress = False
                    leap_gestures.screentap.direction = [screentap.direction[0], screentap.direction[1], screentap.direction[2]]

        return leap_gestures

def spin_default(listener, record_settings):
    time.sleep(0.5)

def spin_record_with_enter(listener, record_settings):
    input()
    listener.record.recording_requests.append(record_settings)


def main():
    parser=argparse.ArgumentParser(description='')
    parser.add_argument('--print', default=False, type=bool, help='(default=%(default)s)')
    parser.add_argument('--record_with_enter', default=False, type=bool, help='(default=%(default)s)', choices=[True, False])
    parser.add_argument('--recording_length', default=1., type=float, help='(default=%(default)s)')
    parser.add_argument('--directory', default='', type=str, help='(default=%(default)s)')
    parser.add_argument('--save_method', default='numpy', type=str, help='(default=%(default)s)')
    # TODO:
    #parser.add_argument("--send_infrared_image", default='false', typ)

    try:
        args=parser.parse_args()
        print_on = args.print
        record_with_enter = args.record_with_enter
        recording_length = args.recording_length
        directory = args.directory
        save_method = args.save_method
    except:
        print("[Leap] parse_args failed -> parse_known_args")
        args=parser.parse_known_args()
        print_on = False
        record_with_enter = False
        recording_length = 1.0
        directory = ''
        save_method = 'numpy'

    if directory == '':
        #directory = os.path.abspath(os.path.join(os.path.dirname(inspect.getabsfile(inspect.currentframe())), '..', '..'))+'/include/data/learning/'
        directory = GlobalPaths().learn_path

    print(f"[Leap] {print_on}, record_with_enter: {record_with_enter}, recording_length {recording_length}, directory {directory}, save_method {save_method}")

    record_settings = [directory, save_method, recording_length]

    if ros_enabled():
        rclpy.init(args=None)
    else:
        print(f"ROS is not enabled! Check also sourcing the teleop_gesture_toolbox package")

    # Create a sample listener and controller
    listener = SampleListener()
    listener.print_on = print_on
    controller = Leap.Controller()

    # Have the sample listener receive events from the controller
    controller.add_listener(listener)

    if record_with_enter:
        spin = spin_record_with_enter
    else:
        spin = spin_default
    if ros_enabled():
        while rclpy.ok():
            spin(listener, record_settings)
    else:
        try:
            while True:
                spin(listener, record_settings)
        except KeyboardInterrupt:
            pass
    controller.remove_listener(listener)

if __name__ == "__main__":
    main()
    print("[leap] Exit")
