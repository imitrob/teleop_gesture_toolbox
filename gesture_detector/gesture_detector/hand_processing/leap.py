#!/usr/bin/env python
import numpy as np
import Leap
import time, argparse, threading

import gesture_detector
from gesture_detector.utils.saving import Recording
from gesture_detector.utils.utils import ros_enabled
from gesture_detector.hand_processing import frame_lib
import ctypes

if ros_enabled():
    import rclpy
    from rclpy.node import Node
    import gesture_msgs.msg as rosm
    import gesture_msgs.srv as ross
    from sensor_msgs.msg import Image
    print("ROS Enabled!")
else:
    class Node():
        pass
    print("ROS Disabled!")

class LeapPublisherNode(Node):
    def __init__(self, recorder):
        super().__init__('leap_publisher')
        self.frame_publisher = self.create_publisher(rosm.Frame, '/hand_frame', 5)
        self.srv = self.create_service(ross.SaveHandRecord, '/save_hand_record', self.save_hand_record_callback)

        self.image_publisher = self.create_publisher(Image, '/leap_image', 5)
        self.recorder = recorder

    def save_hand_record_callback(self, request, response):
        print("save hand record callback")
        self.recorder.recording_requests.append([request.directory, request.recording_length])
        response.success = True
        return response

class SampleListener(Leap.Listener):
    def on_init(self, controller):
        self.recorder = Recording()
        if ros_enabled():
            self.rosnode = LeapPublisherNode(self.recorder)
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

        if self.send_images:
            img = frame.images[0]

            imgros = Image()
            imgros.height = img.height
            imgros.width = img.width
            imgros.encoding='mono8'
            imgros.header.stamp = self.rosnode.get_clock().now().to_msg()
            imgros.step = img.width * 1 * 1
            imgros.header.frame_id = ""

            image_buffer_ptr = img.data_pointer
            ctype_array_def = ctypes.c_ubyte * img.width * img.height
            as_ctype_array = ctype_array_def.from_address(int(image_buffer_ptr))
            data = np.ctypeslib.as_array(as_ctype_array).flatten().tolist()

            imgros.data = data
            self.rosnode.image_publisher.publish(imgros)
        self.recorder.auto_handle(f)
        if self.printer:
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
    listener.recorder.recording_requests.append(record_settings)

def main(args):
    
    directory = f"{gesture_detector.path}/gesture_data/{args['recording_gesture_name']}"

    print(f"[Leap] printer: {args['printer']}")
    print(f"[Leap] record_with_enter: {args['record_with_enter']}")
    print(f"[Leap] recording_length {args['recording_length']}")
    
    record_settings = [directory, args['recording_length']]

    if ros_enabled():
        rclpy.init(args=None)
    else:
        print(f"ROS is not enabled! Check also sourcing the teleop_gesture_toolbox package")

    # Create a sample listener and controller
    listener = SampleListener()
    listener.send_images = args['send_images']
    listener.printer = args['printer']
    controller = Leap.Controller()

    # Have the sample listener receive events from the controller
    if args['send_images']:
        controller.set_policy(Leap.Controller.POLICY_BACKGROUND_FRAMES)
        controller.set_policy(Leap.Controller.POLICY_IMAGES)
    controller.add_listener(listener)

    if ros_enabled():
        spinning_thread = threading.Thread(target=rclpy.spin, args=(listener.rosnode, ), daemon=True)
        spinning_thread.start()

    if args['record_with_enter']:
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

def ros_run():
    main({
        'printer': False,
        'record_with_enter': False,
        'recording_length': 1.0,
        'send_images': False,
        'recording_gesture_name': "",
    })

if __name__ == "__main__":
    parser=argparse.ArgumentParser(description='')

    parser.add_argument("--printer", action="store_true")
    parser.add_argument("--no_printer", dest="printer", action="store_false")
    parser.set_defaults(printer=False)

    parser.add_argument("--record_with_enter", action="store_true")
    parser.add_argument("--no_record_with_enter", dest="record_with_enter", action="store_false")
    parser.set_defaults(record_with_enter=False)

    parser.add_argument('--recording_gesture_name', default="", type=str)
    parser.add_argument('--recording_length', default=1.0, type=float, help='(default=%(default)s)') # [s]
    # TODO:
    parser.add_argument("--send_images", default=False, type=bool)

    main(vars(parser.parse_args()))
    print("[Leap] Exit")
