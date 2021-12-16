#!/usr/bin/env python3.8
import sys, Leap, time, argparse

sys.path.append('../os_and_utils')
from saving import Recording
from utils import ROS_ENABLED
import frame_lib

if ROS_ENABLED():
    import rospy
    import mirracle_gestures.msg as rosm
    import mirracle_gestures.srv as ross
    #ross.SaveHandRecord, ross.SaveHandRecordResponse

global args

'''
TODO:
    - [x] Add also the ROS publisher of the data
    - [x] Saved data to Frame are not ROS dependent
    - [x] Hand class Frame and Hand needs to be independent from Leap Class --> All values needs to be saved to own defined objects
    - [x] Setup service to save recordings
    - [ ] Test solution

Problems:
    - ui_lib is reffering to some variables that are not in Frame anymore
        -> Solutions: need to change these variables to their substitution function
    -
'''

class SampleListener(Leap.Listener):
    def on_init(self, controller):
        self.record = Recording()
        global args
        self.is_printing = args.print
        if ROS_ENABLED():
            self.frame_publisher = rospy.Publisher("/hand_frame", rosm.Frame, queue_size=5)
            rospy.Service('save_hand_record', ross.SaveHandRecord, self.save_hand_record_callback)
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

        if ROS_ENABLED():
            self.frame_publisher.publish(f.to_ros())

        self.record.auto_handle(f)

        if self.is_printing:
            print(f)

    def save_hand_record_callback(self, msg):
        self.record.recording_requests.append([msg.directory, msg.save_method, msg.recording_length])
        return ross.SaveHandRecordResponse(True)

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
    parser.add_argument('--directory', default='/home/pierro/Downloads/asd', type=str, help='(default=%(default)s)')
    parser.add_argument('--save_method', default='numpy', type=str, help='(default=%(default)s)')
    global args
    args=parser.parse_args()
    print(args)
    record_settings = [args.directory, args.save_method, args.recording_length]

    if ROS_ENABLED():
        rospy.init_node('leap', anonymous=True)
    else:
        print(f"ROS is not enabled! Check also sourcing the mirracle_gestures package")

    # Create a sample listener and controller
    listener = SampleListener()
    controller = Leap.Controller()

    # Have the sample listener receive events from the controller
    controller.add_listener(listener)

    if args.record_with_enter:
        spin = spin_record_with_enter
    else:
        spin = spin_default

    if ROS_ENABLED():
        while not rospy.is_shutdown():
            spin(listener, record_settings)
    else:
        try:
            while True: spin(listener, record_settings)
        except KeyboardInterrupt:
            pass
    controller.remove_listener(listener)

if __name__ == "__main__":
    main()
    print("[leap] Exit")
