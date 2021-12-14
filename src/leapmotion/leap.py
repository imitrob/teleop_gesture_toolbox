#!/usr/bin/env python3.8
import sys, Leap, time

sys.path.append('../os_and_utils')
from saving import save_recording
from utils import ROS_ENABLED
import frame_lib

if ROS_ENABLED():
    import rospy
    import mirracle_gestures.msg as rosm

'''
TODO:
    - [x] Add also the ROS publisher of the data
    - [x] Saved data to Frame are not ROS dependent
    - [x] Hand class FrameAdv and HandAdv needs to be independent from Leap Class --> All values needs to be saved to own defined objects
    - [ ] Setup service saverecordings
    - [ ] Test solution

Problems:
    - ui_lib is reffering to some variables that are not in FrameAdv anymore
        -> Solutions: need to change these variables to their substitution function
    -
'''

class SampleListener(Leap.Listener):
    def on_init(self, controller):
        self.recording = []
        self.is_recording = False
        self.is_printing = True
        if ROS_ENABLED():
            self.frame_publisher = rospy.Publisher("/leap_frame", rosm.Frame, queue_size=5)
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
        frame = controller.frame() # Current Frame
        frame1 = controller.frame(1) # Last frame
        leapgestures = LeapMotionGestures.extract(frame.gestures(), frame1)
        f = frame_lib.Frame(frame, leapgestures)

        if ROS_ENABLED():
            self.frame_publisher.publish(f.to_ros())
        if self.is_recording:
            self.recording.append(f)
        if self.is_printing:
            print(f)


    def save_recording_src_callback(self, msg):
        self.recording = True
        time.sleep(msg.recording_length)
        self.recording = False
        save_recording(msg.directory, self.recording, save=msg.save_method)


class LeapMotionGestures():
    @staticmethod
    def extract(gestures, frame1):
        ''' Leap Motion gestures extraction
        '''
        leap_gestures = frame_lib.LeapGestures()

        for gesture in gestures:
            if gesture.type == Leap.Gesture.TYPE_CIRCLE:
                leap_gestures.circle.id = True
                circle = Leap.CircleGesture(gesture)
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
                leap_gestures.swipe.id = True
                swipe = Leap.SwipeGesture(gesture)
                leap_gestures.swipe.state = swipe.state
                if gesture.state != Leap.Gesture.STATE_START:
                    leap_gestures.swipe.in_progress = True
                if gesture.state == Leap.Gesture.STATE_STOP:
                    leap_gestures.swipe.in_progress = False
                    leap_gestures.swipe.direction = [swipe.direction[0], swipe.direction[1], swipe.direction[2]]
                    leap_gestures.swipe.speed = swipe.speed

            if gesture.type == Leap.Gesture.TYPE_KEY_TAP:
                leap_gestures.keytap.id = True
                keytap = Leap.KeyTapGesture(gesture)
                leap_gestures.keytap.state = keytap.state
                if gesture.state != Leap.Gesture.STATE_START:
                    leap_gestures.keytap.in_progress = True
                if gesture.state == Leap.Gesture.STATE_STOP:
                    leap_gestures.keytap.in_progress = False
                    leap_gestures.keytap.direction = [keytap.direction[0], keytap.direction[1], keytap.direction[2]]

            if gesture.type == Leap.Gesture.TYPE_SCREEN_TAP:
                leap_gestures.screentap.id = True
                screentap = Leap.ScreenTapGesture(gesture)
                leap_gestures.screentap.state = screentap.state
                if gesture.state != Leap.Gesture.STATE_START:
                    leap_gestures.screentap.in_progress = True
                if gesture.state == Leap.Gesture.STATE_STOP:
                    leap_gestures.screentap.in_progress = False
                    leap_gestures.screentap.direction = [screentap.direction[0], screentap.direction[1], screentap.direction[2]]

        return leap_gestures

def main():
    print("ROS_ENABLED: ", ROS_ENABLED())
    if ROS_ENABLED():
        rospy.init_node('leap', anonymous=True)

    # Create a sample listener and controller
    listener = SampleListener()
    controller = Leap.Controller()

    # Have the sample listener receive events from the controller
    controller.add_listener(listener)

    if ROS_ENABLED():
        while not rospy.is_shutdown():
            time.sleep(0.5)
    else:
        try:
            while True: time.sleep(0.5)
        except KeyboardInterrupt:
            pass
    controller.remove_listener(listener)

if __name__ == "__main__":
    main()
    print("[leap] Exit")
