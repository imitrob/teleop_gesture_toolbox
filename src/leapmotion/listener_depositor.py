#!/usr/bin/env python2
import tf

import sys
from os.path import expanduser, isfile
import numpy as np
from copy import deepcopy
from itertools import permutations, combinations

import Leap, time
from Leap import CircleGesture, KeyTapGesture, ScreenTapGesture, SwipeGesture

sys.path.append('../os_and_utils')
from saving import save_recording
from utils import ROS_ENABLED
import hand_classes

if ROS_ENABLED():
    from mirracle_gestures.msg import Frame as Framemsg
    from mirracle_gestures.msg import Bone as Bonemsg
exit()

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
        if ROS_ENABLED():
            self.frame_publisher = rospy.publisher("/leap_frame")
        print("[leapmotion.listener_depositor] Initialized")

    def on_connect(self, controller):
        print("[leapmotion.listener_depositor] Connected")
        # Enable gestures
        controller.enable_gesture(Leap.Gesture.TYPE_CIRCLE)
        controller.enable_gesture(Leap.Gesture.TYPE_KEY_TAP)
        controller.enable_gesture(Leap.Gesture.TYPE_SCREEN_TAP)
        controller.enable_gesture(Leap.Gesture.TYPE_SWIPE)

    def on_disconnect(self, controller):
        # Note: not dispatched when running in a debugger.
        print("[leapmotion.listener_depositor] Disconnected")

    def on_exit(self, controller):
        print("[leapmotion.listener_depositor] Exited")

    def on_frame(self, controller):
        frame = controller.frame()
        f = hand_classes.Frame(frame)
        f.leapgestures = self.leap_gestures_extract(frame.gestures())

        if ROS_ENABLED():
            self.ros_publish(f)
        if self.is_recording:
            self.recording.append(f)

    def save_recording_src_callback(self, msg):
        self.recording = True
        time.sleep(msg.recording_length)
        self.recording = False
        save_recording(msg.directory, self.recording, save=msg.save_method)

    def ros_publish(self, f):
        # f hand_classes.Frame() -> framemsg mirracle_gestures.msg/Frame
        framemsg = Framemsg()
        framemsg.fps = f.fps
        framemsg.hands = f.hands
        framemsg.header.secs = f.secs
        framemsg.header.nsecs = f.nsecs
        framemsg.header.seq = f.seq

        framemsg.leapgestures.circle_toggle = f.leapgestures.circle.toggle
        framemsg.leapgestures.circle_in_progress = f.leapgestures.circle.in_progress
        framemsg.leapgestures.circle_clockwise = f.leapgestures.circle.clockwise
        framemsg.leapgestures.circle_progress = f.leapgestures.circle.progress
        framemsg.leapgestures.circle_angle = f.leapgestures.circle.angle
        framemsg.leapgestures.circle_radius = f.leapgestures.circle.radius
        framemsg.leapgestures.circle_state = f.leapgestures.circle.state
        framemsg.leapgestures.swipe_toggle = f.leapgestures.swipe.toggle
        framemsg.leapgestures.swipe_in_progress = f.leapgestures.swipe.in_progress
        framemsg.leapgestures.swipe_direction = f.leapgestures.swipe.direction
        framemsg.leapgestures.swipe_speed = f.leapgestures.swipe.speed
        framemsg.leapgestures.swipe_state = f.leapgestures.swipe.state
        framemsg.leapgestures.keytap_toggle = f.leapgestures.keytap.toggle
        framemsg.leapgestures.keytap_in_progress = f.leapgestures.keytap.in_progress
        framemsg.leapgestures.keytap_direction = f.leapgestures.keytap.direction
        framemsg.leapgestures.keytap_state = f.leapgestures.keytap.state
        framemsg.leapgestures.screentap_toggle = f.leapgestures.screentap.toggle
        framemsg.leapgestures.screentap_in_progress = f.leapgestures.screentap.in_progress
        framemsg.leapgestures.screentap_direction = f.leapgestures.screentap.direction
        framemsg.leapgestures.screentap_state = f.leapgestures.screentap.state

        for (handmsg, hand) in [(framemsg.l, f.l), (framemsg.r, f.r)]:
            handmsg.id = hand.id
            handmsg.is_left = hand.is_left
            handmsg.is_right = hand.is_right
            handmsg.is_valid = hand.is_valid
            handmsg.grab_strength = hand.grab_strength
            handmsg.pinch_strength = hand.pinch_strength
            handmsg.confidence = hand.confidence
            handmsg.palm_normal = hand.palm_normal()
            handmsg.direction = hand.direction()
            handmsg.palm_position = hand.palm_position()

            handmsg.finger_bones = []
            for finger in hand.fingers:
                for bone in finger.bones:
                    bonemsg = Bonemsg()

                    basis = bone.basis[0](), bone.basis[1](), bone.basis[2]()
                    bonemsg.basis = [item for sublist in basis for item in sublist]
                    bonemsg.direction = bone.direction()
                    bonemsg.next_joint = bone.next_joint()
                    bonemsg.prev_joint = bone.prev_joint()
                    bonemsg.center = bone.center()
                    bonemsg.is_valid = bone.is_valid
                    bonemsg.length = bone.length
                    bonemsg.width = bone.width

                    handmsg.finger_bones.append(bonemsg)

            handmsg.palm_velocity = hand.palm_velocity()
            basis = hand.basis[0](), hand.basis[1](), hand.basis[2]()
            handmsg.basis = [item for sublist in basis for item in sublist]
            handmsg.palm_width = hand.palm_width
            handmsg.sphere_center = hand.sphere_center()
            handmsg.sphere_radius = hand.sphere_radius
            handmsg.stabilized_palm_position = hand.stabilized_palm_position()
            handmsg.time_visible = hand.time_visible
            handmsg.wrist_position = hand.wrist_position()

        self.frame_publisher.publish(framemsg)

    def leap_gestures_extract(self, gestures):
        ''' Leap Motion gestures extraction
        '''
        leap_gestures = hand_classes.LeapGestures()

        for gesture in gestures:
            if gesture.type == Leap.Gesture.TYPE_CIRCLE:
                leap_gestures.circle.toggle = True
                circle = CircleGesture(gesture)
                leap_gestures.circle.state = circle.state
                # Determine clock direction using the angle between the pointable and the circle normal
                if circle.pointable.direction.angle_to(circle.normal) <= Leap.PI/2:
                    leap_gestures.circle.clockwise = True
                else:
                    leap_gestures.circle.clockwise = False

                # Calculate the angle swept since the last frame
                swept_angle = 0
                if circle.state != Leap.Gesture.STATE_START:
                    leap_gestures.circle.in_progress = True
                    previous_update = CircleGesture(controller.frame(1).gesture(circle.id))
                    swept_angle =  (circle.progress - previous_update.progress) * 2 * Leap.PI
                    leap_gestures.circle.progress = circle.progress
                    leap_gestures.circle.angle = swept_angle * Leap.RAD_TO_DEG
                    leap_gestures.circle.radius = circle.radius
                if circle.state == Leap.Gesture.STATE_STOP:
                    leap_gestures.circle.in_progress = False
                    leap_gestures.circle.progress = circle.progress
                    leap_gestures.circle.angle = swept_angle * Leap.RAD_TO_DEG
                    leap_gestures.circle.radius = circle.radius

            if gesture.type == Leap.Gesture.TYPE_SWIPE:
                leap_gestures.swipe.toggle = True
                swipe = SwipeGesture(gesture)
                leap_gestures.swipe.state = swipe.state
                if gesture.state != Leap.Gesture.STATE_START:
                    leap_gestures.swipe.in_progress = True
                if gesture.state == Leap.Gesture.STATE_STOP:
                    leap_gestures.swipe.in_progress = False
                    leap_gestures.swipe.direction = [swipe.direction[0], swipe.direction[1], swipe.direction[2]]
                    leap_gestures.swipe.speed = swipe.speed

            if gesture.type == Leap.Gesture.TYPE_KEY_TAP:
                leap_gestures.keytap.toggle = True
                keytap = KeyTapGesture(gesture)
                leap_gestures.keytap.state = keytap.state
                if gesture.state != Leap.Gesture.STATE_START:
                    leap_gestures.pin.in_progress = True
                if gesture.state == Leap.Gesture.STATE_STOP:
                    leap_gestures.pin.in_progress = False
                    leap_gestures.pin.direction = [keytap.direction[0], keytap.direction[1], keytap.direction[2]]

            if gesture.type == Leap.Gesture.TYPE_SCREEN_TAP:
                leap_gestures.screentap.toggle = True
                screentap = ScreenTapGesture(gesture)
                leap_gestures.screentap.state = screentap.state
                if gesture.state != Leap.Gesture.STATE_START:
                    leap_gestures.touch.in_progress = True
                if gesture.state == Leap.Gesture.STATE_STOP:
                    leap_gestures.touch.in_progress = False
                    leap_gestures.touch.direction = [screentap.direction[0], screentap.direction[1], screentap.direction[2]]

        return leap_gestures


def main():
    # Create a sample listener and controller
    listener = SampleListener()
    controller = Leap.Controller()

    # Have the sample listener receive events from the controller
    controller.add_listener(listener)
    while True: pass
    try:
        sys.stdin.readline()
    except KeyboardInterrupt:
        pass
    finally:
        # Remove the sample listener when done
        controller.remove_listener(listener)


if __name__ == "__main__":
    main()
    print("[leapmotion.listener_depositor] Exit")
