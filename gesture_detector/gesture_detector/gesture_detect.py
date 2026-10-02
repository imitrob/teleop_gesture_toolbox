#!/usr/bin/env python
from gesture_detector.gesture_classification.gestures_lib import DYNAMIC_WINDOW, GestureDataDetection
from gesture_meaning.one_to_one_mapping import user_settings
import sys, os, time, threading, rclpy

GESTURE_DETECTOR_RATE = 10

def main():
    rclpy.init(args=None)

    gd = GestureDataDetection(silent=False)
    
    # Read from launch file
    # Recognizers (static,dynamic) to work on left,right,both hands
    gd.declare_parameter('l_hand_mode', 'static+dynamic')
    l_hand_mode = gd.get_parameter('l_hand_mode').get_parameter_value().string_value
    gd.declare_parameter('r_hand_mode', 'static+dynamic')
    r_hand_mode = gd.get_parameter('r_hand_mode').get_parameter_value().string_value

    # The longest dynamic gesture that has to fit in one window. The postures that
    # switch off dynamic detection are not a parameter: they are the user's
    # adaptive_setup, which gestures_lib reads from the sentence maker.
    gd.declare_parameter('dynamic_window', DYNAMIC_WINDOW)
    dynamic_window = gd.get_parameter('dynamic_window').get_parameter_value().double_value
    print(f"[Gesture Detect] Dynamic gesture window {dynamic_window}s")

    # How long a gesture has to be held, from this user's links file -- the same
    # value the sentence maker triggers on. This node publishes the activation
    # evidence the dashboard draws as a progress bar, so a default here would
    # fill the bar at a different count than the one that commands the robot.
    # Set after the constructor because the value is a parameter, which needs
    # the node to exist; nothing spins until the thread below starts.
    gd.declare_parameter('user_name', '')
    name_user = gd.get_parameter('user_name').get_parameter_value().string_value
    gd.activate_length = user_settings(name_user, "Gesture Detector").get(
        "activate_length", gd.activate_length)
    print(f"[Gesture Detector] Gesture activates after {gd.activate_length} detections")

    spinning_thread = threading.Thread(target=spinning_threadfn, args=(gd, ), daemon=True)
    spinning_thread.start()

    rate = gd.create_rate_(GESTURE_DETECTOR_RATE) 
    while rclpy.ok():
        # print("..")
        if gd.present():
            gd.send_g_data(l_hand_mode, r_hand_mode, dynamic_detection_window=dynamic_window)
        gd.send_state()
        rate.sleep()
        
    print("quit")

def spinning_threadfn(gd):
    while rclpy.ok():
        gd.spin_once(sem=True)
        time.sleep(0.01)

if __name__ == '__main__':
    main()
    print('[Main] Interrupted')
    try:
        sys.exit(0)
    except SystemExit:
        os._exit(0)
