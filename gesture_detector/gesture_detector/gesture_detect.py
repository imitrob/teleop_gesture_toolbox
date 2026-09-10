#!/usr/bin/env python
from gesture_detector.gesture_classification.gestures_lib import DYNAMIC_WINDOW, GestureDataDetection
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
