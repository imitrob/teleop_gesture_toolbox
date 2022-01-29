#!/usr/bin/env python3.8
'''
Loads informations about robot, configurations, move data, gestures, ...
All configurations are loaded from:
    - /include/custom_settings/robot_move.yaml
    - /include/custom_settings/gesture_recording.yaml
    - /include/custom_settings/application.yaml
    and
    - /include/custom_settings/*paths*.yaml
    - /include/custom_settings/*scenes*.yaml
    - /include/custom_settings/poses.yaml

Import this file and call init() to access parameters, info and data
'''
import numpy as np
import yaml
from os_and_utils.utils import ordered_load, GlobalPaths, load_params
from os_and_utils.parse_yaml import ParseYAML

def init():
    ''' Initialize the shared data across threads (Leap, UI, Control)
    '''
    global Gs, GsK, configGestures, paths
    # 1. Initialize File/Folder Paths
    paths = GlobalPaths()

    # 2. Loads config. data to gestures
    # - gesture_recording.yaml
    configGestures = ParseYAML.load_gesture_config_file(paths.custom_settings_yaml)
    recordingConfig = ParseYAML.load_recording_file(paths.custom_settings_yaml)
    Gs, GsK = ParseYAML.load_gestures_file(paths.custom_settings_yaml)

    # 3. Loads config. about robot
    # - ROSparams /mirracle_config/*
    # - robot_move.yaml.yaml
    global robot, simulator, gripper, plot, inverse_kinematics, inverse_kinematics_topic, gesture_detection_on, launch_gesture_detection, launch_ui
    robot, simulator, gripper, plot, inverse_kinematics, inverse_kinematics_topic, gesture_detection_on, launch_gesture_detection, launch_ui = load_params()

    global vel_lim, effort_lim, lower_lim, upper_lim, joint_names, base_link, group_name, eef, grasping_group, tac_topic, joint_states_topic
    vel_lim, effort_lim, lower_lim, upper_lim, joint_names, base_link, group_name, eef, grasping_group, tac_topic, joint_states_topic = ParseYAML.load_robot_move_file(paths.custom_settings_yaml, robot, simulator)

    print("[Settings] Workspace folder is set to: "+paths.ws_folder)



    ## Fixed Conditions
    global position_mode, orientation_mode, print_path_trace
    # ORIENTATION_MODE options:
    #   - 'fixed', eef has fixed eef orientaion based on chosen environment (md.ENV)
    orientation_mode = 'fixed'

    # POSITION_MODE options:
    #   - '', default
    #   - 'sim_camera', uses simulator camera to transform position
    position_mode = ''
    # When turned on, rViz marker array of executed trajectory is published
    print_path_trace = False


    ## User Interface Data
    #   - From application.yaml
    with open(paths.custom_settings_yaml+"application.yaml", 'r') as stream:
        app_data_loaded = ordered_load(stream, yaml.SafeLoader)
    # Configuration page values
    global NumConfigBars, VariableValues
    NumConfigBars = [app_data_loaded['ConfigurationPage']['Rows'], app_data_loaded['ConfigurationPage']['Columns']]
    VariableValues = np.zeros(NumConfigBars)

    # Status Bar
    global HoldAnchor, HoldPrevState, HoldValue, currentPose, WindowState, leavingAction, w, h, ui_scale, record_with_keys
    WindowState = 0.0  # 0-Main page, 1-Config page
    HoldAnchor = 0.0  # For moving status bar
    HoldPrevState = False  # --||--
    HoldValue = 0
    currentPose = 0
    leavingAction = False
    w, h = 1000, 800 # Will be set dynamically to proper value
    ui_scale = app_data_loaded['Scale']

    record_with_keys = False # Bool, Enables recording with keys in UI
    print("[Settings] done")
