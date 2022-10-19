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

def init(change_working_directory=True):
    ''' Initialize the shared data across threads (Leap, UI, Control)
    '''
    global paths
    # 1. Initialize File/Folder Paths
    paths = GlobalPaths(change_working_directory=change_working_directory)

    # 3. Loads config. about robot
    # - ROSparams /gtoolbox_config/*
    # - robot_move.yaml.yaml
    global robot, simulator, gripper, plot, inverse_kinematics, inverse_kinematics_topic, gesture_detection_on, launch_gesture_detection, launch_ui
    robot, simulator, gripper, plot, inverse_kinematics, inverse_kinematics_topic, gesture_detection_on, launch_gesture_detection, launch_ui = load_params()

    global vel_lim, effort_lim, lower_lim, upper_lim, joint_names, base_link, group_name, eef, grasping_group, tac_topic, joint_states_topic
    vel_lim, effort_lim, lower_lim, upper_lim, joint_names, base_link, group_name, eef, grasping_group, tac_topic, joint_states_topic = ParseYAML.load_robot_move_file(paths.custom_settings_yaml, robot, simulator)

    ## YAML files loaded
    global yaml_config_gestures, yaml_config_recording
    yaml_config_gestures = ParseYAML.load_gesture_config_file(paths.custom_settings_yaml)
    yaml_config_recording = ParseYAML.load_recording_file(paths.custom_settings_yaml)

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
    global WindowState, w, h, ui_scale, record_with_keys
    WindowState = 0.0  # 0-Main page, 1-Config page
    w, h = 1000, 800 # Will be set dynamically to proper value
    ui_scale = app_data_loaded['Scale']
    record_with_keys = False # Bool, Enables recording with keys in UI
    print("[Settings] Workspace folder is set to: " + paths.ws_folder)

def get_network_file(type='static'):
    main_set_name = yaml_config_gestures['using_set']
    chosen_set = yaml_config_gestures['sets'][main_set_name]
    chosen_set['gestures']
    try:
        network_set_name = chosen_set['network_set']
        yaml_config_gestures['network_sets'][network_set_name][f'{type}_network_file']
    except KeyError:
        return None

    return yaml_config_gestures['network_sets'][network_set_name][f'{type}_network_file']

def get_detection_approach(type='static'):
    main_set_name = yaml_config_gestures['using_set']
    chosen_set = yaml_config_gestures['sets'][main_set_name]
    chosen_set['gestures']
    try:
        network_set_name = chosen_set['network_set']
    except KeyError:
        return None

    try:
        return yaml_config_gestures['network_sets'][network_set_name][f'{type}_detection_approach']
    except KeyError:
        return None

def get_hand_mode():
    main_set_name = yaml_config_gestures['using_set']
    chosen_set = yaml_config_gestures['sets'][main_set_name]
    chosen_set = {}
    try:
        hand_mode_set_name = chosen_set['hand_mode_sets']
    except KeyError:
        ''' Assign default option '''
        default_mode_set = {
            'l': 'static',
            'r': 'dynamic'
            }
        return default_mode_set

    return dict(yaml_config_gestures['hand_mode_sets'][hand_mode_set_name])

def get_gesture_mapping():
    main_set_name = yaml_config_gestures['using_set']
    chosen_set = yaml_config_gestures['sets'][main_set_name]
    try:
        mapping_set_name = chosen_set['mapping_sets']
    except KeyError:
        return {}
    return dict(yaml_config_gestures['mapping_sets'][mapping_set_name])
