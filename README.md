
# Teleoperation gesture toolbox v1.3

Welcome to **teleoperation gesture toolbox** package made for **Leap Motion Controller** or D400 series RealSense.
Most of the package utilize **ROS2**. 

News and updates:
- Hand visualization web dashboard (`localhost:6357`) updated with real hand visualization.
- Gesture Meaning: Added a mapping game (`python -m gesture_meaning.link_game`).

## Installation 

Install Leap Motion SDK and API for Python (v3.11), see [script](gesture_detector/leap_motion_install.sh).

I use [miniconda](docs.anaconda.com/miniconda) packaging. Dependency packages are stored in `environment.yml` file.
```Shell
conda install mamba -c conda-forge
mamba env create -f environment.yml
mamba activate teleopenv
```

Build as ROS2 package:
```Shell
mkdir -p ~/teleop_ws/src
cd ~/teleop_ws/src
git clone https://github.com/imitrob/teleop_gesture_toolbox.git --depth 1
cd ..
colcon build --symlink-install --cmake-args -DPython3_FIND_VIRTUALENV=ONLY
```

I use following alias to source the environment. The `*_PATH` variables define recordings, trained models, scenes:
```Shell
alias teleopenv='conda activate teleopenv; export GESTURE_DATA_PATH=~/teleop_ws/src/teleop_gesture_toolbox/gesture_detector/gesture_data; export GESTURE_MODELS_PATH=~/teleop_ws/src/teleop_gesture_toolbox/gesture_detector/saved_models; export SCENES_PATH=~/teleop_ws/src/teleop_gesture_toolbox/scene_getter/scene_getter/scene_makers/scenes; source ~/teleop_ws/install/setup.bash'
```

See Leap Motion rigged hands by using [leapjs-rigged-hand](https://github.com/leapmotion/leapjs-rigged-hand).

## Common Gestures dataset

Sample trained model (containing common gestures) is included with the repository (`gesture_detector/saved_models`) and is loaded by default. For more information about training on new dataset, look at section "Gesture dataset collection and detector training".

## Usage 

### Gesture detector (not requires robotics setup)

Run Leap Motion backend: `sudo leapd`

Run gesture detector:
```Shell
teleopenv; ros2 launch gesture_detector gesture_detect_launch.py sensor:=leap # or realsense
```

See the gesture detections on your browser `localhost:6357`.

### Deictic gesture (Pointing object selection)

Pointing on objects on the scene with your hand will select it. Run: `ros2 run pointing_object_selection selector_node`.

Deictic selector requires scene publisher, publishing the scene object locations. Run `ros2 run scene_getter mocked_scene` to publish mocked scene, or see the script [mocked_scene_maker.py](scene_getter/scene_getter/scene_makers/mocked_scene_maker.py) how it is done.

Secondly, calibration of the Leap Motion Controller with your scene base frame is needed. Transforms are defined in ([saved_config folder](pointing_object_selection/pointing_object_selection/saved_setups/)). `a404.yaml` is valid for example setup (see image [setup.jpg](setup.jpg)) when the Leap Motion controller is opposite from base.  

Example setup
![setup.jpg](setup.jpg)

### Gesture sentence processor

Combining Gesture detector and Pointing object selection.
Multiple gesture types and a gesture sentence generation. When pointing gesture is detected, object selection is activated. See example video [here](http://imitrob.ciirc.cvut.cz/publications/chi23/2023_IROS_GESTURE_SENTENCE_VIDEO.mp4).

Requires gesture detector (`teleopenv; ros2 launch gesture_detector gesture_detect_launch.py`) and deictic node (`ros2 run pointing_object_selection selector_node`) running.

Then gesture sentence processor is launch with

```
ros2 run gesture_sentence_maker sentence_maker
```

After gesture sentence finishes (hand no longer visible), processed gestures are sent and you should see `HRI Command original` results on your browser (`localhost:6357`).

### Mapping gestures to Robotic Actions

A gesture meaning are defined in `links.yaml`.

Try without any hardware &mdash; click gestures, see which action fires:

```Shell
python -m gesture_meaning.link_game  # http://127.0.0.1:8078
```

### Action execution by the robotic manipulator

Part that executes the actions with robitic manipulator is moved to separate [repository](https://github.com/imitrob/imitrob_templates) compatibility with this package is currently under development.


### Gesture Direct Teleoperation (requires robotics setup)

Direct teleoperation is a separate subpackage (*live_teleoperation* folder)

Tested robot is Franka Emika Panda and [panda_py](https://github.com/JeanElsner/panda-py) `pip install panda-python`. See implementation in `robot.py`.

Servoing happens in task space (cartesian controller).

#### Usage:

1. Run Leap Motion backend: `sudo leapd`
2. Run Leap Motion ROS2 publisher: `teleopenv; ros2 run gesture_detector leap`
3. Run servo: `python servoing.py`
    - Teleoperate robot with fist (grab) gesture, so close hand and move the robot.
    - Open your hand for the robot to stop following your hand movements.
    - Right hand for teleoperation, Left hand to close and open gripper.

## Gesture dataset collection and detector training

The sample dataset can be downloaded from [link](https://drive.google.com/file/d/17L5KEuhW9kLYC073t11jctynQQ6z2Qm0/view?usp=sharing). The dataset needs to be saved to `gesture_detector/gesture_data` folder.

To create your owndataset, run:

`teleopenv; python gesture_detector/hand_processing/leap.py --record_with_enter --recording_gesture_name <your gesture name>`

Use Enter to record 1 sec long gesture demonstration saved as hand movement to `gesture_detector/gesture_data` folder.

I like to also run the marker publisher: `teleopenv; ros2 launch gesture_detector hand_marker_pub`
and rviz to see the hand: `teleopenv; rviz2 -d gesture_detector/live_display/hand_cfg.rviz`

To train the static gestures, run:

`teleopenv; python gesture_detector/gesture_classification/torch_lib.py --gestures <gesture 1 name> <gesture 2 name> <gesture n name>` script, where gesture names are your gesture names. By default, gesture names are the ones from sample dataset.

After training is done, see the  model in `gesture_detector/saved_models` folder. To set the model, adjust model in `launch/gesture_detect_launch.py` file.

