
# Teleoperation gesture toolbox v1.1

Welcome to **teleoperation gesture toolbox** package made for **Leap Motion Controller**.
Most of the package utilize **ROS2**. 

## Installation 

Install Leap Motion SDK and API, see [script](gesture_detector/leap_motion_install.sh).

Dependency packages are stored in `environment.yml` file.
```Shell
conda install mamba -c conda-forge
mamba create -f environment.yml # Installs ROS2 Humble via RoboStack
mamba activate teleopenv
```

Build as ROS2 package:
```Shell
mkdir -p <your_ws>/src
git clone https://github.com/imitrob/teleop_gesture_toolbox.git --depth 1 --branch dev
cd <your_ws>
colcon build --symlink-install
```

I use following alias to sourcing the environment:
```Shell
alias teleopenv='conda activate teleopenv;
LD_LIBRARY_PATH=\$LD_LIBRARY_PATH:$HOME/LeapAPI/lib/x64/;
source install/setup.bash'
```

See Leap Motion rigged hands by using [leapjs-rigged-hand](https://github.com/leapmotion/leapjs-rigged-hand).

## SampleTraining dataset

Download dataset to `gesture_detector/gesture_data` folder from [link](https://drive.google.com/file/d/17L5KEuhW9kLYC073t11jctynQQ6z2Qm0/view?usp=sharing).

Then train static gestures by using `pymc_lib.py` script.

## Usage 

Run Leap Motion backend: `sudo leapd`

Run gesture detector:
```Shell
teleopenv; ros2 launch gesture_detector gesture_detect_launch.py
```

See the gesture detections in web:

```Shell
ros2 launch rosbridge_server rosbridge_websocket_launch.xml
```

```Shell
python -m http.server
```

### Create link for saved_models

```Shell
rm ~/teleop_2_ws/build/gesture_detector/gesture_detector
ln -s ~/teleop_2_ws/src/teleop_gesture_toolbox/gesture_detector/saved_models ~/teleop_2_ws/build/gesture_detector/gesture_detector
```