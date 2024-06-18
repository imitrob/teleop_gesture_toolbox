
# teleop_gesture_toolbox v1.1

Welcome to teleop_gesture_toolbox package made for Leap Motion Controller.
Most of the package utilize ROS2 Humble. 

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

Make alias for sourcing environment:
```Shell
alias teleopenv='conda activate teleopenv;
LD_LIBRARY_PATH=\$LD_LIBRARY_PATH:$HOME/CoppeliaSim:$HOME/LeapSDK/lib/x64/;
source install/setup.bash'
```

See Leap Motion rigged hands by using [leapjs-rigged-hand](https://github.com/leapmotion/leapjs-rigged-hand).

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