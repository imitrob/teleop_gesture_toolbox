# Robot control via gestures

## Installation

Download installation script [here](https://gitlab.ciirc.cvut.cz/imitrob/mirracle/mirracle_gestures/-/raw/master/installation.sh?inline=false) and run with `bash installation.sh`.


If error occurs or application is black, try to create conda environment. If you don't have anaconda, download it from [link](https://repo.anaconda.com/archive/Anaconda3-2021.05-Linux-x86_64.sh), install with `bash Anaconda3-2021.05-Linux-x86_64.sh`

```
cd ~/$YOUR_WS/src/mirracle_gestures
conda env create -f environment.yml
conda activate gestures_env
conda install -c anaconda pyqt
```
Feel free to open an issue, if something is not working.

## Launch the robot

### Simulator

```
# Term 1, Leap Controller
sudo leapd
# Term 2, Simulator demo
source /opt/ros/melodic/setup.bash
roslaunch mirracle_gestures demo.launch simulator:=coppelia
```

Add `gripper:=franka_gripper` for gripper option or `gripper:=franka_gripper_with_camera` for gripper with mounted _RealSense_ camera.

### Real

- Check the Panda Robot light is blue
```
# Term 1, Panda Robot Start
source /opt/ros/melodic/setup.bash
source ~/<panda_ws>/setup.sh
roslaunch panda_launch start_robot.launch
# Term 2, Leap Controller
sudo leapd
# Term 3, Workspace with "motion" package
source /opt/ros/melodic/setup.bash
source ~/<your_ws>/setup.sh
roslaunch mirracle_gestures demo.launch simulator:=real
```

## Application

1. Motion planning
  - Path execution: Pick 'Play path' on Drop-down list
  - Path selection is made in second Drop-down list
  - To Add/Edit paths in this list, go to [here](#manage-paths)

2. Live mapping with _Leap Motion_ and gesture detection
  - Pick 'Live hand' in left Drop-down list
  - Note: Leap Motion needs to be connected to see this option and Leap software (`sudo leapd`) must be running
  - On the left panel can be seen what gestures has been detected, pointing arrow indicates probabilistic solution

Features:
1. Switch between environments in _Menu->Robot config.->Environment_
2. Switch between scenes _Menu->Scene_, more about scenes in [here](#manage-scenes)
3. Switch between gesture detection NN files in _Menu->Gestures->Pick detection network_, new NN files can be downloaded (_Menu->Gestures->Download networks from gdrive_)

## Manage Gestures

Gesture configurations YAML file is saved in `<gestures pkg>/include/custom_settings/gesture_recording.yaml`. Add new gesture by creating new record under `staticGestures` or `dynamicGestures` (below). Note that gesture list is updated when program starts.

`gesture_recording.yaml`:
```
staticGestures:
  <gesture 1>:
    key: <keyboard key for recording>
    filename: <picture (icon) of gesture showed in UI>
  <gesture 2>:
    ...
  ...
dynamicGestures:
  <gesture 1>:
    ...
  ...
```

TODO: Right now all gestures loaded from `network.pkl` needs to have record in this yaml file. Make it independent.



All available networks can be downloaded from google drive with button from UI menu and they are saved in folder `<gestures pkg>/include/data/Trained_network/`.
To get information about network, run: `rosrun mirracle_gestures get_info_about_network.py` then specify network (e.g. network0.pkl).

Train new NN with script (src/learning/learn.py). Running script via Jupyter notebook (IPython) is advised for better orientation in learning processes.

Gesture management described above can be summarized into graph:
```mermaid
graph LR;
classDef someclass fill:#f96;
classDef someclass2 fill:#00FF00;

A("Train network *<br>src/learning/train.py<br>"):::someclass --> B("Folder containing network data files<br>include/data/Trained_network/<network x>.pkl")
C("Download networks (gdrive)<br><inside user interface>") --> B
B --> D("Sampling thread<br>src/learning/sample.py<br>(launched by Main<br>src/demo.launch)")
E("Info about gestures **<br>/include/custom_settings/gesture_recording.yaml"):::someclass2 --> D
B --> F("Get info about <network x>.pkl *<br>src/learning/get_info_about_network.py"):::someclass

G("* python files executed separately (Python3)"):::someclass
H("** Configuration text file"):::someclass2

I("Recorded gestures folder<br>include/data/learning/<gesture n>/<recording m>") -.-> A
D --> I
```

## Manage Scenes

Scene configurations YAML file is saved in `<this_pkg>/include/custom_settings/scenes.yaml`.

The object needs to have at least _pose_ and _size_.

Scenes are loaded with start of a program and can be switched from menu.

Create new scenes in this file as follows:
```yaml
<Sample scene1 name>:
  <Sample object 1 name>:
    pose:
      # Loads position as dictionary, uses Cartesian coordinates
      position: {'x': 1.0, 'y': 1.0, 'z': 1.0}
      # Loads orientation as dictionary, uses Quaternion values
      orientation: {'x': 1.0, 'y': 1.0, 'z': 1.0, 'w': }
    size: [1.,1.,1.]
    shape: cube # Loads shape object (or sphere, cylinder, cone)
    mass: 0.1 # [kg]
    friction: 0.3 # [-] Coefficient
    pub_info: true # Info about object is published to topic (/coppelia/object_info)
  <Sample object 2 name>:
    pose:
      # Loads position as list or tuple, uses Cartesian coordinates (x,y,z)
      position: [1., 1., 1.]
      # Loads orientation as list or tuple (uses notation x,y,z,w)
      orientation: {'x': 1.0, 'y': 1.0, 'z': 1.0, 'w': 1.0}
    size: [1.,1.,1.]
    mesh: <mesh file>.obj # Loads mesh file from include/models
  <Sample object 3 name>:
    pose:
      # Loads position as saved in poses.yaml name
      position: home
      # Loads orientation as saved in poses.yaml name
      orientation: home
    size: [1.,1.,1.]
  <Sample object 4 name>:
    # Loads both position and orientation as saved in poses.yaml name
    pose: home
    size: [1.,1.,1.]
   ...
<Sample scene2 name>:
  ...
```

## Manage Paths
Path configurations YAML file is saved in `<this_pkg>/include/custom_settings/paths.yaml`.

Create new paths in this file as follows:
```yaml
<Sample path1 name>:
  <Sample pose 1 name>:
    pose:
      # Loads position as dictionary, uses Cartesian coordinates
      position: {'x': 1.0, 'y': 1.0, 'z': 1.0}
      # Loads orientation as dictionary, uses Quaternion values
      orientation: {'x': 1.0, 'y': 1.0, 'z': 1.0, 'w': }
    # Grab/Release object
    action:
  <Sample pose 2 name>:
    pose:
      # Loads position as list or tuple, uses Cartesian coordinates (x,y,z)
      position: [1., 1., 1.]
      # Loads orientation as list or tuple (uses notation x,y,z,w)
      orientation: {'x': 1.0, 'y': 1.0, 'z': 1.0, 'w': 1.0}
  <Sample pose 3 name>:
    pose:
      # Loads position as saved in poses.yaml name
      position: home
      # Loads orientation as saved in poses.yaml name
      orientation: home
  <Sample pose 4 name>:
    # Loads both position and orientation as saved in poses.yaml name
    pose: home
   ...
<Sample path2 name>:
  ...
```

## Frequent issues
- `segmentation fault` when launching (or the `thread 17 ended with error`). If Conda environment in use, try:
```
conda update --all
```
- `RuntimeError: RobotInterfacePython: invalid robot model` The robot MoveIt "server" is not launched

- `[Err] [REST.cc:205] Error in REST request
libcurl: (51) SSL: no alternative certificate subject name matches target host name ‘api.ignitionfuel.org’`
Solution:
Open ~/.ignition/fuel/config.yaml:
replace: api.ignitionfuel.org
to: fuel.ignitionrobotics.org

- Try upgrade pip with:
```
pip install --upgrade pip
pip3 install --upgrade pip
```

## GPU Accelerated learning:
- Check kernel version -> need to be 5.4.0
```
uname -a
```
- Check gcc version -> need to be 7.5.0
```
gcc --version
```
- Check what driver suites the linux distribution
```
sudo lshw -C display
```
- Find the packages available
```
sudo apt list nvidia-driver-*
```
- Install the drivers
```
sudo add-apt-repository ppa:graphics-drivers/ppa
sudo apt update
sudo apt install -y nvidia-driver-450
```
- Reboot & Check Installation
```
sudo reboot
sudo nvidia-smi
```
[Source](https://medium.com/@sreenithyc21/nvidia-driver-installation-for-ubuntu-18-04-2020-2918be830d0f)
