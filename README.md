# Teleoperation gesture toolbox v0.01

Python gesture toolbox for teleoperating robotic arm. The toolbox covers both static and dynamic gestures in several representations, including ProMP.

## Installation

Tested on Linux Ubuntu 20.04.

- Conda, e.g. Miniconda [download](https://docs.conda.io/en/latest/miniconda.html)
- [Coppelia Sim](https://www.coppeliarobotics.com/) simulator ([install](include/scripts/coppelia_sim_install.sh))
  - (Recommended) Use version 4.1 (PyRep can have problems with newer versions)
  - Please install Coppelia Sim files to your home folder: `~/CoppeliaSim` (as shown below):
```
cd ~
wget --no-check-certificate https://www.coppeliarobotics.com/files/CoppeliaSim_Edu_V4_1_0_Ubuntu20_04.tar.xz
tar -xf CoppeliaSim_Edu_V4_1_0_Ubuntu20_04.tar.xz
mv CoppeliaSim_Edu_V4_1_0_Ubuntu20_04 CoppeliaSim
rm CoppeliaSim_Edu_V4_1_0_Ubuntu20_04.tar.xz
```

- [Leap Motion Controller](https://www.ultraleap.com/product/leap-motion-controller/) as a hand sensor ([install](include/scripts/leap_motion_install.sh))

- Packages install with Mamba:

```
conda install mamba -c conda-forge # Install mamba

mamba create -n teleopenv python=3.9
conda activate teleopenv

export ws=<path/to/your/colcon/ws>
mkdir -p $ws/src
cd $ws/src
git clone git@gitlab.ciirc.cvut.cz:imitrob/mirracle/teleop_gesture_toolbox.git
git clone git@gitlab.ciirc.cvut.cz:imitrob/mirracle/coppelia_sim_ros_interface.git

cd $ws/src/teleop_gesture_toolbox
mamba env update -n teleopenv --file environment.yml

# Reactivate conda env before proceeding.
conda deactivate
conda activate teleopenv  

cd $ws
rosdep init
rosdep update
rosdep install --from-paths src --ignore-src -r -y
colcon build --symlink-install

source $ws/install/setup.bash

# make activation script
echo "export ws=$ws
conda activate teleopenv
source $ws/install/setup.bash
export COPPELIASIM_ROOT=$HOME/CoppeliaSim
export LD_LIBRARY_PATH=$HOME/CoppeliaSim;
export QT_QPA_PLATFORM_PLUGIN_PATH=$HOME/CoppeliaSim;
export LD_PRELOAD=$HOME/LeapSDK/lib/x64/libLeap.so;" > ~/activate_teleop.sh
source ~/activate_teleop.sh

cd $ws/src
git clone https://github.com/imitrob/PyRep.git
cd PyRep
pip install .

cd ~/LeapSDK
gdown https://drive.google.com/uc?id=1sOEsgkofRgy_nhZvp2DrdIvltdMCwIQV
tar -xvzf Leap.tgz
cd Leap
pip install .
```

- Sample dataset example:
  - Download [dataset (1.2GB)](https://drive.google.com/file/d/1Jitk-MxzczreZ81PuO86xTapuSkBMOb-/view?usp=sharing) and move the `learning` folder to match the `<gesture_toolbox>/include/data/learning` folder
- Sample trained network example:
  - Download [model (22MB)](https://drive.google.com/file/d/1jyDatUJy10sdXmLjPEdHL1cfSo-RZ4ct/view?usp=share_link) and move the file to `/include/data/trained_networks` folder

## Launch the robot

- Aliases (suggestion):
```Shell
alias copqt='export LD_LIBRARY_PATH=/home/<user>/CoppeliaSim; export QT_QPA_PLATFORM_PLUGIN_PATH=/home/<user>/CoppeliaSim;'
alias srcgestures='conda activate <your conda ws>; source ~/<your_ws>/install/setup.bash'
```
- Terminal 1, Leap Controller:
```Shell
sudo leapd
```
- Terminal 2, Coppelia Sim:
```Shell
srcgestures; copqt
cd <your ws>/src/coppelia_sim_ros_interface/coppelia_sim_ros_interface
python coppelia_sim_ros_server.py
```
- Terminal 3, Gestures Detection Threads:
```Shell
srcgestures
cd <your ws>/src/teleop_gesture_toolbox/teleop_gesture_toolbox/gesture_classification
python main_sample_thread.py
# and
python dynamic_sample_thread.py
```
- Terminal 4, Main + GUI:
```Shell
srcgestures
cd <your ws>/src/teleop_gesture_toolbox/teleop_gesture_toolbox
python main_coppelia.py
# or
python main.py
# or
python example_grec.py
```

## Recognizers

### Static gestures with Probabilistic Neural Network

- (Default)
- Preprocessing (based on config chosen in given script):
```Shell
cd <your ws>/src/teleop_gesture_toolbox/teleop_gesture_toolbox/learning
python pymc3_train.py
```

### Dynamic gestures with Dynamic Time Warping

- (Default)
- Preprocessing not needed


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



All available networks can be downloaded from google drive with button from UI menu and they are saved in folder `<gestures pkg>/include/data/trained_networks/`.
To get information about network, run: `rosrun mirracle_gestures get_info_about_network.py` then specify network (e.g. network0.pkl).

Train new NN with script (src/learning/pymc3_train.py). Running script via Jupyter notebook (IPython) is advised for better orientation in learning processes. TODO: Publish recorded data-set to gdrive

Gesture management described above can be summarized into graph:
```mermaid
graph LR;
classDef someclass fill:#f96;
classDef someclass2 fill:#00FF00;

A("Train network *<br>src/learning/train.py<br>"):::someclass --> B("Folder containing network data files<br>include/data/trained_networks/<network x>.pkl")
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

Scene configurations YAML file is saved in `<gestures_pkg>/include/custom_settings/scenes.yaml`.

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
Path configurations YAML file is saved in `<gestures_pkg>/include/custom_settings/paths.yaml`.

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
