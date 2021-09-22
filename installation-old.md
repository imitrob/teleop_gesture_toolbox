
## ROS Melodic Installation
```
sudo sh -c 'echo "deb http://packages.ros.org/ros/ubuntu $(lsb_release -sc) main" > /etc/apt/sources.list.d/ros-latest.list'
sudo apt-key adv --keyserver 'hkp://keyserver.ubuntu.com:80' --recv-key C1CF6E31E6BADE8868B172B4F42ED6FBAB17C654
sudo apt update
sudo apt install ros-melodic-desktop-full
echo "source /opt/ros/melodic/setup.bash" >> ~/.bashrc
source ~/.bashrc
sudo apt install python-rosdep python-rosinstall python-rosinstall-generator python-wstool build-essential
sudo rosdep init
rosdep update
sudo apt install ros-melodic-moveit-core ros-melodic-moveit-ros-planning-interface ros-melodic-moveit-visual-tools ros-melodic-control-toolbox ros-melodic-controller-interface ros-melodic-controller-manager ros-melodic-joint-limits-interface ros-melodic-industrial-msgs ros-melodic-moveit-simple-controller-manager ros-melodic-ompl ros-melodic-moveit-planners-ompl ros-melodic-moveit-ros-visualization ros-melodic-joint-state-controller ros-melodic-ros-controllers ros-melodic-moveit-commander ros-melodic-robot-state-publisher python-catkin-tools ros-melodic-joint-state-publisher-gui
echo "export ROSLAUNCH_SSH_UNKNOWN=1" >> ~/.bashrc
```

## Install dependencies

```
sudo apt-get install python-pip # Python install packages
python2 -m pip install ctypes-callable # For Leap Motion
python2 -m pip install python-fcl scikit-learn # RelaxedIK
```

## Workspace setup script
<details><summary>(should be done already) SSH setup (to be able to download from GitLab)</summary>
<code>mkdir ~/.ssh
cd ~/.ssh
ssh-keygen -t ed25519 -C "Computer123"
</code>
You will be asked for a name to a file, type: Computer123

You will be asked for a password, this is optional and you can proceed with: enter, enter

<code>sudo apt install xclip
 xclip -sel clip < Computer.pub</code>
- GitLab -> Settings -> SSK Keys -> Add key
- Test if working:
<code>cd ~/.ssh
ssh -T git@gitlab.ciirc.cvut.cz
</code>
should print out Welcome message.
</details>

- Pick the robot setup: _KUKA iiwa_ or _Franka Emika Panda_

<details><summary>1. iiwa setup</summary>
<code>ssh -T git@gitlab.ciirc.cvut.cz

mkdir -p ~/<your_ws>/src
cd ~/<your_ws>/src
git clone git@gitlab.ciirc.cvut.cz:capek/capek_testbed.git
git clone git@gitlab.ciirc.cvut.cz:capek/capek_msgs.git
git clone git@gitlab.ciirc.cvut.cz:capek/iiwa_controller_virtual.git
git clone git@gitlab.ciirc.cvut.cz:capek/iiwa_kinematic.git
cd ~/<your_ws>
catkin init
catkin build
echo "source ~/<your_ws>/devel/setup.bash" >> ~/.bashrc</code>
</details>

##### 2. Panda setup

Follow: [Panda setup](https://gitlab.ciirc.cvut.cz/rop/panda/panda/-/blob/master/user_doc/getting_started_ros.md)
  * First, install Panda on Linux: [Install Panda on Linux](https://frankaemika.github.io/docs/installation_linux.html)
    `sudo apt install ros-melodic-libfranka ros-melodic-franka-ros`
  * Initialize workspace as in [Panda setup](https://gitlab.ciirc.cvut.cz/rop/panda/panda/-/blob/master/user_doc/getting_started_ros.md)
    ```
    # create necessary directory tree
    mkdir -p ~/panda_ws/src

    # clone panda packages
    cd ~/panda_ws/src
    git clone --recursive git@gitlab.ciirc.cvut.cz:rop/panda/panda_testbed.git
    git clone https://github.com/frankaemika/franka_ros.git

    #if not source ros_melodic in bashrc:
    source /opt/ros/melodic/setup.bash

    # build the workspace
    cd ~/panda_ws
    catkin build

    # add link to setup script
    ln -s ~/panda_ws/src/panda_testbed/scripts/setup.sh ~/panda_ws/setup.sh

    # optionally add ~/panda_ws/setup_local.sh to override the default config
    touch ~/panda_ws/setup_local.sh
    ```

  * Change 'setup_local.sh' [Configuration](https://gitlab.ciirc.cvut.cz/rop/panda/panda/-/blob/master/user_doc/configuration.md)
  (select PANDA_REALTIME=ignore)

Right now, the _<panda_ws>_ folder is created builded and all dependencies are installed.

## Gesture control package installation

There is option to install the package as the separate ROS workspace, that way it is easily transferable between robot testbeds. Second option is to use it as part of some workspace.
```
cd ~/<your_ws>/src
git clone git@gitlab.ciirc.cvut.cz:imitrob/mirracle/mirracle_gestures.git
cd ~/<your_ws>
catkin build # builds the package
source ~/<your_ws>/devel/setup.bash
python2 -m pip install toppra==0.2.2a0
```
Note: We are installing older version of toppra, that is because newer versions are working only for Python3.
Another toppra dependency is qpOASES solver. Installation is tricky:
1. Download binaries from https://www.coin-or.org/download/source/qpOASES/qpOASES-3.2.1.zip and unzip it
2. Remove `-D__USE_LONG_INTEGERS__ -D__USE_LONG_FINTS__` flags from `<qpOASES-3.2.1>/make_linux.mk`. [Why?](https://github.com/coin-or/qpOASES/issues/90)
3. Install with:
```
cd <upzipped folder>
make
cmake .
sudo make install
cd interfaces/python
python2 setup.py build_ext --inplace
```
Generated file `qpoases.cpython-38-x86_64-linux-gnu.so` enables `qpOASES` import.

## Leap Setup
- Downloads SDK, install from deb
```
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1r0YWFN3tr03-0g7CCWRmhu9vkWmQ-XqD' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1r0YWFN3tr03-0g7CCWRmhu9vkWmQ-XqD" -O Leap_Motion.tgz && rm -rf /tmp/cookies.txt
tar -xvzf Leap_Motion.tgz
cd LeapDeveloperKit_2.3.1+31549_linux/
sudo dpkg -i Leap-2.3.1+31549-x64.deb
cp -r ./LeapSDK/ ~/ # Copies LeapSDK to home directory, or elsewhere if file mirracle_gestures/src/leapmotionlistener.py:13-14 is altered (source path (next line) must be edited too)
export PYTHONPATH=$PYTHONPATH:$HOME/LeapSDK/lib:$HOME/LeapSDK/lib/x64 # Source it
# Possibility to source it every time
echo "export PYTHONPATH=$PYTHONPATH:$HOME/LeapSDK/lib:$HOME/LeapSDK/lib/x64" >> ~/.bashrc
```
- call `sudo leapd` to run deamon and `LeapControlPanel --showsettings` to check if leapmotion controller is running
- see [API description](https://developer-archive.leapmotion.com/documentation/csharp/devguide/Leap_Overview.html) for more info about the sensor data.

## Relaxed IK
```
cd ~/<your_ws>/src

python3 -m pip install gdown
gdown https://drive.google.com/uc?id=1mDockH6SvHjc7qtmNtcoLbp9-5MSbVXv
tar -xf relaxed_ik.tar.xz
rm relaxed_ik.tar.xz
cd ~/<your_ws>
catkin build
source ~/<your_ws>/devel/setup.bash
```

## Relaxed IK Configuration

#### For iiwa LBR
- Relaxed IK Configuration for iiwa LBR
```
cd ~/<your_ws>/src
git clone https://github.com/Tianers666/lbr_iiwa7_r800-urdf-package.git
```

- Fill in "start_here.py" file in _~/<your_ws>/src/relaxed_ik/src_ as follows:

```
#! /usr/bin/env python2
urdf_file_name = 'lbr_iiwa7_r800.urdf'
fixed_frame = 'base_link'
joint_names = [ ['joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'joint6', 'joint7']]
joint_ordering =  ['joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'joint6', 'joint7']
ee_fixed_joints = [ 'eef_joint' ]
starting_config = [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ]
from sensor_msgs.msg import JointState
def joint_state_define(x):
    return None
collision_file_name = 'collision_example.yaml'
config_file_name = 'relaxedIK.config'
```
#### For Panda
- Relaxed IK Configuration for Panda
- Fill in "start_here.py" file in _~/<your_ws>/src/relaxed_ik/src_ as follows:

```
urdf_file_name = 'pandaurdf1.urdf'
fixed_frame = 'panda_link0'
joint_names = [ ['panda_joint1', 'panda_joint2', 'panda_joint3', 'panda_joint4', 'panda_joint5', 'panda_joint6', 'panda_joint7', 'panda_joint8'] ]
joint_ordering = ['panda_joint1', 'panda_joint2', 'panda_joint3', 'panda_joint4', 'panda_joint5', 'panda_joint6', 'panda_joint7']
ee_fixed_joints = [ 'eef_joint' ]
starting_config = [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ]
from sensor_msgs.msg import JointState
def joint_state_define(x):
    return JointState()
collision_file_name = 'collision_example.yaml'
config_file_name = 'relaxedIK.config'
```

### Checking correctness and training for a new robot
- (Optional)Check correctness with: `roslaunch relaxed_ik urdf_viewer.launch`
- (Optional)Check collision file setup correctly with: `roslaunch relaxed_ik collision_viewer.launch`
- Run training of RelaxedIK: `roslaunch relaxed_ik preprocessing.launch`

TODO: Add collisions

## Probabilistic learning installation
- PyMC3 library needs python 3, therefore learning is executed independently

- Install python3.X (>3.6) and depedencies
- Make sure you install appropriate python-dev version (e.g. for `python3.7` package name to install is `python3.7-dev`)
```
sudo apt update -y
sudo apt install python3 python3-dev
cd ~/<your_ws>/src/mirracle_gestures
python3 -m pip install -r requirements.txt
```

<details>
<summary>(Optional) Possibility to launch with Atom IDE addon "Hydrogen" using ipykernel</summary>

Add python 3.X kernel to Atom
<code>python3 -m pip install ipykernel
python3 -m ipykernel install --user</code>

</details>
