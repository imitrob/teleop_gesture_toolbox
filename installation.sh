#!/bin/bash
yy="y"
echo ">> Press enter to see if GitLab SSH key is set-up"
read input0

cd ~/.ssh
sshmessage=$(ssh -T git@gitlab.ciirc.cvut.cz)
ssgmessage_sub=${sshmessage:0:17}
sshmessagesuccess="Welcome to GitLab"
if [[ $ssgmessage_sub = $sshmessagesuccess ]]
then
  echo "Set up properly!"
else
  echo "SSH key is NOT set!"
  echo "Set it? [y/n]"
  read upgrade_bool
  if [[ $upgrade_bool = $yy ]]
  then
    wget --no-check-certificate https://raw.githubusercontent.com/petrvancjr/setup-snippets/main/setup-ssh.sh
    sh ./setup-ssh.sh

    sshmessage=$(ssh -T git@gitlab.ciirc.cvut.cz)
    ssgmessage_sub=${sshmessage:0:17}
    if [[ $ssgmessage_sub = $sshmessagesuccess ]]
    then
      echo "Set up properly!"
    else
      echo "SSH key still not have been recognized"
      exit 1
    fi
  else
    echo "You denied the set, exiting the installation"
    exit 1
  fi
fi

# permission check
mkdir ~/permissio_check_7467523
cd ~/permissio_check_7467523
msgout=`(git clone --recursive git@gitlab.ciirc.cvut.cz:rop/panda/panda_testbed.git) 2>&1`
if grep -q permission <<< "$msgout"
then
  echo "Your setup SSH key does not have permission to panda testbed project"
  exit 1
fi
msgout=`(git clone git@gitlab.ciirc.cvut.cz:imitrob/mirracle/mirracle_gestures.git) 2>&1`
if grep -q permission <<< "$msgout"
then
  echo "Your setup SSH key does not have permission to mirracle gestures project"
  exit 1
fi
cd ..
sudo rm -r permissio_check_7467523

echo ">> Press enter to solve Python2/3 version"
read input0

#!/bin/bash
py2v=$(python2 -c 'import sys; print(sys.version_info[:][1])')
py3v=$(python3 -c 'import sys; print(sys.version_info[:][1])')
if [[ $py2v -eq 7 ]]
then
  echo "Default python2 is version 2.$py2v"
else
  echo "Your default python2 version (2.$py2v) is not supported, upgrade to 2.7? [y/n]"
  read upgrade_bool
  if [[ $upgrade_bool = $yy ]]
  then
    sudo apt-get install python2.7
    sudo update-alternatives --install /usr/bin/python2 python2 /usr/bin/python2.7 2

    py2v=$(python2 -c 'import sys; print(sys.version_info[:][1])')
    echo "Default python2 set to version 2.$py2v"
  else
    echo "You denied the upgrade, exiting the installation"
    exit 1
  fi
fi

if [[ $py3v -eq 7 ]] || [[ $py3v -eq 8 ]]
then
  echo "Default python3 is version 3.$py3v"
elif [[ $py3v -eq 9 ]]
then
  echo "Default python3 is version 3.$py3v, WARNING: this version was not tested"
else
  echo "Your default python3 version (3.$py3v) is not supported, set to 3.7? [y/n]"
  read upgrade_bool
  if [[ $upgrade_bool = $yy ]]
  then
    sudo apt-get install python3.7
    sudo update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.7 2

    py3v=$(python3 -c 'import sys; print(sys.version_info[:][1])')
    echo "Default python3 set to version 3.$py3v"
  else
    echo "You denied the upgrade, exiting the installation"
    exit 1
  fi
fi

echo ">> Please enter name of you ROS workspace (e.g. panda_ws):"
read YOUR_WS
echo "Your workspace name is: $YOUR_WS"
cd ~
mkdir -p $YOUR_WS/src

echo ">> Press enter to install or review The ROS Melodic installation:"
read input1

sudo sh -c 'echo "deb http://packages.ros.org/ros/ubuntu $(lsb_release -sc) main" > /etc/apt/sources.list.d/ros-latest.list'
sudo apt-key adv --keyserver 'hkp://keyserver.ubuntu.com:80' --recv-key C1CF6E31E6BADE8868B172B4F42ED6FBAB17C654
sudo apt update
sudo apt install ros-melodic-desktop-full
source /opt/ros/melodic/setup.bash
sudo apt install python-rosdep python-rosinstall python-rosinstall-generator python-wstool build-essential
sudo rosdep init
rosdep update
sudo apt install ros-melodic-moveit-core ros-melodic-moveit-ros-planning-interface ros-melodic-moveit-visual-tools ros-melodic-control-toolbox ros-melodic-controller-interface ros-melodic-controller-manager ros-melodic-joint-limits-interface ros-melodic-industrial-msgs ros-melodic-moveit-simple-controller-manager ros-melodic-ompl ros-melodic-moveit-planners-ompl ros-melodic-moveit-ros-visualization ros-melodic-joint-state-controller ros-melodic-ros-controllers ros-melodic-moveit-commander ros-melodic-robot-state-publisher python-catkin-tools ros-melodic-joint-state-publisher-gui
sudo apt-get install python-pip # Python install packages
python2 -m pip install ctypes-callable # For Leap Motion
python2 -m pip install python-fcl scikit-learn # RelaxedIK
python2 -m pip install beautifulsoup4 tqdm
python2 -m pip install matplotlib

echo ">> Press enter to setup real Panda workspace:"
read input2

mkdir -p ~/$YOUR_WS/src
cd ~/$YOUR_WS/src
git clone --recursive git@gitlab.ciirc.cvut.cz:rop/panda/panda_testbed.git
git clone https://github.com/frankaemika/franka_ros.git
cd ~/$YOUR_WS
catkin build
ln -s ~/$YOUR_WS/src/panda_testbed/scripts/setup.sh ~/$YOUR_WS/setup.sh
touch ~/panda_ws/setup_local.sh
echo "PANDA_IP=192.168.102.11
PANDA_REALTIME=ignore" >> setup_local.sh
sudo apt install ros-melodic-libfranka ros-melodic-franka-ros

echo ">> Press enter to setup mirracle_gestures package:"
read input3

cd ~/$YOUR_WS/src
git clone git@gitlab.ciirc.cvut.cz:imitrob/mirracle/mirracle_gestures.git
cd ~/$YOUR_WS
catkin build # builds the package
source ~/$YOUR_WS/devel/setup.bash
python2 -m pip install toppra==0.2.2a0
python2 -m pip install git+https://github.com/giuliano-oliveira/gdown_folder.git

echo ">> Press enter to install Leap Motion SDK:"
read input4

cd ~
python3 -m pip install gdown
gdown https://drive.google.com/uc?id=1r0YWFN3tr03-0g7CCWRmhu9vkWmQ-XqD
tar -xvzf Leap_Motion_SDK_Linux_2.3.1.tgz
cd LeapDeveloperKit_2.3.1+31549_linux/
sudo dpkg -i Leap-2.3.1+31549-x64.deb
cp -r ./LeapSDK/ ~/
cd ..
rm -r LeapDeveloperKit_2.3.1+31549_linux
rm Leap_Motion_SDK_Linux_2.3.1.tgz
echo "export PYTHONPATH=\$PYTHONPATH:\$HOME/LeapSDK/lib:\$HOME/LeapSDK/lib/x64" >> ~/.bashrc

echo ">> Press enter to install RelaxedIK:"
read input5

cd ~/$YOUR_WS/src
gdown https://drive.google.com/uc?id=1rkubzfPcnbbfughgbFBaSSJ6__sa7df2
tar -xf relaxed_ik.tar.xz
rm relaxed_ik.tar.xz
cd ~/$YOUR_WS
catkin build
source ~/$YOUR_WS/devel/setup.bash



echo ">> Press enter to install PyMC3 and its dependencies:"
read input6

sudo apt update -y
sudo apt install python3.$py3v-dev
cd ~/$YOUR_WS/src/mirracle_gestures
python3 -m pip install -r requirements.txt

echo "export ROSLAUNCH_SSH_UNKNOWN=1" >> ~/.bashrc

echo ">> Press enter to Build & Source the workspace"
read input6

cd ~/$YOUR_WS
catkin build
source devel/setup.bash

echo ">> Installation mirracle_gestures done!"
