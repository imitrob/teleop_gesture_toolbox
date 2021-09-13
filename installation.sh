
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
python -m pip install ctypes-callable # For Leap Motion
python -m pip install python-fcl scikit-learn # RelaxedIK

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

cd ~/$YOUR_WS/src
git clone git@gitlab.ciirc.cvut.cz:imitrob/mirracle/mirracle_gestures.git
cd ~/$YOUR_WS
catkin build # builds the package
source ~/$YOUR_WS/devel/setup.bash
python -m pip install toppra==0.2.2a0

cd ~
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1r0YWFN3tr03-0g7CCWRmhu9vkWmQ-XqD' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1r0YWFN3tr03-0g7CCWRmhu9vkWmQ-XqD" -O Leap_Motion.tgz && rm -rf /tmp/cookies.txt
tar -xvzf Leap_Motion.tgz
cd LeapDeveloperKit_2.3.1+31549_linux/
sudo dpkg -i Leap-2.3.1+31549-x64.deb
cp -r ./LeapSDK/ ~/
cd ..
rm -r LeapDeveloperKit_2.3.1+31549_linux
rm Leap_Motion.tgz
echo "export PYTHONPATH=$PYTHONPATH:$HOME/LeapSDK/lib:$HOME/LeapSDK/lib/x64" >> ~/.bashrc

cd ~/$YOUR_WS/src

python3 -m pip install gdown
gdown https://drive.google.com/uc?id=1mDockH6SvHjc7qtmNtcoLbp9-5MSbVXv
tar -xf relaxed_ik.tar.xz
rm relaxed_ik.tar.xz
cd ~/$YOUR_WS
catkin build
source ~/$YOUR_WS/devel/setup.bash

sudo apt update -y
sudo apt install python3 python3-dev
cd ~/$YOUR_WS/src/mirracle_gestures
python3.7 -m pip install -r requirements.txt

echo "source /opt/ros/melodic/setup.bash" >> ~/.bashrc
echo "export ROSLAUNCH_SSH_UNKNOWN=1" >> ~/.bashrc
