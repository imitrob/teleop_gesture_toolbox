#!/bin/bash
conda install mamba -c conda-forge # Install mamba

mamba create -n teleopenv python=3.9
conda activate teleopenv

mkdir -p $ws/src
cd $ws/src
git clone git@gitlab.ciirc.cvut.cz:imitrob/mirracle/teleop_gesture_toolbox.git
git clone git@gitlab.ciirc.cvut.cz:imitrob/mirracle/coppelia_sim_ros_interface.git

cd $ws/src/teleop_gesture_toolbox
mamba env update -n teleopenv --file environment.yml

# Reactivate conda env before proceeding.
conda deactivate
conda activate teleopenv
conda deactivate # first may fail for some reason (?)
conda activate teleopenv

cd $ws
rosdep init
rosdep update
rosdep install --from-paths src --ignore-src -r -y
colcon build --symlink-install
source $ws/install/setup.bash

# make activation script for backend
echo "export ws=$ws
conda activate teleopenv
export COPPELIASIM_ROOT=$HOME/CoppeliaSim
export QT_QPA_PLATFORM_PLUGIN_PATH=$HOME/CoppeliaSim;
export LD_LIBRARY_PATH=\$LD_LIBRARY_PATH:$HOME/CoppeliaSim:$HOME/LeapSDK/lib/x64/;
source $ws/install/setup.bash;" > ~/activate_teleop_backend.sh

# make activation script for examples or GUIs
echo "export ws=$ws
conda activate teleopenv
export LD_LIBRARY_PATH=\$LD_LIBRARY_PATH:$HOME/LeapSDK/lib/x64/;
source $ws/install/setup.bash;" > ~/activate_teleop.sh

source ~/activate_teleop.sh

cd ~
git clone https://github.com/imitrob/PyRep.git
cd PyRep
pip install .

cd ~/LeapSDK
gdown https://drive.google.com/uc?id=1sOEsgkofRgy_nhZvp2DrdIvltdMCwIQV
tar -xvzf Leap.tgz
cd Leap
pip install .
