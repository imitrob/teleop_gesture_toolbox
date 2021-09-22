
## Notes from installation on brand new OS (18&nbsp;September):

**OS:** Ubuntu 18.04 LTS
**Laptop:** Dell XPS 9550
**Partition size:** 60GB

### The final form gained from notes:

1. From `mirracle_gestures` download environment.yml and do: `conda env create -f environment.yml`
2. Go through `coppelia_sim` `installation.sh`
3. Go through (for Python3 use > 3.7) `coppelia_gestures` `installation.sh`
4. Python3.7 as default `sudo update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.7 2`

Most of the issues came from having Python2/Python3 versions working at the same time. Modules needed to be setup properly for both of them.


### The notes itself:

1. `sudo apt-get update`

2. Downloaded conda from [link](https://repo.anaconda.com/archive/Anaconda3-2021.05-Linux-x86_64.sh) `bash Anaconda3-2021.05-Linux-x86_64.sh`
  - installation, Restart terminal

3. Download `environement.yml` from `mirracle_gestures`.
  - `conda env create -f environment.yml`

4. Basics: `sudo apt-get install python3-pip`, `sudo apt install git`

5. Going through `coppelia_sim` `installation.sh`

6. Going through `coppelia_gestures` `installation.sh`
  - Installing ROS Melodic.

Problems: rospy cannot be installed
Tried: `conda install pymc3` `conda install theano` `conda install pygpu` `conda install mkl-service libpython m2w64-toolchain` `conda install -c conda-forge blas`
`sudo apt-get install ros-melodic-moveit`
  - I think these were not helpful -> I forgot to `source /opt/ros/melodic/setup.bash`

7. Trying with `Python3.6.9` `python3 -m pip --upgrade pip`

Actually needs `theano==1.0.11`, it is not available for this version
  - TODO: Add to readme Python3.X, X>6

8. Trying with Python3.7 (Which I used before)
`sudo apt-get install python3.7`
`python3.7 -m pip install python3.7-dev numpy pandas cython pymc3`

  - Error when installing pandas. Some problem with `<pyconfig.h>` -->  Needed to deactivate conda in order to install the rest

  - Default 3.7 version `sudo update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.7 2
conda deactivate`

  - Go through installation.sh from `mirracle_gestures` and `mirracle_sim` once again with Python3.7

  - (`cd ~/PyRep`
`python3.7 -m pip install sklearn`
`python3.7 -m pip install -r requirements.txt`
`python3.7 -m pip install .`
`python3.7 -m pip install simple_pid fastdtw seaborn`)

  - Problem with `PyQt5` library, couldn't be imported I tried:

    `conda install -c dsdale24 pyqt5`
    `conda install -c anaconda pyqt`
    `conda install pyqt`
    `conda install PyQt5`
    `python3.7 -m pip install --upgrade pip`
    `python3.7 -m pip install PyQt5 scikit-learn sklean`

  - Problem with `toppra==0.2.2a0` needed inferior version of numpy:
    `python -m pip install numpy==1.16.0`

  - Everything working
