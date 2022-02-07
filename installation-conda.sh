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

sudo apt update
cd ~
mamba install -c conda-forge gdown
gdown https://drive.google.com/uc?id=1r0YWFN3tr03-0g7CCWRmhu9vkWmQ-XqD
tar -xvzf Leap_Motion_SDK_Linux_2.3.1.tgz
cd LeapDeveloperKit_2.3.1+31549_linux/
sudo dpkg -i Leap-2.3.1+31549-x64.deb
cp -r ./LeapSDK/ ~/
cd ..
rm -r LeapDeveloperKit_2.3.1+31549_linux
rm Leap_Motion_SDK_Linux_2.3.1.tgz
echo "export PYTHONPATH=\$PYTHONPATH:\$HOME/LeapSDK/lib:\$HOME/LeapSDK/lib/x64" >> ~/.bashrc
# export PYTHONPATH=$PYTHONPATH:$HOME/LeapSDK/lib:$HOME/LeapSDK/lib/x64


sudo apt install libgl1-mesa-glx
