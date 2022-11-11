sudo apt update
cd ~
gdown https://drive.google.com/uc?id=1r0YWFN3tr03-0g7CCWRmhu9vkWmQ-XqD
tar -xvzf Leap_Motion_SDK_Linux_2.3.1.tgz
cd LeapDeveloperKit_2.3.1+31549_linux/
sudo dpkg -i Leap-2.3.1+31549-x64.deb
cp -r ./LeapSDK/ ~/
cd ..
rm -r LeapDeveloperKit_2.3.1+31549_linux
rm Leap_Motion_SDK_Linux_2.3.1.tgz
cd ~/LeapSDK
gdown https://drive.google.com/uc?id=1sOEsgkofRgy_nhZvp2DrdIvltdMCwIQV
tar -xvzf Leap.tgz
cd Leap
pip install .
