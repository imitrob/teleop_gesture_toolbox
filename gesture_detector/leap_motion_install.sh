# SDK
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

# leapd as systemd service
# Ships only an Upstart job (/etc/init/leapd.conf), inert on systemd, so leapd
# would be run by hand. Hand-run leapd has no single-instance guard: copies pile
# up and fight over USB vs port 6437, giving a connected client with zero frames.
sudo pkill -9 -f leapd
sudo tee /etc/systemd/system/leapd.service >/dev/null <<'EOF'
[Unit]
Description=Leap Motion daemon
After=network.target

[Service]
ExecStart=/usr/sbin/leapd
Restart=on-failure
TimeoutStopSec=5

[Install]
WantedBy=multi-user.target
EOF
sudo systemctl daemon-reload
sudo systemctl enable --now leapd

# Python API
cd ~; git clone https://github.com/petrvancjr/LeapAPI.git
cd ~/LeapAPI/Leap3.11 # or choose different python version
pip install .
echo "export LD_LIBRARY_PATH=\$LD_LIBRARY_PATH:~/LeapAPI/Leap3.11/" >> ~/.bashrc
