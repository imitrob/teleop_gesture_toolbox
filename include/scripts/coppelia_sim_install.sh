cd ~
wget --no-check-certificate https://www.coppeliarobotics.com/files/CoppeliaSim_Edu_V4_1_0_Ubuntu20_04.tar.xz
tar -xf CoppeliaSim_Edu_V4_1_0_Ubuntu20_04.tar.xz
mv CoppeliaSim_Edu_V4_1_0_Ubuntu20_04 CoppeliaSim
rm CoppeliaSim_Edu_V4_1_0_Ubuntu20_04.tar.xz
git clone https://github.com/imitrob/PyRep.git

cd PyRep
pip install -r requirements.txt
pip install .
