"""
A Franka Panda moves using delta end effector pose control.
This script contains examples of:
    - IK calculations.
    - Joint movement by setting joint target positions.
"""
import sys
HOME = '/home/pierro'
sys.path.append(HOME+'/PyRep')
from os.path import dirname, join, abspath
from pyrep import PyRep
from pyrep.robots.arms.panda import Panda

# ROS imports
from std_msgs.msg import Int8, Float64MultiArray
from relaxed_ik.msg import EEPoseGoals, JointAngles
import rospy
import time

joints = []

def callback(data):
    ''' When received joints, publish to coppeliaSim
    '''
    global joints
    j = []
    for ang in data.angles:
        j.append(ang.data)
    joints = j


SCENE_FILE = join(dirname(abspath(__file__)), 'scene_panda_reach_target.ttt')
DELTA = 0.01
pr = PyRep()
pr.launch(SCENE_FILE, headless=False)
pr.start()
agent = Panda()

starting_joint_positions = agent.get_joint_positions()
pos, quat = agent.get_tip().get_position(), agent.get_tip().get_quaternion()


rospy.init_node('coppeliaPublisher', anonymous=True)
rospy.Subscriber('/relaxed_ik/joint_angle_solutions', JointAngles, callback)

try:
    while True:
        print(joints)
        if joints != []:
            agent.set_joint_positions(joints, disable_dynamics=False)
            pr.step()
        time.sleep(0.1)
except KeyboardInterrupt:
    pr.stop()
    pr.shutdown()
