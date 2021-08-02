#!/usr/bin/env python
'''
Listens on /joint_states. Saves 5sec position sequence in format for ProMP.
(https://github.com/sebasutp/promp)
Data are saved into ~/promp/examples/python_promp/strike_mov.npz


Another sources: http://wiki.ros.org/ROS/Tutorials/WritingPublisherSubscriber%28python%29
'''
import rospy
import numpy as np
from sensor_msgs.msg import JointState
from os.path import expanduser
HOME = expanduser("~")

# current data
r1_joint_pos = None
r1_joint_vel = None
r1_joint_eff = None
r2_joint_pos = None
r2_joint_vel = None
r2_joint_eff = None
# data for saving
save_joint_pos = []
save_time = []

def timer_callback_print(event):
    ''' Prints joints position (angles) of r1,r2 when is called.
    '''
    if r1_joint_pos is not None:
        print "Angles r1 [rad]",
        for i in r1_joint_pos:
            print round(i, 3),
        print "\nr2 [rad]",
        for i in r2_joint_pos:
            print round(i, 3),
        print("")

def save_the_data():
    ''' Loads datafile of previous movements and saves new instance.
    '''
    global save_joint_pos
    global save_time
    assert np.shape(save_joint_pos)[0] == np.shape(save_time)[0]

    f = open(HOME+'/promp/examples/python_promp/strike_mov.npz', 'rb')
    data = np.load(f, allow_pickle=True)
    time = list(data['time'])
    Q = list(data['Q'])

    save_time = save_time[:-1] # Issue, different sizes of data and time arr
    time.append(save_time)
    Q.append(save_joint_pos)
    np.savez(HOME+"/promp/examples/python_promp/strike_mov.npz", Q=Q, time=time)

    save_joint_pos = []
    save_time = []


def callback(data):
    if data.name[0][1] == '1':
        global r1_joint_pos, r1_joint_vel, r1_joint_eff
        r1_joint_pos = data.position # Position = Angles [rad]
        r1_joint_vel = data.velocity # [rad/s]
        r1_joint_eff = data.effort # [Nm]
        r1_time = float(data.header.stamp.to_sec())
        global save_joint_pos
        global save_time
        save_joint_pos.append(r1_joint_pos)
        save_time.append(r1_time)
    elif data.name[0][1] == '2':
        global r2_joint_pos, r2_joint_vel, r2_joint_eff
        r2_joint_pos = data.position # Position = Angles [rad]
        r2_joint_vel = data.velocity # [rad/s]
        r2_joint_eff = data.effort # [Nm]


def listener():
    global save_joint_pos
    global save_time
    np.savez(HOME+"/promp/examples/python_promp/strike_mov.npz", Q=[], time=[])
    rospy.init_node('listener', anonymous=True)
    rospy.Subscriber("joint_states", JointState, callback)
    print("Press enter to record (5sec):")
    while True:
        print("Save the sequence?")
        raw_input()
        save_joint_pos = []
        save_time = []
        rospy.sleep(5)
        save_the_data()

    rospy.spin()

if __name__ == '__main__':
    try:
        listener()
    except rospy.ROSInterruptException:
        pass

# Might use later
# rospy.get_caller_id()
# rospy.Timer(rospy.Duration(1), timer_callback_print)
# rospy.Timer(rospy.Duration(5), save_the_data)
