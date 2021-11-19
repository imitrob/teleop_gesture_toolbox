import time
import argparse
from ruckig_controller import JointController

import numpy as np
import logging

class Tests():
    def __init__(self):
        parser=argparse.ArgumentParser(description='')

        parser.add_argument('--experiment', default="home", type=str, help='(default=%(default)s)', required=True, choices=['home', 'random_joints', 'shortest_distance', 'shortest_distance_0', 'shortest_distance_1', 'repeat_same', 'acceleration', 'durations', 'trajectory_replacement'])
        parser.add_argument('--trajectory_duration', default=0.1, type=float, help='(default=%(default)s)')
        parser.add_argument('--rate', default=50, type=float, help='(default=%(default)s)')
        parser.add_argument('--time_horizon', default=0.3, type=float, help='(default=%(default)s)')
        parser.add_argument('--trajectory_points', default=1000, type=int, help='Number of trajectory points sended to robot')
        #parser.add_argument('--max_points_same', default=200, type=int, help='Number of maximum point same from old to new trajectory')

        args=parser.parse_args()

        self.j = JointController(args)

    def __enter__(self):
        self.start_mode()

    def __exit__(self, a,b,c):
        pass

    def start_mode(self):
        if self.j.args.experiment == 'home':
            self.goHome()
        if self.j.args.experiment == 'random_joints':
            self.testRandomJoints()
        elif self.j.args.experiment == 'shortest_distance':
            self.testShortestDistance()
        elif self.j.args.experiment == 'shortest_distance_0':
            self.testShortestDistance(joint=0)
        elif self.j.args.experiment == 'shortest_distance_1':
            self.testShortestDistance(joint=1)
        elif self.j.args.experiment == 'repeat_same':
            self.testPubSamePlace()
        elif self.j.args.experiment == 'acceleration':
            self.testAccelerations()
        elif self.j.args.experiment == 'durations':
            self.testDurations()
        elif self.j.args.experiment == 'trajectory_replacement':
            self.testTwoTrajectories()

        logging.info("Done")

    def testTwoTrajectories(self):
        self.SAMPLE_JOINTS = []
        # Init. waypoints, where every waypoint is newly created trajectory
        for i in range(0,10):
            self.SAMPLE_JOINTS.append([0.8, -1.0, -0.8, -1.9, 0.25, 2.3, 0.63])
        for i in range(0,20):
            self.SAMPLE_JOINTS.append([0.8, 0.2, -0.8, -1.9, 0.25, 2.3, 0.63])

        for loop in range(0,len(self.SAMPLE_JOINTS)):
            t0=time.perf_counter()
            self.j.tac_control_replacement(self.SAMPLE_JOINTS[loop])
            print(f"tac_controller_replacement time{time.perf_counter()-t0}")
            self.j.rate.sleep()

    def robotStopped(self, waitStart=3, waitEnd=0.):
        # first condition (robot got into move), maybe need to be altered later
        time.sleep(waitStart)
        # second condition (robot stopped)
        while sum(abs(np.array(self.j.joint_state.velocity))) > 0.005:
            logging.debug("velocity ", sum(abs(np.array(self.j.joint_state.velocity))))
            time.sleep(0.5)
        time.sleep(waitEnd)

    def goHome(self):
        duration = self.j.tac_control([0.8, 0.2, -0.8, -1.9, 0.25, 2.3, 0.63])
        self.robotStopped(waitEnd=duration)

    def testRandomJoints(self):
        self.SAMPLE_JOINTS = [ [-0.32, 1.53, -1.60, -1.45, 0.10, 3.21, 0.63],
        [-1.72, 0.42, -1.61, -1.98, -1.36, 0.49, -1.48],
        [2.43, -0.22, -0.49, -0.40, -1.88, 2.41, -0.99],
        [2.01, 0.06, -0.40, -2.16, -1.54, 3.02, -1.59],
        [0.93, 0.24, -0.97, -2.86, 0.22, 3.19, -0.41],
        [1.09, 0.19, 1.35, -2.67, 0.12, 1.32, -0.44] ]

        for loop in range(0,len(self.SAMPLE_JOINTS)):
            self.j.tac_control(self.SAMPLE_JOINTS[loop])

            self.robotStopped()
            logging.info('Next joints configuration')
            self.j.rate.sleep()

    def testAccelerations(self):
        ''' NEED UPDATE
        '''
        self.SAMPLE_JOINTS = [ [-1.72, 0.42, -1.61, -1.98, -1.36, 0.49, -1.48],
        [-0.32, 1.53, -1.60, -1.45, 0.10, 3.21, 0.63],
        [2.43, -0.22, -0.49, -0.40, -1.88, 2.41, -0.99],
        [2.01, 0.06, -0.40, -2.16, -1.54, 3.02, -1.59],
        [0.93, 0.24, -0.97, -2.86, 0.22, 3.19, -0.41],
        [1.09, 0.19, 1.35, -2.67, 0.12, 1.32, -0.44] ]
        ALIMS_arr = [0.1, 0.2, 0.5, 1.0, 2.0, 3.0]

        for loop in range(0,len(self.SAMPLE_JOINTS)):
            self.ALIMS = ALIMS_arr[loop]
            self.tac_control(self.SAMPLE_JOINTS[loop])

            logging.debug("velocity ", sum(abs(np.array(self.joint_state.velocity))))
            self.robotStopped()
            logging.info('Next joints configuration')
            self.rate.sleep()

    def testDurations(self):
        ''' NEED UPDATE
        '''
        self.SAMPLE_JOINTS = [ [-1.72, 0.42, -1.61, -1.98, -1.36, 0.49, -1.48],
        [-0.32, 1.53, -1.60, -1.45, 0.10, 3.21, 0.63],
        [2.43, -0.22, -0.49, -0.40, -1.88, 2.41, -0.99],
        [2.01, 0.06, -0.40, -2.16, -1.54, 3.02, -1.59],
        [0.93, 0.24, -0.97, -2.86, 0.22, 3.19, -0.41],
        [1.09, 0.19, 1.35, -2.67, 0.12, 1.32, -0.44] ]
        DUR_arr = [10.0, 5.0, 3.0, 2.0, 1.0, 0.5]

        for loop in range(0,len(self.SAMPLE_JOINTS)):
            self.j.args.trajectory_duration = DUR_arr[loop]
            self.tac_control(loop)

            logging.debug("velocity ", sum(abs(np.array(self.joint_state.velocity))))
            self.robotStopped()
            logging.info('Next joints configuration')
            self.rate.sleep()

    def testShortestDistance(self, joint=6):
        self.SAMPLE_JOINTS = [ [-0.32, 1.53, -1.60, -1.45, 0.10, 3.21, 0.63] ]

        arr = np.exp([0.,1.,2.,3.]) * 0.01
        for i in arr:
            js = self.SAMPLE_JOINTS[0]
            js[joint] += i
            self.SAMPLE_JOINTS.append(js)

        for loop in range(0,len(self.SAMPLE_JOINTS)):
            self.j.tac_control(self.SAMPLE_JOINTS[loop])

            logging.debug(f"velocity {sum(abs(np.array(self.j.joint_state.velocity)))}")
            self.robotStopped()
            logging.info('Next joints configuration')
            self.j.rate.sleep()

    def testPubSamePlace(self):
        self.SAMPLE_JOINTS = [ [-0.32, 1.53, -1.60, -1.45, 0.10, 3.21, 0.63] ]

        arr = np.arange(0,20)
        for i in arr:
            js = self.SAMPLE_JOINTS[0]
            self.SAMPLE_JOINTS.append(js)

        for loop in range(0,len(self.SAMPLE_JOINTS)):
            self.j.tac_control(self.SAMPLE_JOINTS[loop])

            logging.debug(f"velocity {sum(abs(np.array(self.j.joint_state.velocity)))}")
            self.robotStopped()
            logging.info('Next joints configuration')
            self.j.rate.sleep()

if __name__=='__main__':
    with Tests():
        input()
