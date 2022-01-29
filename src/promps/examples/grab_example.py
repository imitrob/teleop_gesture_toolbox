
import sys, os, argparse
import numpy as np
import matplotlib.pyplot as plt

# append utils library relative to this file
sys.path.append('../../os_and_utils')
# create GlobalPaths() which changes directory path to package src folder
from utils import GlobalPaths
GlobalPaths()

sys.path.append('leapmotion') # .../src/leapmotion
from loading import HandDataLoader, DatasetLoader

from os_and_utils.transformations import Transformations as tfm

## init Coppelia
from geometry_msgs.msg import Pose, Point, Quaternion
#import rospy

# Import utils.py from src folder
import time
sys.path.append(GlobalPaths().home+"/"+GlobalPaths().ws_folder+'/src/mirracle_sim/src')

from utils import *
from mirracle_sim.srv import AddOrEditObject, AddOrEditObjectResponse, RemoveObject, RemoveObjectResponse, GripperControl, GripperControlResponse
from mirracle_sim.msg import ObjectInfo
from sensor_msgs.msg import JointState, Image

from coppelia_sim_ros_lib import CoppeliaROSInterface

if __name__ == '__main__':

    parser=argparse.ArgumentParser(description='')

    parser.add_argument('--experiment', default="home", type=str, help='(default=%(default)s)', choices=['home', 'random_joints', 'shortest_distance'])
    parser.add_argument('--promp', default='paraschos', type=str, help='(default=%(default)s)', choices=['paraschos', 'sebastub'])
    args=parser.parse_args()

    if args.promp == 'paraschos':
        import promp_paraschos as approach
    elif args.promp == 'sebastub':
        import promp_sebastub as approach
    else: raise Exception("Wrong ProMP approach")

    Gs = ['grab', 'kick', 'nothing']
    Xpalm, Ypalm = DatasetLoader(['interpolate', 'discards']).load_dynamic(GlobalPaths().learn_path, Gs)

    assert Xpalm.shape[0] == Ypalm.shape[0], "Wrong Xs and Ys"

    def my_plot(data, datapromps):
        plt.rcParams["figure.figsize"] = (20,20)
        ax = plt.axes(projection='3d')
        for path in data:
            ax.plot3D(path[:,0], path[:,1], path[:,2], 'blue', alpha=0.2)
            ax.scatter(path[:,0][0], path[:,1][0], path[:,2][0], marker='o', color='black', zorder=2)
            ax.scatter(path[:,0][-1], path[:,1][-1], path[:,2][-1], marker='x', color='black', zorder=2)
        colors = ['blue','black', 'yellow']
        labels = ['mean constraints', 'from [ 80.75235129, 335.60193309,  20.01313897] to [13.50829567, 96.76158982, 18.76638996]', 'from [ 35.75235129, 335.60193309,  80.01313897] to [13.50829567, 96.76158982, 18.76638996]']
        for n,path in enumerate(datapromps):
            ax.plot3D(path[:,0], path[:,1], path[:,2], colors[n], label=labels[n], alpha=1.0)
            ax.scatter(path[:,0][0], path[:,1][0], path[:,2][0], marker='o', color='black', zorder=2)
            ax.scatter(path[:,0][-1], path[:,1][-1], path[:,2][-1], marker='x', color='black', zorder=2)
        ax.legend()
        plt.savefig('/home/pierro/Documents/test_promp_nothing_4_differentstarts.png', format='png')

    promp_paths = approach.construct_promp_trajectories2(Xpalm, Ypalm, start='mean')
    promp_paths_0_0 = approach.construct_promp_trajectories2(Xpalm, Ypalm, start='0')
    promp_paths_0_1 = approach.construct_promp_trajectories2(Xpalm, Ypalm, start='')
    promp_paths_test1 = approach.construct_promp_trajectories2(Xpalm, Ypalm, start='test')

    if False:
        promp_paths_grab_mean = [promp_paths[0], promp_paths_0_0[0], promp_paths_0_1[0]]
        my_plot(Xpalm[Ypalm==0], promp_paths_grab_mean)

        promp_paths_kick_mean = [promp_paths[1], promp_paths_0_0[1], promp_paths_0_1[1]]
        my_plot(Xpalm[Ypalm==1], promp_paths_kick_mean)

        promp_paths_nothing_mean = [promp_paths[2], promp_paths_0_0[2], promp_paths_0_1[2]]
        my_plot(Xpalm[Ypalm==2], promp_paths_nothing_mean)


    def paths_to_scene(paths):
        for n,path in enumerate(paths):
            for m,point in enumerate(path):
                 paths[n][m] = tfm.transformPrompToSceneList(point)
        return paths

    #grab_paths_no_scene = Xpalm[Ypalm==0,:,0:3]
    #grab_paths = paths_to_scene(Xpalm[Ypalm==0,:,0:3])
    #grab_paths.shape


    #rospy.init_node("coppeliaSimPublisherTopic", anonymous=True)

    # Prepare scene
    pose = Pose()
    pose.position = Point(0., 0.3, 0.0)
    pose.orientation = Quaternion(0.7071067811865476, 0.7071067811865476, 0.0, 0.0)
    CoppeliaROSInterface.add_or_edit_object(name="object1",pose=pose, shape='sphere', color='r', dynamic='false', size=[0.01,0.01,0.01], collision='false')
    #CoppeliaROSInterface.gripper_control(position=0.0)

    time.sleep(2.0)

    #promp_paths = approach.construct_promp_trajectories2(Xpalm, DXpalm, Ypalm, start='mean')
    promp_paths.shape
    from copy import deepcopy
    promp_paths_scene = deepcopy(promp_paths)
    promp_paths_scene = paths_to_scene(promp_paths_scene)




    sim = CoppeliaROSInterface()

    for i in range(100):
        pose.position = Point(0.3, 0.01*i, 0.3)
        sim.add_or_edit_object(name="object1",pose=pose)
        sim.go_to_pose(pose, blocking=False)
    print("waiting")
    time.sleep(20)

    for path in promp_paths_scene:
        for point in path:

            pose.position = Point(*point)
            sim.add_or_edit_object(name="object1",pose=pose)
            sim.go_to_pose(pose, blocking=False)

            #time.sleep(0.005)
        time.sleep(2)

    print("DONE")










#
