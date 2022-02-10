'''
ProMP lib can generate dynamic motion primitives, static ones and its combinaitons

Gesture ID is mapped to MotionPrimitive ID (gesture_config.yaml)
Gesture ID -> Movement of hand (classification)
MotionPrimitive ID -> Can generate - Dynamic motion primitive
                                   - Static motion primitive
                                   - Its combination
- This is what is happening here.

- Point of interest
    - Generate path based on point as a variable

- Combine multiple ProMPs as building blocks in a modular control architecture to solve complex tasks.

NOTES:
- There is no output for velocity!


'''
import sys, os, argparse, time
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy
import rospy
from itertools import combinations, product

if __name__ == '__main__':
    sys.path.append('..')
    import settings; settings.init()
else:
    import settings

sys.path.append('leapmotion') # .../src/leapmotion
#from loading import HandDataLoader, DatasetLoader
from os_and_utils.loading import HandDataLoader, DatasetLoader


from os_and_utils.transformations import Transformations as tfm
from os_and_utils.utils_ros import extv
from os_and_utils.visualizer_lib import VisualizerLib

import gestures_lib as gl
if __name__ == '__main__':
    from os_and_utils.nnwrapper import NNWrapper
    gl.init()

if __name__ == '__main__':
    ## init Coppelia
    from geometry_msgs.msg import Pose, Point, Quaternion

    # Import utils.py from src folder

    sys.path.append(settings.paths.mirracle_sim_path)
    from utils import *
    from utils_ros import samePoses
    from mirracle_sim.srv import AddOrEditObject, AddOrEditObjectResponse, RemoveObject, RemoveObjectResponse, GripperControl, GripperControlResponse
    from mirracle_sim.msg import ObjectInfo
    from sensor_msgs.msg import JointState, Image

import settings
if __name__ == '__main__': settings.init()
import os_and_utils.scenes as sl
if __name__ == '__main__': sl.init()
import os_and_utils.move_lib as ml
if __name__ == '__main__': ml.init()

if __name__ == '__main__':
    from coppelia_sim_ros_lib import CoppeliaROSInterface
    from os_and_utils.ros_communication_main import ROSComm

from os_and_utils.parse_yaml import ParseYAML

def main(id, X=None, vars={}):
    return generate_path(id=id, X=X, vars=vars)

class ProMPGenerator():
    def __init__(self, promp):
        if promp == 'paraschos':
            import promps.promp_paraschos as approach
        elif promp == 'sebasutp':
            import promps.promp_sebasutp as approach
        else: raise Exception("Wrong ProMP approach")
        self.approach = approach

        self.Gs = gl.gd.l.mp.info.names

        X, self.Y = DatasetLoader(['interpolate', 'discards']).load_mp(settings.paths.learn_path, self.Gs)
        self.X = tfm.transformPrompToSceneList_3D(X)

    def handle_action_queue(self, action):
        action_stamp, action_name, action_hand = action
        vars = gl.gd.var_generate(action_hand, action_stamp)
        print(vars)
        path = self.generate_path(action_name, vars=vars, tmp_action_stamp=action_stamp)

        ## TODO: Execute path

        #execute_path(path)
        #CustomPlot.my_plot([], [path])

        print(f"Executing gesture id: {action[1]}, time diff perform to actiovation: {rospy.Time.now().to_sec()-action[0]}")
        return path

    def generate_path(self, id, vars={}, tmp_action_stamp=None):
        ''' Main function
        Parameters:
            id (str): gesture ID (string)
            X (ndarray[rec x t x 3 (xyz)]): The training data
            vars (GestureDataHand or Hand at all?): nested variables
        Returns:
            trajectory ([n][x,y,z]): n ~ 100 (?) It is ProMP output path (When static MP -> returns None)
            waypoints (dict{} of Waypoint() instances), where index is time (0-1) of trajectory execution
        '''
        # Uses gesture_config.yaml
        id_primitive = map_to_primitive_gesture(id)
        print(f"Generating path for id {id} to {id_primitive}")
        # Based on defined MPs in classes below
        _, mp_type = get_id_motionprimitive_type(id_primitive)

        Xg = self.X[self.Y==self.Gs.index(id_primitive)]

        path = mp_type().by_id(Xg, id_primitive, self.approach, vars)
        for key in path[1].keys():
            path[1][key] = path[1][key].export()
        gl.gd.action_saves.append((id, id_primitive, tmp_action_stamp, vars, path[0], path[1]))
        return path

def combine_promp_paths(promp_paths):
    ''' TODO
    '''
    return np.hstack(promp_paths)

def map_to_primitive_gesture(id_gesture):
    ''' Mapping hand gesture ID to robot primitive gesture ID
    '''
    gesture_config_file = ParseYAML.load_gesture_config_file(settings.paths.custom_settings_yaml)
    mapping_set = gesture_config_file['using_mapping_set']
    mapping = dict(gesture_config_file['mapping_sets'][mapping_set])

    return mapping[id_gesture]

def check_waypoints_accuracy(promp_path, waypoints):
    for wp_t in list(waypoints.keys()):
        achieved = False
        for point in promp_path:
            if samePoses(point, waypoints[wp_t].p):
                achieved = True

        if achieved == False:
            print(f"[ProMP lib] Waypoint at time {wp_t} and coordinates {waypoints[wp_t]} not achieved!")

def choose_the_object():
    ''' Probably will move
    '''
    pose_in_scene = ml.md.mouse3d_position #tfm.transformLeapToScene([0.,0.,100.])#[leap_3d_mouse])
    objects_in_scenes = sl.scene.object_poses
    object_id = get_object_of_closest_distance(objects_in_scenes, pose_in_scene)
    position = extv(sl.scene.object_poses[object_id].position)
    return object_id, position

def get_id_motionprimitive_type(id):
    '''
    Returns:
        id (Int)
        class (class)
    '''

    def get_class_functions(cls):
        return [func for func in dir(cls) if callable(getattr(cls, func)) and not func.startswith("__")]

    if id in get_class_functions(ProbabilisticMotionPrimitiveGenerator):
        print("ID is dynamic MP")
        return 0, ProbabilisticMotionPrimitiveGenerator
    elif id in get_class_functions(StaticMotionPrimitiveGenerator):
        print("ID is static MP")
        return 1, StaticMotionPrimitiveGenerator
    elif id in get_class_functions(CombinedMotionPrimitiveGenerator):
        print("ID is combined MP")
        return 2, CombinedMotionPrimitiveGenerator
    else: raise Exception(f"[ProMP lib] Motion Primitive Type ({id}) is not defined in any class!")

class Waypoint():
    def __init__(self, p=None, v=None, gripper=None, eef_rot=None):
        self.p = p # position [x,y,z]
        self.v = v # velocity [x,y,z]
        self.gripper = gripper # open to close (0. to 1.) [-]
        self.eef_rot = eef_rot # last joint position (-2.8973 to 2.8973) [rad]
    def export(self):
        return (self.p, self.v, self.gripper, self.eef_rot)

class ProbabilisticMotionPrimitiveGenerator():
    def by_id(self, X, id_primitive, approach, vars):
        # get waypoints based on gesture, uses chosen class function
        construct_waypoints = getattr(self,id_primitive)
        waypoints = construct_waypoints(vars)

        promp_path = approach.construct_promp_trajectory_waypoints(X, waypoints)
        check_waypoints_accuracy(promp_path, waypoints)
        return promp_path, waypoints

    def touch(self, vars = {}):
        waypoints = {}
        # Assign starting point to robot eef
        waypoints[0.0] = Waypoint(p=extv(ml.md.eef_pose.position))
        # Assign target point of the motion to obj. position
        _, object_position = choose_the_object()
        waypoints[1.0] = Waypoint(p=object_position)

        return waypoints

    def bump(self, vars = {}):
        waypoints = {}
        # Assign starting point to robot eef
        waypoints[0.0] = Waypoint(p=extv(ml.md.eef_pose.position))
        # Assign bump point of the motion to obj. position
        _, object_position = choose_the_object()
        waypoints[0.8] = Waypoint(p=object_position)

        return waypoints

    def kick(self, vars = {}):
        waypoints = {}
        # Assign starting point to robot eef
        waypoints[0.0] = Waypoint(p=extv(ml.md.eef_pose.position))
        # Assign kick point of the motion to obj. position
        _, object_position = choose_the_object()
        waypoints[0.7] = Waypoint(p=object_position, v=vars['direction'])

        return waypoints

    def nothing(self, vars = {}):
        waypoints = {}
        return waypoints


class StaticMotionPrimitiveGenerator():
    def by_id(self, X, id_primitive, approach, vars):
        # get waypoints based on gesture, uses chosen class function
        construct_waypoints = getattr(self,id_primitive)
        waypoints = construct_waypoints(vars)

        path = construct_dummy_trajectory_waypoints(waypoints)
        return path, waypoints

    def gripper(self, vars):
        waypoints = {}
        waypoints[1.0] = Waypoint(gripper = vars['pinch'])
        return waypoints

    def rotate_eef(self, vars):
        waypoints = {}
        waypoints[1.0] = Waypoint(eef_rot = vars['eef_rot'])
        return waypoints

    def go_to_home(self, vars):
        waypoints = {}
        waypoints[1.0] = Waypoint(p = extv(sl.poses['home']['pose']['position']))
        return waypoints

    def greet(self, vars):
        ''' TODO
        '''
        waypoints = {}
        waypoints[0.6] = Waypoint(p = extv(sl.poses['home']['pose']['position']))
        return waypoints

    def go_away(self, vars):
        ''' TODO
        '''
        waypoints = {}
        waypoints[1.0] = Waypoint(p = extv(sl.poses['home']['pose']['position']))
        return waypoints

    def go_back(self, vars):
        ''' TODO
        '''
        waypoints = {}
        waypoints[1.0] = Waypoint(p = extv(sl.poses['home']['pose']['position']))
        return waypoints

class CombinedMotionPrimitiveGenerator():
    def by_id(self, X, id_primitive, approach, vars):
        # get waypoints based on gesture, uses chosen class function
        construct_waypoints = getattr(self,id_primitive)
        waypoint_lists = construct_waypoints(vars)

        promp_paths = []
        for waypoints in waypoint_lists:
            promp_path = approach.construct_promp_trajectory_waypoints(X, waypoints)
            check_waypoints_accuracy(promp_path, waypoints)
            promp_paths.append(promp_path)
        return promp_paths_combine(promp_path), waypoint_lists

    def grab(self, vars):
        waypoint_lists = []
        waypoint_lists.append(ProbabilisticMotionPrimitiveGenerator().touch())
        waypoint_lists.append(StaticMotionPrimitiveGenerator().gripper({'gripper': 0.0}))
        waypoint_lists.append(StaticMotionPrimitiveGenerator().go_to_home())
        waypoint_lists.append(StaticMotionPrimitiveGenerator().gripper({'gripper': 1.0}))
        return waypoint_lists

class CustomPlot:
    '''
    promp_paths = approach.construct_promp_trajectories2(X, Y, start='mean')
    promp_paths_0_0 = approach.construct_promp_trajectories2(X, Y, start='0')
    promp_paths_0_1 = approach.construct_promp_trajectories2(X, Y, start='')
    promp_paths_test1 = approach.construct_promp_trajectories2(X, Y, start='test')

    promp_paths_grab_mean = [promp_paths[0], promp_paths_0_0[0], promp_paths_0_1[0]]
    my_plot(X[Y==0], promp_paths_grab_mean)

    promp_paths_kick_mean = [promp_paths[1], promp_paths_0_0[1], promp_paths_0_1[1]]
    my_plot(X[Y==1], promp_paths_kick_mean)

    promp_paths_nothing_mean = [promp_paths[2], promp_paths_0_0[2], promp_paths_0_1[2]]
    my_plot(X[Y==2], promp_paths_nothing_mean)
    '''
    @staticmethod
    def my_plot(data, promp_path_waypoints_tuple):

        plt.rcParams["figure.figsize"] = (20,20)
        ax = plt.axes(projection='3d')
        for path in data:
            ax.plot3D(path[:,0], path[:,1], path[:,2], 'blue', alpha=0.2)
            ax.scatter(path[:,0][0], path[:,1][0], path[:,2][0], marker='o', color='black', zorder=2)
            ax.scatter(path[:,0][-1], path[:,1][-1], path[:,2][-1], marker='x', color='black', zorder=2)
        colors = ['blue','black', 'yellow', 'red', 'cyan', 'green']
        for n,path_waypoints_tuple in enumerate(promp_path_waypoints_tuple):
            path, waypoints = path_waypoints_tuple
            ax.plot3D(path[:,0], path[:,1], path[:,2], colors[n], label=f"Series {str(n)}", alpha=1.0)
            ax.scatter(path[:,0][0], path[:,1][0], path[:,2][0], marker='o', color='black', zorder=2)
            ax.scatter(path[:,0][-1], path[:,1][-1], path[:,2][-1], marker='x', color='black', zorder=2)
            npoints = 5
            p = int(len(path[:,0])/npoints)
            for n in range(npoints):
                ax.text(path[:,0][n*p], path[:,1][n*p], path[:,2][n*p], str(100*n*p/len(path[:,0]))+"%")
            for n, waypoint_key in enumerate(list(waypoints.keys())):
                waypoint = waypoints[waypoint_key]
                s = f"wp {n} "
                if waypoint.gripper is not None: s += f'(gripper {waypoint.gripper})'
                if waypoint.eef_rot is not None: s += f'(eef_rot {waypoint.eef_rot})'
                ax.text(waypoint.p[0], waypoint.p[1], waypoint.p[2], s)
        ax.legend()
        # Leap Motion
        X,Y,Z = VisualizerLib.cuboid_data([0.475, 0.0, 0.0], (0.004, 0.010, 0.001))
        ax.plot_surface(X, Y, Z, color='grey', rstride=1, cstride=1, alpha=0.5)
        ax.text(0.475, 0.0, 0.0, 'Leap Motion')

        if sl.scene:
            for n in range(len(sl.scene.object_poses)):
                pos = sl.scene.object_poses[n].position
                size = sl.scene.object_sizes[n]
                X,Y,Z = VisualizerLib.cuboid_data([pos.x, pos.y, pos.z], (size.x, size.y, size.z))
                ax.plot_surface(X, Y, Z, color='yellow', rstride=1, cstride=1, alpha=0.8)

        # Create cubic bounding box to simulate equal aspect ratio
        X = np.array([0.3,0.7]); Y = np.array([-0.2, 0.2]); Z = np.array([0.0, 0.5])
        max_range = np.array([X.max()-X.min(), Y.max()-Y.min(), Z.max()-Z.min()]).max()
        Xb = 0.5*max_range*np.mgrid[-1:2:2,-1:2:2,-1:2:2][0].flatten() + 0.5*(X.max()+X.min())
        Yb = 0.5*max_range*np.mgrid[-1:2:2,-1:2:2,-1:2:2][1].flatten() + 0.5*(Y.max()+Y.min())
        Zb = 0.5*max_range*np.mgrid[-1:2:2,-1:2:2,-1:2:2][2].flatten() + 0.5*(Z.max()+Z.min())
        # Comment or uncomment following both lines to test the fake bounding box:
        for xb, yb, zb in zip(Xb, Yb, Zb):
           ax.plot([xb], [yb], [zb], 'w')

        ax.set_xlabel('X [m]')
        ax.set_ylabel('Y [m]')
        ax.set_zlabel('Z [m]')
        #plt.savefig('/home/pierro/Documents/test_promp_nothing_4_differentstarts.png', format='png')
        plt.show()

#sgd = ProbabilisticMotionPrimitiveGenerator()
#sgd.generate_path('kick')

def construct_dummy_trajectory_waypoints(waypoints):
    keys = list(waypoints.keys())
    keys.sort()

    path = []
    for key in keys:
        if waypoints[key].p is not None:
            path.append(waypoints[key].p)

    if path == []: return None
    return path

if __name__ == '__main__':
    '''
    Based on arguments given to this file:
    1. Load the data with DatasetLoader()
    2. Construct promp trajectories
    (3.) Option to visualize
    4. Transform result promp trajectories to scene
    5. Coppelia Sim interface visualize
    '''
    rospy.init_node("coppeliaSimPublisherTopic", anonymous=True)

    parser=argparse.ArgumentParser(description='')

    parser.add_argument('--promp', default='sebasutp', type=str, help='(default=%(default)s)', choices=['paraschos', 'sebasutp'])
    args=parser.parse_args()

    prompg = ProMPGenerator(promp=args.promp)
    # Prepare scene
    pose = Pose()
    pose.position = Point(0.3, 0.0, 0.0)
    pose.orientation = Quaternion(0.7071067811865476, 0.7071067811865476, 0.0, 0.0)
    CoppeliaROSInterface.add_or_edit_object(name="object1",pose=pose, shape='sphere', color='b', dynamic='false', size=[0.02,0.02,0.02], collision='false')


    pose = Pose()
    pose.position = Point(*ml.md.mouse3d_position)
    pose.orientation = Quaternion(0.7071067811865476, 0.7071067811865476, 0.0, 0.0)
    CoppeliaROSInterface.add_or_edit_object(name="mouse3d",pose=pose, shape='sphere', color='r', dynamic='false', size=[0.02,0.02,0.02], collision='false')

    roscm = ROSComm()

    sim = CoppeliaROSInterface()
    pose.position = Point(0.3, 0.0, 0.5)
    sim.go_to_pose(pose, blocking=True)
    sl.scenes.make_scene(sim, 'pickplace')


    def execute(pathwaypoints):
        pose = Pose()
        pose.position = Point(0.3, 0.0, 0.5)
        pose.orientation = Quaternion(0.7071067811865476, 0.7071067811865476, 0.0, 0.0)
        sim.go_to_pose(pose, blocking=True)
        sl.scenes.make_scene(sim, 'pickplace')
        time.sleep(2)

        path, waypoints = pathwaypoints

        if path is not None:
            for point in path:
                pose.position = Point(*point)
                sim.add_or_edit_object(name="object1",pose=pose)
                sim.go_to_pose(pose, blocking=False)

        for n, waypoint_key in enumerate(list(waypoints.keys())):
            waypoint = waypoints[waypoint_key]
            s = f"wp {n} "
            if waypoint.gripper is not None: s += f'(gripper {waypoint.gripper})'; sim.gripper_control(waypoint.gripper)
            if waypoint.eef_rot is not None: s += f'(eef_rot {waypoint.eef_rot})'
            print(s)

        time.sleep(2)

    '''
    I'm able to generate ProMP -> dynamic, static, need to test Combines

    It returnes waypoints also, actions are generated

    '''
    CustomPlot.my_plot(prompg.X[prompg.Y==[0]], [])
    CustomPlot.my_plot(prompg.X[prompg.Y==[1]], [])
    CustomPlot.my_plot(prompg.X[prompg.Y==[2]], [])
    CustomPlot.my_plot(prompg.X[prompg.Y==[3]], [])
    CustomPlot.my_plot(prompg.X[prompg.Y==[4]], [])

    '''    vars = {'pinch': 0.0}
    promp_paths_waypoints_tuple = prompg.generate_path('pinch', vars=vars)
    execute(promp_paths_waypoints_tuple)

    vars = {'pinch': 1.0}
    promp_paths_waypoints_tuple = prompg.generate_path('pinch', vars=vars)
    execute(promp_paths_waypoints_tuple)

    promp_paths_waypoints_tuple = prompg.generate_path('victory', vars=vars)
    execute(promp_paths_waypoints_tuple)

    print("DDD")
    exit()'''

    promp_paths_waypoints_tuple = []
    vars = {'direction': (0.3,0.0,0.0)}
    ### TODO: PLOT UNCERTAINTY !
    promp_paths_waypoints_tuple.append(prompg.generate_path('nothing', vars))
    CustomPlot.my_plot(Xnothing, promp_paths_waypoints_tuple)

    for promp_path_waypoints_tuple in promp_paths_waypoints_tuple:
        execute(promp_path_waypoints_tuple)

    promp_paths_waypoints_tuple = []
    vars = {'direction': (0.3,0.0,0.0)}
    promp_paths_waypoints_tuple.append(prompg.generate_path('grab', vars))
    CustomPlot.my_plot(Xgrab, promp_paths_waypoints_tuple)

    for promp_path_waypoints_tuple in promp_paths_waypoints_tuple:
        execute(promp_path_waypoints_tuple)

    promp_path_waypoints_tuple = []
    vars = {'direction': (0.3,0.0,0.0)}
    promp_path_waypoints_tuple.append(prompg.generate_path('kick', vars))
    vars = {'direction': (0.0,0.3,0.0)}
    promp_path_waypoints_tuple.append(prompg.generate_path('kick', vars))
    vars = {'direction': (0.0,0.0,0.3)}
    promp_path_waypoints_tuple.append(prompg.generate_path('kick', vars))

    CustomPlot.my_plot(Xkick, promp_path_waypoints_tuple)

    for promp_path_waypoints_tuple in promp_paths_waypoints_tuple:
        execute(promp_path_waypoints_tuple)

    #promp_paths = approach.construct_promp_trajectories(X, Y)
    #print(f"X {X.shape}, promp_paths {promp_paths.shape}")

    #promp_path2 = ProbabilisticMotionPrimitiveGenerator().prompg.generate_path('kick')
    #CustomPlot.my_plot(X[Y==1], [promp_path2])

    #promp_path3 = ProbabilisticMotionPrimitiveGenerator().prompg.generate_path('nothing')
    #CustomPlot.my_plot(X[Y==2], [promp_path3])




    print("DONE")










#
