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
from os_and_utils.path_def import Waypoint

import gestures_lib as gl
if __name__ == '__main__':
    from os_and_utils.nnwrapper import NNWrapper
    gl.init()

# TEMP:
from geometry_msgs.msg import Pose, Point, Quaternion

if __name__ == '__main__':
    ## init Coppelia
    from geometry_msgs.msg import Pose, Point, Quaternion

    # Import utils.py from src folder

    sys.path.append(settings.paths.mirracle_sim_path)
    from utils import *

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
from os_and_utils.utils import get_object_of_closest_distance
from os_and_utils.utils_ros import samePoses, extv

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
        print("MP gestures", self.Gs)

        self.X, self.Y, self.robot_promps = DatasetLoader(['interpolate', 'discards']).load_mp(settings.paths.learn_path, self.Gs, approach)
        print("Xlen", self.X.shape, self.robot_promps)


    def handle_action_queue(self, action):
        action_stamp, action_name, action_hand = action
        vars = gl.gd.var_generate(action_hand, action_stamp)

        path = self.generate_path(action_name, vars=vars, tmp_action_stamp=action_stamp)

        '''
        def NormalizeData(data):
            return (data - np.min(data)) / (np.max(data) - np.min(data))

        ## Visualize and analyze Variables attached
        from os_and_utils.visualizer_lib import VisualizerLib
        viz = VisualizerLib()
        viz.visualize_new_fig("Gesture attached variables", dim=2)
        for key in ['velocities', 'grab', 'pinch', 'holding_sphere_radius']:
            var = vars[key]
            viz.visualize_2d(list(zip(np.linspace(0, 1, len(var)), NormalizeData(var))),label=f"{key}", xlabel='0-1', ylabel='var values', start_stop_mark=False)
        viz.savefig('1Dvarvalues')
        viz.visualize_new_fig("Gesture attached variables", dim=3)
        for key in ['point_direction', 'palm_euler']:
            var = vars[key]
            viz.visualize_3d(var,label=f"{key}", xlabel='X', ylabel='Y', zlabel='Z')
        viz.savefig('3Dvarvalues')
        '''
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

        try:
            robot_promps = self.robot_promps[self.Gs.index(id_primitive)]
        except ValueError:
            robot_promps = None # It is static gesture

        path = mp_type().by_id(robot_promps, id_primitive, self.approach, vars)
        path_ = deepcopy(path)
        for key in path[1].keys():
            path[1][key] = path[1][key].export()
        gl.gd.action_saves.append((id, id_primitive, tmp_action_stamp, vars, path[0], path[1]))
        return path_

def combine_promp_paths(promp_paths):
    ''' TODO
    '''
    return np.hstack(promp_paths)

def map_to_primitive_gesture(id_gesture):
    ''' Mapping hand gesture ID to robot primitive gesture ID
    '''
    mapping = settings.get_gesture_mapping()
    return mapping[id_gesture]

def check_waypoints_accuracy(promp_path, waypoints):
    achieved_all = True
    for wp_t in list(waypoints.keys()):
        if not waypoints[wp_t].p: continue
        achieved = False
        for point in promp_path:
            if samePoses(point, waypoints[wp_t].p):
                achieved = True

        if achieved == False:
            print(f"[ProMP lib] Waypoint at time {wp_t} and coordinates {waypoints[wp_t]} not achieved!")
            achieved_all = False
    return achieved_all

def choose_the_object():
    ''' Probably will move
    '''
    #pose_in_scene = ml.md.mouse3d_position #tfm.transformLeapToScene([0.,0.,100.])#[leap_3d_mouse])
    #objects_in_scenes = sl.scene.object_poses[ml.md.object_focus_id]
    #object_id = get_object_of_closest_distance(objects_in_scenes, pose_in_scene)
    object_id = ml.md.object_focus_id
    position = extv(sl.scene.object_poses[object_id].position)
    return object_id, position

def handle_build_structure():
    '''
    # movement touch is applied
    Returns:
        position
    '''
    def get_structure_id(touch_id):
        for n,structure in enumerate(ml.md.structures):
            if structure.base_id == touch_id:
                return n
        return None

    gripper = None
    position = [0.,0.,0.]
    if ml.md.attached: # Item must be attached to build
        if ml.md.object_focus_id != ml.md.object_touch_id: # Focused object cannot be the object currently visiting
            id = get_structure_id(ml.md.object_focus_id)
            if id is not None:
                ''' Attached object, target is struct -> add block to structure '''
                position = ml.md.structures[id].add(ml.md.object_touch_id)
                print(f"------------------\nAdding block to a structure\nItem is attached;; focused and touched DIFFERENT;; touched ID is member of struct;; structure id {id}, touched id {ml.md.object_touch_id}, focused id {ml.md.object_focus_id}, position {position}, structures[id].n {ml.md.structures[id].n}")
                gripper = 1.0
            else:
                ''' Attached object and target object will create new structure '''
                ml.md.structures.append(ml.Structure(type=ml.md.build_mode, id=ml.md.object_focus_id, base_position=extv(sl.scene.object_poses[ml.md.object_focus_id].position)))
                position = ml.md.structures[0].add(ml.md.object_touch_id)
                print(f"------------------\nCreating new structure\n Item is attached;; focused and touched DIFFERENT;; touched ID is NOT member of struct;; structure id {id}, touched id {ml.md.object_touch_id}, focused id {ml.md.object_focus_id}, position {position}, structures {ml.md.structures}")
                gripper = 1.0
        else:
            ''' Pointing target is same as focused -> do nothing '''
            print(f"------------------\nDo nothing\n Item is attached;; focused and touched are SAME;; touched ID is member of struct;; touched id {ml.md.object_touch_id}, focused id {ml.md.object_focus_id}, structures {ml.md.structures}")
            return None, None
    else:
        id = get_structure_id(ml.md.object_focus_id)
        if id is not None: # member of structure
            ''' No item is attached, touch the latest object of the structure '''
            position = ml.md.structures[id].get_position(ml.md.object_focus_id)
            print(f"------------------\nTouch the latest object of the structure\n Item is NOT attached;; touched ID is member of struct;; structure id {id}, touched id {ml.md.object_touch_id}, focused id {ml.md.object_focus_id}, position {position}, structures.n {ml.md.structures[id].n}")
            gripper = 0.0
        else:
            ''' No item is attached, touch simple object '''
            _, position = choose_the_object()
            print(f"------------------\nTouch simple object\n Item is NOT attached;; touched ID is NOT member of struct;; structure id {id}, touched id {ml.md.object_touch_id}, focused id {ml.md.object_focus_id}, position {position}, structures {ml.md.structures}")
            gripper = 0.0
    return position, gripper



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

class ProbabilisticMotionPrimitiveGenerator():
    def by_id(self, robot_promps, id_primitive, approach, vars):
        # get waypoints based on gesture, uses chosen class function
        construct_waypoints = getattr(self,id_primitive)
        waypoints = construct_waypoints(vars)
        print("waypoints ", [waypoints[wp].export() for wp in waypoints])
        promp_path = approach.sample_promp_trajectory_waypoints(robot_promps, waypoints)
        if not check_waypoints_accuracy(promp_path, waypoints):
            print("Waypoints not accurate, assigning default waypoints!")
            promp_path = []
            for wpkey in waypoints:
                wp = waypoints[wpkey]
                promp_path.append(wp.p)
        return promp_path, waypoints

    def update_by_id(self, robot_promps, id_primitive, approach, vars):
        # get waypoints based on gesture, uses chosen class function
        construct_waypoints = getattr(self,f"{id_primitive}_update")
        waypoints = construct_waypoints(vars)

        promp_path = approach.sample_promp_trajectory_waypoints(robot_promps, waypoints)
        check_waypoints_accuracy(promp_path, waypoints)
        return promp_path, waypoints

    def touch(self, vars = {}):
        waypoints = {}
        # Assign starting point to robot eef
        waypoints[0.0] = Waypoint(p=extv(ml.md.eef_pose.position))
        # Assign target point of the motion to obj. position
        object_id, object_position = choose_the_object()

        # # TODO: apply real object len
        #over_object_position = (object_position[0], object_position[1], object_position[2]+0.1)
        #waypoints[0.7] = Waypoint(p=over_object_position)

        object_position, gripper = handle_build_structure()

        # TODO: Assigning should not be here
        ml.md.object_touch_id = object_id
        waypoints[1.0] = Waypoint(p=object_position, gripper=gripper)

        return waypoints
    '''
    def bump(self, vars = {}):
        waypoints = {}
        # Assign starting point to robot eef
        waypoints[0.0] = Waypoint(p=extv(ml.md.eef_pose.position))
        # Assign bump point of the motion to obj. position
        _, object_position = choose_the_object()
        waypoints[1.0] = Waypoint(p=object_position)

        return waypoints
    '''
    def kick(self, vars = {}):
        waypoints = {}
        # Assign starting point to robot eef
        waypoints[0.0] = Waypoint(p=extv(ml.md.eef_pose.position))
        # Assign kick point of the motion to obj. position
        _, object_position = choose_the_object()
        waypoints[0.8] = Waypoint(p=object_position)

        return waypoints


class StaticMotionPrimitiveGenerator():
    def by_id(self, robot_promps, id_primitive, approach, vars):
        # get waypoints based on gesture, uses chosen class function
        construct_waypoints = getattr(self,id_primitive)
        waypoints = construct_waypoints(vars)

        path = construct_dummy_trajectory_waypoints(waypoints)
        return path, waypoints

    def update_by_id(self, robot_promps, id_primitive, approach, vars):
        # get waypoints based on gesture, uses chosen class function
        construct_waypoints = getattr(self,f"{id_primitive}_update")
        waypoints = construct_waypoints(vars)

        path = construct_dummy_trajectory_waypoints(waypoints)
        return path, waypoints

    def build_switch(self, vars):
        return {}

    def gripper(self, vars):
        waypoints = {}

        # choose the average from second half
        #pinch = vars['pinch']
        #gripper_value = 1-np.mean(pinch[-1])
        #waypoints[1.0] = Waypoint(gripper = gripper_value)
        return waypoints

    def gripper_update(self, vars):
        pass

    def focus(self, vars):
        waypoints = {}
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

    def nothing(self, vars = {}):
        waypoints = {}
        return waypoints

    def direction_move(self, vars = {}):
        waypoints = {}

        point_direction = vars['point_direction']
        last_point_direction = point_direction[-1]
        if last_point_direction[0] < 0: # observarion
            print("%%% GOING RIGHT")
        else:
            print("%%% GOING LEFT")

        # Assign starting point to robot eef
        px, py, pz = extv(ml.md.eef_pose.position)
        waypoints[0.0] = Waypoint(p=[px,py,pz])
        # Assign kick point of the motion to obj. position
        #_, object_position = choose_the_object()
        px += 0.1
        ml.md.eef_pose.position = Point(px, py, pz)
        waypoints[1.0] = Waypoint(p=[px,py,pz])

        return waypoints

class CombinedMotionPrimitiveGenerator():
    def by_id(self, robot_promps, id_primitive, approach, vars):
        # get waypoints based on gesture, uses chosen class function
        construct_waypoints = getattr(self,id_primitive)
        waypoint_lists = construct_waypoints(vars)

        promp_paths = []
        for waypoints in waypoint_lists:
            promp_path = approach.sample_promp_trajectory_waypoints(robot_promps, waypoints)
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
