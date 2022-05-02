import numpy as np
import tf
from os_and_utils.utils_ros import extq
## ROS dependent lib
from geometry_msgs.msg import Quaternion, Point, Pose
import settings
from copy import deepcopy

class Transformations():
    @staticmethod
    def transformLeapToScene(data, env, scale, camera_orientation):
        '''
        '''
        if isinstance (data, Pose):
            return Transformations.transformLeapToScenePose(data, env, scale, camera_orientation)
        elif isinstance(data, (list,tuple, np.ndarray)):
            return Transformations.transformLeapToSceneList(data, env, scale)
        else: raise Exception("Wrong datatype to function transformLeapToScene(), need Pose, List, Tuple or np.ndarray")


    @staticmethod
    def transformPrompToSceneList_3D(paths):
        ''' paths to scene, might edit later
        '''
        for n,path in enumerate(paths):
            for m,point in enumerate(path):
                 paths[n][m] = Transformations.transformPrompToSceneList(point)
        return paths

    @staticmethod
    def transformPrompToSceneList(xyz):
        ''' TODO: Orientaiton
        '''
        x = xyz[0]; y = xyz[1]; z = xyz[2]

        # ProMP to rViz center point
        x_ = x/1000
        y_ = -z/1000
        z_ = y/1000

        x__ = y_ + 0.5
        y__ = x_
        z__ = z_
        return [x__, y__, z__]

    @staticmethod
    def transformLeapToSceneList(list, env, scale):
        ''' TODO: Orientaiton
        Parameters:
            env (): ml.md.ENV
            scale (Float): ml.md.scale
        '''
        x = list[0]; y = list[1]; z = list[2]
        if len(list) == 7:
            rx = list[3]; ry = list[4]; rz = list[5]; rw = list[6]
            raise Exception("TODO:")

        # Leap to rViz center point
        x = x/1000
        y = -z/1000
        z = y/1000
        x_ = np.dot([x,y,z], env['axes'][0])*scale + env['start'].x
        y_ = np.dot([x,y,z], env['axes'][1])*scale + env['start'].y
        z_ = np.dot([x,y,z], env['axes'][2])*scale + env['start'].z
        return [x_, y_, z_]

    @staticmethod
    def transformLeapToBase_3D(paths):
        ''' paths to scene, might edit later
        '''
        for n,path in enumerate(paths):
            for m,point in enumerate(path):
                 paths[n][m] = Transformations.transformLeapToBase(point)
        return paths

    @staticmethod
    def transformLeapToBase_2D(path):
        ''' paths to scene, might edit later
        '''
        for m,point in enumerate(path):
             path[m] = Transformations.transformLeapToBase(point)
        return path

    @staticmethod
    def transformLeapToBase(pose, out=''):
        ''' # REVIEW:
        '''
        if isinstance(pose, (list,np.ndarray, tuple)):
            if len(pose) == 3:
                return [pose[0]/1000, -pose[2]/1000, pose[1]/1000]
            else:
                return [pose[0]/1000, -pose[2]/1000, pose[1]/1000] + pose[3:]

        if isinstance(pose, Point):
            x = pose.x/1000
            y = -pose.z/1000
            z = pose.y/1000
            pose.x = x
            pose.y = y
            pose.z = z
            return pose

        x = pose.position.x/1000
        y = -pose.position.z/1000
        z = pose.position.y/1000
        pose.position.x = x
        pose.position.y = y
        pose.position.z = z
        return pose

    @staticmethod
    def transformLeapToScenePose(pose, env, scale, camera_orientation, out='pose'):
        ''' Leap -> rViz -> Scene
        '''
        if isinstance(pose, list):
            lenposelist = len(pose)
            x, y, z = pose[0], pose[1], pose[2]
            pose = Pose()
            pose.position.x = x
            pose.position.y = y
            pose.position.z = z
            if lenposelist == 7:
                pose.orientation.x = pose[3]
                pose.orientation.y = pose[4]
                pose.orientation.z = pose[5]
                pose.orientation.w = pose[6]

        pose_ = deepcopy(pose)
        pose_.position.x = 0.0
        pose_.position.y = 0.0
        pose_.position.z = 0.0
        # Leap to rViz center point
        x = pose.position.x/1000
        y = -pose.position.z/1000
        z = pose.position.y/1000

        ## Camera rotation from CoppeliaSim
        ## TODO: Optimize for all environments
        if settings.position_mode == 'sim_camera':
            x = -pose.position.x/1000
            y = pose.position.y/1000
            z = -pose.position.z/1000
            camera = camera_orientation
            camera_matrix = tf.transformations.euler_matrix(camera.x, camera.y, camera.z, 'rxyz')
            camera_matrix = np.array(camera_matrix)[0:3,0:3]
            x_cop = np.dot([x,y,z], camera_matrix[0])
            y_cop = np.dot([x,y,z], camera_matrix[1])
            z_cop = np.dot([x,y,z], camera_matrix[2])
            x,y,z = x_cop,y_cop,z_cop

        # Linear transformation to point with rotation
        # How the Leap position will affect system
        pose_.position.x = np.dot([x,y,z], env['axes'][0])*scale + env['start'].x
        pose_.position.y = np.dot([x,y,z], env['axes'][1])*scale + env['start'].y
        pose_.position.z = np.dot([x,y,z], env['axes'][2])*scale + env['start'].z

        # apply rotation
        alpha, beta, gamma = tf.transformations.euler_from_quaternion([pose.orientation.x,pose.orientation.y,pose.orientation.z,pose.orientation.w])
        Rx = tf.transformations.rotation_matrix(alpha, env['ori_live'][0])
        Ry = tf.transformations.rotation_matrix(beta,  env['ori_live'][1])
        Rz = tf.transformations.rotation_matrix(gamma, env['ori_live'][2])
        R = tf.transformations.concatenate_matrices(Rx, Ry, Rz)
        euler = tf.transformations.euler_from_matrix(R, 'rxyz')

        [alpha, beta, gamma] = euler
        Rx = tf.transformations.rotation_matrix(alpha, [1,0,0]) #env['axes'][0])
        Ry = tf.transformations.rotation_matrix(beta,  [0,1,0]) #env['axes'][1])
        Rz = tf.transformations.rotation_matrix(gamma, [0,0,1]) #env['axes'][2])
        R = tf.transformations.concatenate_matrices(Rx, Ry, Rz)
        euler = tf.transformations.euler_from_matrix(R, 'rxyz')

        pose_.orientation = Quaternion(*tf.transformations.quaternion_multiply(tf.transformations.quaternion_from_euler(*euler), extq(env['ori'])))

        if settings.orientation_mode == 'fixed':
            pose_.orientation = env['ori']

        if out == 'position':
            return [pose_.position.x, pose_.position.y, pose_.position.z]
        if out == 'list':
            return [pose_.position.x, pose_.position.y, pose_.position.z, pose_.orientation.x, pose_.orientation.y, pose_.orientation.z, pose_.orientation.w]
        # only for this situtaiton
        return pose_

    @staticmethod
    def transformSceneToUI(pose, env, view='view'):
        ''' Scene -> rViz -> UI
        '''
        pose_ = Pose()
        pose_.orientation = pose.orientation
        p = Point(pose.position.x-env['start'].x, pose.position.y-env['start'].y, pose.position.z-env['start'].z)
        # View transformation
        x = (np.dot([p.x,p.y,p.z], env[view][0]) )*settings.ui_scale
        y = (np.dot([p.x,p.y,p.z], env[view][1]) )*settings.ui_scale
        z = (np.dot([p.x,p.y,p.z], env[view][2]) )*settings.ui_scale
        # Window to center, y is inverted
        pose_.position.x = x + settings.w/2
        pose_.position.y = -y + settings.h
        pose_.position.z = round(-(z-200)/10)
        return pose_

    @staticmethod
    def transformLeapToUIsimple(pose, out='pose'):
        ''' Leap -> UI
        Parameters:
            pose (Pose() or list)
            out (Str): 'pose' -> output as Pose, 'list' -> output as list
        Returns:
            xyz (Pose or list - based on out Param)
        '''
        if isinstance(pose, list):
            x, y, z = pose[0], pose[1], pose[2]
        else:
            x, y, z = pose.position.x, pose.position.y, pose.position.z
        x_ = 2*x + settings.w/2
        y_ = 2*z + settings.h/2
        z_ = round(-(y-300)/10)
        if out == 'pose':
            pose_ = Pose()
            pose_.orientation = pose.orientation
            pose_.position.x, pose_.position.y, pose_.position.z = x_, y_, z_
        elif out == 'list':
            pose_ = x_, y_, z_
        else: raise Exception(f"transformLeapToUIsimple wrong argumeter: {out}")

        return pose_

    @staticmethod
    def transformLeapToUI(pose, env, scale):
        ''' Leap -> UI
        '''
        pose_ = Transformations.transformLeapToScene(pose, env, scale)
        pose__ = Transformations.transformSceneToUI(pose_, env)
        return pose__

    @staticmethod
    def eulerToVector(euler):
        ''' Check if there are no exception
        '''
        roll, pitch, yaw = euler
        x = np.cos(yaw)*np.cos(pitch)
        y = np.sin(yaw)*np.cos(pitch)
        z = np.sin(pitch)
        return x,y,z

    @staticmethod
    def pointToScene(point, env, scale):
        x,y,z = point.x, point.y, point.z
        point_ = Point()
        point_.x = np.dot([x,y,z], env['axes'][0])*scale + env['start'].x
        point_.y = np.dot([x,y,z], env['axes'][1])*scale + env['start'].y
        point_.z = np.dot([x,y,z], env['axes'][2])*scale + env['start'].z
        return point_
