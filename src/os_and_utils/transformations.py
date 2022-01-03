import numpy as np
import tf
from os_and_utils.utils_ros import extq
## ROS dependent lib
from geometry_msgs.msg import Quaternion, Point, Pose
import os_and_utils.move_lib as ml
import settings
from copy import deepcopy

class Transformations():
    @staticmethod
    def transformLeapToScene(pose):
        ''' Leap -> rViz -> Scene
        '''
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
            camera = ml.md.camera_orientation
            camera_matrix = tf.transformations.euler_matrix(camera.x, camera.y, camera.z, 'rxyz')
            camera_matrix = np.array(camera_matrix)[0:3,0:3]
            x_cop = np.dot([x,y,z], camera_matrix[0])
            y_cop = np.dot([x,y,z], camera_matrix[1])
            z_cop = np.dot([x,y,z], camera_matrix[2])
            x,y,z = x_cop,y_cop,z_cop

        # Linear transformation to point with rotation
        # How the Leap position will affect system
        pose_.position.x = np.dot([x,y,z], ml.md.ENV['axes'][0])*ml.md.scale + ml.md.ENV['start'].x
        pose_.position.y = np.dot([x,y,z], ml.md.ENV['axes'][1])*ml.md.scale + ml.md.ENV['start'].y
        pose_.position.z = np.dot([x,y,z], ml.md.ENV['axes'][2])*ml.md.scale + ml.md.ENV['start'].z

        # apply rotation
        alpha, beta, gamma = tf.transformations.euler_from_quaternion([pose.orientation.x,pose.orientation.y,pose.orientation.z,pose.orientation.w])
        Rx = tf.transformations.rotation_matrix(alpha, ml.md.ENV['ori_live'][0])
        Ry = tf.transformations.rotation_matrix(beta,  ml.md.ENV['ori_live'][1])
        Rz = tf.transformations.rotation_matrix(gamma, ml.md.ENV['ori_live'][2])
        R = tf.transformations.concatenate_matrices(Rx, Ry, Rz)
        euler = tf.transformations.euler_from_matrix(R, 'rxyz')

        [alpha, beta, gamma] = euler
        Rx = tf.transformations.rotation_matrix(alpha, [1,0,0]) #ml.md.ENV['axes'][0])
        Ry = tf.transformations.rotation_matrix(beta,  [0,1,0]) #ml.md.ENV['axes'][1])
        Rz = tf.transformations.rotation_matrix(gamma, [0,0,1]) #ml.md.ENV['axes'][2])
        R = tf.transformations.concatenate_matrices(Rx, Ry, Rz)
        euler = tf.transformations.euler_from_matrix(R, 'rxyz')

        pose_.orientation = Quaternion(*tf.transformations.quaternion_multiply(tf.transformations.quaternion_from_euler(*euler), extq(ml.md.ENV['ori'])))

        if settings.orientation_mode == 'fixed':
            pose_.orientation = ml.md.ENV['ori']

        # only for this situtaiton
        return pose_

    @staticmethod
    def transformSceneToUI(pose, view='view'):
        ''' Scene -> rViz -> UI
        '''
        pose_ = Pose()
        pose_.orientation = pose.orientation
        p = Point(pose.position.x-ml.md.ENV['start'].x, pose.position.y-ml.md.ENV['start'].y, pose.position.z-ml.md.ENV['start'].z)
        # View transformation
        x = (np.dot([p.x,p.y,p.z], ml.md.ENV[view][0]) )*settings.ui_scale
        y = (np.dot([p.x,p.y,p.z], ml.md.ENV[view][1]) )*settings.ui_scale
        z = (np.dot([p.x,p.y,p.z], ml.md.ENV[view][2]) )*settings.ui_scale
        # Window to center, y is inverted
        pose_.position.x = x + settings.w/2
        pose_.position.y = -y + settings.h
        pose_.position.z = round(-(z-200)/10)
        return pose_

    @staticmethod
    def transformLeapToUIsimple(pose):
        x, y, z = pose.position.x, pose.position.y, pose.position.z
        x_ = 2*x + settings.w/2
        y_ = -2*y + settings.h
        z_ = round(-(z-200)/10)
        pose_ = Pose()
        pose_.orientation = pose.orientation
        pose_.position.x, pose_.position.y, pose_.position.z = x_, y_, z_
        return pose_

    @staticmethod
    def transformLeapToUI(self, pose):
        ''' Leap -> UI
        '''
        pose_ = Transformations.transformLeapToScene(pose)
        pose__ = Transformations.transformSceneToUI(pose_)
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
    def pointToScene(point):
        x,y,z = point.x, point.y, point.z
        point_ = Point()
        point_.x = np.dot([x,y,z], ml.md.ENV['axes'][0])*ml.md.SCALE + ml.md.ENV['start'].x
        point_.y = np.dot([x,y,z], ml.md.ENV['axes'][1])*ml.md.SCALE + ml.md.ENV['start'].y
        point_.z = np.dot([x,y,z], ml.md.ENV['axes'][2])*ml.md.SCALE + ml.md.ENV['start'].z
        return point_
