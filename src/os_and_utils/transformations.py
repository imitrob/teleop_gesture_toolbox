import numpy as np

class Transformations():
    def transformLeapToScene(self, pose):
        ''' Leap -> rViz -> Scene
        '''
        assert isinstance(pose, Pose), "pose not right datatype"
        pose_ = Pose()
        pose_.orientation = deepcopy(pose.orientation)
        # Leap to rViz center point
        x = pose.position.x/1000
        y = -pose.position.z/1000
        z = pose.position.y/1000

        ## Camera rotation from CoppeliaSim
        ## TODO: Optimize for all environments
        if settings.POSITION_MODE == 'sim_camera':
            x = -pose.position.x/1000
            y = pose.position.y/1000
            z = -pose.position.z/1000
            camera = settings.md.camera_orientation
            camera_matrix = tf.transformations.euler_matrix(camera.x, camera.y, camera.z, 'rxyz')
            camera_matrix = np.array(camera_matrix)[0:3,0:3]
            x_cop = np.dot([x,y,z], camera_matrix[0])
            y_cop = np.dot([x,y,z], camera_matrix[1])
            z_cop = np.dot([x,y,z], camera_matrix[2])
            x,y,z = x_cop,y_cop,z_cop

        # Linear transformation to point with rotation
        # How the Leap position will affect system
        pose_.position.x = np.dot([x,y,z], settings.md.ENV['axes'][0])*settings.md.SCALE + settings.md.ENV['start'].x
        pose_.position.y = np.dot([x,y,z], settings.md.ENV['axes'][1])*settings.md.SCALE + settings.md.ENV['start'].y
        pose_.position.z = np.dot([x,y,z], settings.md.ENV['axes'][2])*settings.md.SCALE + settings.md.ENV['start'].z

        #hand.palm_normal.roll, hand.direction.pitch, hand.direction.yaw
        ## Orientation

        # apply rotation

        alpha, beta, gamma = tf.transformations.euler_from_quaternion([pose.orientation.x,pose.orientation.y,pose.orientation.z,pose.orientation.w])
        Rx = tf.transformations.rotation_matrix(alpha, settings.md.ENV['ori_live'][0])
        Ry = tf.transformations.rotation_matrix(beta,  settings.md.ENV['ori_live'][1])
        Rz = tf.transformations.rotation_matrix(gamma, settings.md.ENV['ori_live'][2])
        R = tf.transformations.concatenate_matrices(Rx, Ry, Rz)
        euler = tf.transformations.euler_from_matrix(R, 'rxyz')

        [alpha, beta, gamma] = euler
        Rx = tf.transformations.rotation_matrix(alpha, [1,0,0]) #settings.md.ENV['axes'][0])
        Ry = tf.transformations.rotation_matrix(beta,  [0,1,0]) #settings.md.ENV['axes'][1])
        Rz = tf.transformations.rotation_matrix(gamma, [0,0,1]) #settings.md.ENV['axes'][2])
        R = tf.transformations.concatenate_matrices(Rx, Ry, Rz)
        euler = tf.transformations.euler_from_matrix(R, 'rxyz')

        pose_.orientation = Quaternion(*tf.transformations.quaternion_multiply(tf.transformations.quaternion_from_euler(*euler), settings.extq(settings.md.ENV['ori'])))

        if settings.ORIENTATION_MODE == 'fixed':
            pose_.orientation = settings.md.ENV['ori']

        # only for this situtaiton
        return pose_

    def transformSceneToUI(self, pose, view='view'):
        ''' Scene -> rViz -> UI
        '''
        pose_ = Pose()
        pose_.orientation = pose.orientation
        p = Point(pose.position.x-settings.md.ENV['start'].x, pose.position.y-settings.md.ENV['start'].y, pose.position.z-settings.md.ENV['start'].z)
        # View transformation
        x = (np.dot([p.x,p.y,p.z], settings.md.ENV[view][0]) )*settings.ui_scale
        y = (np.dot([p.x,p.y,p.z], settings.md.ENV[view][1]) )*settings.ui_scale
        z = (np.dot([p.x,p.y,p.z], settings.md.ENV[view][2]) )*settings.ui_scale
        # Window to center, y is inverted
        pose_.position.x = x + settings.w/2
        pose_.position.y = -y + settings.h
        pose_.position.z = round(-(z-200)/10)
        return pose_

    def transformLeapToUIsimple(self, pose):
        x, y, z = pose.position.x, pose.position.y, pose.position.z
        x_ = 2*x + settings.w/2
        y_ = -2*y + settings.h
        z_ = round(-(z-200)/10)
        pose_ = Pose()
        pose_.orientation = pose.orientation
        pose_.position.x, pose_.position.y, pose_.position.z = x_, y_, z_
        return pose_

    def transformLeapToUI(self, pose):
        ''' Leap -> UI
        '''
        pose_ = self.transformLeapToScene(pose)
        pose__ = self.transformSceneToUI(pose_)
        return pose__
        x, y, z = pose.position.x, pose.position.y, pose.position.z
        x_ = 2*x + settings.w/2
        y_ = -2*y + settings.h
        z_ = round(-(z-200)/10)
        pose_ = Pose()
        pose_.orientation = pose.orientation
        pose_.position.x, pose_.position.y, pose_.position.z = x_, y_, z_
        return pose_

    def eulerToVector(self, euler):
        ''' Check if there are no exception
        '''
        roll, pitch, yaw = euler
        x = np.cos(yaw)*np.cos(pitch)
        y = np.sin(yaw)*np.cos(pitch)
        z = np.sin(pitch)
        return x,y,z
