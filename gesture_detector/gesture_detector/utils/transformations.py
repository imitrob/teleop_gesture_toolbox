import numpy as np

class Transformations():
    @staticmethod
    def transformLeapToBase_3D(paths):
        ''' paths to scene, might edit later
        '''
        for n,path in enumerate(paths):
            for m,point in enumerate(path):
                 paths[n][m] = Transformations.transformLeapToBase(point)
        return paths

    @staticmethod
    def transformLeapToBase(pose, out=''):
        ''' # REVIEW:
        '''
        if isinstance(pose, (list,np.ndarray, tuple)):
            if len(pose) == 3:
                return [pose[0]/1000, -pose[2]/1000, pose[1]/1000]
            else:
                return [pose[0]/1000, -pose[2]/1000, pose[1]/1000] + pose[3:]

        def is_type_Pose(pose):
            try:
                pose.position
                pose.orientation
                pose.position.x
                pose.orientation.x
            except AttributeError:
                return False
            return True

        ''' Input is Pose '''
        if is_type_Pose(pose):
            x = pose.position.x/1000
            y = -pose.position.z/1000
            z = pose.position.y/1000
            pose.position.x = x
            pose.position.y = y
            pose.position.z = z
            return pose

        ''' Input is Point '''
        x = pose.x/1000
        y = -pose.z/1000
        z = pose.y/1000
        pose.x = x
        pose.y = y
        pose.z = z
        return pose
