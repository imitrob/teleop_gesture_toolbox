# Pose = np.array([x,y,z,roll,pitch,yaw])
# Point = np.array([x,y,z]) or Vector3 np.array([xlen, ylen, zlen])

from typing import Iterable
import numpy as np

class SceneObject():
    ''' Static attributes
    '''
    def __init__(self, name: str, # String
                       position: Iterable[float], # Point
                       orientation: Iterable[float] = np.array([0., 0., 0., 1.]), # x y z w
                       ):

        self.position = np.array(position)
        self.orientation = np.array(orientation)

        self.name = name

        assert isinstance(position, Iterable) and len(position) == 3
        assert isinstance(orientation, Iterable) and len(orientation) == 4

    @property
    def type(self):
        return self.__class__.__name__

    @property
    def quaternion(self):
        return self.orientation

    @property
    def info(self):
        print(self.__str__())

    def __str__(self):
        return f'{self.name},\t{self.type},\t{np.array(self.position).round(2)}'
