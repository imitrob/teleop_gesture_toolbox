# Pose = np.array([x,y,z,roll,pitch,yaw])
# Point = np.array([x,y,z]) or Vector3 np.array([xlen, ylen, zlen])

import numpy as np
from numpy import array as a

class SceneObject():
    ''' Static attributes
    '''
    def __init__(self, name, # String
                       position = None, # Point
                       random = True,
                       orientation = np.array([0., 0., 0., 1.]), # x y z w
                       size = 0.05,
                       # Additional
                       ycb = False,
                       color = 'b',
                       scale = 1.,
                       shape_type = None, # Optional
                       mass = None, # Optional
                       friction = None, # Optional
                       inertia = None, # Optional
                       inertia_transformation = None, # Optional
                       position_real = None,
                       ):

        self.name = name
        self.size = size
        if isinstance(self.size, (tuple,np.ndarray,list)):
            self.size = size[0]
        assert ((position is not None) or (position_real is not None)), "Position is required"

        if position is not None:
            self.position_real = self.pos_grid_to_real(position)
        else:
            self.position_real = np.array(position_real)

        self.type = 'object'
        self.inside_drawer = False
        self.under = None # object
        self.above = None # object

        self.max_allowed_size = 0.0
        self.stackable = True
        self.graspable = True
        self.pushable = True
        self.pourable = 0.1
        self.full = False
        self.color = color
        if random: self.color = np.random.choice(['r','g','b'])

        self.quaternion = orientation

        self.ycb = ycb

        ''' Additional '''
        self.scale = scale
        self.shape_type = shape_type
        self.mass = mass
        self.friction = friction
        self.inertia = inertia
        self.inertia_transformation = inertia_transformation

    @property
    def absolute_location(self):
        return self.position_real

    @property
    def direction(self):
        raise Exception("TODO")

    @property
    def position(self):
        ''' Default option is '''
        return self.position_grid

    @position.setter
    def position(self, position_grid):
        self.position_real = self.pos_grid_to_real(position_grid)

    @property
    def position_grid(self):
        return self.pos_real_to_grid(self.position_real)

    @position_grid.setter
    def position_grid(self, position_grid):
        self.position_real = self.pos_grid_to_real(position_grid)

    @property
    def orientation(self):
        return self.quaternion

    @property
    def type_id(self):
        return self.all_types.index(self.type)

    @property
    def on_top(self):
        if (not self.above and not self.inside_drawer): return True
        return False

    @property
    def free(self):
        return self.on_top

    @property
    def info(self):
        print(self.__str__())

    def __str__(self):
        return f'{self.name},\t{self.type},\t{np.array(self.position_real).round(2)},\t{self.print_structure(out_oneline_str=True)}'
