
import numpy as np
from random import choice

import teleop_gesture_toolbox.scene_getter.scene_getter.Objects as Objects

from copy import deepcopy
from geometry_msgs.msg import Point, Quaternion, Pose

class Scene():
    def __init__(self, objects=[], import_data=None, name="scene0"):
        self.objects = []
        self.objects = objects
        self.name = name

    @classmethod
    def init_from_dict(cls, dict):
        o = cls()
        o.from_dict(dict)
        return o

    @classmethod
    def init_from_ros(cls, rosdata):
        o = cls()
        o.from_ros(rosdata)
        return o

    @property
    def n(self):
        return len(self.O)
    
    @property
    def empty_scene(self):
        if len(self.O) > 0:
            return False
        else: 
            return True
    
    @property
    def info(self):
        print(self.__str__())

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        s = f"Scene info. shape: {self.grid_lens}\n"
        for n, o in enumerate(self.objects):
            s += f'{n}. '
            s += str(o)
            s += '\n'
        return s

    def get_object_id(self, name):
        return self.O.index(name)

    def get_object(self, id):
        return self.O[id]

    @property
    def O(self):
        return [object.name for object in self.objects]

    @property
    def object_positions(self):
        return [obj.position for obj in self.objects]

    @property
    def object_positions_real(self):
        return [obj.position_real for obj in self.objects]

    @property
    def object_sizes(self):
        return [obj.size for obj in self.objects]

    @property
    def object_types(self):
        return [obj.type for obj in self.objects]

    @property
    def object_poses(self):
        return [[*obj.position, *obj.quaternion] for obj in self.objects]

    @property
    def object_poses_ros(self):
        return [Pose(position=Point(x=obj.position[0], y=obj.position[1], z=obj.position[2]), orientation=Quaternion(x=obj.quaternion[0],y=obj.quaternion[1],z=obj.quaternion[2],w=obj.quaternion[3])) for obj in self.objects]

    @property
    def object_names(self):
        return [obj.name for obj in self.objects]

    def get_object_by_type(self, type):
        for obj in self.objects:
            if obj.type == type:
                return obj
        return None

    def get_object_by_name(self, name):
        for obj in self.objects:
            if obj.name == name:
                return obj
        return None

    def has_duplicate_objects(self):
        O = []
        for object in self.objects:
            if object.name in O:
                return True
            O.append(object.name)
        return False

    def in_scene(self, position):
        if (np.array([0,0,0]) <= position).all() and (position < self.grid_lens).all():
            return True
        return False

    def __getattr__(self, attr):
        return self.objects[self.O.index(attr)]

    def to_dict(self):
        scene_state = {}
        scene_state['objects'] = {}
        for o in self.objects:
            scene_state['objects'][o.name] = {}
            scene_state['objects'][o.name]['position'] = o.position
            scene_state['objects'][o.name]['type'] = o.type
            scene_state['objects'][o.name]['graspable'] = o.graspable
            scene_state['objects'][o.name]['pushable'] = o.pushable
            scene_state['objects'][o.name]['free'] = o.free
            scene_state['objects'][o.name]['size'] = o.size
            scene_state['objects'][o.name]['above_str'] = o.above_str
            scene_state['objects'][o.name]['under_str'] = o.under_str

            if o.type == 'drawer':
                scene_state['objects'][o.name]['opened'] = o.opened
                scene_state['objects'][o.name]['contains_list'] = o.contains_list

            if o.type == 'cup':
                scene_state['objects'][o.name]['full'] = o.full

        return scene_state

    def to_ros(self, rosobj=None):
        if rosobj is None: raise Exception("to_ros() function needs Scene ROS object to be filled!")
        for n in range(7):
            rosobj.objects[n].position_real = np.array([0,0,0], dtype=float)
        
        
        for n,o in enumerate(deepcopy(self.objects)):
            if n >= 7:
                print(f"objects in scene: {len(self.objects)} > 7, breaking")
                break
            rosobj.objects[n].name = o.name

            rosobj.objects[n].position_real = np.array(o.position_real, dtype=float)
            rosobj.objects[n].type = o.type
            rosobj.objects[n].graspable = o.graspable
            rosobj.objects[n].pushable = o.pushable
            rosobj.objects[n].free = o.free
            rosobj.objects[n].size = o.size
            rosobj.objects[n].above_str = o.above_str
            rosobj.objects[n].under_str = o.under_str

            if o.type == 'drawer':
                rosobj.objects[n].opened = float(o.opened_)
                if o.contains_list != []:
                    rosobj.objects[n].contains_list = o.contains_list[0]
                else:
                    rosobj.objects[n].contains_list = ''
            if o.type == 'cup':
                rosobj.objects[n].full = float(o.full)

        return rosobj

    def copy(self):
        return Scene(init='from_dict', import_data=self.to_dict())


    def from_dict(self, scene_state):
        objects = scene_state['objects']
        self.objects = []
        for n,name in enumerate(objects.keys()):
            self.objects.append(getattr(Objects, objects[name]['type'].capitalize())(name=name, position=objects[name]['position']))
            self.objects[n].type = objects[name]['type']
            self.objects[n].graspable = objects[name]['graspable']
            self.objects[n].pushable = objects[name]['pushable']
            self.objects[n].size = objects[name]['size']

            if objects[name]['type'] == 'drawer':
                self.objects[n].opened = objects[name]['opened']
            if objects[name]['type'] == 'cup':
                if 'full' in objects[name].keys():
                    self.objects[n].full = objects[name]['full']

        for n,name in enumerate(objects.keys()):
            if 'under_str' in objects[name].keys():
                under_str = objects[name]['under_str']
                for o in self.objects:
                    if o.name == under_str:
                        self.objects[n].under = o
                        break
            if 'above_str' in objects[name].keys():
                above_str = objects[name]['above_str']
                for o in self.objects:
                    if o.name == above_str:
                        self.objects[n].above = o
                        break
            if objects[name]['type'] == 'drawer':
                if 'contains_list' in objects[name]:
                    contains_list = objects[name]['contains_list']
                    for contain_item in contains_list:
                        for o in self.objects:
                            if o.name == contain_item:
                                self.objects[n].contains.append(o)
                                break

    def from_ros(self, scene_state):
        objects = scene_state.objects
        nobj=0
        while True:
            try:
                if scene_state.objects[nobj].name == '':
                    break
            except IndexError:
                break
            nobj+=1

        self.objects = []
        object_names = []
        for n in range(nobj):
            name = objects[n].name
            object_names.append(name)

            self.objects.append(getattr(Objects, objects[n].type.capitalize())(name=name, position_real=objects[n].position_real))
            self.objects[n].type = objects[n].type
            self.objects[n].graspable = objects[n].graspable
            self.objects[n].pushable = objects[n].pushable
            self.objects[n].size = objects[n].size

            if objects[n].type == 'drawer':
                self.objects[n].opened = objects[n].opened
            if objects[n].type == 'cup':
                self.objects[n].full = objects[n].full

        for n in range(nobj):

            under_str = objects[n].under_str
            for o in self.objects:
                if o.name == under_str:
                    self.objects[n].under = o
                    break
            above_str = objects[n].above_str
            for o in self.objects:
                if o.name == above_str:
                    self.objects[n].above = o
                    break
            if objects[n].type == 'drawer':
                contains_list = objects[n].contains_list
                for contain_item in contains_list:
                    for o in self.objects:
                        if o.name == contain_item:
                            self.objects[n].contains.append(o)
                            break

    def __eq__(self, obj2):
        ''' Reward function
        '''

        if len(self.object_positions) != len(obj2.object_positions):
            raise Exception("scenes havent got same objects")
        reward = 0.
        max_reward = 0.
        for n,(o1,o2) in enumerate(zip(self.object_positions, obj2.object_positions)):
            reward -= sum(abs(o1 - o2))

        for n,(o1,o2) in enumerate(zip(self.objects, obj2.objects)):
            max_reward += 2
            if o1.under == o2.under:
                reward += 1
            if o1.above == o2.above:
                reward += 1
            if o1.type == 'drawer':
                if o2.type != 'drawer': raise Exception("scenes havent got same objects")
                reward += len(list(set(o2.contains_list).intersection(o1.contains_list)))
                max_reward += len(o2.contains_list)
                max_reward += 1
                if o1.opened == o2.opened:
                    reward += 1

        if reward == max_reward: return True
        return reward
