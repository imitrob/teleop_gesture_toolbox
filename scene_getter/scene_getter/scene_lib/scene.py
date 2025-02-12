
from typing import Iterable
import numpy as np

from scene_getter.scene_lib.scene_object import SceneObject
import scene_getter.scene_lib.scene_object as scene_object
import scene_msgs.msg as scene_msgs

from copy import deepcopy
from geometry_msgs.msg import Point, Quaternion, Pose

class Scene():
    def __init__(self, 
                 name: str,
                 objects: Iterable[SceneObject] = [],
                 ):
        self.objects = []
        self.objects = objects
        self.name = name

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
        s = f"Scene info:\n"
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

    def get_object_types(self, names):
        ret = []
        for object_name in names:
            o_ = self.get_object_by_name(object_name)
            if o_ is not None:
                ret.append(o_.type)
            else:
                ret.append('object')
        return ret

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
        scene_state['name'] = self.name
        for o in self.objects:
            scene_state['objects'][o.name] = {}
            scene_state['objects'][o.name]['position'] = o.position
            scene_state['objects'][o.name]['orientation'] = o.orientation
            scene_state['objects'][o.name]['type'] = o.type
            # scene_state['objects'][o.name]['graspable'] = o.graspable
            # scene_state['objects'][o.name]['pushable'] = o.pushable
            # scene_state['objects'][o.name]['free'] = o.free
            # scene_state['objects'][o.name]['size'] = o.size
            # scene_state['objects'][o.name]['above_str'] = o.above_str
            # scene_state['objects'][o.name]['under_str'] = o.under_str

            if o.type == 'drawer':
                scene_state['objects'][o.name]['opened'] = o.opened
                scene_state['objects'][o.name]['contains_list'] = o.contains_list

            if o.type == 'cup':
                scene_state['objects'][o.name]['full'] = o.full

        return scene_state

    def to_ros(self):
        sceneros = scene_msgs.Scene()
        
        ros_sceneobjects = []
        for n in range(len(self.objects)):
            o = self.objects[n]
            
            ros_sceneobject = scene_msgs.SceneObject()
            ros_sceneobject.name = o.name
            ros_sceneobject.pose.position = Point(x=o.position[0], y=o.position[1], z=o.position[2])
            ros_sceneobject.pose.orientation = Quaternion(
                x=o.orientation[0],
                y=o.orientation[1],
                z=o.orientation[2],
                w=o.orientation[3],
            )
            ros_sceneobject.type = o.type

            if o.type == 'drawer':
                sceneros.objects[n].opened = float(o.opened_)
                if o.contains_list != []:
                    sceneros.objects[n].contains_list = o.contains_list[0]
                else:
                    sceneros.objects[n].contains_list = ''
            if o.type == 'cup':
                sceneros.objects[n].full = float(o.full)

            ros_sceneobjects.append(ros_sceneobject)

        sceneros.objects = ros_sceneobjects
        sceneros.name = self.name
        return sceneros

    def copy(self):
        return Scene(init='from_dict', import_data=self.to_dict())


    @classmethod
    def from_dict(cls, dict_data):
        o = cls(dict_data['name'])
    
        objects = dict_data['objects']
        o.objects = []
        for n,name in enumerate(objects.keys()):
            o.objects.append(getattr(scene_object, objects[name]['type'])(name=name, position=objects[name]['position']))
            # o.objects[n].graspable = objects[name]['graspable']
            # o.objects[n].pushable = objects[name]['pushable']
            # o.objects[n].size = objects[name]['size']

            if objects[name]['type'] == 'drawer':
                o.objects[n].opened = objects[name]['opened']
            if objects[name]['type'] == 'cup':
                if 'full' in objects[name].keys():
                    o.objects[n].full = objects[name]['full']

            if objects[name]['type'] == 'drawer':
                if 'contains_list' in objects[name]:
                    contains_list = objects[name]['contains_list']
                    for contain_item in contains_list:
                        for o in o.objects:
                            if o.name == contain_item:
                                o.objects[n].contains.append(o)
                                break
        return o

    @classmethod
    def from_ros(cls, scene_state):
        o = cls(scene_state.name)

        objects = scene_state.objects

        o.objects = []
        for n in range(len(objects)):
            name = objects[n].name

            position = [objects[n].pose.position.x, objects[n].pose.position.y, objects[n].pose.position.z]
            orientation = [objects[n].pose.orientation.x, objects[n].pose.orientation.y, objects[n].pose.orientation.z, objects[n].pose.orientation.w]

            o.objects.append(getattr(scene_object, objects[n].type)(name=name, position=position, orientation=orientation))
            # o.objects[n].graspable = objects[n].graspable
            # o.objects[n].pushable = objects[n].pushable
            # o.objects[n].size = objects[n].size

            if objects[n].type == 'drawer':
                o.objects[n].opened = objects[n].opened
            if objects[n].type == 'cup':
                o.objects[n].full = objects[n].full

        return o

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
