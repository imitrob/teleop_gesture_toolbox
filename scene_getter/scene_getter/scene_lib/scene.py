
from typing import Iterable
import numpy as np

from scene_getter.scene_lib.scene_object import SceneObject
import scene_getter.scene_lib.scene_object as scene_object
import scene_msgs.msg as scene_msgs

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
    def n(self): # number of objects in the scene
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
            scene_state['objects'][o.name]['params'] = o.params

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
            ros_sceneobject.params = o.params

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
            o.objects[n].params = objects[name]['params']

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
            
            o.objects[n].params = objects[n].params

        return o

    def __eq__(self, other_scene):
        if len(self.object_positions) != len(other_scene.object_positions):
            return False
        for o1,o2 in zip(self.object_positions, other_scene.object_positions):
            if sum(abs(o1 - o2)) > 0.001:
                return False
        return True

    def get_scene_param_description(self): 
        s = ""
        for o in self.objects:
            s += o.params + " "
        return s