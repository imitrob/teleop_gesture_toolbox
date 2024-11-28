

import numpy as np
from scene_getter.scene_lib.scene_object import SceneObject
from scene_getter.scene_lib.scene import Scene

import scene_msgs.msg as scene_msgs
from rclpy.node import Node

class SceneGetter():
    def __init__(self):
        super(SceneGetter, self).__init__()

        self.scene = Scene(name = "scene")
        self.scene_subscriber = self.create_subscription(scene_msgs.Scene, "/scene", self.save_scene_callback, 5)

    def save_scene_callback(self, msg):
        self.scene = Scene.from_ros(msg)

    def get_scene(self):
        return self.scene
 
class SceneGetterNode(SceneGetter, Node):
    def __init__(self):
        Node.__init__(self, 'scene_getter_node')
        SceneGetter.__init__(self)
