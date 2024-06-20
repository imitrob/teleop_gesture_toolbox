

import numpy as np
from teleop_gesture_toolbox.scene_getter.scene_getter.Objects import SceneObject
from teleop_gesture_toolbox.scene_getter.scene_getter.Scene import Scene


class SceneGetter():
    def __init__(self):
        super().__init__(self)
        
        self.scene = Scene()
        self.scene_subscriber = self.create_subscription(SceneRos, "/scene", self.save_scene_callback, 5)

    def save_scene_callback(self, msg):
        self.scene = msg

    def get_scene(self):
        return self.scene
    
 
