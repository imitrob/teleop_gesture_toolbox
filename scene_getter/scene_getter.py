

import numpy as np
from scene_getter.Objects import SceneObject
from scene_getter.Scene import Scene



class SceneGetter():
    def __init__(self):
        super().__init__(self)
        
        self.scene = Scene()

        

    def get_scene(self):
        return self.scene
    
 
