
from scene_getter.scene_lib.scene import Scene
from skills_manager.ros_utils import SpinningRosNode
import scene_msgs.msg as scene_msgs
import time

SCENE_CHECK_RATE = 0.1

class SceneGetter():
    def __init__(self):
        super(SceneGetter, self).__init__()

        self.scene = Scene(name = "scene")
        self.scene_subscriber = self.create_subscription(scene_msgs.Scene, "/scene", self.save_scene_callback, 5)
        self.last_scene_arrived = 0.0

    def save_scene_callback(self, msg):
        self.scene = Scene.from_ros(msg)
        self.last_scene_arrived = time.time()

    def get_scene(self):
        return self.scene
    
    def wait_for_scene(self):
        last_scene_arrived = self.last_scene_arrived
        while self.last_scene_arrived == last_scene_arrived:
            time.sleep(SCENE_CHECK_RATE)
 
class SceneGetterNode(SceneGetter, SpinningRosNode):
    def __init__(self):
        super(SceneGetterNode, self).__init__()
