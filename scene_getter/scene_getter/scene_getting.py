
from scene_getter.scene_lib.scene import Scene
from skills_manager.ros_utils import SpinningRosNode
import scene_msgs.msg as scene_msgs
from std_srvs.srv import Trigger
from rclpy.qos import QoSProfile, QoSReliabilityPolicy
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

class SceneGetterViaObjectLocalizer(SceneGetter):
    def __init__(self):
        super(SceneGetterViaObjectLocalizer, self).__init__()

        self.start_publishing_scene_call = self.create_client(Trigger, 'start_publishing_scene', qos_profile=QoSProfile(depth=10, reliability=QoSReliabilityPolicy.BEST_EFFORT), callback_group=self.callback_group)
        self.stop_publishing_scene_call = self.create_client(Trigger, 'stop_publishing_scene', qos_profile=QoSProfile(depth=10, reliability=QoSReliabilityPolicy.BEST_EFFORT), callback_group=self.callback_group)


    def start_publishing_scene(self):
        self.start_publishing_scene_call.call(Trigger.Request())

    def stop_publishing_scene(self):
        self.stop_publishing_scene_call.call(Trigger.Request())


class SceneGetterNode(SceneGetterViaObjectLocalizer, SpinningRosNode):
    def __init__(self):
        super(SceneGetterNode, self).__init__()
