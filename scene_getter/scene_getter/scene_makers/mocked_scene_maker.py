import time
import rclpy
from rclpy.node import Node
import scene_msgs.msg as scene_ros
from scene_getter.scene_lib.scene import Scene
from scene_getter.scene_lib.scene_object import SceneObject
import yaml 
import scene_getter

SCENE_FILE = "scene_1"

class MockedScenePublisher(Node):
    def __init__(self):
        super().__init__("mocked_scene_publisher_node")

        self.scene_pub = self.create_publisher(scene_ros.Scene, "/scene", 5)
        
        data_dict = yaml.safe_load(open(f"{scene_getter.path}/scene_makers/scenes/{SCENE_FILE}.yaml", mode="r"))
        scene_objects = []
        for name,objectdata in data_dict.items():
            scene_objects.append(SceneObject.from_dict(name, objectdata))

        self.scene = Scene(name=SCENE_FILE, objects=scene_objects)

    def __call__(self):
        self.scene_pub.publish(self.scene.to_ros())

def main():
    rclpy.init()
    sp = MockedScenePublisher()
    
    while rclpy.ok():
        sp()
        time.sleep(1.0)

if __name__ == '__main__':
    main()