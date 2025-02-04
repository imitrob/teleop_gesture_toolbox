import time
import rclpy
from rclpy.node import Node
import scene_msgs.msg as scene_ros
from scene_getter.scene_lib.scene import Scene
from scene_getter.scene_lib.scene_object import SceneObject
import scene_getter
import yaml

class MockedScenePublisher(Node):
    def __init__(self):
        super().__init__("mocked_scene_publisher_node")

        scene = self.declare_parameter("scene_name", "scene_1")
        self.scene_pub = self.create_publisher(scene_ros.Scene, "/scene", 5)
        
        objs_dict = yaml.safe_load(open(f"{scene_getter.path}/scene_makers/scenes/{scene}.yaml", mode="r"))
        objects = []
        for name,value in objs_dict.items():
            objects.append(SceneObject(name, list(value)))

        self.scene = Scene(name=scene, objects=objects)

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