import time
import rclpy
from rclpy.node import Node
import scene_msgs.msg as scene_ros
from scene_getter.scene_lib.scene import Scene
from scene_getter.scene_lib.scene_object import SceneObject

class MockedScenePublisher(Node):
    def __init__(self):
        super().__init__("mocked_scene_publisher_node")

        self.scene_pub = self.create_publisher(scene_ros.Scene, "/scene", 5)
        
        self.scene = Scene(name="scene_1", objects=[
            SceneObject("open_drawer", [0.5,-0.1,0.06]),
            SceneObject("sponge_wipe", [0.5,0.2,0.06]),
            SceneObject("pick_lid", [0.4,0.2,0.06]),
            SceneObject("handle_screwdriver", [0.4,0.0,0.06]),
            SceneObject("pour_potatoes", [0.4,-0.2,0.06]),
        ])

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