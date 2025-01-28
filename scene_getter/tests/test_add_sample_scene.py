

import time
from scene_getter.scene_getting import SceneGetterNode
from scene_getter.scene_lib.scene import Scene
from scene_getter.scene_lib.scene_object import SceneObject

import scene_msgs.msg as scene_msgs 
import rclpy


def test_scene_generation():
    s = Scene("scene_1")
    print(s)

    s = Scene("scene_1", objects=[
        SceneObject("plastic_cube_1", [0.5,0.0,0.04]),
        SceneObject("robothon_box", [0.5,0.0,0.04]),
        SceneObject("robothon_peg", [0.5,0.0,0.04]),
    ])
    print(s)

    d = s.to_dict()
    print(d)
    
    s2 = Scene.from_dict(d)
    print(s2)

    rosmsg = s.to_ros()

    s3 = Scene.from_ros(rosmsg)
    print(s3)

    rosscene = scene_msgs.Scene(name="asd", objects=[])

def test_scene_getter():
    rclpy.init()
    scene_getter = SceneGetterNode()
    
    print(scene_getter.scene)
    rclpy.spin_once(scene_getter)
    time.sleep(1.0)
    print(scene_getter.scene)
    rclpy.spin_once(scene_getter)
    time.sleep(1.0)
    print(scene_getter.scene)
    rclpy.spin_once(scene_getter)
    time.sleep(1.0)
    print(scene_getter.scene)
    rclpy.spin_once(scene_getter)
    time.sleep(1.0)

if __name__ == "__main__":
    test_scene_generation()
    test_scene_getter()