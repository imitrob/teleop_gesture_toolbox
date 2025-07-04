

import time
from scene_getter.scene_getting import SceneGetterNode
from scene_getter.scene_lib.scene import Scene
from scene_getter.scene_lib.scene_object import SceneObject

import scene_msgs.msg as scene_msgs 
import rclpy
import numpy as np

def test_scene_generation():
    s = Scene(name="scene_1")
    assert s.name == "scene_1"

    s = Scene(name="scene_1", objects=[
        SceneObject(name="plastic_cube_1", position = [0.4,0.0,1.0]),
        SceneObject(name="robothon_box", position = [0.5,0.0,0.04]),
        SceneObject(name="robothon_peg", position = [1.123,0.0,0.04]),
    ])
    assert np.allclose(s.plastic_cube_1.position, [0.4, 0.0, 1.0])
    assert np.allclose(s.robothon_box.position, [0.5, 0.0, 0.04])
    assert np.allclose(s.robothon_peg.position, [1.123, 0.0, 0.04])
    assert np.allclose(s.robothon_peg.orientation, [0.0, 0.0, 0.0, 1.0])

    d = s.to_dict()
    s2 = Scene.from_dict(d)
    assert s == s2

    rosmsg = s.to_ros()
    s3 = Scene.from_ros(rosmsg)
    assert s == s3

    s4 = s.copy()
    assert s == s4

    s = Scene(name="scene_2", objects=[
        SceneObject(name="plastic_cube_1",position = [0.5,0.0,0.04], orientation=[1.0,0.0,0.0,0.0]),
        SceneObject(name="robothon_box",  position = [0.5,0.0,0.04], orientation=[1.0,0.0,0.0,0.0]),
        SceneObject(name="robothon_peg",  position = [0.5,0.0,0.04], orientation=[1.0,0.0,0.0,0.0]),
    ])
    assert np.allclose(s.get_object_by_name("plastic_cube_1").orientation,[1.0, 0.0, 0.0, 0.0])
    assert np.allclose(s.get_object_by_name("robothon_box").orientation,  [1.0, 0.0, 0.0, 0.0])
    assert np.allclose(s.get_object_by_name("robothon_peg").orientation,  [1.0, 0.0, 0.0, 0.0])
    
    s = Scene(name="scene_3", objects=[
        SceneObject(name="plastic_cube_1",position = [0.5,0.0,0.04], orientation=[1.0,0.0,0.0,0.0], params="small green plastic cube")
    ])
    assert s.get_object_by_name("plastic_cube_1").params == "small green plastic cube"

def test_scene_getter():
    rclpy.init()
    scene_getter = SceneGetterNode()
    
    time.sleep(1.0)
    print(scene_getter.scene)

if __name__ == "__main__":
    test_scene_generation()
    test_scene_getter()