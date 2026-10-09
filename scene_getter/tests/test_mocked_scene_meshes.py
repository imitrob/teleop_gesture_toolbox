from scene_getter.scene_lib.scene_object import SceneObject
from scene_getter.scene_makers.mocked_scene_maker import mesh_markers


def test_only_objects_with_a_mesh_are_drawn_on_the_table():
    scene = {
        "cup1": {"position": [0.5, 0.1, 0.04], "params": "cup1 is a red cup.",
                 "mesh": {"url": "models/tabletop/cup.stl", "scale": 0.0014, "color": [0.8, 0.1, 0.1]}},
        "box1": {"position": [0.3, 0.0, 0.04]},
    }
    (marker,) = mesh_markers(scene).markers
    assert marker.mesh_resource == "models/tabletop/cup.stl"
    assert (marker.pose.position.x, marker.pose.position.y, marker.pose.position.z) == (0.5, 0.1, 0.0)
    assert marker.scale.z == 0.0014 and marker.color.a == 1.0
    # The scene itself ignores the `mesh` key.
    assert SceneObject.from_dict("cup1", scene["cup1"]).params == "cup1 is a red cup."
