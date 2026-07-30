import time
import rclpy
from rclpy.node import Node
import scene_msgs.msg as scene_ros
from scene_getter.scene_lib.scene import Scene
from scene_getter.scene_lib.scene_object import SceneObject
import yaml 
import scene_getter

SCENE_FILE = "scene_1"  # used when the user's links file names no scene


def _user_scene(name_user: str) -> str:
    """The scene named in the user's links file, or "" when no user is given or
    hri_manager (an optional dependency) is not on the path."""
    if not name_user:
        return ""
    try:
        from hri_manager.user_links import load_user_links
        return load_user_links(name_user).get("scene", "")
    except Exception as e:  # noqa: BLE001 -- missing package or missing file
        print(f"[Mocked Scene] User settings for {name_user!r} not loaded ({e})", flush=True)
        return ""

class MockedScenePublisher(Node):
    def __init__(self):
        super().__init__("mocked_scene_publisher_node")

        self.scene_pub = self.create_publisher(scene_ros.Scene, "/scene", 5)

        # Which scene to publish comes from links/<user>_links.yaml (key
        # `scene`), so a new cell is configured in yaml rather than here.
        user = self.declare_parameter("user_name", "").get_parameter_value().string_value
        scene_file = _user_scene(user) or SCENE_FILE
        print(f"[Mocked Scene] Publishing scene {scene_file!r} from {scene_getter.scenes_path}",
              flush=True)

        data_dict = yaml.safe_load(open(f"{scene_getter.scenes_path}/{scene_file}.yaml", mode="r"))
        scene_objects = []
        for name,objectdata in data_dict.items():
            scene_objects.append(SceneObject.from_dict(name, objectdata))

        self.scene = Scene(name=scene_file, objects=scene_objects)

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