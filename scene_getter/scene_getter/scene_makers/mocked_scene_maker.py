import rclpy
from rclpy.node import Node
import scene_msgs.msg as scene_ros
from scene_getter.scene_lib.scene import Scene
from scene_getter.scene_lib.scene_object import SceneObject
import yaml 
import scene_getter

SCENE_FILE = "scene_1"  # used only when no user_name is given


def _user_scene(name_user: str) -> str:
    """The scene named in the user's links file, or "" when no user is given.

    A named user that resolves to no scene raises: falling back to SCENE_FILE
    would publish the wrong scene for a whole recorded session, and a user study
    cannot be re-run once the participants have left.
    """
    if not name_user:
        return ""
    from hri_manager.user_links import load_user_links
    scene = load_user_links(name_user).get("scene", "")
    if not scene:
        raise SystemExit(f"[Mocked Scene] links file for user {name_user!r} names no `scene`")
    return scene

class MockedScenePublisher(Node):
    def __init__(self):
        super().__init__("mocked_scene_publisher_node")

        self.scene_pub = self.create_publisher(scene_ros.Scene, "/scene", 5)

        # Which scene to publish comes from links/<user>_links.yaml (key
        # `scene`), so a new cell is configured in yaml rather than here.
        user = self.declare_parameter("user_name", "").get_parameter_value().string_value
        # Switchable at run time (`ros2 param set <node> scene <name>`): user
        # study simple changes the scene between cases.
        self.declare_parameter("scene", _user_scene(user) or SCENE_FILE)
        self.scene = None

    def load(self, scene_file):
        print(f"[Mocked Scene] Publishing scene {scene_file!r} from {scene_getter.scenes_path}",
              flush=True)
        data_dict = yaml.safe_load(open(f"{scene_getter.scenes_path}/{scene_file}.yaml", mode="r"))
        scene_objects = []
        for name,objectdata in data_dict.items():
            scene_objects.append(SceneObject.from_dict(name, objectdata))
        return Scene(name=scene_file, objects=scene_objects)

    def __call__(self):
        scene_file = self.get_parameter("scene").get_parameter_value().string_value
        if self.scene is None or self.scene.name != scene_file:
            self.scene = self.load(scene_file)
        self.scene_pub.publish(self.scene.to_ros())

def main():
    rclpy.init()
    sp = MockedScenePublisher()
    
    while rclpy.ok():
        sp()
        rclpy.spin_once(sp, timeout_sec=1.0)  # serves `ros2 param set`

if __name__ == '__main__':
    main()