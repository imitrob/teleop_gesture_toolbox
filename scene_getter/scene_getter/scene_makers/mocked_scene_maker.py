import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile
import scene_msgs.msg as scene_ros
from visualization_msgs.msg import Marker, MarkerArray
from scene_getter.scene_lib.scene import Scene
from scene_getter.scene_lib.scene_object import SceneObject
import yaml 
import scene_getter

SCENE_FILE = "scene_1"  # used only when no user_name is given
MESH_TOPIC = "/teleop_gesture_toolbox/scene_mesh"  # drawn by the Hand Scene viewer


def mesh_markers(data_dict) -> MarkerArray:
    """One marker per scene object with a `mesh`: {url: STL relative to the
    viewer page, scale, color: [r, g, b]}. Objects without one are not drawn."""
    markers = []
    for data in data_dict.values():
        if not isinstance(data, dict) or "mesh" not in data:
            continue
        mesh = data["mesh"]
        marker = Marker(ns="scene_mesh", id=len(markers), type=Marker.MESH_RESOURCE,
                        action=Marker.ADD, mesh_resource=mesh["url"])
        marker.header.frame_id = "base"
        # z stays 0: the STL origin is its bottom, so the mesh stands on the table.
        marker.pose.position.x, marker.pose.position.y = map(float, data["position"][:2])
        marker.pose.orientation.w = 1.0
        marker.scale.x = marker.scale.y = marker.scale.z = float(mesh["scale"])
        marker.color.r, marker.color.g, marker.color.b = map(float, mesh["color"])
        marker.color.a = 1.0
        markers.append(marker)
    return MarkerArray(markers=markers)


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
        # Latched: the viewer may connect after the scene was loaded.
        self.mesh_pub = self.create_publisher(
            MarkerArray, MESH_TOPIC, QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL))
        self.meshes_shown = False

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
        markers = mesh_markers(data_dict)
        # A scene without meshes publishes nothing (scene_mesh_pub.py may own the
        # topic), except an empty set that clears the previous scene's meshes.
        if markers.markers or self.meshes_shown:
            self.mesh_pub.publish(markers)
            self.meshes_shown = bool(markers.markers)
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