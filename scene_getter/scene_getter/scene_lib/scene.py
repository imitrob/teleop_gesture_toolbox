
from typing import Iterable
import numpy as np

from scene_getter.scene_lib.scene_object import SceneObject
from pydantic import BaseModel, Field, ConfigDict, field_validator, model_validator
from typing import Iterable, List, Sequence

OBJECT__eq__THRESHOLD = 0.02 # 2cm

class Scene(BaseModel):
    name: str = Field(..., description="Scene name")
    objects: List[SceneObject] = Field(
        default_factory=list,
        description="Objects present in the scene",
    )
    model_config = ConfigDict(arbitrary_types_allowed=True, validate_assignment=True)

    @field_validator("objects", mode="before")
    @classmethod
    def _coerce_objects(cls, value: object) -> List[SceneObject]:  # type: ignore[override]
        # Accept iterables of dicts / SceneObjects
        if isinstance(value, Iterable):
            return [SceneObject.model_validate(v) for v in value]
        raise TypeError("`objects` must be an iterable of SceneObject‑like data")

    @model_validator(mode="after")
    def _check_duplicates(self):  # noqa: D401
        names = [obj.name for obj in self.objects]
        if len(names) != len(set(names)):
            raise ValueError("Duplicate object names are not allowed in a Scene")
        return self

    @property
    def n(self) -> int:
        """Number of objects in the scene."""
        return len(self.objects)

    @property
    def empty_scene(self) -> bool:
        return self.n == 0

    @property
    def O(self) -> List[str]:
        """List of object names."""
        return [obj.name for obj in self.objects]

    @property
    def object_positions(self) -> List[np.ndarray]:
        return [obj.position for obj in self.objects]

    @property
    def object_positions_xy(self) -> List[np.ndarray]:
        return [obj.position[0:2] for obj in self.objects]

    @property
    def object_types(self) -> List[str]:
        return [obj.type for obj in self.objects]

    @property
    def object_poses(self) -> List[Sequence[float]]:
        return [[*obj.position, *obj.quaternion] for obj in self.objects]

    @property
    def object_names(self) -> List[str]:
        return self.O


    def get_object_id(self, name: str) -> int:
        return self.O.index(name)

    def get_object_by_type(self, typ: str) -> SceneObject | None:
        return next((obj for obj in self.objects if obj.type == typ), None)

    def get_object_by_name(self, name: str) -> SceneObject | None:
        return next((obj for obj in self.objects if obj.name == name), None)

    def get_scene_param_description(self) -> str:
        return " ".join(obj.params for obj in self.objects)


    @property
    def info(self):
        print(self.__str__())

    def __repr__(self):
        return self.__str__()

    def __str__(self) -> str:  # noqa: D401
        header = "Scene info:" if self.objects else "Scene (empty):"
        lines = [f"{idx}. {obj}" for idx, obj in enumerate(self.objects)]
        return f"{header}\n" + "\n".join(lines)

    def copy(self):
        return self.from_dict(self.to_dict())

    def get_object_types(self, names):
        ret = []
        for object_name in names:
            o_ = self.get_object_by_name(object_name)
            if o_ is not None:
                ret.append(o_.type)
            else:
                ret.append('object')
        return ret

    def in_scene(self, position):
        if (np.array([0,0,0]) <= position).all() and (position < self.grid_lens).all():
            return True
        return False

    def __getattr__(self, attr):  # dot‑access to objects
        if attr in self.O:
            return self.objects[self.get_object_id(attr)]
        raise AttributeError(attr)


    def to_dict(self) -> dict:
        scene_state: dict = {
            "name": self.name,
            "objects": {},
        }
        for obj in self.objects:
            scene_state["objects"][obj.name] = {
                "position": obj.position.tolist(),
                "orientation": obj.orientation.tolist(),
                "type": obj.type,
                "params": obj.params,
            }
        return scene_state

    @classmethod
    def from_dict(cls, data: dict):
        objects_payload = data.get("objects", {})
        objects: list[SceneObject] = []
        for name, spec in objects_payload.items():
            obj_data = {
                "name": name,
                "position": spec["position"],
                "orientation": spec.get("orientation"),
                "params": spec.get("params", ""),
            }
            objects.append(SceneObject(**obj_data))

        return cls(name=data["name"], objects=objects)

    def to_ros(self):  # noqa: D401
        try:
            import scene_msgs.msg as scene_msgs
            from geometry_msgs.msg import Point, Quaternion, Pose
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("ROS dependencies are missing") from exc

        sceneros = scene_msgs.Scene()
        
        ros_sceneobjects = []
        for obj in self.objects:
            ros_obj = scene_msgs.SceneObject()
            ros_obj.name = obj.name
            ros_obj.pose = Pose(
                position=Point(x=float(obj.position[0]), y=float(obj.position[1]), z=float(obj.position[2])),
                orientation=Quaternion(
                    x=float(obj.orientation[0]),
                    y=float(obj.orientation[1]),
                    z=float(obj.orientation[2]),
                    w=float(obj.orientation[3]),
                ),
            )
            ros_obj.type = obj.type
            ros_obj.params = obj.params
            ros_sceneobjects.append(ros_obj)
        sceneros.objects = ros_sceneobjects
        sceneros.name = self.name
        return sceneros
    
    @classmethod
    def from_ros(cls, sceneros):  # noqa: D401
        objs = []
        for ros_obj in sceneros.objects:
            position = [ros_obj.pose.position.x, ros_obj.pose.position.y, ros_obj.pose.position.z]
            orientation = [
                ros_obj.pose.orientation.x,
                ros_obj.pose.orientation.y,
                ros_obj.pose.orientation.z,
                ros_obj.pose.orientation.w,
            ]
            objs.append(
                SceneObject(
                    name=ros_obj.name,
                    position=position,
                    orientation=orientation,
                    params=ros_obj.params,
                )
            )
        return cls(name=sceneros.name, objects=objs)

    def __eq__(self, other):  # noqa: D401
        if not isinstance(other, Scene):
            return NotImplemented
        if len(self.objects) != len(other.objects):
            return False

        for p1, p2 in zip(self.object_positions, other.object_positions):
            if np.linalg.norm(p1 - p2) > OBJECT__eq__THRESHOLD:
                return False
        return True




