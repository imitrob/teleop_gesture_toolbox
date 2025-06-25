from __future__ import annotations
from copy import deepcopy
from typing import List
from pydantic import BaseModel, Field, conlist

import time
import rclpy
import scene_msgs.msg as scene_ros
from skills_manager.ros_utils import SpinningRosNode
from scene_getter.scene_lib.scene import Scene
from scene_getter.scene_lib.scene_object import SceneObject
import numpy as np
from natural_language_processing.text_to_speech.kokoro_model import Chatterbox
from gesture_msgs.msg import DeicticSolution
from typing import Sequence
from std_msgs.msg import Float32

# ====================
# pydantic definitions
# ====================
class Point3D(BaseModel):
    """A single 3‑D point in metres, expressed in the robot/world frame."""
    x: float = Field(..., description="X‑coordinate (m)")
    y: float = Field(..., description="Y‑coordinate (m)")
    z: float = Field(..., description="Z‑coordinate (m)")

class Line3D(BaseModel):
    """A directed line segment between *start* and *end* for one time sample."""
    start: Point3D
    end: Point3D

class PointingSequence(BaseModel):
    """Time‑ordered list of 3‑D line segments that make up the pointing gesture."""
    lines: List[Line3D] = Field(
        ...,
        min_length=1,
        description="Sequence of line segments, one per time sample, oldest → newest.",
    )

class InputSample(BaseModel):
    """All data needed to decide which object the user is referring to."""
    description: str = Field(
        ...,
        description=(
            "Natural‑language description of the target object (can be partial/incomplete)."
        ),
    )
    pointing: PointingSequence

class SelectionResult(BaseModel):
    """Just the ID of the object that the system thinks the user meant."""

    object_id: str = Field(..., description="Unique identifier of the selected object")

# ============================================
# PointingExperiment
# ============================================
class PointingExperimentBase(SpinningRosNode):
    def __init__(self):
        super(PointingExperimentBase, self).__init__()
        self.scene_pub = self.create_publisher(scene_ros.Scene, "/scene", 5)
        self.angle_error_pub = self.create_publisher(Float32, "/angle_error", 5)
        self.deictic_sub = self.create_subscription(DeicticSolution, "/teleop_gesture_toolbox/deictic_solution", self.receive_deictic, 5)

        self.tts = Chatterbox(device="cuda")

    def send_scene(self):
        scene = self.get_random_scene()
        self.scene_pub.publish(scene.to_ros())

    def receive_deictic(self, msg):
        # msg.header # = Header(stamp=self.get_clock().now().to_msg(), frame_id="leapworld"),
        # msg.object_id # = deictic_solution['object_id'],
        # msg.object_name # = deictic_solution['object_name'],
        # msg.object_names # = deictic_solution['object_names'],
        # msg.distances_from_line # = deictic_solution['distances_from_line'],
        # msg.line_point_1 # = Point(x=line_point_1[0], y=line_point_1[1], z=line_point_1[2]),
        # msg.line_point_2 # = Point(x=line_point_2[0], y=line_point_2[1], z=line_point_2[2]),
        # msg.target_object_position # = Point(x=to_position[0],y=to_position[1],z=to_position[2]),
        # msg.hand_velocity # = deictic_solution['hand_velocity'],
        line_point_1 = [msg.line_point_1.x, msg.line_point_1.y, msg.line_point_1.z]
        line_point_2 = [msg.line_point_2.x, msg.line_point_2.y, msg.line_point_2.z]
        target_object_position = [msg.target_object_position.x, msg.target_object_position.y, msg.target_object_position.z]
        
        ang = self.compute_error(line_point_1, line_point_2, target_object_position)
        self.angle_error_pub.publish(Float32(data=ang))

    def compute_error(self,
            hand_cartesian_position: Sequence[float],      # [x, y, z]
            pointing_cartesian_position: Sequence[float],  # [x, y, z]
            object_cartesian_position: Sequence[float],    # [x, y, z]
        ) -> float:
        """
        Angular error (radians) between:
        the ray the user is actually pointing along
        the ray from the hand to the real object

        Returns values in [0, pi]. 0 = perfect aim, pi = 180° in the opposite direction.
        """

        # Build direction vectors that both *originate at the hand*
        v_pointing = np.subtract(pointing_cartesian_position, hand_cartesian_position)
        v_object   = np.subtract(object_cartesian_position,   hand_cartesian_position)

        # Safeguard against zero-length vectors
        if np.allclose(v_pointing, 0) or np.allclose(v_object, 0):
            raise ValueError("Pointing and object vectors must have non-zero length")

        # Compute angle with robust atan2 formulation
        cross_norm = np.linalg.norm(np.cross(v_pointing, v_object))
        dot        = np.dot(v_pointing, v_object)
        angle_rad  = np.arctan2(cross_norm, dot)

        return angle_rad

    def select_target(self, sample: InputSample) -> SelectionResult:
        """Return a dummy object ID.
        selection logic (e.g. language grounding + spatial reasoning).
        """
        return SelectionResult(object_id="none")




class PointingExperiment(PointingExperimentBase):
    def __init__(self):
        super(PointingExperiment, self).__init__()


# =====================================================
# SCENE
# =====================================================
class Loc:
    def __init__(self, xyz):
        self.x, self.y, self.z = xyz[0], xyz[1], xyz[2]
    @property
    def left(self):
        if self.x < 0: return True
        else: return False
    @property
    def right(self):
        if self.x > 0: return True
        else: return False
    @property
    def center(self):
        if self.x == 0 or self.y == 0: return True
        else: return False
    @property
    def front(self):
        if self.y < 0: return True
        else: return False
    @property
    def back(self):
        if self.y > 0: return True
        else: return False
    def __eq__(self, other):
        if (self.left and other.left and self.front and other.front) or \
            (self.right and other.right and self.front and other.front) or \
            (self.right and other.right and self.back and other.back) or \
            (self.left and other.left and self.back and other.back) or \
            (self.left and other.left and self.center and other.center) or \
            (self.right and other.right and self.center and other.center) or \
            (self.front and other.front and self.center and other.center) or \
            (self.back and other.back and self.center and other.center):
            return True
        else:
            return False

def find_dissimilar_loc(loc, locs, offset):
    for i,l in enumerate(locs):
        if Loc(l - offset) != Loc(loc - offset):
            return i
    raise Exception("Not found dissimilar loc")
    
def find_similar_loc(loc, locs, offset):
    for i,l in enumerate(locs):
        if Loc(l - offset) == Loc(loc - offset):
            return i
    raise Exception("Not Found similar loc", loc, "\n\n", locs)

def generate_random_scene(
    num_scene_objects: int, # number of objects
    scene_complexity: int = 1, # has specific definition
    object_complexity_1: list = [
        "small blue plastic cup", 
        "medium red plastic cup", 
        "medium red metal bowl", 
        "medium yellow plastic banana", 
        "medium red plastic apple", 
        "small brown plastic capsule", 
        "small paper box", 
        "small green plastic cube", 
        "medium red plastic cube", 
    ],
    object_complexity_2: list = [
        ["small blue plastic cup", "medium red plastic cup"], 
        ["small green plastic cube", "medium red plastic cube"], 
    ],
    object_complexity_3: list = [
        "medium yellow plastic banana", 
        "medium red plastic apple", 
        "small paper box", 
    ],
    ngon: int = 6, # e.g., 6 is hexagon
    rounds: int = 2, # number of layers from the center
    dist_base: float = 0.15, # 
    offset: np.ndarray = np.array([0.5, 0.0, 0.04])
    ):
    
    # 1. this generates list of cartesian locations
    locs = [offset]
    for round in range(1, rounds + 1):
        dist = round * dist_base
        for deg in range(0, 360, int(10 * ngon / round)):
            locs.append(offset + np.array([np.sin(np.deg2rad(deg))*dist, np.cos(np.deg2rad(deg))*dist, 0.0 ]))
    locs = np.array(locs)

    remaining_num_scene_objects = num_scene_objects
    object_names = []
    object_locs = []
    # 2. assign objects to generates locations
    if scene_complexity == 1: # complexity 1 has all object classes different
        objects_pullup_list = np.array(deepcopy(object_complexity_1))
        # choose n objects from the list
        idx = np.random.choice(len(objects_pullup_list), size=num_scene_objects, replace=False)
        object_locs = locs[idx,:]                  # selected locations
        object_names = objects_pullup_list[idx]  # selected object names

        target_object = np.random.choice(object_names) # choose target form selected

    elif scene_complexity == 2:
        objects_grouped_names = deepcopy(object_complexity_2)
        objects_flatten = [o 
                           for o_ in objects_grouped_names
                           for o in o_]
        # pick target object-class first
        target_object_class = objects_grouped_names[np.random.choice(len(objects_grouped_names))]
        objects_grouped_names.remove(target_object_class)
        # pick target object
        target_object = target_object_class[np.random.choice(len(target_object_class))]
        target_object_class.remove(target_object)
        # assign the locs
        to_locs_id = np.random.choice(len(locs))
        to_loc = locs[to_locs_id,:]
        locs = np.delete(locs, to_locs_id, axis=0)
        
        object_names.append(target_object)
        object_locs.append(to_loc)
        remaining_num_scene_objects -= 1
        # add the grouped-objects and place them so they are not in the similar location
        for o_name in target_object_class:
            loc_idx = find_dissimilar_loc(to_loc, locs, offset)

            object_names.append(o_name)
            object_locs.append(locs[loc_idx])

            locs = np.delete(locs, loc_idx, axis=0)
            remaining_num_scene_objects -= 1

        # remaining object-groups and add remaining number of objects
        objects_flatten = [o 
                           for o_ in objects_grouped_names
                           for o in o_]
        
        objects_idx = np.random.choice(len(objects_flatten), size=remaining_num_scene_objects, replace=False)
        remaining_names = np.array(objects_flatten)[objects_idx]
        rem_loc_idx = np.random.choice(len(remaining_names), size=remaining_num_scene_objects, replace=False)
        rem_loc = locs[rem_loc_idx,:]

        object_names.extend(remaining_names)
        object_locs.extend(rem_loc)
        
    elif scene_complexity == 3:
        objects_pullup_list = deepcopy(object_complexity_3)
        locs = np.delete(locs, 0, axis=0) # delete center location here
        # select target object
        target_object = objects_pullup_list[np.random.choice(len(objects_pullup_list))]
        objects_pullup_list.remove(target_object)
        # make more copies of the same object
        object_names.append(str(target_object))
        object_names.append(str(target_object))
        # assign them location
        loc_idx = int(np.random.randint(len(locs), size=1))
        loc = locs[loc_idx,:]
        locs = np.delete(locs, loc_idx, axis=0)
        object_locs.append(loc)
        loc_id = find_similar_loc(loc, locs, offset)
        object_locs.append(locs[loc_id])
        remaining_num_scene_objects -= 2
        
        # add remaining objects
        objects_idx = np.random.choice(len(objects_pullup_list), size=remaining_num_scene_objects, replace=False)
        objects = np.array(objects_pullup_list)[objects_idx]
        idx = np.random.choice(len(objects), size=remaining_num_scene_objects, replace=False)
        locs = locs[idx,:]

        object_names.extend(list(objects))
        object_locs.extend(locs)
        
    object_names = list(object_names)
    
    # 3. add positional description
    for i, (name, loc) in enumerate(zip(object_names, object_locs)):
        loc_str = " located at "
        if (loc - offset)[0] == 0 or (loc - offset)[1] == 0:
            loc_str += "center "
        elif (loc - offset)[0] < 0:
            loc_str += "back "
        elif (loc - offset)[0] > 0:
            loc_str += "front "
        
        if (loc - offset)[1] < 0:
            loc_str += "left."
        elif (loc - offset)[1] > 0:
            loc_str += "right."
        object_names[i] = object_names[i] + loc_str

    scene_objects = []
    is_there_target_object = False
    for o in range(num_scene_objects):
        if target_object in object_names[o] and not is_there_target_object:
            n = "target_object"
            is_there_target_object = True
        else:
            n = f"object_number{o}"
        
        scene_objects.append( SceneObject.from_dict(n, {"position": object_locs[o], "params": object_names[o] }) )

    assert is_there_target_object == True, "No target object!"
    return Scene(name="target_object_scene", objects=scene_objects)



def main():
    rclpy.init()
    node = PointingExperiment()

    while rclpy.ok():
        
        # 1. Setup scene with one object with random location
        scene = generate_random_scene(3, 1)
        time.sleep(1.0)
        # 2. Put the object on the scene and let the user point to it
        node.tts.speak(f"Put the object on locations")
        input()
        node.tts.speak(f"3, 2, 1, capture!")
        # 3. evaluate continuously
        time_start = time.perf_counter()
        while True:
            predicted_object = node.select_target(sample)

            if predicted_object == scene.target_object:
                return

            time.sleep(0.5)





if __name__ == "__main__":
    main()