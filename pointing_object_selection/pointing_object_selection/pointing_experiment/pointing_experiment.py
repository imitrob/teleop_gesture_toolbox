from __future__ import annotations
from copy import deepcopy
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
from hri_msgs.msg import WhisperText as WhisperTextMSG

from pointing_object_selection.deictic_lib import PointingSequence, SelectionResult
from pointing_object_selection.pointing_experiment.utils import print_table_scene, show_pointing_experiment_plot
from pointing_object_selection.pointing_object_getter import PointingObjectGetter
from skills_manager.lfd import LfD

from scene_getter.scene_getting import SceneGetter

from pydantic import BaseModel, Field, conlist
from typing import List

from panda_control.panda_extended import top_of_object

DEICTIC_HZ = 10

class WhisperText(BaseModel):
    """ Natural‑language description of the target object (can be partial/incomplete). """
    stamp: float = Field(..., description="Timestamps")
    text: str = Field(..., description="Transcription Texts")

class WhisperTextList(BaseModel):
    texts: List[WhisperText] = Field(default_factory=list, description="oldest → newest.")

    def cut(self, t_start: float, t: float):
        t_end = t_start + t
        idx_start = None
        idx_end = None
        for i, text in enumerate(self.texts):
            if idx_start is None and text.stamp > t_start:
                idx_start = i
            if text.stamp > t_end:
                break
        idx_end = i
        return WhisperTextList(texts=self.texts[idx_start:idx_end])


class InputSample(BaseModel):
    """All data needed to decide which object the user is referring to."""
    whisper_texts: WhisperTextList
    pointing: PointingSequence
    scene: Scene
    time_start: float # time.time() float format

    def to_pointing_experiment_plot(self):
        return {
            "T": len(self.pointing.lines),
            "object_names": self.scene.object_names,
            "descriptions": {item.stamp: item.text for item in self.whisper_texts.texts},
            "object_positions": self.scene.object_positions_xy,
            "target_pointing": self.pointing.get_contacts_with_ground(),
            "valid_objects": np.zeros((len(self.pointing.lines), self.scene.n)),
            "pointing_likelihoods": np.array([item.object_likelihoods for item in self.pointing.lines]),
            "time_start": self.time_start,
        }
    

class RealtimeWhisperGetter:
    def __init__(self):
        super(RealtimeWhisperGetter, self).__init__()
        self.create_subscription(WhisperTextMSG, "/nlp/whisper", self.receive_whisper_callback, 5)
        self.transcriptions = WhisperTextList()

    def receive_whisper_callback(self, msg):
        self.transcriptions.texts.append(
            WhisperText(stamp = msg.header.stamp.sec+msg.header.stamp.nanosec*1e-9, text=str(msg.new_text))
        )

class PointingExperimentBase(RealtimeWhisperGetter, SceneGetter, PointingObjectGetter, LfD):
    def __init__(self):
        super(PointingExperimentBase, self).__init__()
        self.angle_error_pub = self.create_publisher(Float32, "/angle_error", 5)
        
        self.tts = Chatterbox(device="cuda")

    def receive_deictic(self, msg):
        ang = self.compute_error([
                self._target_object.line_points.start.x,
                self._target_object.line_points.start.y,
                self._target_object.line_points.start.z
            ], [
                self._target_object.line_points.end.x,
                self._target_object.line_points.end.y,
                self._target_object.line_points.end.z
            ], [
                self._target_object.target_object_position.x, 
                self._target_object.target_object_position.y,
                self._target_object.target_object_position.z
            ])
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

    @staticmethod
    def select_target(sample: InputSample) -> SelectionResult:
        """Return a dummy object ID.
        selection logic (e.g. language grounding + spatial reasoning).
        """
        raise BaseException
        return SelectionResult(object_id="none")


class PointingExperiment(PointingExperimentBase):
    def __init__(self):
        super(PointingExperiment, self).__init__()

    @staticmethod
    def select_target(sample: InputSample) -> SelectionResult:
        """Return a dummy object ID.
        selection logic (e.g. language grounding + spatial reasoning).
        """
        # Convert UTC to relative time from the start
        considered_words = ""
        for text in sample.whisper_texts.texts:
            if text.stamp - sample.time_start > 0.0:
                considered_words += text.text

        object_scores = [0] * sample.scene.n
        for word in considered_words:
            # process word
            word.strip().lower()
            word = ''.join(e for e in word if e.isalnum())
            for i, obj in enumerate(sample.scene.objects):
                if word in obj.params:
                    object_scores[i] += i

        ids = np.argwhere(object_scores == np.amax(object_scores)).flatten().tolist()
        if len(sample.pointing.lines) > 0:
            ids.append(sample.pointing.lines[-1].target_object_id)

        ids = list(set(ids))

        return SelectionResult(object_ids=ids)
    
def main():
    np.random.seed(42)

    rclpy.init()
    node = PointingExperiment()
    node.start()

    node.start_publishing_scene()
    node.wait_for_scene()
    node.stop_publishing_scene()

    node.home()

    while rclpy.ok():
        # 1. Setup scene with one object with random location
        scene, locs, of = generate_random_scene(3, 1)
        node.tts.speak(f"Put the object on locations")
        # 2. Put the object on the scene
        if False:
            while node.scene != scene:
                time.sleep(1.0)
                print_scene = scene.copy()
        
                print_table_scene(print_scene, locs, of, scene2=node.scene)
                print(node.scene)
        node.tts.speak(f"3, 2, 1, capture!")
        # 3. evaluate continuously
        time_start = time.time()
        
        while rclpy.ok():
            obj_id = node.target_object_solutions[-1].target_object_id
            object1 = node.scene.objects[obj_id]

            print("object1", flush=True)
            print(object1, flush=True)
            print(object1.position, flush=True)

            node.go_to_pose_ik(top_of_object(object1))

            # TODO: Check if the correct object is not selected 
            # if predicted_object == scene.target_object:
            #     return
            if time.time() - time_start > 3.0:
                break

        valid_objects = []
        for t in range(len(node.target_object_solutions)):
            # InputSample at this t
            sample = InputSample(whisper_texts= node.transcriptions.cut(time_start, float(t)*(1.0/DEICTIC_HZ)), 
                                scene=node.scene,
                                pointing=PointingSequence(lines=list(node.target_object_solutions)[:t+1]),
                                time_start = time_start,
                                )
            predicted_objects = node.select_target(sample)
            valid_objects.append(predicted_objects.to_bools(sample.scene.n))
            
        plot_data = sample.to_pointing_experiment_plot()
        plot_data["valid_objects"] = valid_objects
        print(valid_objects)
        show_pointing_experiment_plot(**plot_data)
        break

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
    object_classes = [o.split(" ")[-1] for o in object_names]
    
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
            n = f"{object_classes[o]}{o}"
        
        scene_objects.append( SceneObject.from_dict(n, {"position": object_locs[o], "params": object_names[o] }) )

    assert is_there_target_object == True, "No target object!"
    return Scene(name="target_object_scene", objects=scene_objects), locs, offset

if __name__ == "__main__":
    main()