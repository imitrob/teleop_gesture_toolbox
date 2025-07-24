from __future__ import annotations
import time
import rclpy
from scene_getter.scene_lib.scene import Scene
import numpy as np
from natural_language_processing.text_to_speech.kokoro_model import Chatterbox
from typing import Sequence
from std_msgs.msg import Float32
from hri_msgs.msg import WhisperText as WhisperTextMSG

from pointing_object_selection.deictic_lib import PointingSequence, SelectionResult
from pointing_object_selection.pointing_experiment.utils import show_pointing_experiment_plot
from pointing_object_selection.pointing_object_getter import PointingObjectGetter
from pointing_object_selection.pointing_experiment.scene_gen import generate_random_scene
from skills_manager.lfd import LfD

from scene_getter.scene_getting import SceneGetterViaObjectLocalizer

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
        i = None
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

class PointingExperimentBase(RealtimeWhisperGetter, SceneGetterViaObjectLocalizer, PointingObjectGetter, LfD):
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
    node.home()

    while rclpy.ok():
        # 1. Setup scene with one object with random location
        scene, locs, of = generate_random_scene(3, 1, rings=2)
        node.tts.speak(f"Put the object on locations")
        # 2. Put the object on the scene
        node.start_publishing_scene()
        node.wait_for_scene()
        if True:
            while node.scene != scene:
                time.sleep(1.0)
                print_scene = scene.copy()
                print_scene.print_table_scene(locs, of, scene2=node.scene)
                print(node.scene)
        node.stop_publishing_scene()

        node.tts.speak(f"3, 2, 1, capture!")
        while len(node.target_object_solutions) == 0:
            print("No hand data")
            time.sleep(0.1)
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

if __name__ == "__main__":
    main()