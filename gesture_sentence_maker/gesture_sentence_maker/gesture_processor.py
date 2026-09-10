#!/usr/bin/env python
from collections import Counter
from copy import deepcopy
import threading
import time

from scene_getter.scene_getting import SceneGetter
from gesture_detector.gesture_classification.gestures_lib import GestureDataDetection
import numpy as np
import rclpy
from rclpy.node import Node

from gesture_sentence_maker.hricommand_export import (
    export_mapped_to_HRICommand, export_original_to_HRICommand,
    import_original_HRICommand_to_dict)
from gesture_meaning.gesture_icons import GESTURE_ICONS
from gesture_meaning.one_to_one_mapping import DEFAULT_LINKS, OneToOneMapping, load_links
from pointing_object_selection.pointing_object_getter import PointingObjectGetter
from pointing_object_selection.deictic_evidence import EVIDENCE, select as select_deictic
from gesture_sentence_maker.hri_command_msg import (
    HRICommand, HRICommandMSG, HRI_COMMAND_TYPE,
    HRI_COMMAND_ROSBRIDGE_TYPE)
from gesture_sentence_maker.utils import get_dist_by_extremes

try:
    from hri_msgs.srv import AddGestureLink, RemoveGestureLink
except ImportError:
    AddGestureLink = None
    RemoveGestureLink = None
from std_msgs.msg import String
from gesture_detector.utils.utils import CustomDeque

from rclpy.qos import QoSProfile, QoSReliabilityPolicy
import json

try:
    from hri_manager.user_links import add_gesture_link, links_mtime, remove_gesture_link
except ImportError:
    add_gesture_link = None
    links_mtime = None
    remove_gesture_link = None

HRI_MANAGER_AVAILABLE = (
    add_gesture_link is not None and remove_gesture_link is not None)
LINK_SERVICES_AVAILABLE = (
    AddGestureLink is not None and RemoveGestureLink is not None)


def _links_capability(name_user: str):
    """Return whether links can be edited and the filename shown in the UI."""
    if HRI_MANAGER_AVAILABLE:
        return bool(name_user) and LINK_SERVICES_AVAILABLE, (
            f"{name_user}_links.yaml" if name_user else "<user>_links.yaml")
    return False, DEFAULT_LINKS.rsplit("/", 1)[-1]


def _user_settings(name_user: str) -> dict:
    """The user's links file: their vocabulary, gesture links and cell settings.

    With hri_manager, an absent user or unreadable user file gives no mapping.
    Without hri_manager, load the standard links.yaml shipped with
    gesture_meaning, including when no user name was supplied."""
    # The standalone gesture toolbox deliberately uses gesture_meaning's
    # standard links.yaml. A user name only has meaning when hri_manager is
    # installed and owns per-user files.
    if not HRI_MANAGER_AVAILABLE:
        return load_links(name_user)
    if not name_user:
        return {}
    try:
        return load_links(name_user)
    except Exception as e:  # noqa: BLE001 -- missing file or unreadable yaml
        print(f"[Gesture Processor] User settings for {name_user!r} not loaded ({e}), "
              f"using defaults and no gesture meaning", flush=True)
        return {}


class GestureSentence(PointingObjectGetter, SceneGetter, GestureDataDetection):
    def __init__(self,
                 ignored_gestures = ['point', 'no_moving'],
                 # rate=10 in activated_gesture_type_to_action assumes this period:
                 # at 0.2 a mode needed 1.0s to settle rather than the 0.5s intended.
                 step_period = 0.1, # seconds
                 ):
        """
        Args:
            ignored_gestures (list, optional): Activation of these gesture names do not trigger gesture sentence (publisher is not sending). Defaults to ['point', 'no_moving', 'five', 'pinch'].
        """        
        self.topic = "sentence_processor_node"
        super(GestureSentence, self).__init__()

        self.gesture_sentence_publisher = self.create_publisher(HRICommand, "/teleop_gesture_toolbox/hricommand_original", qos_profile=QoSProfile(depth=10, reliability=QoSReliabilityPolicy.RELIABLE))
        # Final output
        self.modality_gestures_publisher = self.create_publisher(HRICommand, "/modality/gestures", qos_profile=QoSProfile(depth=10, reliability=QoSReliabilityPolicy.RELIABLE))
        self.get_logger().info(f"Using {HRI_COMMAND_TYPE} for gesture commands")

        # sentence data
        self.prev_gesture_type = None
        self.prev_deictic_solutions = CustomDeque()
        self.prev_auxgesture_solutions = CustomDeque()
        
        # One entry per finished pointing, each already decided by
        # deictic_evidence.select. Deliberately NOT named target_object_solutions:
        # PointingObjectGetter owns an attribute of that name and appends every
        # raw frame it receives to it, which would overwrite these decisions.
        self.selected_object_solutions = CustomDeque() # Queue of DeicticSolutions
        self.target_auxgesture_solutions = CustomDeque() # Queue of (?)

        self.evidence_gesture_type_to_activate_last_added = 0.
        self.evidence_gesture_type_to_activate = CustomDeque()

        # Per-user settings live in links/<user>_links.yaml, so a new cell is configured by editing yaml rather than this file. The arguments above stay the defaults for when no user is given.
        self.user = self.declare_parameter("user_name", "").get_parameter_value().string_value
        self.user_settings = _user_settings(self.user)
        self.links_editable, self.links_source = _links_capability(self.user)
        self.ignored_gestures = self.user_settings.get("ignored_gestures", ignored_gestures)
        self.activate_length = self.user_settings.get("activate_length", self.activate_length)
        self.activate_length_dynamic = self.user_settings.get("activate_length_dynamic", self.activate_length_dynamic)
        # A gesture means what this user linked it to and nothing else. The
        # dashboard service replaces this mapping after it persists a new link.
        self.mapping = OneToOneMapping(self.user_settings)
        self.settings_lock = threading.RLock()
        self._settings_mtime = self._links_mtime()
        # Which gestures put this user into which mode. Per-user like the rest,
        # so a cell is retuned by editing yaml rather than this file.
        self.adaptive_setup = AdaptiveSetup(self.user_settings.get("adaptive_setup"))
        print(f"[Gesture Processor] Adaptive setup: {self.adaptive_setup.adaptive_setup}")

        self.step_period = step_period

        # The live display (gesture_detector/live_display) reads the user and their links from here.
        self.meaning_info_pub = self.create_publisher(String, "/teleop_gesture_toolbox/gesture_meaning_info", 5)
        self.add_gesture_link_service = None
        self.remove_gesture_link_service = None
        if self.links_editable:
            self.add_gesture_link_service = self.create_service(
                AddGestureLink,
                "/teleop_gesture_toolbox/add_gesture_link",
                self.add_gesture_link_callback)
            self.remove_gesture_link_service = self.create_service(
                RemoveGestureLink,
                "/teleop_gesture_toolbox/remove_gesture_link",
                self.remove_gesture_link_callback)
        # What the current pointing has settled on so far, for the live display.
        self.pending_selection_pub = self.create_publisher(String, "/teleop_gesture_toolbox/pending_object_selection", 5)
        # The mode the user is in right now. Its own topic rather than a field of
        # gesture_meaning_info, which is a 1 Hz config snapshot: a mode arriving
        # up to a second late would read as a stuck sign. Empty means idle.
        self.mode_pub = self.create_publisher(String, "/teleop_gesture_toolbox/gesture_mode", 5)
        threading.Thread(target=self.send_info_thread, daemon=True).start()

        self.continue_episode = self.present
        print(f"[Gesture Processor] Note that gesture processor is discarding gestures: {self.ignored_gestures}")
        print(f"[Gesture Processor] Gesture activates after {self.activate_length} detections")
        print(f"[Gesture Processor] Gesture meaning: {self.mapping.combinations}")
        print("[GS] Done ")

    def send_info_thread(self):
        while rclpy.ok():
            time.sleep(1.0)
            self.publish_meaning_info()

    def _links_mtime(self):
        if not (self.user and links_mtime is not None):
            return None
        try:
            return links_mtime(self.user)
        except OSError:
            return None

    def reload_settings_if_changed(self):
        """Pick up dashboard edits to the links file without a restart."""
        mtime = self._links_mtime()
        if mtime is None or mtime == self._settings_mtime:
            return
        self._settings_mtime = mtime
        settings = _user_settings(self.user)
        mapping = OneToOneMapping(settings)
        with self.settings_lock:
            self.user_settings = settings
            self.mapping = mapping
        self.publish_meaning_info()
        print(f"[Gesture Processor] Reloaded {self.user}_links.yaml", flush=True)

    def meaning_info(self):
        """Configuration snapshot for dashboards, independent of detections."""
        with self.settings_lock:
            return {
                "user": self.user,
                "gesture_icons": GESTURE_ICONS,
                "adaptive_setup": self.adaptive_setup.adaptive_setup,
                "static_gestures": list(self.Gs_static),
                "dynamic_gestures": list(self.Gs_dynamic),
                "actions": list(self.user_settings.get("actions") or []),
                "links": dict(self.user_settings.get("links") or {}),
                "links_editable": self.links_editable,
                "links_source": self.links_source,
                "hri_command_type": HRI_COMMAND_ROSBRIDGE_TYPE,
            }

    def publish_meaning_info(self):
        self.meaning_info_pub.publish(String(data=json.dumps(self.meaning_info())))

    def add_gesture_link_callback(self, request, response):
        """Validate, persist and activate a dashboard-created mapping."""
        try:
            if not self.links_editable or add_gesture_link is None:
                raise RuntimeError("The active mapping file is read-only")
            link_name, settings, created = add_gesture_link(
                self.user,
                request.action_template,
                request.static_gesture,
                request.dynamic_gesture,
                static_gestures=self.Gs_static,
                dynamic_gestures=self.Gs_dynamic)
            mapping = OneToOneMapping(settings)
            with self.settings_lock:
                self.user_settings = settings
                self.mapping = mapping
            response.success = True
            response.link_name = link_name
            response.message = (
                f"Added {link_name}" if created else
                f"Mapping already exists as {link_name}")
            response.links_json = json.dumps(settings.get("links") or {})
            self.publish_meaning_info()
        except Exception as error:  # service boundary: return validation/I/O errors
            response.success = False
            response.message = str(error)
            response.link_name = ""
            response.links_json = ""
            self.get_logger().warning(f"Unable to add gesture link: {error}")
        return response

    def remove_gesture_link_callback(self, request, response):
        """Persist a link removal and activate the reduced mapping."""
        try:
            if not self.links_editable or remove_gesture_link is None:
                raise RuntimeError("The active mapping file is read-only")
            _, settings = remove_gesture_link(self.user, request.link_name)
            mapping = OneToOneMapping(settings)
            with self.settings_lock:
                self.user_settings = settings
                self.mapping = mapping
            response.success = True
            response.message = f"Removed {request.link_name}"
            response.links_json = json.dumps(settings.get("links") or {})
            self.publish_meaning_info()
        except Exception as error:  # service boundary: return validation/I/O errors
            response.success = False
            response.message = str(error)
            response.links_json = ""
            self.get_logger().warning(f"Unable to remove gesture link: {error}")
        return response

    def publish_sentence(self, **kwargs):
        """The sentence, raw and mapped: gesture names on hricommand_original,
        the action the user linked them to on /modality/gestures."""
        msg = export_original_to_HRICommand(self.scene, self.selected_object_solutions, **kwargs)
        self.gesture_sentence_publisher.publish(msg)
        self.modality_gestures_publisher.publish(export_mapped_to_HRICommand(
            import_original_HRICommand_to_dict(msg), self.mapping,
            gesture_names=kwargs.get("gesture_names"),
            gesture_probs=kwargs.get("gesture_probabilities"),
            gesture_timestamps=kwargs.get("gesture_timestamps")))

    def step(self):
        time.sleep(self.step_period)
        self.reload_settings_if_changed()
        if self.continue_episode(): # Hand not visible is condition for Episode to End 
            self.gesturing_step()
        
        else: # End episode
            if not self.gestures_queue.empty:
                time.sleep(0.5)

                # target_object data save
                if not self.prev_deictic_solutions.empty:
                    self.save_accumulated_deictic_gesture_data()

                publish, max_probs, max_timestamps, params = self.gestures_queue.processing(self.ignored_gestures)

                if publish > 0:
                    self.publish_sentence(gesture_probabilities=max_probs,
                                          gesture_timestamps=max_timestamps,
                                          gesture_names=self.Gs, params=params)
                    self.clearing()
                elif len(self.selected_object_solutions) > 0:
                    self.publish_sentence()
                    self.clearing()
                    return

            elif len(self.selected_object_solutions) > 0:
                self.publish_sentence()
        
            # Whenever hand is not seen clearing
            self.clearing(wait=False)

    def gesturing_step(self):
        # A mode is the posture shown now: at 2.0s a finished pointing kept the
        # user in deictic mode for two seconds after the hand had moved on.
        activated_gestures = self.load_all_relevant_activated_gestures(relevant_time=0.4, records=3)

        activated_gesture_type = self.adaptive_setup.get_adaptive_gesture_type(activated_gestures)

        activated_gesture_type_action = self.activated_gesture_type_to_action(activated_gesture_type)

        # The settled mode, not the raw per-frame detection: the raw one flickers
        # between modes several times a second. None means not settled yet, and
        # then the last mode stands rather than the sign blinking to idle.
        if activated_gesture_type_action is not None:
            self.publish_mode(activated_gesture_type_action)

        self.save_accumulated_data_of_unactivated_gesture_types()

        if activated_gesture_type_action == 'deictic':
            self.step_deictic()
        
        # elif activated_gesture_type_action == 'approvement':
        #     self.step_approvement()

        
        else:
            # Evaluate when action gesturing ends
            self.prev_gesture_type = 'action'

    def activated_gesture_type_to_action(self, activated_gesture_type, rate=10, x=5, y=10, threshold=0.9):
        '''
        Parameters:
            rate (Int): Rate of new frames (Hz)
            x (Int): Gesture type evidence to be activated (frames)
            y (Int): How many frames it must be non activated, before the gesture type is activated
            threshold (Float): accuracy threshiold, >% frames gesture type -> activated
        Returns:
            gesture_type (String/None): If fulfills the conditions or None if not

        --------- | --------- | ---------
             aaaaa|dddddddddddddddddddd aaaaa
                   < -------- x ------> x True
             <-y->|< ----- delay -----> y False
        '''
        if (time.time()-self.evidence_gesture_type_to_activate_last_added) > (1/rate):
            self.evidence_gesture_type_to_activate_last_added = time.time()
            self.evidence_gesture_type_to_activate.append(activated_gesture_type)

        gesture_type = self.evidence_gesture_type_to_activate.get_last_common(x, threshold=1.0)
        if gesture_type is not None:
            return gesture_type



    def save_accumulated_data_of_unactivated_gesture_types(self):
        ''' Handle for each gesture type '''

        if (self.prev_gesture_type != 'deictic' and 
            not self.prev_deictic_solutions.empty):

            self.save_accumulated_deictic_gesture_data()
    
        if (self.prev_gesture_type != 'measurement_distance' and 
                not self.prev_auxgesture_solutions.empty):
            
            self.target_auxgesture_solutions.append(get_dist_by_extremes(np.array(self.prev_auxgesture_solutions)))
            self.prev_auxgesture_solutions = CustomDeque()

    def step_deictic(self):
        ''' Activated gesture enabled Deictic gesture mode.
        '''
        deictic_solution = self.get_target_object()

        self.prev_gesture_type = 'deictic'
        self.prev_deictic_solutions.append(deictic_solution)
        self.publish_pending_selection()

    def publish_mode(self, mode: str = ""):
        """The mode the user is in, for the live display's Doing sign.

        Sent every step rather than only on a change, so a dashboard opened
        mid-session fills in within one step instead of waiting for the user to
        switch modes. Empty string is idle -- no hand, nothing settled."""
        self.mode_pub.publish(String(data=mode))

    def publish_pending_selection(self):
        """The object this pointing would contribute if it ended now, for viewers.

        Run through the same select() the sentence uses, over the same buffer, so
        a viewer shows the decision instead of a second guess at it."""
        solution = select_deictic(self.prev_deictic_solutions)
        self.pending_selection_pub.publish(String(
            data=(solution.target_object_name if solution is not None else "")))

    def step_approvement(self):
        res = misc_gesture_handle(f"Approve? (y/n)", new_episode=False)
        print(f"Added approvement {res}")
        self.target_auxgesture_solutions.append(res)

    def clearing(self, wait=True):
        self.gestures_queue.clear()
        self.evaluate_episode = False

        self.selected_object_solutions = CustomDeque()
        self.target_auxgesture_solutions = CustomDeque()
        # Pointing frames belong to the episode they were made in: kept, they
        # would let a previous episode's object win this episode's select().
        self.prev_deictic_solutions = CustomDeque()
        self.prev_auxgesture_solutions = CustomDeque()
        self.publish_pending_selection()
        self.publish_mode()  # episode over: idle until the next gesture settles

        if wait:
            print("Move hand out to end the episode!")
            while self.present():
                time.sleep(0.1)
            print("Episode finished!")
            print("=================")
            print("")


    def save_accumulated_deictic_gesture_data(self):

        if self.prev_deictic_solutions.empty:
            print("No object to be added, returning")
            return

        # The user may point for as long as they like and wander over several
        # objects on the way; what they meant is the last object that stayed
        # selected long enough (see deictic_evidence.py). Never settling on one
        # selects nothing, which beats naming whatever the hand passed last.
        solution = select_deictic(self.prev_deictic_solutions)
        self.prev_deictic_solutions = CustomDeque()
        self.publish_pending_selection()
        if solution is None:
            print(f"No object held for {EVIDENCE} frames, nothing selected")
            return

        self.selected_object_solutions.append(solution)
        print(f"New scene object selected: {solution.target_object_name}")




class AdaptiveSetup():
    """Which gesture modes this user has, read from their links file.

    One entry is `mode: [gesture, ...]` and showing *any* of those gestures puts
    the user in that mode -- a mode is a hand posture, not a combination (unlike
    links.action_gestures, where every gesture of an entry has to be shown).

    Only modes gesturing_step() actually dispatches are accepted. A mode nobody
    handles is dropped with a warning rather than kept: it would be detected,
    fall through to the 'action' branch, and leave the dashboard announcing a
    mode that does nothing.
    """

    # Modes with a working handler in gesturing_step. 'approvement' and
    # 'measurement_distance' are deliberately absent: step_approvement calls an
    # undefined misc_gesture_handle, and nothing ever fills the buffer
    # measurement_distance would read. Add a mode here once its step_ works.
    HANDLED = ('deictic',)
    # Applies when the links file says nothing, so existing files and the
    # no-user standalone path keep pointing without an adaptive_setup key.
    DEFAULT = {'deictic': ['point', 'two']}

    def __init__(self, adaptive_setup: dict | None = None):
        self.adaptive_setup = self._validate(
            self.DEFAULT if not adaptive_setup else adaptive_setup)

    @classmethod
    def _validate(cls, setup: dict) -> dict:
        accepted = {}
        for mode, gestures in setup.items():
            # A bare string is the trap this check exists for: `mode: pinch`
            # yaml-parses to "pinch", and `gesture in "pinch"` then matches on
            # substrings, so a gesture named 'pin' or even 'n' would activate it.
            if isinstance(gestures, str) or not isinstance(gestures, (list, tuple)):
                print(f"[AdaptiveSetup] mode {mode!r} must list its gestures "
                      f"(got {gestures!r}), ignoring it", flush=True)
                continue
            if mode not in cls.HANDLED:
                print(f"[AdaptiveSetup] mode {mode!r} has no handler in "
                      f"gesturing_step, ignoring it", flush=True)
                continue
            accepted[mode] = [str(g) for g in gestures]
        return accepted

    def get_adaptive_gesture_type(self, activated_gestures):
        activated_gesture_types = []

        as_ = self.adaptive_setup
        # activated_gestures = ('point')
        for ag in activated_gestures:
            # as_.keys() = ('deictic', 'approvement', 'measurement')
            for k in as_.keys():
                asi = as_[k]
                # if the adaptive setup item has the activated gesture in its list
                if ag in asi: # gesture which is activated is is adaptive setup gestures
                    # activate the gesture type
                    if k not in activated_gesture_types:
                        activated_gesture_types.append(k)

        if len(activated_gesture_types) > 1:
            # TODO:
            #print(f"[WARNING] More possible gesture types, act: {activated_gesture_types}")
            activated_gesture_type = activated_gesture_types[0]
        elif len(activated_gesture_types) == 0:
            activated_gesture_type = 'action'
        elif len(activated_gesture_types) == 1:
            activated_gesture_type = activated_gesture_types[0]
        else: raise Exception("Cannot happen")

        return activated_gesture_type


import threading

def spinning_threadfn(gd):
    while rclpy.ok():
        gd.spin_once(sem=True)
        time.sleep(0.01)


def main():
    rclpy.init()
    sentence_processor = GestureSentence()
    spinning_thread = threading.Thread(target=spinning_threadfn, args=(sentence_processor, ), daemon=True)
    spinning_thread.start()
    while rclpy.ok():
        sentence_processor.step()


if __name__ == '__main__':
    main()
