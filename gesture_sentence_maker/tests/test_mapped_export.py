#!/usr/bin/env python
"""The /modality/gestures payload, as the mergers read it.

No sensor and no running ROS: the sentence dict is written by hand and the
message is handed straight to HriCommand.from_ros, which is what
reasoning_merger and action_executor do with it. What matters here is the
contract between the two: which slots appear, and that no action appears when
no gesture named one.
"""
import json

import pytest

from gesture_meaning.one_to_one_mapping import OneToOneMapping
from gesture_sentence_maker.hricommand_export import export_mapped_to_HRICommand
from hri_manager.HriCommand import HriCommand

MAPPING = OneToOneMapping({"links": {
    "l1": {"action_template": "pick", "action_gestures": [["grab", "swipe_up"]]},
}, "actions": ["pick", "push"]}, quiet=True)

# As the detector reports an episode: one probability and one timestamp per
# gesture, the stamp being when that gesture was at its most likely.
GS = ["grab", "swipe_up"]
STAMPS = [11.5, 12.0]

# What export_original_to_HRICommand produced for the same episode.
SENTENCE = {
    "target_gesture": "grab",
    "target_gesture_timestamp": 11.5,
    "gesture_names": GS,
    "gesture_probs": [0.9, 0.8],
    "object_names": ["cube1", "bowl1"],
    "object_probs": [0.7, 0.3],
    "target_object": "cube1",
    "target_object_timestamp": 12.4,
}


def mapped(gesture_probs):
    msg = export_mapped_to_HRICommand(SENTENCE, MAPPING, gesture_names=GS,
                                      gesture_probs=gesture_probs,
                                      gesture_timestamps=STAMPS)
    return json.loads(msg.data[0]), HriCommand.from_ros(msg)


def test_a_linked_combination_gives_the_merger_an_action_slot():
    payload, command = mapped([0.9, 0.8])
    assert payload["target_action"] == "pick"
    assert payload["action_names"] == ["pick", "push"]
    assert "action" in command.pv_dict
    assert command.target_action == "pick"
    # The gesture and object slots survive untouched.
    assert command.target_object == "cube1"


def test_the_action_is_stamped_at_the_start_of_the_combination():
    """The merger interleaves this against the voice words by stamp, so the
    action belongs where the user began commanding it."""
    payload, command = mapped([0.9, 0.8])
    assert payload["target_action_timestamp"] == min(STAMPS)
    assert command.get_stamp("action") == min(STAMPS)


def test_half_a_combination_names_no_action():
    """swipe_up was never shown. Publishing the argmax of an all-zero
    distribution would name the first action of the vocabulary as a confident
    command."""
    payload, command = mapped([0.95, 0.0])
    assert "action_names" not in payload and "target_action" not in payload
    assert "action" not in command.pv_dict
    # The pointed-at object still reaches the merger.
    assert command.target_object == "cube1"


def test_a_pointing_only_episode_is_published_unchanged():
    sentence = {k: v for k, v in SENTENCE.items() if not k.startswith(("gesture", "target_gesture"))}
    msg = export_mapped_to_HRICommand(sentence, MAPPING)
    assert json.loads(msg.data[0]) == sentence
    assert "action" not in HriCommand.from_ros(msg).pv_dict


def test_the_payload_is_json_the_consumers_can_parse():
    """from_ros reads it with json.loads and the live display with JSON.parse,
    so the payload has to be real JSON -- not str(dict)."""
    msg = export_mapped_to_HRICommand(dict(SENTENCE, parameter_flag=True), MAPPING,
                                      gesture_names=GS, gesture_probs=[0.9, 0.8],
                                      gesture_timestamps=STAMPS)
    assert json.loads(msg.data[0])["parameter_flag"] is True
