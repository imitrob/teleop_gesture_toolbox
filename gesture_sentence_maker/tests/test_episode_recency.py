#!/usr/bin/env python
"""Two commands in one episode: the last one performed is the one that counts.

No sensor and no running ROS. The episode is built the way
GestureDataDetection.activation_postprocessing builds it -- one entry per
activated gesture, each carrying the whole probability vector as it stood at
that moment -- and read the way the sentence maker reads it.
"""
from gesture_detector.gesture_classification.episodic_accumulation import AccumulatedGestures
from gesture_meaning.one_to_one_mapping import OneToOneMapping

MAPPING = OneToOneMapping({"links": {
    "l1": {"action_template": "pick", "action_gestures": [["grab", "swipe_up"]]},
    "l2": {"action_template": "close", "action_gestures": [["five", "swipe_left"]]},
}, "actions": ["pick", "close"]}, quiet=True)

GS = ["grab", "five", "swipe_up", "swipe_left"]


def episode(*shown):
    """(stamp, static, dynamic) per command, both gestures saturated at 1.0."""
    queue = AccumulatedGestures()
    for stamp, static, dynamic in shown:
        probs = [1.0 if g in (static, dynamic) else 0.0 for g in GS]
        for name in (static, dynamic):
            queue.append({"stamp": stamp, "name": name, "hand": "r",
                          "probs": probs, "params": {}})
    return queue


def action_of(queue):
    publish, probs, _, _ = queue.processing(ignored_gestures=[])
    assert publish
    return MAPPING.best_combination(GS, list(probs))[1], probs


def test_one_command_is_unweighted():
    action, probs = action_of(episode((10.0, "grab", "swipe_up")))
    assert action == "pick"
    assert max(probs) == 1.0


def test_the_command_performed_last_wins():
    """Both saturate at 1.0, so without recency this is decided by dict order."""
    action, _ = action_of(episode((10.0, "grab", "swipe_up"),
                                  (12.0, "five", "swipe_left")))
    assert action == "close"
    action, _ = action_of(episode((10.0, "five", "swipe_left"),
                                  (12.0, "grab", "swipe_up")))
    assert action == "pick"


def test_redoing_the_same_command_does_not_weaken_it():
    action, probs = action_of(episode((10.0, "grab", "swipe_up"),
                                      (12.0, "grab", "swipe_up")))
    assert action == "pick"
    assert max(probs) == 1.0


def test_an_unsure_correction_does_not_beat_a_confident_command():
    """The penalty is small on purpose: it breaks ties, it does not overrule."""
    queue = episode((10.0, "grab", "swipe_up"))
    for name in ("five", "swipe_left"):
        queue.append({"stamp": 12.0, "name": name, "hand": "r", "params": {},
                      "probs": [0.0, 0.5, 0.0, 0.5]})
    assert action_of(queue)[0] == "pick"


if __name__ == "__main__":
    for name, fn in sorted(list(globals().items())):
        if name.startswith("test_"):
            fn()
            print(f"ok {name}")
