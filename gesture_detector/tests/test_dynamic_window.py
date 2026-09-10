import json

import numpy as np

import gesture_detector
from gesture_detector.gesture_classification.gestures_lib import (
    DYNAMIC_WINDOW, DynamicGs, GestureMorphClassStamped, dynamic_sample_indices,
    mode_posture_shown)
from gesture_detector.gesture_classification.timewarp_lib import fastdtw_
from gesture_detector.utils.utils import transform_leap_to_leapdynamicdetector
from gesture_msgs.msg import DetectionSolution

FPS = 100.0
DETECTOR_RATE = 10.0
WINDOW = DYNAMIC_WINDOW
TIME_SAMPLES = 10
ACTIVATE_LENGTH_DYNAMIC = 3


def _model():
    with open(gesture_detector.saved_models_path + "directional_swipes.json") as f:
        config = json.load(f)
    sampler = fastdtw_()
    sampler.init(np.load(gesture_detector.saved_models_path + "directional_swipes.npz"), config)
    return sampler, config["gestures"]


def _circle(t, period, radius):
    a = 2 * np.pi * t / period
    return [radius * 1000 * np.sin(a), 300 + radius * 1000 * (1 - np.cos(a)), 0.0]


def _swipe_up(t, period, radius):
    return [0.0, 300 + radius * 1000 * 2 * (t / period), 0.0]


def _swipe_left(t, period, radius):
    return [-radius * 1000 * 2 * (t / period), 300.0, 0.0]


def _still(t, period, radius):
    return [0.0, 300.0, 0.0]


def _still_then_swipe_up(t, period, radius):
    """What the user actually does: hand held in view, then a quick swipe."""
    if t < 1.0:
        return _still(t, period, radius)
    return _swipe_up(min(t - 1.0, period), period, radius)


def _classify_live(traj, period, radius, duration=6.0, time_samples=TIME_SAMPLES, window=WINDOW):
    """Replays the /teleop_gesture_toolbox/dynamic_detection_observations composition of send_g_data."""
    sampler, Gs = _model()
    points = [traj(t, period, radius) for t in np.arange(0, duration, 1 / FPS)]
    n = int(window * FPS)

    labels = []
    for tick in np.arange(window, duration, 1 / DETECTOR_RATE):
        frames = points[int(tick * FPS) - n : int(tick * FPS)]
        composition = np.array([transform_leap_to_leapdynamicdetector(frames[i]) for i in dynamic_sample_indices(n, time_samples)])
        composition -= composition[len(composition) // 2]
        pred, _ = sampler.sample(composition.flatten())
        labels.append(Gs[pred])
    return labels, Gs


def _longest_run(labels, name):
    best = current = 0
    for label in labels:
        current = current + 1 if label == name else 0
        best = max(best, current)
    return best


def _evidence(labels, Gs, gesture, activate_length):
    """Feeds labels through the real activation counter and returns its peak ratio."""
    gs = DynamicGs({"Gs": Gs})
    peak = 0.0
    for label in labels:
        solution = DetectionSolution()
        solution.id = Gs.index(label)
        solution.probabilities.data = [1.0 if g == label else 0.0 for g in Gs]
        gs.data_queue.append(GestureMorphClassStamped(solution, Gs))
        peak = max(peak, gs.count_activ_evidence(Gs.index(gesture), activate_length) / activate_length)
    return peak


def _add(queue, gesture_id, gestures=6):
    """One detection solution with this gesture as top-1."""
    solution = DetectionSolution()
    solution.id = gesture_id
    solution.probabilities.data = [1.0 if n == gesture_id else 0.0 for n in range(gestures)]
    queue.data_queue.append(GestureMorphClassStamped(solution, queue.Gs))


def test_sample_indices_span_and_count():
    indices = dynamic_sample_indices(150, TIME_SAMPLES)
    assert len(indices) == TIME_SAMPLES
    assert indices == sorted(indices)
    assert indices[-1] == -1
    assert indices[0] == -int(150 * (TIME_SAMPLES - 1) / TIME_SAMPLES)


# DTW cannot align a cyclic shift, so a periodic path is only recognized while the
# window holds about one whole period: dynamic_window has to be ~1.25x the longest
# gesture. The 0.6s default covers a swipe; a circle this slow needs the knob raised.
CIRCLE_PERIOD = 0.8
CIRCLE_WINDOW = 1.0


def test_circle_becomes_top_1():
    labels, _ = _classify_live(_circle, period=CIRCLE_PERIOD, radius=0.05, window=CIRCLE_WINDOW)
    assert _longest_run(labels, "circle") >= ACTIVATE_LENGTH_DYNAMIC


def test_five_samples_cannot_see_a_circle():
    """Five samples a window is too sparse to hold circle long enough to activate."""
    labels, Gs = _classify_live(_circle, period=CIRCLE_PERIOD, radius=0.05, time_samples=5, window=CIRCLE_WINDOW)
    assert _longest_run(labels, "circle") < ACTIVATE_LENGTH_DYNAMIC
    assert _evidence(labels, Gs, "circle", ACTIVATE_LENGTH_DYNAMIC) < 1.0


def test_circle_activates_only_with_dynamic_activate_length():
    labels, Gs = _classify_live(_circle, period=CIRCLE_PERIOD, radius=0.05, window=CIRCLE_WINDOW)
    assert _evidence(labels, Gs, "circle", ACTIVATE_LENGTH_DYNAMIC) >= 1.0
    assert _evidence(labels, Gs, "circle", 10) < 1.0


def test_straight_swipe_unaffected():
    labels, _ = _classify_live(_swipe_up, period=0.4, radius=0.05)
    assert set(labels) == {"swipe_up"}


def test_left_swipe_recognized():
    labels, _ = _classify_live(_swipe_left, period=0.4, radius=0.03)
    assert set(labels) == {"swipe_left"}


def test_swipe_amplitude_does_not_decide():
    """A 60mm swipe is the same gesture as a 400mm one: unit_scale drops the scale."""
    for radius in (0.03, 0.2):
        labels, _ = _classify_live(_swipe_up, period=0.4, radius=radius)
        assert set(labels) == {"swipe_up"}, radius


def test_a_moving_hand_is_never_resting():
    """The resting class is out of the running above rest_displacement.

    This is the bug: on DTW distance alone the all-zero resting template beat every
    swipe that did not fill the whole window."""
    for traj, radius, window in ((_swipe_up, 0.05, WINDOW),
                                 (_swipe_left, 0.03, WINDOW),
                                 (_circle, 0.05, CIRCLE_WINDOW)):
        labels, _ = _classify_live(traj, period=0.4 if traj is not _circle else CIRCLE_PERIOD,
                                   radius=radius, window=window)
        assert "no_moving" not in labels, traj.__name__


def test_a_resting_hand_is_the_resting_gesture():
    """It stays reachable, users link it to actions ([five, no_moving] = stop)."""
    labels, Gs = _classify_live(_still, period=0.4, radius=0.05)
    assert set(labels) == {"no_moving"}
    assert _evidence(labels, Gs, "no_moving", ACTIVATE_LENGTH_DYNAMIC) >= 1.0
    # A hand that drifts by less than rest_displacement is still a resting hand
    labels, _ = _classify_live(_swipe_up, period=0.4, radius=0.01)  # 20mm of travel
    assert set(labels) == {"no_moving"}


def test_quick_swipe_after_waiting_is_not_lost_to_the_resting_gesture():
    """The reported bug: hand held in view, then a quick swipe.

    Waiting reads as resting, which is correct, and the swipe that follows still
    gets its own activation instead of the window staying stuck on resting."""
    labels, Gs = _classify_live(_still_then_swipe_up, period=0.4, radius=0.05)
    assert labels[0] == "no_moving"
    assert _longest_run(labels, "swipe_up") >= ACTIVATE_LENGTH_DYNAMIC
    assert _evidence(labels, Gs, "swipe_up", ACTIVATE_LENGTH_DYNAMIC) >= 1.0


def test_mode_postures_stop_dynamic_detection_and_nothing_else_does():
    """Pointing owns the hand, every other posture leaves the two streams independent."""
    modes = {"point", "two"}
    assert mode_posture_shown(["point"] * 6, modes)
    assert mode_posture_shown(["grab"] * 3 + ["point"] * 3, modes)  # pointing starts
    assert not mode_posture_shown(["grab"] * 6, modes)
    assert not mode_posture_shown(["grab", "pinch", "grab"], modes)  # static flicker
    assert not mode_posture_shown(["five"] * 6, modes)               # five + no_moving = stop
    assert not mode_posture_shown([None] * 6, modes)                 # no static prediction
    assert not mode_posture_shown(["point"] * 6, set())              # no links loaded yet


def test_evidence_counts_uninterrupted_detections():
    """activate_length detections mean activate_length, not one less."""
    _, Gs = _model()
    gs = DynamicGs({"Gs": Gs})

    def add(gesture_id):
        _add(gs, gesture_id)

    add(1)                                        # some other gesture
    assert gs.count_activ_evidence(0, 3) == 0.0   # not this one, no evidence at all
    add(0)
    assert gs.count_activ_evidence(0, 3) == 1.0
    add(0)
    assert gs.count_activ_evidence(0, 3) / 3 < 1.0  # two of three is not activation
    add(0)
    assert gs.count_activ_evidence(0, 3) / 3 == 1.0
    add(1)
    assert gs.count_activ_evidence(0, 3) == 0.0   # interrupted, evidence starts over

    # A queue too short to judge cannot report full evidence either
    short = DynamicGs({"Gs": Gs})
    _add(short, 0)
    _add(short, 0)
    assert short.count_activ_evidence(0, 3) / 3 < 1.0


if __name__ == "__main__":
    for name, fn in sorted(list(globals().items())):
        if name.startswith("test_"):
            fn()
            print(f"ok {name}")
    print("ok")
