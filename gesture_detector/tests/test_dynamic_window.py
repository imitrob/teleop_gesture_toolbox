import json

import numpy as np

import gesture_detector
from gesture_detector.gesture_classification.gestures_lib import DynamicGs, GestureMorphClassStamped, dynamic_sample_indices
from gesture_detector.gesture_classification.timewarp_lib import fastdtw_
from gesture_detector.utils.utils import transform_leap_to_leapdynamicdetector
from gesture_msgs.msg import DetectionSolution

FPS = 100.0
DETECTOR_RATE = 10.0
WINDOW = 1.5
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


def _classify_live(traj, period, radius, duration=6.0, time_samples=TIME_SAMPLES):
    """Replays the /teleop_gesture_toolbox/dynamic_detection_observations composition of send_g_data."""
    sampler, Gs = _model()
    points = [traj(t, period, radius) for t in np.arange(0, duration, 1 / FPS)]
    n = int(WINDOW * FPS)

    labels = []
    for tick in np.arange(WINDOW, duration, 1 / DETECTOR_RATE):
        window = points[int(tick * FPS) - n : int(tick * FPS)]
        composition = np.array([transform_leap_to_leapdynamicdetector(window[i]) for i in dynamic_sample_indices(n, time_samples)])
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


def test_sample_indices_span_and_count():
    indices = dynamic_sample_indices(150, TIME_SAMPLES)
    assert len(indices) == TIME_SAMPLES
    assert indices == sorted(indices)
    assert indices[-1] == -1
    assert indices[0] == -int(150 * (TIME_SAMPLES - 1) / TIME_SAMPLES)


def test_circle_becomes_top_1():
    labels, _ = _classify_live(_circle, period=1.2, radius=0.05)
    assert _longest_run(labels, "circle") >= ACTIVATE_LENGTH_DYNAMIC


def test_five_samples_cannot_see_a_circle():
    labels, _ = _classify_live(_circle, period=1.2, radius=0.05, time_samples=5)
    assert "circle" not in labels


def test_circle_activates_only_with_dynamic_activate_length():
    labels, Gs = _classify_live(_circle, period=1.2, radius=0.05)
    assert _evidence(labels, Gs, "circle", ACTIVATE_LENGTH_DYNAMIC) >= 1.0
    assert _evidence(labels, Gs, "circle", 10) < 1.0


def test_straight_swipe_unaffected():
    labels, _ = _classify_live(_swipe_up, period=1.0, radius=0.03)
    assert set(labels) == {"swipe_up"}


def test_left_swipe_recognized():
    labels, _ = _classify_live(_swipe_left, period=1.0, radius=0.03)
    assert set(labels) == {"swipe_left"}


if __name__ == "__main__":
    test_sample_indices_span_and_count()
    test_circle_becomes_top_1()
    test_five_samples_cannot_see_a_circle()
    test_circle_activates_only_with_dynamic_activate_length()
    test_straight_swipe_unaffected()
    test_left_swipe_recognized()
    print("ok")
