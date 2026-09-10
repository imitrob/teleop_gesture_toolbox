import collections
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest
from gesture_msgs.msg import DetectionSolution
from launch import LaunchContext

from gesture_detector.gesture_classification import gestures_lib
from gesture_detector.gesture_classification.gestures_lib import (
    DetectionFreshness,
    GestureDataDetection,
    TemplateGs,
)


def _record(stamp, received_at=1.0):
    return SimpleNamespace(
        header=SimpleNamespace(stamp=stamp, received_at=received_at))


def _solution(sensor_seq=1):
    solution = DetectionSolution()
    solution.header.frame_id = "r"
    solution.header.stamp.sec = 1
    solution.sensor_seq = sensor_seq
    solution.probabilities.data = [1.0]
    return solution


def _detector(replay_mode):
    detector = object.__new__(GestureDataDetection)
    detector.replay_mode = replay_mode
    detector.freshness = DetectionFreshness(replay_mode=replay_mode)
    detector.hand_frames = collections.deque([SimpleNamespace(seq=250)])
    detector.r = SimpleNamespace(
        static=TemplateGs({"Gs": ["grab"]}, freshness=detector.freshness))
    detector.activation_postprocessing = lambda *_args, **_kwargs: None
    return detector


def test_replay_mode_keeps_old_records_relevant(monkeypatch):
    monkeypatch.setattr(gestures_lib.time, "time", lambda: 1000.0)
    monkeypatch.setattr(gestures_lib.time, "monotonic", lambda: 10.0)
    old_record = _record(stamp=1.0, received_at=9.5)

    live = TemplateGs({"Gs": []})
    live.data_queue.append(old_record)
    replay = TemplateGs(
        {"Gs": []}, freshness=DetectionFreshness(replay_mode=True))
    replay.data_queue.append(old_record)

    assert live.relevant() is None
    assert replay.relevant() is old_record


def test_replay_mode_accepts_solution_with_old_sensor_sequence():
    live = _detector(replay_mode=False)
    replay = _detector(replay_mode=True)
    solution = _solution()

    GestureDataDetection.new_record(live, solution)
    GestureDataDetection.new_record(replay, solution)

    assert live.r.static.n == 0
    assert replay.r.static.n == 1


def test_replay_mode_ignores_header_age_when_loading_recent_records(monkeypatch):
    monkeypatch.setattr(gestures_lib.time, "time", lambda: 1000.0)
    monkeypatch.setattr(gestures_lib.time, "monotonic", lambda: 10.0)
    freshness = DetectionFreshness(replay_mode=True)
    records = TemplateGs({"Gs": []}, freshness=freshness)
    records.data_queue.extend([
        _record(1.0, received_at=9.0),
        _record(2.0, received_at=9.5),
    ])
    detector = object.__new__(GestureDataDetection)
    detector.replay_mode = True
    detector.freshness = freshness
    detector.r = SimpleNamespace(static=records)

    assert detector.relevant(hand="r", type="static") == [records[-1]]


def test_replay_record_expires_by_receipt_time(monkeypatch):
    monkeypatch.setattr(gestures_lib.time, "monotonic", lambda: 10.0)
    records = TemplateGs(
        {"Gs": []}, freshness=DetectionFreshness(replay_mode=True))
    records.data_queue.append(_record(stamp=1.0, received_at=8.0))

    assert records.relevant(last_secs=1.0) is None


def _load_detector_launch():
    launch_path = Path(__file__).parents[1] / "launch" / "gesture_detect_launch.py"
    spec = importlib.util.spec_from_file_location("gesture_detect_launch", launch_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_bag_input_does_not_start_a_hardware_publisher(monkeypatch, tmp_path):
    monkeypatch.setenv("ROS_LOG_DIR", str(tmp_path))
    module = _load_detector_launch()
    context = LaunchContext()
    context.launch_configurations["sensor"] = "bag"

    assert module.generate_nodes(context) == []


def test_unknown_input_source_is_rejected(monkeypatch, tmp_path):
    monkeypatch.setenv("ROS_LOG_DIR", str(tmp_path))
    module = _load_detector_launch()
    context = LaunchContext()
    context.launch_configurations["sensor"] = "camera42"

    with pytest.raises(ValueError, match="camera42"):
        module.generate_nodes(context)
