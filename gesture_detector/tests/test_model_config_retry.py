from types import SimpleNamespace

from gesture_detector.gesture_classification import gestures_lib
from gesture_detector.gesture_classification.gestures_lib import GestureDataDetection


class FakeFuture:
    def __init__(self, gestures=None):
        self._result = (
            SimpleNamespace(gestures=gestures)
            if gestures is not None
            else None
        )

    def done(self):
        return self._result is not None

    def result(self):
        return self._result


class LostThenSuccessfulClient:
    def __init__(self):
        self.futures = [FakeFuture(), FakeFuture(["grab", "point"])]
        self.calls = 0
        self.removed = []

    def call_async(self, _request):
        future = self.futures[self.calls]
        self.calls += 1
        return future

    def remove_pending_request(self, future):
        self.removed.append(future)


def test_model_config_request_retries_after_a_lost_response(monkeypatch):
    client = LostThenSuccessfulClient()
    detector = object.__new__(GestureDataDetection)
    detector.get_static_model_config = client

    spin_timeouts = []
    monkeypatch.setattr(gestures_lib.rclpy, "ok", lambda: True)
    monkeypatch.setattr(
        gestures_lib.rclpy,
        "spin_until_future_complete",
        lambda _node, _future, timeout_sec: spin_timeouts.append(timeout_sec),
    )

    gestures = detector.call_static_model_config_service()

    assert gestures == ["grab", "point"]
    assert client.calls == 2
    assert client.removed == [client.futures[0]]
    assert all(timeout is not None for timeout in spin_timeouts)
