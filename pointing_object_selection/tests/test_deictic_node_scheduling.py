from pointing_object_selection import deictic_node


def test_main_spins_continuously_with_a_deictic_timer(monkeypatch):
    events = []

    class FakeDeicticNode:
        def __init__(self, hand):
            events.append(("node", hand))

        def step(self):
            events.append(("step",))

        def create_timer(self, period, callback):
            events.append(("timer", period, callback))
            return object()

        def destroy_node(self):
            events.append(("destroy",))

    monkeypatch.setattr(deictic_node, "DeicticLibRos", FakeDeicticNode)
    monkeypatch.setattr(
        deictic_node.rclpy,
        "init",
        lambda: events.append(("init",)),
    )
    monkeypatch.setattr(
        deictic_node.rclpy,
        "spin",
        lambda node: events.append(("spin", node)),
    )
    monkeypatch.setattr(
        deictic_node.rclpy,
        "spin_once",
        lambda _node: (_ for _ in ()).throw(
            AssertionError("manual spin_once starves subscription callbacks")
        ),
    )
    monkeypatch.setattr(
        deictic_node.rclpy,
        "shutdown",
        lambda: events.append(("shutdown",)),
    )

    deictic_node.main({"hand": "lr", "frequency": 10})

    timer_event = next(event for event in events if event[0] == "timer")
    assert timer_event[1] == 0.1
    assert getattr(timer_event[2], "__self__", None).__class__ is FakeDeicticNode
    assert [event[0] for event in events] == [
        "init",
        "node",
        "timer",
        "spin",
        "destroy",
        "shutdown",
    ]
