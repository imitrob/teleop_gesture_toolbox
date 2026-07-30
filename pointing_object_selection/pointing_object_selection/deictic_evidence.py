#!/usr/bin/env python
"""How much evidence there is that the user means one object.

The pointing ray flickers between neighbouring objects, and the last frames of
a point are the hand already moving away, so no single frame can be trusted to
name the object. An object is only accepted once it has stayed the closest one
for EVIDENCE frames in a row. Users point for as long as they like, so the one
that counts is the *last* object that reached the threshold; if none ever does,
nothing was selected.

The per-frame count is published in DeicticSolution.evidence, so a viewer can
show a selection growing (0 -> threshold) instead of a binary highlight.
"""

EVIDENCE = 20        # consecutive frames the same object must stay on top
RESET_AFTER = 0.5   # s without an update -> the hand left, start a new point


class DeicticEvidence():
    """Streak counter over a stream of per-frame closest-object names."""

    def __init__(self, threshold: int = EVIDENCE, reset_after: float = RESET_AFTER):
        self.threshold = threshold
        self.reset_after = reset_after
        self.reset()

    def reset(self):
        self.name = None        # object of the current streak
        self.evidence = 0       # how long the streak is
        self.selected = None    # last object that reached the threshold
        self._last_update = None

    def update(self, name: str | None, now: float) -> int:
        """Feed one frame's closest object, get its evidence so far."""
        if self._last_update is not None and (now - self._last_update) > self.reset_after:
            self.reset()  # gap in the stream: previous point ended
        self._last_update = now

        if not name:
            self.name, self.evidence = None, 0
            return 0

        self.evidence = self.evidence + 1 if name == self.name else 1
        self.name = name
        if self.evidence >= self.threshold:
            self.selected = name
        return self.evidence


def select(solutions, threshold: int = EVIDENCE):
    """The solution a whole pointing run meant: the last one whose evidence
    reached the threshold, or None when the user never settled on an object.

    `solutions` are DeicticSolutions in time order, each carrying the evidence
    counted by the publisher (see DeicticEvidence)."""
    for solution in reversed(list(solutions)):
        if solution is not None and getattr(solution, "evidence", 0) >= threshold:
            return solution
    return None


if __name__ == "__main__":
    class _Sol():  # stand-in for DeicticSolution: select() only reads .evidence
        def __init__(self, name, evidence):
            self.target_object_name, self.evidence = name, evidence

    counter = DeicticEvidence(threshold=3)
    stream = ["cube", "bowl", "cube", "cube", "cube", "cube", "box", "box"]
    got = [counter.update(name, now=t * 0.1) for t, name in enumerate(stream)]
    assert got == [1, 1, 1, 2, 3, 4, 1, 2], got
    assert counter.selected == "cube"  # box never got 3 frames in a row

    counter.update("box", now=10.0)  # a gap: new point, streak starts over
    assert counter.evidence == 1 and counter.selected is None

    counter = DeicticEvidence(threshold=3)
    assert counter.update(None, now=0.0) == 0 and counter.selected is None

    run = [_Sol("cube", e) for e in (1, 2, 3, 4)] + [_Sol("bowl", e) for e in (1, 2)]
    assert select(run, threshold=3).target_object_name == "cube"  # bowl too short
    assert select(run + [_Sol("bowl", 3)], threshold=3).target_object_name == "bowl"
    assert select([_Sol("cube", 1)], threshold=3) is None
    assert select([], threshold=3) is None
    print("deictic evidence checks ok")
