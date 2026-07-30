import test from "node:test";
import assert from "node:assert/strict";

import { TfGraph } from "../core/tf_graph.mjs";
import { applyTransform } from "../core/transforms.mjs";

function tf(parent, child, translation, rotation) {
  return {
    header: { frame_id: parent },
    child_frame_id: child,
    transform: {
      translation: {
        x: translation[0],
        y: translation[1],
        z: translation[2],
      },
      rotation: {
        x: rotation[0],
        y: rotation[1],
        z: rotation[2],
        w: rotation[3],
      },
    },
  };
}

const closeTo = (actual, expected, epsilon = 1e-6) => {
  actual.forEach((value, index) => {
    assert.ok(Math.abs(value - expected[index]) <= epsilon);
  });
};

test("TF graph resolves the direct a404 leapworld to base transform", () => {
  const graph = new TfGraph();
  graph.updateMessage({
    transforms: [tf("base", "leapworld", [1.07, 0.4, 0.01], [0, 0, 1, 0])],
  });

  const baseFromLeapworld = graph.lookup("base", "leapworld");
  closeTo(baseFromLeapworld.translation, [1.07, 0.4, 0.01]);
  closeTo(applyTransform(baseFromLeapworld, [1, 0, 0]), [0.07, 0.4, 0.01]);
});

test("TF graph composes the multi-hop b300 chain", () => {
  const graph = new TfGraph();
  graph.updateMessage({
    transforms: [
      tf(
        "xtion_rgb_optical_frame",
        "leapworld",
        [0, 0, 1],
        [Math.SQRT1_2, 0, Math.SQRT1_2, 0],
      ),
      tf("base_footprint", "xtion_rgb_optical_frame", [0, 0, 0], [0, 0, 0, 1]),
      tf("base", "base_footprint", [0, 0, 0], [0, 0, 0, 1]),
    ],
  });

  const baseFromLeapworld = graph.lookup("base", "leapworld");
  closeTo(baseFromLeapworld.translation, [0, 0, 1]);
  closeTo(
    baseFromLeapworld.rotation,
    [Math.SQRT1_2, 0, Math.SQRT1_2, 0],
  );
});

test("TF graph returns null when no transform path exists", () => {
  const graph = new TfGraph();
  graph.updateMessage({
    transforms: [tf("base", "camera", [0, 0, 0], [0, 0, 0, 1])],
  });

  assert.equal(graph.lookup("base", "leapworld"), null);
});
