import test from "node:test";
import assert from "node:assert/strict";

import {
  applyTransform,
  composeTransforms,
  rawLeapPointToLeapworld,
} from "../core/transforms.mjs";

const closeTo = (actual, expected, epsilon = 1e-9) => {
  assert.equal(actual.length, expected.length);
  actual.forEach((value, index) => {
    assert.ok(
      Math.abs(value - expected[index]) <= epsilon,
      `component ${index}: expected ${expected[index]}, got ${value}`,
    );
  });
};

test("raw Leap millimetres use the toolbox leapworld convention", () => {
  closeTo(rawLeapPointToLeapworld([100, 200, 300]), [-0.3, -0.1, 0.2]);
});

test("a404 transform maps a known Leap point into base", () => {
  const baseFromLeapworld = {
    translation: [1.07, 0.4, 0.01],
    rotation: [0, 0, 1, 0],
  };

  closeTo(
    applyTransform(
      baseFromLeapworld,
      rawLeapPointToLeapworld([100, 200, 300]),
    ),
    [1.37, 0.5, 0.21],
  );
});

test("transform composition applies child transform before parent transform", () => {
  const baseFromCamera = {
    translation: [1, 0, 0],
    rotation: [0, 0, 0, 1],
  };
  const cameraFromHand = {
    translation: [0, 2, 0],
    rotation: [0, 0, Math.SQRT1_2, Math.SQRT1_2],
  };

  const baseFromHand = composeTransforms(baseFromCamera, cameraFromHand);
  closeTo(baseFromHand.translation, [1, 2, 0]);
  closeTo(applyTransform(baseFromHand, [1, 0, 0]), [1, 3, 0]);
});

test("transform composition preserves the order of two real rotations", () => {
  const rootFromMiddle = {
    translation: [0, 0, 0],
    rotation: [0, 0, Math.SQRT1_2, Math.SQRT1_2],
  };
  const middleFromChild = {
    translation: [0, 0, 0],
    rotation: [Math.SQRT1_2, 0, 0, Math.SQRT1_2],
  };

  const rootFromChild = composeTransforms(
    rootFromMiddle,
    middleFromChild,
  );
  closeTo(rootFromChild.rotation, [0.5, 0.5, 0.5, 0.5]);
  closeTo(applyTransform(rootFromChild, [0, 1, 0]), [0, 0, 1]);
});
