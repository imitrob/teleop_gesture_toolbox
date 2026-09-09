import test from "node:test";
import assert from "node:assert/strict";

import { getHandRateConfig, supersamplePixelRatio } from "../core/config.mjs";

test("hand rate defaults to 30 Hz", () => {
  assert.deepEqual(getHandRateConfig(""), {
    hz: 30,
    throttleMs: 34,
    warning: "",
  });
});

test("hand rate accepts a safe URL override", () => {
  assert.deepEqual(getHandRateConfig("?hand_hz=60"), {
    hz: 60,
    throttleMs: 17,
    warning: "",
  });
});

test("hand rate rejects values outside 1 to 120 Hz", () => {
  assert.deepEqual(getHandRateConfig("?hand_hz=240"), {
    hz: 30,
    throttleMs: 34,
    warning: "Invalid hand_hz=240; using 30 Hz.",
  });
});

test("pixel ratio supersamples a card-sized canvas", () => {
  assert.equal(supersamplePixelRatio(430, 430, 1), 2);
  assert.equal(supersamplePixelRatio(430, 430, 4), 3);
});

test("pixel ratio stops supersampling a maximized canvas", () => {
  assert.equal(supersamplePixelRatio(1900, 950, 1), 1);
  assert.equal(supersamplePixelRatio(1900, 950, 3), 1.5);
});

test("pixel ratio survives a missing devicePixelRatio", () => {
  assert.equal(supersamplePixelRatio(430, 430, undefined), 2);
  assert.equal(supersamplePixelRatio(1900, 950, 0), 1);
});
