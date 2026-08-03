import test from "node:test";
import assert from "node:assert/strict";

import { getHandRateConfig } from "../core/config.mjs";

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
