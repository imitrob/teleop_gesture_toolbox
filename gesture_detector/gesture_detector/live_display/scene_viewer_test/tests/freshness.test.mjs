import test from "node:test";
import assert from "node:assert/strict";

import { FreshnessTracker } from "../core/freshness.mjs";

test("hand and beam data expire after 500 milliseconds", () => {
  const freshness = new FreshnessTracker(500);
  freshness.mark("hand", 1000);
  freshness.mark("beam", 1100);

  assert.equal(freshness.isFresh("hand", 1499), true);
  assert.equal(freshness.isFresh("hand", 1500), false);
  assert.equal(freshness.isFresh("beam", 1500), true);
  assert.equal(freshness.age("beam", 1500), 400);
});

test("never-seen and cleared channels are stale", () => {
  const freshness = new FreshnessTracker(500);
  assert.equal(freshness.isFresh("hand", 0), false);
  assert.equal(freshness.age("hand", 0), Infinity);

  freshness.mark("hand", 100);
  freshness.clear("hand");
  assert.equal(freshness.isFresh("hand", 101), false);
});
