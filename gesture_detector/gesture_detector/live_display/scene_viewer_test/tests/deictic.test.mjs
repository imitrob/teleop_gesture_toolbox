import assert from "node:assert/strict";
import test from "node:test";

import {
  getSelectedObjectName,
  getSelectionStrength,
} from "../core/deictic.mjs";

test("uses the selected object name carried by a deictic solution", () => {
  assert.equal(
    getSelectedObjectName({
      target_object_id: 1,
      target_object_name: "robothon_box",
      object_names: ["plastic_cube_1", "robothon_box"],
    }),
    "robothon_box",
  );
});

test("falls back to target id when the name is unavailable", () => {
  assert.equal(
    getSelectedObjectName({
      target_object_id: 0,
      target_object_name: "",
      object_names: ["plastic_cube_1"],
    }),
    "plastic_cube_1",
  );
});

test("rejects incomplete deictic solutions", () => {
  assert.equal(getSelectedObjectName(null), null);
  assert.equal(
    getSelectedObjectName({
      target_object_id: 4,
      target_object_name: "",
      object_names: ["plastic_cube_1"],
    }),
    null,
  );
});

test("scales the highlight by how much evidence the object has", () => {
  assert.equal(getSelectionStrength({ evidence: 0, evidence_threshold: 5 }), 0);
  assert.equal(getSelectionStrength({ evidence: 1, evidence_threshold: 5 }), 0.2);
  assert.equal(getSelectionStrength({ evidence: 5, evidence_threshold: 5 }), 1);
  assert.equal(getSelectionStrength({ evidence: 9, evidence_threshold: 5 }), 1);
});

test("highlights fully when the publisher counts no evidence", () => {
  assert.equal(getSelectionStrength({ target_object_name: "cube" }), 1);
  assert.equal(getSelectionStrength(null), 1);
});
