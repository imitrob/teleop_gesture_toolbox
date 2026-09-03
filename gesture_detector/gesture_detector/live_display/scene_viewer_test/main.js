import { getHandRateConfig, isDemoMode } from "./core/config.mjs";
import { DemoSceneSource } from "./demo_scene_source.js";
import { RosSceneSource } from "./ros_scene_source.js";
import { SceneViewer } from "./scene_viewer.js";

const handRate = getHandRateConfig(window.location.search);
const demoMode = isDemoMode(window.location.search);
const viewer = new SceneViewer(document.getElementById("viewer"));

const callbacks = {
  onHands: (hands) => viewer.setHands(hands),
  onScene: (objects) => viewer.setSceneObjects(objects),
  onBeam: (points) => viewer.setBeam(points),
  onSelection: (selection) => viewer.setSelectedObject(selection),
};
const source = demoMode
  ? new DemoSceneSource({ handRate, ...callbacks })
  : new RosSceneSource({ handRate, ...callbacks });

document.getElementById("resetCamera").addEventListener(
  "click",
  () => viewer.resetCamera(),
);
const diagnosticsPanel = document.getElementById("diagnostics");
const toggleStats = document.getElementById("toggleStats");
toggleStats.addEventListener("click", () => {
  const show = diagnosticsPanel.hidden;
  diagnosticsPanel.hidden = !show;
  toggleStats.setAttribute("aria-expanded", String(show));
});

const warning = document.getElementById("warning");
if (handRate.warning) {
  warning.textContent = handRate.warning;
  warning.hidden = false;
}

const fields = {
  source: document.getElementById("diagSource"),
  connection: document.getElementById("diagConnection"),
  transform: document.getElementById("diagTransform"),
  cap: document.getElementById("diagCap"),
  handRate: document.getElementById("diagHandRate"),
  renderFps: document.getElementById("diagRenderFps"),
  hands: document.getElementById("diagHands"),
  handAge: document.getElementById("diagHandAge"),
  objects: document.getElementById("diagObjects"),
  candidate: document.getElementById("diagCandidate"),
  selected: document.getElementById("diagSelected"),
  beamAge: document.getElementById("diagBeamAge"),
};

function formatAge(age, channel) {
  if (!Number.isFinite(age)) {
    return "Waiting…";
  }
  if (age >= 500) {
    return `${channel} data stale (${Math.round(age)} ms)`;
  }
  return `${Math.round(age)} ms`;
}

function updateDiagnostics() {
  const now = performance.now();
  const diagnostics = source.getDiagnostics(now);
  fields.source.textContent = diagnostics.source;
  fields.connection.textContent = diagnostics.connection;
  fields.transform.textContent = diagnostics.transform;
  fields.cap.textContent = `${handRate.hz} Hz`;
  fields.handRate.textContent = `${diagnostics.handRate} Hz`;
  fields.renderFps.textContent = `${viewer.getRenderFps(now)} FPS`;
  fields.hands.textContent = String(diagnostics.visibleHands);
  fields.handAge.textContent = formatAge(diagnostics.handAge, "Hand");
  fields.objects.textContent = String(diagnostics.objectCount);
  fields.candidate.textContent = diagnostics.candidateObject || "—";
  fields.selected.textContent = diagnostics.selectedObject || "—";
  fields.beamAge.textContent = formatAge(diagnostics.beamAge, "Beam");
}

source.start();
updateDiagnostics();
const diagnosticsTimer = window.setInterval(updateDiagnostics, 250);

window.addEventListener("beforeunload", () => {
  window.clearInterval(diagnosticsTimer);
  source.stop();
});
