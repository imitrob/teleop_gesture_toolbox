import { FreshnessTracker } from "./core/freshness.mjs";

function openHand(offsetZ) {
  const palm = [0.58, 0, 0.35 + offsetZ];
  const fingerOffsets = [-0.054, -0.028, 0, 0.028, 0.052];
  const fingerLengths = [
    [0.035, 0.032, 0.027, 0.022],
    [0.042, 0.038, 0.031, 0.025],
    [0.046, 0.042, 0.034, 0.027],
    [0.043, 0.039, 0.032, 0.025],
    [0.037, 0.033, 0.027, 0.022],
  ];
  const bones = [];

  for (let finger = 0; finger < 5; finger += 1) {
    let start = [
      palm[0] - (finger === 0 ? 0.015 : 0),
      fingerOffsets[finger],
      palm[2],
    ];
    for (let bone = 0; bone < 4; bone += 1) {
      const thumbSpread = finger === 0 ? -0.45 : 0;
      const length = fingerLengths[finger][bone];
      const end = [
        start[0] + length * Math.cos(thumbSpread),
        start[1] + length * Math.sin(thumbSpread),
        start[2] + (finger === 0 ? -0.004 : 0.003),
      ];
      bones.push({
        valid: true,
        start: [...start],
        end: [...end],
        width: Math.max(0.009, 0.016 - bone * 0.0018),
      });
      start = end;
    }
  }

  return {
    handedness: "left",
    wrist: [0.49, 0, 0.35 + offsetZ],
    palm: {
      position: palm,
      basis: [
        [0, 1, 0],
        [0, 0, 1],
        [1, 0, 0],
      ],
      width: 0.085,
    },
    bones,
  };
}

export class DemoSceneSource {
  constructor({ handRate, onHands, onScene, onBeam, onSelection }) {
    this.handRate = handRate;
    this.onHands = onHands;
    this.onScene = onScene;
    this.onBeam = onBeam;
    this.onSelection = onSelection;
    this.freshness = new FreshnessTracker(500);
    this.timer = null;
    this.handArrivals = [];
    this.objectCount = 3;
  }

  start() {
    this.onScene([
      { name: "plastic_cube_1", position: [0.9, -0.28, 0.22] },
      { name: "robothon_box", position: [0.92, 0.08, 0.15] },
      { name: "robothon_peg", position: [1.12, -0.02, 0.3] },
    ]);
    this.onSelection("robothon_box");
    const publishHand = () => {
      const now = performance.now();
      const offsetZ = Math.sin(now / 700) * 0.012;
      this.onHands([openHand(offsetZ)]);
      this.onBeam([
        [0.72, -0.028, 0.36 + offsetZ],
        [1.35, -0.14, 0.2],
      ]);
      this.freshness.mark("hand", now);
      this.freshness.mark("beam", now);
      this.handArrivals.push(now);
      this.pruneHandArrivals(now);
    };
    publishHand();
    this.timer = window.setInterval(
      publishHand,
      this.handRate.throttleMs,
    );
  }

  stop() {
    window.clearInterval(this.timer);
    this.onSelection(null);
  }

  pruneHandArrivals(now) {
    const cutoff = now - 1000;
    while (this.handArrivals.length && this.handArrivals[0] < cutoff) {
      this.handArrivals.shift();
    }
  }

  getDiagnostics(now = performance.now()) {
    this.pruneHandArrivals(now);
    return {
      source: "Demo",
      connection: "Not used",
      transform: "Synthetic base",
      handRate: this.handArrivals.length,
      visibleHands: 1,
      handAge: this.freshness.age("hand", now),
      objectCount: this.objectCount,
      selectedObject: "robothon_box",
      candidateObject: "robothon_box",
      beamAge: this.freshness.age("beam", now),
    };
  }
}
