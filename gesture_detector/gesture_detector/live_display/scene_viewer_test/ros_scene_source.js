import { FreshnessTracker } from "./core/freshness.mjs";
import {
  getPendingSelection,
  getSelectedObjectName,
  getSelectionStrength,
} from "./core/deictic.mjs";
import { TfGraph } from "./core/tf_graph.mjs";
import {
  applyTransform,
  rawLeapPointToLeapworld,
  rawLeapVectorToLeapworld,
  rotateVector,
} from "./core/transforms.mjs";

const ROSBRIDGE_URL = "ws://127.0.0.1:9090";
const STALE_TIMEOUT_MS = 500;
const RECONNECT_MS = 1000;
const TOPICS = Object.freeze({
  hand: "/teleop_gesture_toolbox/hand_frame",
  scene: "/scene",
  beam: "/teleop_gesture_toolbox/line_marker",
  selection: "/teleop_gesture_toolbox/deictic_solution",
  pendingSelection: "/teleop_gesture_toolbox/pending_object_selection",
  tfStatic: "/tf_static",
});

function pointFromRos(point) {
  return [point.x, point.y, point.z];
}

function normalizeVector(vector) {
  const length = Math.hypot(...vector);
  return length > 1e-9
    ? vector.map((component) => component / length)
    : [0, 0, 0];
}

export class RosSceneSource {
  constructor({ handRate, onHands, onScene, onBeam, onSelection }) {
    this.handRate = handRate;
    this.onHands = onHands;
    this.onScene = onScene;
    this.onBeam = onBeam;
    this.onSelection = onSelection;

    this.tfGraph = new TfGraph();
    this.freshness = new FreshnessTracker(STALE_TIMEOUT_MS);
    this.connection = "disconnected";
    this.baseFromLeapworld = null;
    this.latestHandFrame = null;
    this.visibleHands = 0;
    this.objectCount = 0;
    this.selectedObject = null;
    this.candidateObject = null;
    this.candidateStrength = 0;
    this.sceneSignature = "";
    this.handArrivals = [];
    this.socket = null;
    this.reconnectTimer = null;
    this.stopped = true;
    this.handsShown = false;
    this.beamShown = false;
    this.handExpiryTimer = null;
    this.beamExpiryTimer = null;
    this.selectionExpiryTimer = null;
  }

  start() {
    this.stopped = false;
    this.connect();
  }

  stop() {
    this.stopped = true;
    window.clearTimeout(this.reconnectTimer);
    window.clearTimeout(this.handExpiryTimer);
    window.clearTimeout(this.beamExpiryTimer);
    window.clearTimeout(this.selectionExpiryTimer);
    if (this.socket) {
      const socket = this.socket;
      this.socket = null;
      socket.close();
    }
  }

  connect() {
    if (this.stopped) {
      return;
    }
    this.connection = "connecting";
    const socket = new WebSocket(ROSBRIDGE_URL);
    this.socket = socket;

    socket.addEventListener("open", () => {
      if (this.stopped || socket !== this.socket) {
        socket.close();
        return;
      }
      this.connection = "connected";
      const subscriptions = [
        {
          id: "scene_viewer_hand",
          topic: TOPICS.hand,
          type: "gesture_msgs/Frame",
          throttle_rate: this.handRate.throttleMs,
        },
        {
          id: "scene_viewer_scene",
          topic: TOPICS.scene,
          type: "scene_msgs/Scene",
          throttle_rate: 0,
        },
        {
          id: "scene_viewer_beam",
          topic: TOPICS.beam,
          type: "visualization_msgs/MarkerArray",
          throttle_rate: 50,
        },
        {
          id: "scene_viewer_selection",
          topic: TOPICS.selection,
          type: "gesture_msgs/DeicticSolution",
          throttle_rate: 50,
        },
        {
          id: "scene_viewer_pending_selection",
          topic: TOPICS.pendingSelection,
          type: "std_msgs/String",
          throttle_rate: 0,
        },
        {
          id: "scene_viewer_tf_static",
          topic: TOPICS.tfStatic,
          type: "tf2_msgs/TFMessage",
          throttle_rate: 0,
        },
      ];
      for (const subscription of subscriptions) {
        socket.send(JSON.stringify({
          op: "subscribe",
          queue_length: 1,
          ...subscription,
        }));
      }
    });

    socket.addEventListener("message", (event) => {
      if (this.stopped || socket !== this.socket) {
        return;
      }
      try {
        this.handlePacket(JSON.parse(event.data));
      } catch (error) {
        console.error("Unable to process rosbridge message", error);
      }
    });

    socket.addEventListener("error", () => {
      socket.close();
    });

    socket.addEventListener("close", () => {
      this.handleDisconnect(socket);
    });
  }

  handleDisconnect(socket) {
    if (socket !== this.socket) {
      return;
    }
    this.socket = null;
    if (this.stopped) {
      return;
    }
    this.connection = "disconnected";
    this.tfGraph.clear();
    this.baseFromLeapworld = null;
    this.latestHandFrame = null;
    this.visibleHands = 0;
    this.freshness.clear("hand");
    this.freshness.clear("beam");
    this.freshness.clear("selection");
    window.clearTimeout(this.handExpiryTimer);
    window.clearTimeout(this.beamExpiryTimer);
    window.clearTimeout(this.selectionExpiryTimer);
    this.handsShown = false;
    this.beamShown = false;
    this.onHands([]);
    this.onBeam(null);
    this.selectedObject = null;
    this.candidateObject = null;
    this.candidateStrength = 0;
    this.onSelection(null);
    this.reconnectTimer = window.setTimeout(
      () => this.connect(),
      RECONNECT_MS,
    );
  }

  handlePacket(packet) {
    if (packet.op !== "publish") {
      return;
    }
    const now = performance.now();
    if (packet.topic === TOPICS.tfStatic) {
      this.tfGraph.updateMessage(packet.msg);
      this.baseFromLeapworld = this.tfGraph.lookup("base", "leapworld");
      if (
        this.baseFromLeapworld &&
        this.latestHandFrame &&
        this.freshness.isFresh("hand", now)
      ) {
        this.publishHands(this.latestHandFrame);
      }
      return;
    }

    if (packet.topic === TOPICS.hand) {
      this.latestHandFrame = packet.msg;
      this.freshness.mark("hand", now);
      this.handArrivals.push(now);
      this.pruneHandArrivals(now);
      this.scheduleHandExpiry();
      if (this.baseFromLeapworld) {
        this.publishHands(packet.msg);
      }
      return;
    }

    if (packet.topic === TOPICS.scene) {
      const objects = (packet.msg.objects || []).map((object) => ({
        name: object.name,
        position: pointFromRos(object.pose.position),
      }));
      this.objectCount = objects.length;
      const signature = JSON.stringify(objects);
      if (signature !== this.sceneSignature) {
        this.sceneSignature = signature;
        this.onScene(objects);
      }
      return;
    }

    if (packet.topic === TOPICS.beam) {
      const marker = (packet.msg.markers || []).find(
        (candidate) =>
          candidate.action !== 2 &&
          candidate.action !== 3 &&
          candidate.points?.length >= 2,
      );
      if (!marker) {
        this.freshness.clear("beam");
        window.clearTimeout(this.beamExpiryTimer);
        this.beamShown = false;
        this.onBeam(null);
        return;
      }
      this.freshness.mark("beam", now);
      this.scheduleBeamExpiry();
      this.beamShown = true;
      this.onBeam(marker.points.map(pointFromRos));
      return;
    }

    if (packet.topic === TOPICS.pendingSelection) {
      // The sentence maker's decision for the pointing in progress.
      this.selectedObject = getPendingSelection(packet.msg);
      this.emitSelection();
      return;
    }

    if (packet.topic === TOPICS.selection) {
      // Where the ray is right now, which flickers between neighbours and is
      // not a selection on its own.
      const candidate = getSelectedObjectName(packet.msg);
      window.clearTimeout(this.selectionExpiryTimer);
      if (!candidate) {
        this.freshness.clear("selection");
        this.candidateObject = null;
        this.candidateStrength = 0;
        this.emitSelection();
        return;
      }
      this.freshness.mark("selection", now);
      this.candidateObject = candidate;
      // Strength is how much evidence the candidate has gathered so far, so the
      // highlight grows while the user keeps pointing at the same thing.
      this.candidateStrength = getSelectionStrength(packet.msg);
      this.emitSelection();
      this.scheduleSelectionExpiry();
    }
  }

  publishHands(frame) {
    const hands = [];
    for (const [handedness, hand] of [
      ["left", frame.l],
      ["right", frame.r],
    ]) {
      if (!hand?.visible) {
        continue;
      }
      hands.push(this.normalizeHand(handedness, hand));
    }
    this.visibleHands = hands.length;
    this.handsShown = hands.length > 0;
    this.onHands(hands);
  }

  normalizeHand(handedness, hand) {
    const transformPoint = (point) =>
      applyTransform(
        this.baseFromLeapworld,
        rawLeapPointToLeapworld(point),
      );
    const transformAxis = (axis) =>
      normalizeVector(
        rotateVector(
          this.baseFromLeapworld.rotation,
          rawLeapVectorToLeapworld(axis),
        ),
      );
    const basis = Array.from({ length: 3 }, (_, index) =>
      transformAxis(hand.basis.slice(index * 3, index * 3 + 3))
    );

    return {
      handedness,
      wrist: transformPoint(hand.wrist_position),
      palm: {
        position: transformPoint(hand.palm_position),
        basis,
        width: Math.max(0, hand.palm_width / 1000),
      },
      bones: (hand.finger_bones || []).map((bone) => ({
        valid: Boolean(bone.is_valid),
        start: transformPoint(bone.prev_joint),
        end: transformPoint(bone.next_joint),
        width: Math.max(0, bone.width / 1000),
      })),
    };
  }

  scheduleHandExpiry() {
    window.clearTimeout(this.handExpiryTimer);
    this.handExpiryTimer = window.setTimeout(() => {
      if (!this.freshness.isFresh("hand") && this.handsShown) {
        this.handsShown = false;
        this.visibleHands = 0;
        this.onHands([]);
      }
    }, STALE_TIMEOUT_MS);
  }

  scheduleBeamExpiry() {
    window.clearTimeout(this.beamExpiryTimer);
    this.beamExpiryTimer = window.setTimeout(() => {
      if (!this.freshness.isFresh("beam") && this.beamShown) {
        this.beamShown = false;
        this.onBeam(null);
      }
    }, STALE_TIMEOUT_MS);
  }

  emitSelection() {
    this.onSelection(
      this.candidateObject || this.selectedObject
        ? {
          name: this.candidateObject,
          strength: this.candidateStrength,
          confirmed: this.selectedObject,
        }
        : null,
    );
  }

  scheduleSelectionExpiry() {
    window.clearTimeout(this.selectionExpiryTimer);
    this.selectionExpiryTimer = window.setTimeout(() => {
      if (!this.freshness.isFresh("selection")) {
        this.candidateObject = null;
        this.candidateStrength = 0;
        this.emitSelection();
      }
    }, STALE_TIMEOUT_MS);
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
      source: "ROS",
      connection: this.connection,
      transform: this.baseFromLeapworld
        ? "base ← leapworld"
        : "Waiting…",
      handRate: this.handArrivals.length,
      visibleHands: this.visibleHands,
      handAge: this.freshness.age("hand", now),
      objectCount: this.objectCount,
      selectedObject: this.selectedObject,
      candidateObject: this.candidateObject,
      beamAge: this.freshness.age("beam", now),
    };
  }
}
