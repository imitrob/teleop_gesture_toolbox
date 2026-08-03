import {
  composeTransforms,
  IDENTITY_TRANSFORM,
  invertTransform,
  transformFromRos,
} from "./transforms.mjs";

function normalizeFrame(frame) {
  return String(frame || "").replace(/^\/+/, "");
}

export class TfGraph {
  constructor() {
    this.transformsByChild = new Map();
  }

  clear() {
    this.transformsByChild.clear();
  }

  updateMessage(message) {
    for (const stamped of message.transforms || []) {
      const parent = normalizeFrame(stamped.header?.frame_id);
      const child = normalizeFrame(stamped.child_frame_id);
      if (!parent || !child || !stamped.transform) {
        continue;
      }
      this.transformsByChild.set(child, {
        parent,
        child,
        parentFromChild: transformFromRos(stamped.transform),
      });
    }
  }

  lookup(targetFrame, sourceFrame) {
    const target = normalizeFrame(targetFrame);
    const source = normalizeFrame(sourceFrame);
    if (!target || !source) {
      return null;
    }
    if (target === source) {
      return {
        translation: [...IDENTITY_TRANSFORM.translation],
        rotation: [...IDENTITY_TRANSFORM.rotation],
      };
    }

    const adjacency = new Map();
    const addEdge = (from, to, toFromFrom) => {
      if (!adjacency.has(from)) {
        adjacency.set(from, []);
      }
      adjacency.get(from).push({ frame: to, transform: toFromFrom });
    };

    for (const edge of this.transformsByChild.values()) {
      addEdge(edge.child, edge.parent, edge.parentFromChild);
      addEdge(
        edge.parent,
        edge.child,
        invertTransform(edge.parentFromChild),
      );
    }

    const queue = [
      {
        frame: source,
        frameFromSource: {
          translation: [...IDENTITY_TRANSFORM.translation],
          rotation: [...IDENTITY_TRANSFORM.rotation],
        },
      },
    ];
    const visited = new Set([source]);

    while (queue.length > 0) {
      const current = queue.shift();
      for (const edge of adjacency.get(current.frame) || []) {
        if (visited.has(edge.frame)) {
          continue;
        }
        const neighborFromSource = composeTransforms(
          edge.transform,
          current.frameFromSource,
        );
        if (edge.frame === target) {
          return neighborFromSource;
        }
        visited.add(edge.frame);
        queue.push({
          frame: edge.frame,
          frameFromSource: neighborFromSource,
        });
      }
    }

    return null;
  }
}
