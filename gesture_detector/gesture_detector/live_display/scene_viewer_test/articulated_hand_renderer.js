import * as THREE from "three";

const LEFT_COLOR = 0x26c6da;
const RIGHT_COLOR = 0xec407a;
const SMOOTHING_SECONDS = 0.045;

function clonePoint(point) {
  return point ? [...point] : [0, 0, 0];
}

function cloneHand(hand) {
  return {
    ...hand,
    wrist: clonePoint(hand.wrist),
    palm: {
      ...hand.palm,
      position: clonePoint(hand.palm.position),
      basis: hand.palm.basis.map(clonePoint),
    },
    bones: hand.bones.map((bone) => ({
      ...bone,
      start: clonePoint(bone.start),
      end: clonePoint(bone.end),
    })),
  };
}

class HandVisual {
  constructor(scene, handedness, color) {
    this.handedness = handedness;
    this.group = new THREE.Group();
    this.group.name = `${handedness}-articulated-hand`;
    scene.add(this.group);

    const material = new THREE.MeshStandardMaterial({
      color,
      roughness: 0.58,
      metalness: 0.02,
    });
    const cylinderGeometry = new THREE.CylinderGeometry(1, 1, 1, 12, 1);
    const jointGeometry = new THREE.SphereGeometry(1, 12, 8);
    this.bones = Array.from({ length: 20 }, () => {
      const segment = new THREE.Mesh(cylinderGeometry, material);
      const joint = new THREE.Mesh(jointGeometry, material);
      segment.visible = false;
      joint.visible = false;
      this.group.add(segment, joint);
      return { segment, joint };
    });

    this.palm = new THREE.Mesh(new THREE.BoxGeometry(1, 1, 1), material);
    this.wrist = new THREE.Mesh(jointGeometry, material);
    this.group.add(this.palm, this.wrist);
    this.group.visible = false;

    this.target = null;
    this.current = null;
    this.yAxis = new THREE.Vector3(0, 1, 0);
  }

  setTarget(hand) {
    if (!hand) {
      this.target = null;
      this.current = null;
      this.group.visible = false;
      return;
    }
    this.target = cloneHand(hand);
    if (!this.current) {
      this.current = cloneHand(hand);
    }
    this.group.visible = true;
  }

  update(deltaSeconds) {
    if (!this.target || !this.current) {
      return;
    }

    const alpha = 1 - Math.exp(-deltaSeconds / SMOOTHING_SECONDS);
    const lerpPoint = (current, target) => {
      for (let index = 0; index < 3; index += 1) {
        current[index] += (target[index] - current[index]) * alpha;
      }
    };

    lerpPoint(this.current.wrist, this.target.wrist);
    lerpPoint(this.current.palm.position, this.target.palm.position);
    this.current.palm.width +=
      (this.target.palm.width - this.current.palm.width) * alpha;
    for (let index = 0; index < 3; index += 1) {
      lerpPoint(
        this.current.palm.basis[index],
        this.target.palm.basis[index],
      );
    }

    const palmPosition = new THREE.Vector3(...this.current.palm.position);
    this.palm.position.copy(palmPosition);
    const basis = this.current.palm.basis.map((axis) =>
      new THREE.Vector3(...axis).normalize()
    );
    const basisMatrix = new THREE.Matrix4().makeBasis(
      basis[0],
      basis[1],
      basis[2],
    );
    this.palm.quaternion.setFromRotationMatrix(basisMatrix);
    const palmWidth = Math.max(this.current.palm.width, 0.04);
    this.palm.scale.set(palmWidth, palmWidth * 0.16, palmWidth * 0.72);

    this.wrist.position.set(...this.current.wrist);
    const wristRadius = Math.max(palmWidth * 0.11, 0.007);
    this.wrist.scale.setScalar(wristRadius);

    for (let index = 0; index < this.bones.length; index += 1) {
      const visual = this.bones[index];
      const targetBone = this.target.bones[index];
      const currentBone = this.current.bones[index];
      if (!targetBone?.valid || !currentBone) {
        visual.segment.visible = false;
        visual.joint.visible = false;
        continue;
      }

      lerpPoint(currentBone.start, targetBone.start);
      lerpPoint(currentBone.end, targetBone.end);
      currentBone.width += (targetBone.width - currentBone.width) * alpha;

      const start = new THREE.Vector3(...currentBone.start);
      const end = new THREE.Vector3(...currentBone.end);
      const direction = end.clone().sub(start);
      const length = direction.length();
      if (length < 1e-5) {
        visual.segment.visible = false;
        visual.joint.visible = false;
        continue;
      }

      const radius = Math.max(currentBone.width * 0.5, 0.0035);
      visual.segment.visible = true;
      visual.segment.position.copy(start).add(end).multiplyScalar(0.5);
      visual.segment.quaternion.setFromUnitVectors(
        this.yAxis,
        direction.normalize(),
      );
      visual.segment.scale.set(radius, length, radius);

      visual.joint.visible = true;
      visual.joint.position.copy(end);
      visual.joint.scale.setScalar(radius * 1.08);
    }
  }
}

export class ArticulatedHandRenderer {
  constructor(scene) {
    this.hands = {
      left: new HandVisual(scene, "left", LEFT_COLOR),
      right: new HandVisual(scene, "right", RIGHT_COLOR),
    };
  }

  setHands(hands) {
    const byHandedness = new Map(
      hands.map((hand) => [hand.handedness, hand]),
    );
    this.hands.left.setTarget(byHandedness.get("left") || null);
    this.hands.right.setTarget(byHandedness.get("right") || null);
  }

  hideAll() {
    this.setHands([]);
  }

  update(deltaSeconds) {
    this.hands.left.update(deltaSeconds);
    this.hands.right.update(deltaSeconds);
  }
}
