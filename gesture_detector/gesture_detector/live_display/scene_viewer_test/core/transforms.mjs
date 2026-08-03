export const IDENTITY_TRANSFORM = Object.freeze({
  translation: Object.freeze([0, 0, 0]),
  rotation: Object.freeze([0, 0, 0, 1]),
});

export function rawLeapPointToLeapworld([x, y, z]) {
  return [-z / 1000, -x / 1000, y / 1000];
}

export function rawLeapVectorToLeapworld([x, y, z]) {
  return [-z, -x, y];
}

export function normalizeQuaternion([x, y, z, w]) {
  const length = Math.hypot(x, y, z, w);
  if (length === 0) {
    return [0, 0, 0, 1];
  }
  return [x / length, y / length, z / length, w / length];
}

export function multiplyQuaternions(left, right) {
  const [lx, ly, lz, lw] = left;
  const [rx, ry, rz, rw] = right;
  return normalizeQuaternion([
    lw * rx + lx * rw + ly * rz - lz * ry,
    lw * ry - lx * rz + ly * rw + lz * rx,
    lw * rz + lx * ry - ly * rx + lz * rw,
    lw * rw - lx * rx - ly * ry - lz * rz,
  ]);
}

export function rotateVector(rotation, [x, y, z]) {
  const [qx, qy, qz, qw] = normalizeQuaternion(rotation);
  const tx = 2 * (qy * z - qz * y);
  const ty = 2 * (qz * x - qx * z);
  const tz = 2 * (qx * y - qy * x);
  return [
    x + qw * tx + (qy * tz - qz * ty),
    y + qw * ty + (qz * tx - qx * tz),
    z + qw * tz + (qx * ty - qy * tx),
  ];
}

export function applyTransform(transform, point) {
  const rotated = rotateVector(transform.rotation, point);
  return rotated.map((value, index) => value + transform.translation[index]);
}

export function composeTransforms(parentFromMiddle, middleFromChild) {
  const childOriginInParent = applyTransform(
    parentFromMiddle,
    middleFromChild.translation,
  );
  return {
    translation: childOriginInParent,
    rotation: multiplyQuaternions(
      parentFromMiddle.rotation,
      middleFromChild.rotation,
    ),
  };
}

export function invertTransform(parentFromChild) {
  const [x, y, z, w] = normalizeQuaternion(parentFromChild.rotation);
  const inverseRotation = [-x, -y, -z, w];
  const inverseTranslation = rotateVector(
    inverseRotation,
    parentFromChild.translation.map((value) => -value),
  );
  return {
    translation: inverseTranslation,
    rotation: inverseRotation,
  };
}

export function transformFromRos(rosTransform) {
  const { translation, rotation } = rosTransform;
  return {
    translation: [translation.x, translation.y, translation.z],
    rotation: normalizeQuaternion([
      rotation.x,
      rotation.y,
      rotation.z,
      rotation.w,
    ]),
  };
}
