import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";

import { ArticulatedHandRenderer } from "./articulated_hand_renderer.js";

const CAMERA = {
  distance: 1.645825386,
  focal: [0.58095634, -0.11324476, -0.27380267],
  pitch: 0.5753982,
  yaw: 0.810398,
};

function cameraPosition() {
  const horizontal = CAMERA.distance * Math.cos(CAMERA.pitch);
  return [
    CAMERA.focal[0] + horizontal * Math.cos(CAMERA.yaw),
    CAMERA.focal[1] + horizontal * Math.sin(CAMERA.yaw),
    CAMERA.focal[2] + CAMERA.distance * Math.sin(CAMERA.pitch),
  ];
}

function makeLabel(name) {
  const canvas = document.createElement("canvas");
  const context = canvas.getContext("2d");
  const font = "600 42px system-ui, sans-serif";
  context.font = font;
  const textWidth = Math.ceil(context.measureText(name).width);
  canvas.width = Math.max(180, textWidth + 44);
  canvas.height = 78;

  context.font = font;
  context.fillStyle = "rgba(32, 33, 36, 0.88)";
  context.beginPath();
  context.roundRect(2, 2, canvas.width - 4, canvas.height - 4, 14);
  context.fill();
  context.strokeStyle = "rgba(255, 255, 255, 0.35)";
  context.lineWidth = 3;
  context.stroke();
  context.fillStyle = "#ffffff";
  context.textAlign = "center";
  context.textBaseline = "middle";
  context.fillText(name, canvas.width / 2, canvas.height / 2 + 1);

  const texture = new THREE.CanvasTexture(canvas);
  texture.colorSpace = THREE.SRGBColorSpace;
  const material = new THREE.SpriteMaterial({
    map: texture,
    transparent: true,
    depthTest: false,
    depthWrite: false,
    sizeAttenuation: false,
  });
  const sprite = new THREE.Sprite(material);
  const aspect = canvas.width / canvas.height;
  sprite.scale.set(0.052 * aspect, 0.052, 1);
  sprite.renderOrder = 100;
  return sprite;
}

export class SceneViewer {
  constructor(container) {
    this.container = container;
    this.scene = new THREE.Scene();
    this.scene.background = new THREE.Color(0x303030);
    this.camera = new THREE.PerspectiveCamera(45, 1, 0.005, 100);
    this.camera.up.set(0, 0, 1);

    this.renderer = new THREE.WebGLRenderer({ antialias: true });
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    this.renderer.outputColorSpace = THREE.SRGBColorSpace;
    this.container.appendChild(this.renderer.domElement);

    this.controls = new OrbitControls(
      this.camera,
      this.renderer.domElement,
    );
    this.controls.enableDamping = true;
    this.controls.dampingFactor = 0.08;
    this.controls.screenSpacePanning = true;
    this.controls.minDistance = 0.15;
    this.controls.maxDistance = 12;

    this.scene.add(new THREE.HemisphereLight(0xffffff, 0x202124, 1.8));
    const keyLight = new THREE.DirectionalLight(0xffffff, 2.2);
    keyLight.position.set(1.5, -1, 2.5);
    this.scene.add(keyLight);

    const grid = new THREE.GridHelper(3, 30, 0x7a7f85, 0x50545a);
    grid.rotation.x = Math.PI / 2;
    grid.position.z = 0;
    this.scene.add(grid);
    this.scene.add(new THREE.AxesHelper(0.22));

    this.objectGroup = new THREE.Group();
    this.objectGroup.name = "scene-object-centers";
    this.scene.add(this.objectGroup);
    this.objectCount = 0;
    this.objectCenterGeometry = new THREE.SphereGeometry(0.022, 18, 12);
    this.objectCenterMaterial = new THREE.MeshStandardMaterial({
      color: 0xffa726,
      roughness: 0.5,
    });

    const beamGeometry = new THREE.BufferGeometry().setFromPoints([
      new THREE.Vector3(),
      new THREE.Vector3(),
    ]);
    this.beam = new THREE.Line(
      beamGeometry,
      new THREE.LineBasicMaterial({ color: 0x35dc72 }),
    );
    this.beam.visible = false;
    this.scene.add(this.beam);

    this.handRenderer = new ArticulatedHandRenderer(this.scene);
    this.clock = new THREE.Clock();
    this.renderFrames = [];

    this.resetCamera();
    this.resizeObserver = new ResizeObserver(() => this.resize());
    this.resizeObserver.observe(this.container);
    this.resize();
    this.renderer.setAnimationLoop(() => this.render());
  }

  resetCamera() {
    this.camera.position.set(...cameraPosition());
    this.controls.target.set(...CAMERA.focal);
    this.controls.update();
  }

  resize() {
    const width = Math.max(1, this.container.clientWidth);
    const height = Math.max(1, this.container.clientHeight);
    this.camera.aspect = width / height;
    this.camera.updateProjectionMatrix();
    this.renderer.setSize(width, height, false);
  }

  setHands(hands) {
    this.handRenderer.setHands(hands);
  }

  hideHands() {
    this.handRenderer.hideAll();
  }

  setSceneObjects(objects) {
    for (const child of [...this.objectGroup.children]) {
      this.objectGroup.remove(child);
      if (child.isSprite) {
        child.material.map.dispose();
        child.material.dispose();
      }
    }

    for (const object of objects) {
      const center = new THREE.Mesh(
        this.objectCenterGeometry,
        this.objectCenterMaterial,
      );
      center.position.set(...object.position);
      center.name = object.name;
      this.objectGroup.add(center);

      const label = makeLabel(object.name);
      label.position.set(
        object.position[0],
        object.position[1],
        object.position[2] + 0.055,
      );
      this.objectGroup.add(label);
    }
    this.objectCount = objects.length;
  }

  setBeam(points) {
    if (!points || points.length < 2) {
      this.beam.visible = false;
      return;
    }
    this.beam.geometry.setFromPoints(
      points.map((point) => new THREE.Vector3(...point)),
    );
    this.beam.geometry.computeBoundingSphere();
    this.beam.visible = true;
  }

  hideBeam() {
    this.beam.visible = false;
  }

  getRenderFps(nowMs = performance.now()) {
    const cutoff = nowMs - 1000;
    while (this.renderFrames.length && this.renderFrames[0] < cutoff) {
      this.renderFrames.shift();
    }
    return this.renderFrames.length;
  }

  render() {
    const delta = Math.min(this.clock.getDelta(), 0.1);
    this.handRenderer.update(delta);
    this.controls.update();
    this.renderer.render(this.scene, this.camera);
    this.renderFrames.push(performance.now());
  }
}
