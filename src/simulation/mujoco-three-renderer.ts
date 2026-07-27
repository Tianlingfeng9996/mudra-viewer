import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import type { MujocoRuntime } from "./mujoco-runtime";
import type { ShadowHandModel } from "./shadow-hand-model";

export interface MujocoThreeRenderer {
  sync(): void;
  dispose(): void;
}

function createMeshGeometry(
  model: ShadowHandModel["model"],
  meshId: number,
): THREE.BufferGeometry {
  const vertexStart = model.mesh_vertadr[meshId] * 3;
  const vertexCount = model.mesh_vertnum[meshId] * 3;
  const faceStart = model.mesh_faceadr[meshId] * 3;
  const faceCount = model.mesh_facenum[meshId] * 3;
  const sourceVertices = model.mesh_vert as Float64Array;
  const sourceFaces = model.mesh_face as Int32Array;
  const positions = new Float32Array(vertexCount);
  const indices = new Uint32Array(faceCount);

  for (let index = 0; index < vertexCount; index++) {
    positions[index] = sourceVertices[vertexStart + index];
  }
  for (let index = 0; index < faceCount; index++) {
    indices[index] = sourceFaces[faceStart + index];
  }

  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute("position", new THREE.BufferAttribute(positions, 3));
  geometry.setIndex(new THREE.BufferAttribute(indices, 1));
  geometry.computeVertexNormals();
  geometry.computeBoundingSphere();
  return geometry;
}

function applyMujocoTransform(
  object: THREE.Object3D,
  position: ArrayLike<number>,
  rotation: ArrayLike<number>,
) {
  object.matrix.set(
    rotation[0], rotation[1], rotation[2], position[0],
    rotation[3], rotation[4], rotation[5], position[1],
    rotation[6], rotation[7], rotation[8], position[2],
    0, 0, 0, 1,
  );
  object.matrixAutoUpdate = false;
  object.matrixWorldNeedsUpdate = true;
}

export function createMujocoThreeRenderer(
  container: HTMLElement,
  runtime: MujocoRuntime,
  hand: ShadowHandModel,
): MujocoThreeRenderer {
  const { module } = runtime;
  const { model, data } = hand;
  const mujocoOption = new module.MjvOption();
  const mujocoPerturb = new module.MjvPerturb();
  const mujocoCamera = new module.MjvCamera();
  const mujocoScene = new module.MjvScene(model, 256);

  for (let group = 0; group < 6; group++) {
    mujocoOption.geomgroup[group] = group === 2 ? 1 : 0;
  }
  module.mj_forward(model, data);
  module.mjv_updateScene(
    model,
    data,
    mujocoOption,
    mujocoPerturb,
    mujocoCamera,
    module.mjtCatBit.mjCAT_ALL.value,
    mujocoScene,
  );

  const scene = new THREE.Scene();
  scene.background = new THREE.Color(0xf4f2ea);
  const camera = new THREE.PerspectiveCamera(38, 1, 0.001, 100);
  const renderer = new THREE.WebGLRenderer({
    antialias: true,
    powerPreference: "high-performance",
  });
  renderer.outputColorSpace = THREE.SRGBColorSpace;
  renderer.toneMapping = THREE.ACESFilmicToneMapping;
  renderer.toneMappingExposure = 1.15;
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
  renderer.shadowMap.enabled = true;
  renderer.shadowMap.type = THREE.PCFSoftShadowMap;
  renderer.domElement.className = "myoarm-hand-canvas";
  renderer.domElement.setAttribute(
    "aria-label",
    "Interactive three-dimensional Shadow Hand model",
  );

  scene.add(new THREE.HemisphereLight(0xffffff, 0x6f7369, 2.4));
  const keyLight = new THREE.DirectionalLight(0xffffff, 4);
  keyLight.position.set(1.2, 1.8, 1.4);
  keyLight.castShadow = true;
  keyLight.shadow.mapSize.set(1024, 1024);
  scene.add(keyLight);
  const fillLight = new THREE.DirectionalLight(0xc9ddff, 2);
  fillLight.position.set(-1, 0.4, -0.8);
  scene.add(fillLight);

  const modelRoot = new THREE.Group();
  modelRoot.rotation.x = -Math.PI / 2;
  scene.add(modelRoot);

  const geometryByMeshId = new Map<number, THREE.BufferGeometry>();
  const materials = new Set<THREE.Material>();
  const renderedGeoms: Array<{
    mesh: THREE.Mesh;
    sceneGeomIndex: number;
  }> = [];

  for (let index = 0; index < mujocoScene.ngeom; index++) {
    const geom = mujocoScene.geoms.get(index);
    if (!geom || geom.type !== module.mjtGeom.mjGEOM_MESH.value) continue;
    const meshId = model.geom_dataid[geom.objid];
    if (meshId < 0 || meshId >= model.nmesh) {
      throw new Error(
        `Visual geom ${geom.objid} references invalid mesh ${meshId}`,
      );
    }

    let geometry = geometryByMeshId.get(meshId);
    if (!geometry) {
      geometry = createMeshGeometry(model, meshId);
      geometryByMeshId.set(meshId, geometry);
    }

    const material = new THREE.MeshStandardMaterial({
      color: new THREE.Color(geom.rgba[0], geom.rgba[1], geom.rgba[2]),
      metalness: Math.min(0.65, Math.max(0, geom.specular * 0.55)),
      roughness: Math.min(0.9, Math.max(0.22, 1 - geom.shininess * 0.7)),
      opacity: geom.rgba[3],
      transparent: geom.rgba[3] < 0.999,
      side: THREE.DoubleSide,
    });
    materials.add(material);

    const mesh = new THREE.Mesh(geometry, material);
    mesh.castShadow = true;
    mesh.receiveShadow = true;
    applyMujocoTransform(mesh, geom.pos, geom.mat);
    modelRoot.add(mesh);
    renderedGeoms.push({ mesh, sceneGeomIndex: index });
  }

  if (!renderedGeoms.length) {
    renderer.dispose();
    mujocoScene.delete();
    mujocoCamera.delete();
    mujocoPerturb.delete();
    mujocoOption.delete();
    throw new Error("Shadow Hand contains no renderable visual mesh geoms");
  }

  modelRoot.updateWorldMatrix(true, true);
  const bounds = new THREE.Box3().setFromObject(modelRoot);
  const center = bounds.getCenter(new THREE.Vector3());
  const size = bounds.getSize(new THREE.Vector3());
  const maxDimension = Math.max(size.x, size.y, size.z);
  const fitDistance =
    (maxDimension / (2 * Math.tan(THREE.MathUtils.degToRad(camera.fov / 2)))) *
    1.45;
  camera.near = Math.max(maxDimension / 200, 0.0001);
  camera.far = Math.max(maxDimension * 100, 10);
  camera.position.set(
    center.x + fitDistance * 0.8,
    center.y + fitDistance * 0.45,
    center.z + fitDistance,
  );
  camera.updateProjectionMatrix();

  const controls = new OrbitControls(camera, renderer.domElement);
  controls.target.copy(center);
  controls.enableDamping = true;
  controls.dampingFactor = 0.07;
  controls.minDistance = maxDimension * 0.65;
  controls.maxDistance = maxDimension * 8;
  controls.update();

  const ground = new THREE.Mesh(
    new THREE.PlaneGeometry(maxDimension * 5, maxDimension * 5),
    new THREE.ShadowMaterial({ color: 0x35372f, opacity: 0.13 }),
  );
  ground.rotation.x = -Math.PI / 2;
  ground.position.y = bounds.min.y - maxDimension * 0.025;
  ground.receiveShadow = true;
  scene.add(ground);

  const resize = () => {
    const width = Math.max(container.clientWidth, 1);
    const height = Math.max(container.clientHeight, 1);
    renderer.setSize(width, height, false);
    camera.aspect = width / height;
    camera.updateProjectionMatrix();
  };
  const resizeObserver = new ResizeObserver(resize);
  resizeObserver.observe(container);
  container.replaceChildren(renderer.domElement);
  resize();
  renderer.setAnimationLoop(() => {
    controls.update();
    renderer.render(scene, camera);
  });

  let disposed = false;
  return {
    sync() {
      if (disposed) return;
      module.mj_forward(model, data);
      module.mjv_updateScene(
        model,
        data,
        mujocoOption,
        mujocoPerturb,
        mujocoCamera,
        module.mjtCatBit.mjCAT_ALL.value,
        mujocoScene,
      );
      for (const renderedGeom of renderedGeoms) {
        const geom = mujocoScene.geoms.get(renderedGeom.sceneGeomIndex);
        if (geom) {
          applyMujocoTransform(renderedGeom.mesh, geom.pos, geom.mat);
        }
      }
    },
    dispose() {
      if (disposed) return;
      disposed = true;
      renderer.setAnimationLoop(null);
      resizeObserver.disconnect();
      controls.dispose();
      ground.geometry.dispose();
      ground.material.dispose();
      for (const geometry of geometryByMeshId.values()) geometry.dispose();
      for (const material of materials) material.dispose();
      renderer.dispose();
      renderer.forceContextLoss();
      renderer.domElement.remove();
      mujocoScene.delete();
      mujocoCamera.delete();
      mujocoPerturb.delete();
      mujocoOption.delete();
    },
  };
}
