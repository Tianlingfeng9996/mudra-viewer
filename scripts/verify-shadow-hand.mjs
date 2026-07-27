import { readdirSync, readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import createMujocoModule from "@mujoco/mujoco";

const modelRoot = fileURLToPath(
  new URL("../public/models/shadow-hand/", import.meta.url),
);
const module = await createMujocoModule();
const vfs = new module.MjVFS();
let model = null;
let data = null;
let visualOption = null;
let visualPerturb = null;
let visualCamera = null;
let visualScene = null;

try {
  vfs.addBuffer(
    "right_hand.xml",
    Array.from(readFileSync(`${modelRoot}/right_hand.xml`)),
  );
  for (const name of readdirSync(`${modelRoot}/assets`)) {
    vfs.addBuffer(
      `assets/${name}`,
      Array.from(readFileSync(`${modelRoot}/assets/${name}`)),
    );
  }

  model = module.MjModel.from_xml_path("right_hand.xml", vfs);
  data = new module.MjData(model);

  const actual = {
    version: module.mj_versionString(),
    degreesOfFreedom: model.nv,
    joints: model.njnt,
    actuators: model.nu,
    bodies: model.nbody,
    geoms: model.ngeom,
    meshes: model.nmesh,
  };
  const expected = {
    version: "3.10.0",
    degreesOfFreedom: 24,
    joints: 24,
    actuators: 20,
    bodies: 26,
    geoms: 62,
    meshes: 13,
  };

  if (JSON.stringify(actual) !== JSON.stringify(expected)) {
    throw new Error(
      `Unexpected Shadow Hand model structure:\n${JSON.stringify(actual, null, 2)}`,
    );
  }

  visualOption = new module.MjvOption();
  visualPerturb = new module.MjvPerturb();
  visualCamera = new module.MjvCamera();
  visualScene = new module.MjvScene(model, 256);
  for (let group = 0; group < 6; group++) {
    visualOption.geomgroup[group] = group === 2 ? 1 : 0;
  }
  module.mj_forward(model, data);
  module.mjv_updateScene(
    model,
    data,
    visualOption,
    visualPerturb,
    visualCamera,
    module.mjtCatBit.mjCAT_ALL.value,
    visualScene,
  );

  const lower = [Infinity, Infinity, Infinity];
  const upper = [-Infinity, -Infinity, -Infinity];
  for (let geomIndex = 0; geomIndex < visualScene.ngeom; geomIndex++) {
    const geom = visualScene.geoms.get(geomIndex);
    if (!geom || geom.type !== module.mjtGeom.mjGEOM_MESH.value) {
      throw new Error(`Unexpected visual geom at index ${geomIndex}`);
    }
    const meshId = geom.dataid;
    const vertexAddress = model.mesh_vertadr[meshId];
    const vertexCount = model.mesh_vertnum[meshId];
    for (let vertexIndex = 0; vertexIndex < vertexCount; vertexIndex++) {
      const offset = (vertexAddress + vertexIndex) * 3;
      const x = model.mesh_vert[offset];
      const y = model.mesh_vert[offset + 1];
      const z = model.mesh_vert[offset + 2];
      for (let axis = 0; axis < 3; axis++) {
        const transformed =
          geom.pos[axis] +
          geom.mat[axis * 3] * x +
          geom.mat[axis * 3 + 1] * y +
          geom.mat[axis * 3 + 2] * z;
        lower[axis] = Math.min(lower[axis], transformed);
        upper[axis] = Math.max(upper[axis], transformed);
      }
    }
  }
  const visualSize = upper.map((value, axis) => value - lower[axis]);
  const maxDimension = Math.max(...visualSize);
  if (
    visualScene.ngeom !== 24 ||
    !visualSize.every(Number.isFinite) ||
    maxDimension < 0.1 ||
    maxDimension > 1
  ) {
    throw new Error(
      `Unexpected render scene: ${JSON.stringify({
        visualGeoms: visualScene.ngeom,
        lower,
        upper,
        visualSize,
      })}`,
    );
  }

  console.log("Shadow Hand model verified:", {
    ...actual,
    visualGeoms: visualScene.ngeom,
    visualSize,
  });
} finally {
  visualScene?.delete();
  visualCamera?.delete();
  visualPerturb?.delete();
  visualOption?.delete();
  data?.delete();
  model?.delete();
  vfs.delete();
}
