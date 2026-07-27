import { readdirSync, readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import createMujocoModule from "@mujoco/mujoco";
import poseConfiguration from "../src/simulation/shadow-hand-poses.json" with {
  type: "json",
};

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

  if (
    poseConfiguration.actuatorNames.length !== model.nu ||
    poseConfiguration.poses.length !== 3
  ) {
    throw new Error("Unexpected Shadow Hand action preset structure");
  }
  const settledPoses = [];
  for (const pose of poseConfiguration.poses) {
    module.mj_resetData(model, data);
    if (pose.controls.length !== model.nu) {
      throw new Error(`${pose.label} does not control all ${model.nu} actuators`);
    }
    for (let actuatorIndex = 0; actuatorIndex < model.nu; actuatorIndex++) {
      const actuator = model.actuator(actuatorIndex);
      try {
        const expectedName = poseConfiguration.actuatorNames[actuatorIndex];
        const control = pose.controls[actuatorIndex];
        if (actuator.name !== expectedName) {
          throw new Error(
            `Actuator ${actuatorIndex} is ${actuator.name}, expected ${expectedName}`,
          );
        }
        if (
          control < model.actuator_ctrlrange[actuatorIndex * 2] ||
          control > model.actuator_ctrlrange[actuatorIndex * 2 + 1]
        ) {
          throw new Error(
            `${pose.label} control is outside ${actuator.name}'s range`,
          );
        }
        data.ctrl[actuatorIndex] = control;
      } finally {
        actuator.delete();
      }
    }
    for (let step = 0; step < 1_000; step++) {
      module.mj_step(model, data);
    }
    const qpos = Array.from(data.qpos);
    if (!qpos.every(Number.isFinite)) {
      throw new Error(`${pose.label} produced non-finite joint positions`);
    }
    const readJointPositions = (jointNames) => {
      const positions = {};
      for (const jointName of jointNames) {
        const joint = model.jnt(jointName);
        try {
          positions[jointName] = data.qpos[joint.qposadr];
        } finally {
          joint.delete();
        }
      }
      return positions;
    };
    const mcpFlexion = readJointPositions([
      "rh_FFJ3",
      "rh_MFJ3",
      "rh_RFJ3",
      "rh_LFJ3",
    ]);
    const mcpAbduction = readJointPositions([
      "rh_FFJ4",
      "rh_MFJ4",
      "rh_RFJ4",
      "rh_LFJ4",
    ]);
    const readFingertip = (bodyName, distalLength) => {
      const body = data.body(bodyName);
      try {
        return [
          body.xpos[0] + body.xmat[2] * distalLength,
          body.xpos[1] + body.xmat[5] * distalLength,
          body.xpos[2] + body.xmat[8] * distalLength,
        ];
      } finally {
        body.delete();
      }
    };
    const thumbTip = readFingertip("rh_thdistal", 0.028);
    const indexTip = readFingertip("rh_ffdistal", 0.028);
    const middleTip = readFingertip("rh_mfdistal", 0.028);
    const oppositionTarget = indexTip.map(
      (value, index) => (value + middleTip[index]) / 2,
    );
    const thumbOppositionDistance = Math.sqrt(
      thumbTip.reduce((sum, value, index) => {
        const difference = value - oppositionTarget[index];
        return sum + difference * difference;
      }, 0),
    );
    if (
      pose.name === "grasp" &&
      (
        Object.values(mcpFlexion).some((value) => value < 0.8) ||
        Math.max(...Object.values(mcpAbduction)) < 0.1 ||
        thumbOppositionDistance > 0.025
      )
    ) {
      throw new Error(
        "Grasp did not rotate both MCP axes and oppose the thumb",
      );
    }
    settledPoses.push({
      name: pose.name,
      qpos,
      mcpFlexion,
      mcpAbduction,
      thumbOppositionDistance,
    });
  }
  for (let first = 0; first < settledPoses.length; first++) {
    for (let second = first + 1; second < settledPoses.length; second++) {
      const distance = Math.sqrt(
        settledPoses[first].qpos.reduce((sum, value, index) => {
          const difference = value - settledPoses[second].qpos[index];
          return sum + difference * difference;
        }, 0),
      );
      if (distance < 0.5) {
        throw new Error(
          `${settledPoses[first].name} and ${settledPoses[second].name} settle to indistinguishable poses`,
        );
      }
    }
  }
  module.mj_resetData(model, data);

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
  const visualMeshIds = [];
  for (let geomIndex = 0; geomIndex < visualScene.ngeom; geomIndex++) {
    const geom = visualScene.geoms.get(geomIndex);
    if (!geom || geom.type !== module.mjtGeom.mjGEOM_MESH.value) {
      throw new Error(`Unexpected visual geom at index ${geomIndex}`);
    }
    const meshId = model.geom_dataid[geom.objid];
    if (meshId < 0 || meshId >= model.nmesh) {
      throw new Error(
        `Visual geom ${geom.objid} references invalid mesh ${meshId}`,
      );
    }
    visualMeshIds.push(meshId);
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
  const expectedVisualMeshIds = [
    0, 1, 3, 4, 5, 6, 7, 8,
    5, 6, 7, 8, 5, 6, 7, 8,
    9, 5, 6, 7, 8, 10, 11, 12,
  ];
  const hasExpectedAssemblySize =
    visualSize[0] > 0.4 &&
    visualSize[0] < 0.5 &&
    visualSize[1] > 0.15 &&
    visualSize[1] < 0.2 &&
    visualSize[2] > 0.1 &&
    visualSize[2] < 0.16;
  if (
    visualScene.ngeom !== 24 ||
    JSON.stringify(visualMeshIds) !== JSON.stringify(expectedVisualMeshIds) ||
    !visualSize.every(Number.isFinite) ||
    !hasExpectedAssemblySize ||
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
    visualMeshIds,
    visualSize,
    actionPoses: settledPoses.map(
      ({ name, mcpFlexion, mcpAbduction, thumbOppositionDistance }) => ({
        name,
        mcpFlexion: Object.values(mcpFlexion)
          .map((value) => value.toFixed(3))
          .join(", "),
        mcpAbduction: Object.values(mcpAbduction)
          .map((value) => value.toFixed(3))
          .join(", "),
        thumbOppositionDistance: thumbOppositionDistance.toFixed(3),
      }),
    ),
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
