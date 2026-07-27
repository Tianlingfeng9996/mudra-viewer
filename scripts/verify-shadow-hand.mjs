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

try {
  vfs.addBuffer(
    "right_hand.xml",
    readFileSync(`${modelRoot}/right_hand.xml`),
  );
  for (const name of readdirSync(`${modelRoot}/assets`)) {
    vfs.addBuffer(
      `assets/${name}`,
      readFileSync(`${modelRoot}/assets/${name}`),
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
  console.log("Shadow Hand model verified:", actual);
} finally {
  data?.delete();
  model?.delete();
  vfs.delete();
}
