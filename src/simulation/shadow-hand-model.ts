import type { MjData, MjModel } from "@mujoco/mujoco";
import type { MujocoRuntime } from "./mujoco-runtime";

const MODEL_ROOT = "models/shadow-hand";
const MODEL_FILE = "right_hand.xml";
const MODEL_ASSET_FILES = [
  "assets/forearm_0.obj",
  "assets/forearm_1.obj",
  "assets/forearm_collision.obj",
  "assets/f_distal_pst.obj",
  "assets/f_knuckle.obj",
  "assets/f_middle.obj",
  "assets/f_proximal.obj",
  "assets/lf_metacarpal.obj",
  "assets/mounting_plate.obj",
  "assets/palm.obj",
  "assets/th_distal_pst.obj",
  "assets/th_middle.obj",
  "assets/th_proximal.obj",
  "assets/wrist.obj",
] as const;

const MODEL_FILES = [MODEL_FILE, ...MODEL_ASSET_FILES] as const;

export interface ShadowHandModelStats {
  degreesOfFreedom: number;
  joints: number;
  actuators: number;
  bodies: number;
  geoms: number;
  meshes: number;
}

export interface ShadowHandModel {
  model: MjModel;
  data: MjData;
  stats: ShadowHandModelStats;
  dispose(): void;
}

async function fetchModelFile(relativePath: string): Promise<Uint8Array> {
  const url = `${import.meta.env.BASE_URL}${MODEL_ROOT}/${relativePath}`;
  const response = await fetch(url);
  if (!response.ok) {
    throw new Error(
      `Shadow Hand asset "${relativePath}" failed to load (${response.status})`,
    );
  }
  return new Uint8Array(await response.arrayBuffer());
}

export async function loadShadowHandModel(
  runtime: MujocoRuntime,
): Promise<ShadowHandModel> {
  const files = await Promise.all(
    MODEL_FILES.map(async (path) => ({
      path,
      bytes: await fetchModelFile(path),
    })),
  );
  const vfs = new runtime.module.MjVFS();
  let model: MjModel | null = null;
  let data: MjData | null = null;

  try {
    for (const file of files) {
      vfs.addBuffer(file.path, file.bytes);
    }
    model = runtime.module.MjModel.from_xml_path(MODEL_FILE, vfs);
    data = new runtime.module.MjData(model);

    const stats: ShadowHandModelStats = {
      degreesOfFreedom: model.nv,
      joints: model.njnt,
      actuators: model.nu,
      bodies: model.nbody,
      geoms: model.ngeom,
      meshes: model.nmesh,
    };
    let disposed = false;

    return {
      model,
      data,
      stats,
      dispose() {
        if (disposed) return;
        disposed = true;
        data?.delete();
        model?.delete();
        data = null;
        model = null;
      },
    };
  } catch (error) {
    data?.delete();
    model?.delete();
    throw error;
  } finally {
    vfs.delete();
  }
}
