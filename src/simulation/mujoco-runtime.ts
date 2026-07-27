import type { MainModule } from "@mujoco/mujoco";
import mujocoWasmUrl from "@mujoco/mujoco/mujoco.wasm?url";

export interface MujocoRuntime {
  module: MainModule;
  version: string;
  versionNumber: number;
  threaded: false;
}

let runtimePromise: Promise<MujocoRuntime> | null = null;

export function loadMujocoRuntime(): Promise<MujocoRuntime> {
  if (!runtimePromise) {
    runtimePromise = import("@mujoco/mujoco")
      .then(({ default: createMujocoModule }) =>
        createMujocoModule({
          locateFile: (path: string) =>
            path.endsWith(".wasm") ? mujocoWasmUrl : path,
        }),
      )
      .then((module) => ({
        module,
        version: module.mj_versionString(),
        versionNumber: module.mj_version(),
        threaded: false as const,
      }))
      .catch((error: unknown) => {
        runtimePromise = null;
        throw error;
      });
  }

  return runtimePromise;
}
