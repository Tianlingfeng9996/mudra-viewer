import type { MujocoRuntime } from "./mujoco-runtime";
import type { ShadowHandModel } from "./shadow-hand-model";
import poseConfiguration from "./shadow-hand-poses.json";

export type ShadowHandPoseName = "open" | "grasp" | "pinch";

export interface ShadowHandPose {
  name: ShadowHandPoseName;
  label: string;
  controls: readonly number[];
}

export interface ShadowHandActionController {
  readonly activePose: ShadowHandPoseName;
  setPose(name: ShadowHandPoseName): void;
  step(deltaSeconds: number): void;
}

export const SHADOW_HAND_ACTUATOR_NAMES =
  poseConfiguration.actuatorNames as readonly string[];

export const SHADOW_HAND_POSES =
  poseConfiguration.poses as readonly ShadowHandPose[];

const POSE_BY_NAME = new Map(
  SHADOW_HAND_POSES.map((pose) => [pose.name, pose]),
);

function validateActuators(model: ShadowHandModel["model"]) {
  if (model.nu !== SHADOW_HAND_ACTUATOR_NAMES.length) {
    throw new Error(
      `Expected ${SHADOW_HAND_ACTUATOR_NAMES.length} Shadow Hand actuators, received ${model.nu}`,
    );
  }

  for (let index = 0; index < model.nu; index++) {
    const actuator = model.actuator(index);
    try {
      const expectedName = SHADOW_HAND_ACTUATOR_NAMES[index];
      if (actuator.name !== expectedName) {
        throw new Error(
          `Expected actuator ${index} to be ${expectedName}, received ${actuator.name}`,
        );
      }
      for (const pose of SHADOW_HAND_POSES) {
        const control = pose.controls[index];
        if (
          control < model.actuator_ctrlrange[index * 2] ||
          control > model.actuator_ctrlrange[index * 2 + 1]
        ) {
          throw new Error(
            `${pose.label} control ${control} is outside ${expectedName}'s range`,
          );
        }
      }
    } finally {
      actuator.delete();
    }
  }
}

export function createShadowHandActionController(
  runtime: MujocoRuntime,
  hand: ShadowHandModel,
): ShadowHandActionController {
  const { model, data } = hand;
  validateActuators(model);
  let activePose: ShadowHandPoseName = "open";
  let accumulatedSeconds = 0;

  const setPose = (name: ShadowHandPoseName) => {
    const pose = POSE_BY_NAME.get(name);
    if (!pose) throw new Error(`Unknown Shadow Hand pose: ${name}`);
    for (let index = 0; index < model.nu; index++) {
      data.ctrl[index] = pose.controls[index];
    }
    activePose = name;
  };

  setPose(activePose);

  return {
    get activePose() {
      return activePose;
    },
    setPose,
    step(deltaSeconds) {
      accumulatedSeconds += Math.min(Math.max(deltaSeconds, 0), 0.05);
      const timestep = model.opt.timestep;
      let stepCount = 0;
      while (accumulatedSeconds >= timestep && stepCount < 25) {
        runtime.module.mj_step(model, data);
        accumulatedSeconds -= timestep;
        stepCount++;
      }
      if (stepCount === 25) accumulatedSeconds = 0;
    },
  };
}
