import type { ClassificationPrediction } from "../myoarm/classifier";
import type { GestureLabel } from "../myoarm/types";
import type { ShadowHandPoseName } from "./shadow-hand-actions";
import configuration from "./ann-hand-control.json" with { type: "json" };

export interface AnnHandControlGate {
  reset(): void;
  accept(prediction: ClassificationPrediction): AnnHandControlResult;
}

export type AnnHandControlResult =
  | {
      status: "unsupported" | "low-confidence";
      label: GestureLabel;
      confidence: number;
    }
  | {
      status: "confirming";
      label: GestureLabel;
      pose: ShadowHandPoseName;
      confidence: number;
      confirmationCount: number;
      requiredConfirmations: number;
    }
  | {
      status: "triggered" | "steady";
      label: GestureLabel;
      pose: ShadowHandPoseName;
      confidence: number;
    };

export const ANN_HAND_CONTROL_CONFIG = {
  confidenceThreshold: configuration.confidenceThreshold,
  requiredConfirmations: configuration.requiredConfirmations,
  poseByLabel: configuration.poseByLabel as Partial<
    Record<GestureLabel, ShadowHandPoseName>
  >,
} as const;

export function createAnnHandControlGate(): AnnHandControlGate {
  let candidatePose: ShadowHandPoseName | null = null;
  let confirmationCount = 0;
  let activePose: ShadowHandPoseName | null = null;

  const resetCandidate = () => {
    candidatePose = null;
    confirmationCount = 0;
  };

  return {
    reset() {
      resetCandidate();
      activePose = null;
    },
    accept(prediction) {
      const pose = ANN_HAND_CONTROL_CONFIG.poseByLabel[prediction.label];
      if (!pose) {
        resetCandidate();
        return {
          status: "unsupported",
          label: prediction.label,
          confidence: prediction.confidence,
        };
      }
      if (
        prediction.confidence < ANN_HAND_CONTROL_CONFIG.confidenceThreshold
      ) {
        resetCandidate();
        return {
          status: "low-confidence",
          label: prediction.label,
          confidence: prediction.confidence,
        };
      }
      if (activePose === pose) {
        resetCandidate();
        return {
          status: "steady",
          label: prediction.label,
          pose,
          confidence: prediction.confidence,
        };
      }
      if (pose !== candidatePose) {
        candidatePose = pose;
        confirmationCount = 1;
      } else {
        confirmationCount++;
      }
      if (
        confirmationCount < ANN_HAND_CONTROL_CONFIG.requiredConfirmations
      ) {
        return {
          status: "confirming",
          label: prediction.label,
          pose,
          confidence: prediction.confidence,
          confirmationCount,
          requiredConfirmations:
            ANN_HAND_CONTROL_CONFIG.requiredConfirmations,
        };
      }
      resetCandidate();
      activePose = pose;
      return {
        status: "triggered",
        label: prediction.label,
        pose,
        confidence: prediction.confidence,
      };
    },
  };
}
