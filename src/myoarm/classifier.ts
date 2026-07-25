import type { PreprocessedWindow } from "./preprocessing";
import type { GestureLabel } from "./types";

export interface ClassificationScore {
  label: GestureLabel;
  probability: number;
}

export interface ClassificationPrediction {
  label: GestureLabel;
  confidence: number;
  scores: ClassificationScore[];
}

export interface ClassificationMetrics {
  loss: number;
  accuracy: number;
  windowCount: number;
}

/**
 * Runtime contract shared by the baseline ANN and future TF2AngleNet adapter.
 * Views and live inference do not need to know how a model is implemented.
 */
export interface TrainedMyoArmClassifier {
  readonly id: string;
  readonly inputValueCount: number;
  readonly labels: readonly GestureLabel[];
  predict(values: Float32Array): ClassificationPrediction;
  evaluate(windows: readonly PreprocessedWindow[]): ClassificationMetrics;
}
