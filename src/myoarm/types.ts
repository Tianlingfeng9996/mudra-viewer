/**
 * Stable data contracts for the MyoArm collection and training pipeline.
 *
 * Raw decoded samples stay as int32 values until preprocessing. This preserves
 * the exact output from mudraka and lets training and live inference share the
 * same normalization code later.
 */

import type {
  CaptureSource,
  ChannelName,
} from "../signal/types";

export type {
  CaptureSource,
  ChannelName,
  SampleChunk,
} from "../signal/types";

export type GestureLabel =
  | "rest"
  | "open"
  | "grasp"
  | "pinch"
  | "pronation"
  | "supination";

export type ArmSide = "left" | "right" | "unknown";
export type EffortLevel = "relaxed" | "light" | "medium" | "strong" | "unknown";

export type QualityFlag =
  | "clipping"
  | "dropped-frames"
  | "too-short"
  | "contains-transition"
  | "manual-reject";

export interface GestureDefinition {
  id: GestureLabel;
  displayName: string;
  description: string;
}

export interface CaptureProtocol {
  id: string;
  labels: readonly GestureLabel[];
  countdownMs: number;
  activeMs: number;
  recoveryMs: number;
  repetitionsPerLabel: number;
}

export interface SegmentQuality {
  accepted: boolean;
  flags: QualityFlag[];
  notes?: string;
}

export interface MyoArmSegment {
  id: string;
  sessionId: string;
  label: GestureLabel;
  effort: EffortLevel;
  repetition: number;
  startOffsetMs: number;
  durationMs: number;
  sampleRateHz: number;
  channels: readonly ChannelName[];
  sampleCount: number;
  /** Sample-major interleaved int32 values. */
  samples: Int32Array;
  quality: SegmentQuality;
}

export interface MyoArmSession {
  id: string;
  datasetId: string;
  /** Pseudonymous local identifier; do not put a participant name here. */
  participantId: string;
  source: CaptureSource;
  sourceName?: string;
  armSide: ArmSide;
  startedAt: string;
  protocolId: string;
  sampleRateHz: number;
  channels: readonly ChannelName[];
  appVersion: string;
  segments: MyoArmSegment[];
}

export interface MyoArmDataset {
  schemaVersion: number;
  id: string;
  name: string;
  createdAt: string;
  updatedAt: string;
  labelSet: readonly GestureLabel[];
  sessions: MyoArmSession[];
}
