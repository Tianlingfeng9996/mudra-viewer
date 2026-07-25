import type {
  CaptureProtocol,
  EffortLevel,
  GestureDefinition,
  GestureLabel,
} from "./types";
import {
  EMG_CHANNELS,
  EMG_SAMPLE_RATE_HZ,
} from "../signal/types";

export const MYOARM_DATASET_SCHEMA_VERSION = 1;

export const MYOARM_CHANNELS = EMG_CHANNELS;
export const MYOARM_SAMPLE_RATE_HZ = EMG_SAMPLE_RATE_HZ;

export const GESTURES: readonly GestureDefinition[] = [
  {
    id: "rest",
    displayName: "Rest",
    description: "Arm and hand relaxed without an intentional gesture.",
  },
  {
    id: "open",
    displayName: "Open",
    description: "Open the hand without overextending the fingers.",
  },
  {
    id: "grasp",
    displayName: "Grasp",
    description: "Close the hand into a natural grasp.",
  },
  {
    id: "pinch",
    displayName: "Pinch",
    description: "Bring the thumb and index finger together.",
  },
  {
    id: "pronation",
    displayName: "Pronation",
    description: "Rotate the forearm so the palm turns downward.",
  },
  {
    id: "supination",
    displayName: "Supination",
    description: "Rotate the forearm so the palm turns upward.",
  },
];

export const GESTURE_LABELS = GESTURES.map((gesture) => gesture.id);

export const CAPTURE_PROTOCOL_V1: CaptureProtocol = {
  id: "myoarm-discrete-gesture-v1",
  labels: GESTURE_LABELS,
  countdownMs: 3_000,
  activeMs: 3_000,
  recoveryMs: 2_000,
  repetitionsPerLabel: 10,
};

const FIXTURE_PATTERN =
  /^(open|grasp|pinch|pronation|supination)(?: \((strong)\))?$/;

/**
 * Converts today's fixture folder names into protocol labels. Fixtures do not
 * include a valid Rest recording, so no synthetic Rest mapping is provided.
 */
export function classifyFixtureName(
  name: string,
): { label: GestureLabel; effort: EffortLevel } | null {
  const match = FIXTURE_PATTERN.exec(name);
  if (!match) return null;
  return {
    label: match[1] as GestureLabel,
    effort: match[2] === "strong" ? "strong" : "medium",
  };
}
