import {
  ANN_HAND_CONTROL_CONFIG,
  createAnnHandControlGate,
} from "../src/simulation/ann-hand-control.ts";

const prediction = (label, confidence) => ({
  label,
  confidence,
  scores: [{ label, probability: confidence }],
});
const gate = createAnnHandControlGate();

const lowConfidence = gate.accept(
  prediction("grasp", ANN_HAND_CONTROL_CONFIG.confidenceThreshold - 0.01),
);
if (lowConfidence.status !== "low-confidence") {
  throw new Error("Low-confidence predictions must not drive the hand");
}

const unsupported = gate.accept(prediction("pronation", 0.99));
if (unsupported.status !== "unsupported") {
  throw new Error("Unmapped predictions must hold the current hand pose");
}

for (
  let confirmation = 1;
  confirmation < ANN_HAND_CONTROL_CONFIG.requiredConfirmations;
  confirmation++
) {
  const result = gate.accept(prediction("grasp", 0.9));
  if (
    result.status !== "confirming" ||
    result.confirmationCount !== confirmation
  ) {
    throw new Error("ANN hand control triggered before enough confirmations");
  }
}
const triggered = gate.accept(prediction("grasp", 0.9));
if (triggered.status !== "triggered" || triggered.pose !== "grasp") {
  throw new Error("Stable grasp predictions did not trigger Grasp");
}

const steady = gate.accept(prediction("grasp", 0.9));
if (steady.status !== "steady") {
  throw new Error("Repeated grasp predictions should hold the current pose");
}

gate.reset();
gate.accept(prediction("open", 0.9));
const interrupted = gate.accept(prediction("pinch", 0.9));
if (
  interrupted.status !== "confirming" ||
  interrupted.confirmationCount !== 1
) {
  throw new Error("A label change must restart confirmation");
}

console.log("ANN hand control verified:", ANN_HAND_CONTROL_CONFIG);
