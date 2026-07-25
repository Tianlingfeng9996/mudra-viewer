import type { PreprocessedWindow } from "./preprocessing";
import type { GestureLabel } from "./types";
import type {
  ClassificationMetrics,
  TrainedMyoArmClassifier,
} from "./classifier";

export interface TrainingEpochMetrics {
  epoch: number;
  train: ClassificationMetrics;
  validation: ClassificationMetrics;
}

export interface BaselineAnnConfig {
  id: string;
  hiddenUnitCount: number;
  epochCount: number;
  batchSize: number;
  learningRate: number;
  l2Regularization: number;
  seed: string;
}

export interface BaselineAnnTrainingResult {
  model: TrainedMyoArmClassifier;
  history: TrainingEpochMetrics[];
}

export interface BaselineAnnTrainingOptions {
  trainWindows: readonly PreprocessedWindow[];
  validationWindows: readonly PreprocessedWindow[];
  labels: readonly GestureLabel[];
  inputValueCount: number;
  config?: Partial<BaselineAnnConfig>;
  signal?: AbortSignal;
  onEpoch?(metrics: TrainingEpochMetrics): void;
  yieldAfterEpoch?(): Promise<void>;
}

export const BASELINE_ANN_CONFIG_V1: BaselineAnnConfig = {
  id: "myoarm-baseline-ann-v1",
  hiddenUnitCount: 16,
  epochCount: 30,
  batchSize: 16,
  learningRate: 0.003,
  l2Regularization: 0.0001,
  seed: "myoarm-baseline-ann-v1",
};

interface AnnParameters {
  inputValueCount: number;
  hiddenUnitCount: number;
  outputUnitCount: number;
  inputWeights: Float32Array;
  hiddenBias: Float32Array;
  outputWeights: Float32Array;
  outputBias: Float32Array;
}

interface ForwardPass {
  hidden: Float32Array;
  probabilities: Float32Array;
}

const hashString = (value: string) => {
  let hash = 0x811c9dc5;
  for (let index = 0; index < value.length; index++) {
    hash ^= value.charCodeAt(index);
    hash = Math.imul(hash, 0x01000193);
  }
  return hash >>> 0;
};

const createRandom = (seed: string) => {
  let state = hashString(seed) || 0x9e3779b9;
  return () => {
    state ^= state << 13;
    state ^= state >>> 17;
    state ^= state << 5;
    return (state >>> 0) / 4_294_967_296;
  };
};

const shuffledIndices = (count: number, random: () => number) => {
  const indices = Array.from({ length: count }, (_, index) => index);
  for (let index = indices.length - 1; index > 0; index--) {
    const other = Math.floor(random() * (index + 1));
    [indices[index], indices[other]] = [indices[other], indices[index]];
  }
  return indices;
};

const initializeParameters = (
  inputValueCount: number,
  hiddenUnitCount: number,
  outputUnitCount: number,
  random: () => number,
): AnnParameters => {
  const inputWeights = new Float32Array(
    hiddenUnitCount * inputValueCount,
  );
  const outputWeights = new Float32Array(
    outputUnitCount * hiddenUnitCount,
  );
  const inputScale = Math.sqrt(
    6 / (inputValueCount + hiddenUnitCount),
  );
  const outputScale = Math.sqrt(
    6 / (hiddenUnitCount + outputUnitCount),
  );

  for (let index = 0; index < inputWeights.length; index++) {
    inputWeights[index] = (random() * 2 - 1) * inputScale;
  }
  for (let index = 0; index < outputWeights.length; index++) {
    outputWeights[index] = (random() * 2 - 1) * outputScale;
  }

  return {
    inputValueCount,
    hiddenUnitCount,
    outputUnitCount,
    inputWeights,
    hiddenBias: new Float32Array(hiddenUnitCount),
    outputWeights,
    outputBias: new Float32Array(outputUnitCount),
  };
};

const forward = (
  parameters: AnnParameters,
  values: Float32Array,
): ForwardPass => {
  if (values.length !== parameters.inputValueCount) {
    throw new Error(
      `ANN expected ${parameters.inputValueCount} values, received ${values.length}`,
    );
  }

  const hidden = new Float32Array(parameters.hiddenUnitCount);
  for (let hiddenIndex = 0; hiddenIndex < parameters.hiddenUnitCount; hiddenIndex++) {
    let activation = parameters.hiddenBias[hiddenIndex];
    const weightOffset = hiddenIndex * parameters.inputValueCount;
    for (let inputIndex = 0; inputIndex < parameters.inputValueCount; inputIndex++) {
      activation +=
        parameters.inputWeights[weightOffset + inputIndex] *
        values[inputIndex];
    }
    hidden[hiddenIndex] = Math.max(0, activation);
  }

  const probabilities = new Float32Array(parameters.outputUnitCount);
  let maximumLogit = -Infinity;
  for (let outputIndex = 0; outputIndex < parameters.outputUnitCount; outputIndex++) {
    let logit = parameters.outputBias[outputIndex];
    const weightOffset = outputIndex * parameters.hiddenUnitCount;
    for (let hiddenIndex = 0; hiddenIndex < parameters.hiddenUnitCount; hiddenIndex++) {
      logit +=
        parameters.outputWeights[weightOffset + hiddenIndex] *
        hidden[hiddenIndex];
    }
    probabilities[outputIndex] = logit;
    maximumLogit = Math.max(maximumLogit, logit);
  }

  let probabilityTotal = 0;
  for (let outputIndex = 0; outputIndex < probabilities.length; outputIndex++) {
    const probability = Math.exp(
      probabilities[outputIndex] - maximumLogit,
    );
    probabilities[outputIndex] = probability;
    probabilityTotal += probability;
  }
  for (let outputIndex = 0; outputIndex < probabilities.length; outputIndex++) {
    probabilities[outputIndex] /= probabilityTotal;
  }
  return { hidden, probabilities };
};

const highestProbabilityIndex = (probabilities: Float32Array) => {
  let bestIndex = 0;
  for (let index = 1; index < probabilities.length; index++) {
    if (probabilities[index] > probabilities[bestIndex]) bestIndex = index;
  }
  return bestIndex;
};

const validateWindows = (
  windows: readonly PreprocessedWindow[],
  labels: readonly GestureLabel[],
  inputValueCount: number,
  partitionName: string,
) => {
  if (!windows.length) {
    throw new Error(`${partitionName} requires at least one window`);
  }
  const labelSet = new Set(labels);
  for (const window of windows) {
    if (window.values.length !== inputValueCount) {
      throw new Error(
        `${partitionName} window "${window.segmentId}" has ${window.values.length} values; expected ${inputValueCount}`,
      );
    }
    if (!labelSet.has(window.label)) {
      throw new Error(
        `${partitionName} window label "${window.label}" is not in the model label set`,
      );
    }
  }
};

const evaluateParameters = (
  parameters: AnnParameters,
  windows: readonly PreprocessedWindow[],
  labelIndices: ReadonlyMap<GestureLabel, number>,
): ClassificationMetrics => {
  if (!windows.length) {
    return { loss: Number.NaN, accuracy: Number.NaN, windowCount: 0 };
  }
  let loss = 0;
  let correct = 0;
  for (const window of windows) {
    const targetIndex = labelIndices.get(window.label);
    if (targetIndex === undefined) {
      throw new Error(`Unknown model label "${window.label}"`);
    }
    const { probabilities } = forward(parameters, window.values);
    loss -= Math.log(Math.max(probabilities[targetIndex], 1e-7));
    if (highestProbabilityIndex(probabilities) === targetIndex) correct++;
  }
  return {
    loss: loss / windows.length,
    accuracy: correct / windows.length,
    windowCount: windows.length,
  };
};

const updateWithAdam = (
  values: Float32Array,
  gradients: Float32Array,
  firstMoment: Float32Array,
  secondMoment: Float32Array,
  step: number,
  learningRate: number,
  gradientScale: number,
) => {
  const beta1 = 0.9;
  const beta2 = 0.999;
  const firstCorrection = 1 - beta1 ** step;
  const secondCorrection = 1 - beta2 ** step;
  for (let index = 0; index < values.length; index++) {
    const gradient = gradients[index] * gradientScale;
    firstMoment[index] =
      beta1 * firstMoment[index] + (1 - beta1) * gradient;
    secondMoment[index] =
      beta2 * secondMoment[index] + (1 - beta2) * gradient * gradient;
    const correctedFirst = firstMoment[index] / firstCorrection;
    const correctedSecond = secondMoment[index] / secondCorrection;
    values[index] -=
      learningRate * correctedFirst / (Math.sqrt(correctedSecond) + 1e-8);
  }
};

const defaultYieldAfterEpoch = () =>
  new Promise<void>((resolve) => setTimeout(resolve, 0));

export async function trainBaselineAnn(
  options: BaselineAnnTrainingOptions,
): Promise<BaselineAnnTrainingResult> {
  const config: BaselineAnnConfig = {
    ...BASELINE_ANN_CONFIG_V1,
    ...options.config,
  };
  if (options.labels.length < 2) {
    throw new Error("Baseline ANN training requires at least two labels");
  }
  if (
    options.inputValueCount <= 0 ||
    config.hiddenUnitCount <= 0 ||
    config.epochCount <= 0 ||
    config.batchSize <= 0 ||
    config.learningRate <= 0 ||
    config.l2Regularization < 0
  ) {
    throw new Error("Baseline ANN training configuration is invalid");
  }
  validateWindows(
    options.trainWindows,
    options.labels,
    options.inputValueCount,
    "Train partition",
  );
  validateWindows(
    options.validationWindows,
    options.labels,
    options.inputValueCount,
    "Validation partition",
  );

  const labels = [...options.labels];
  const labelIndices = new Map(
    labels.map((label, index) => [label, index]),
  );
  const random = createRandom(config.seed);
  const parameters = initializeParameters(
    options.inputValueCount,
    config.hiddenUnitCount,
    labels.length,
    random,
  );
  const parameterArrays = [
    parameters.inputWeights,
    parameters.hiddenBias,
    parameters.outputWeights,
    parameters.outputBias,
  ];
  const gradients = parameterArrays.map(
    (values) => new Float32Array(values.length),
  );
  const firstMoments = parameterArrays.map(
    (values) => new Float32Array(values.length),
  );
  const secondMoments = parameterArrays.map(
    (values) => new Float32Array(values.length),
  );
  const history: TrainingEpochMetrics[] = [];
  let optimizerStep = 0;

  for (let epoch = 1; epoch <= config.epochCount; epoch++) {
    if (options.signal?.aborted) {
      throw new Error("ANN training cancelled");
    }
    const indices = shuffledIndices(options.trainWindows.length, random);
    for (
      let batchStart = 0;
      batchStart < indices.length;
      batchStart += config.batchSize
    ) {
      for (const gradient of gradients) gradient.fill(0);
      const batchEnd = Math.min(
        batchStart + config.batchSize,
        indices.length,
      );

      for (let position = batchStart; position < batchEnd; position++) {
        const window = options.trainWindows[indices[position]];
        const targetIndex = labelIndices.get(window.label)!;
        const { hidden, probabilities } = forward(
          parameters,
          window.values,
        );
        probabilities[targetIndex] -= 1;

        for (let outputIndex = 0; outputIndex < labels.length; outputIndex++) {
          const outputGradient = probabilities[outputIndex];
          gradients[3][outputIndex] += outputGradient;
          const outputOffset = outputIndex * config.hiddenUnitCount;
          for (
            let hiddenIndex = 0;
            hiddenIndex < config.hiddenUnitCount;
            hiddenIndex++
          ) {
            gradients[2][outputOffset + hiddenIndex] +=
              outputGradient * hidden[hiddenIndex];
          }
        }

        for (
          let hiddenIndex = 0;
          hiddenIndex < config.hiddenUnitCount;
          hiddenIndex++
        ) {
          if (hidden[hiddenIndex] <= 0) continue;
          let hiddenGradient = 0;
          for (let outputIndex = 0; outputIndex < labels.length; outputIndex++) {
            hiddenGradient +=
              probabilities[outputIndex] *
              parameters.outputWeights[
                outputIndex * config.hiddenUnitCount + hiddenIndex
              ];
          }
          gradients[1][hiddenIndex] += hiddenGradient;
          const inputOffset = hiddenIndex * options.inputValueCount;
          for (
            let inputIndex = 0;
            inputIndex < options.inputValueCount;
            inputIndex++
          ) {
            gradients[0][inputOffset + inputIndex] +=
              hiddenGradient * window.values[inputIndex];
          }
        }
      }

      for (let index = 0; index < parameters.inputWeights.length; index++) {
        gradients[0][index] +=
          config.l2Regularization * parameters.inputWeights[index];
      }
      for (let index = 0; index < parameters.outputWeights.length; index++) {
        gradients[2][index] +=
          config.l2Regularization * parameters.outputWeights[index];
      }

      optimizerStep++;
      const gradientScale = 1 / (batchEnd - batchStart);
      for (let arrayIndex = 0; arrayIndex < parameterArrays.length; arrayIndex++) {
        updateWithAdam(
          parameterArrays[arrayIndex],
          gradients[arrayIndex],
          firstMoments[arrayIndex],
          secondMoments[arrayIndex],
          optimizerStep,
          config.learningRate,
          gradientScale,
        );
      }
    }

    const metrics: TrainingEpochMetrics = {
      epoch,
      train: evaluateParameters(
        parameters,
        options.trainWindows,
        labelIndices,
      ),
      validation: evaluateParameters(
        parameters,
        options.validationWindows,
        labelIndices,
      ),
    };
    history.push(metrics);
    options.onEpoch?.(metrics);
    await (options.yieldAfterEpoch ?? defaultYieldAfterEpoch)();
  }

  const model: TrainedMyoArmClassifier = {
    id: config.id,
    inputValueCount: options.inputValueCount,
    labels,
    predict(values) {
      const { probabilities } = forward(parameters, values);
      const bestIndex = highestProbabilityIndex(probabilities);
      return {
        label: labels[bestIndex],
        confidence: probabilities[bestIndex],
        scores: labels.map((label, index) => ({
          label,
          probability: probabilities[index],
        })),
      };
    },
    evaluate(windows) {
      return evaluateParameters(parameters, windows, labelIndices);
    },
  };

  return { model, history };
}
