import {
  CAPTURE_PROTOCOL_V1,
  GESTURES,
} from "../myoarm/protocol";
import {
  EMG_SAMPLE_RATE_HZ,
  type CaptureSource,
  type SampleChunk,
} from "../signal/types";
import type { CollectorSnapshot } from "../myoarm/collector";
import type { MyoArmSegment } from "../myoarm/types";
import {
  PREPROCESSING_CONFIG_V1,
  prepareMyoArmDataset,
  type PreparedMyoArmDataset,
  type PreprocessingIssueCode,
} from "../myoarm/preprocessing";
import {
  DATASET_SPLIT_CONFIG_V1,
  splitPreparedDataset,
  type DatasetGroupBy,
  type DatasetPartition,
  type DatasetPartitionName,
  type GroupedDatasetSplit,
} from "../myoarm/split";
import {
  BASELINE_ANN_CONFIG_V1,
  trainBaselineAnn,
  type TrainingEpochMetrics,
} from "../myoarm/baseline-ann";
import type {
  ClassificationMetrics,
  ClassificationPrediction,
  TrainedMyoArmClassifier,
} from "../myoarm/classifier";
import { createLiveWindowStream } from "../myoarm/live-inference";
import {
  calculateChannelStatistics,
  createMyoArmSegmentPlot,
} from "./myoarm-segment";
import { loadMujocoRuntime } from "../simulation/mujoco-runtime";
import {
  loadShadowHandModel,
  type ShadowHandModel,
} from "../simulation/shadow-hand-model";

export interface MyoArmView {
  acceptSamples(chunk: SampleChunk): void;
  dispose(): void;
  setCollectedSegments(segments: readonly MyoArmSegment[]): void;
  setCollectionState(snapshot: CollectorSnapshot): void;
  setPersistenceState(
    status: PersistenceStatus,
    message: string,
  ): void;
  setSourceState(
    source: CaptureSource | null,
    active: boolean,
    sourceName?: string,
  ): void;
}

interface MyoArmViewOptions {
  fixtures: readonly string[];
  onCollectFixture(fixtureName: string): void;
  onClearSavedSegments(): void;
  onExportDataset(): void;
  onImportDataset(file: File): void;
}

type PersistenceStatus =
  | "loading"
  | "ready"
  | "ready-error"
  | "saving"
  | "error";

const sourceLabel = (source: CaptureSource | null) => {
  if (source === "bluetooth") return "Bluetooth";
  if (source === "fixture") return "Fixture";
  if (source === "import") return "Imported data";
  return "No source";
};

export function createMyoArmView(
  root: HTMLElement,
  options: MyoArmViewOptions,
): MyoArmView {
  const protocol = CAPTURE_PROTOCOL_V1;
  const gestureNames = GESTURES.map((gesture) => gesture.displayName).join(", ");

  root.setAttribute("aria-labelledby", "myoarm-title");
  root.innerHTML = `
    <div class="myoarm-header">
      <h2 id="myoarm-title">MyoArm workspace</h2>
      <p>Start with the built-in recordings as a virtual sensor; connect real hardware later without changing the dataset contract.</p>
    </div>
    <div class="myoarm-grid">
      <article class="myoarm-card ready">
        <span class="myoarm-state" data-role="source-state">Waiting</span>
        <h3>Data source</h3>
        <p data-role="source-summary">Play a built-in fixture or connect a Mudra Link to verify the shared decoded-sample pipeline.</p>
        <dl class="myoarm-source-stats">
          <div><dt>Available fixtures</dt><dd>${options.fixtures.length}</dd></div>
          <div><dt>Current source</dt><dd data-role="source-name">No source</dd></div>
          <div><dt>Decoded samples</dt><dd data-role="sample-count">0</dd></div>
          <div><dt>Sample rate</dt><dd>${EMG_SAMPLE_RATE_HZ} Hz</dd></div>
        </dl>
      </article>
      <article class="myoarm-card ready">
        <span class="myoarm-state">Protocol v1 defined</span>
        <h3>Data collection</h3>
        <p>${gestureNames}</p>
        <ul>
          <li>${protocol.countdownMs / 1_000} s countdown</li>
          <li>${protocol.activeMs / 1_000} s stable capture</li>
          <li>${protocol.recoveryMs / 1_000} s recovery</li>
          <li>${protocol.repetitionsPerLabel} repetitions per label</li>
        </ul>
        <div class="myoarm-collector">
          <label>
            Fixture
            <select data-role="fixture-select" aria-label="Fixture to collect"></select>
          </label>
          <button type="button" data-role="collect-fixture">Collect fixture</button>
          <p data-role="collection-summary">Ready to collect a fixture.</p>
          <div class="myoarm-storage">
            <span data-role="storage-summary">Opening the local dataset…</span>
            <div class="myoarm-storage-actions">
              <button type="button" data-role="export-dataset" disabled>Export</button>
              <button type="button" data-role="import-dataset" disabled>Import</button>
              <button type="button" data-role="clear-segments" disabled>Clear</button>
            </div>
            <input data-role="import-file" type="file" accept=".zip,application/zip" hidden>
          </div>
          <ul class="myoarm-segment-list" data-role="segment-list">
            <li>No saved segments in the local fixture dataset.</li>
          </ul>
        </div>
      </article>
      <article class="myoarm-card ready myoarm-inspector">
        <div class="myoarm-inspector-header">
          <div>
            <h3>Saved segment inspector</h3>
            <p>Select a saved segment to inspect its unprocessed samples before preprocessing or training.</p>
          </div>
          <span class="myoarm-quality" data-role="preview-quality">No selection</span>
        </div>
        <div class="myoarm-preview-empty" data-role="preview-empty">
          Collect or import a segment, then select it from the list above.
        </div>
        <div data-role="preview-content" hidden>
          <dl class="myoarm-preview-meta">
            <div><dt>Gesture</dt><dd data-role="preview-gesture">—</dd></div>
            <div><dt>Effort</dt><dd data-role="preview-effort">—</dd></div>
            <div><dt>Repetition</dt><dd data-role="preview-repetition">—</dd></div>
            <div><dt>Duration</dt><dd data-role="preview-duration">—</dd></div>
            <div><dt>Samples</dt><dd data-role="preview-samples">—</dd></div>
            <div><dt>Sample rate</dt><dd data-role="preview-rate">—</dd></div>
            <div><dt>Quality flags</dt><dd data-role="preview-flags">—</dd></div>
            <div><dt>Segment ID</dt><dd data-role="preview-id">—</dd></div>
          </dl>
          <div class="myoarm-waveform-wrap">
            <canvas data-role="segment-waveform"></canvas>
          </div>
          <div class="myoarm-channel-stats" data-role="channel-stats"></div>
          <p class="myoarm-preview-note">
            Raw decoded counts are shown without filtering, normalization, or calibration to µV.
            Boundary hits report samples at the signed 16-bit limits and do not change the saved quality decision.
          </p>
        </div>
      </article>
      <article class="myoarm-card ready myoarm-training-card">
        <span class="myoarm-state" data-role="preprocessing-state">No windows</span>
        <h3>Model training</h3>
        <p>Shared preprocessing prepares the same input shape for offline training and future live inference.</p>
        <dl class="myoarm-preprocessing-stats">
          <div><dt>Usable segments</dt><dd data-role="preprocessing-segments">0</dd></div>
          <div><dt>Skipped segments</dt><dd data-role="preprocessing-skipped">0</dd></div>
          <div><dt>Prepared windows</dt><dd data-role="preprocessing-windows">0</dd></div>
          <div><dt>Values / window</dt><dd data-role="preprocessing-inputs">0</dd></div>
        </dl>
        <p class="myoarm-preprocessing-rule" data-role="preprocessing-rule"></p>
        <div class="myoarm-label-distribution" data-role="label-distribution"></div>
        <p class="myoarm-preprocessing-warning" data-role="preprocessing-warning">
          Collect or import accepted segments to prepare model inputs.
        </p>
        <div class="myoarm-split-controls">
          <label>
            Group split by
            <select data-role="split-grouping" aria-label="Dataset split grouping">
              <option value="segment">Segment / repetition</option>
              <option value="session">Recording session</option>
            </select>
          </label>
          <span class="myoarm-split-state" data-role="split-state">Not ready</span>
        </div>
        <p class="myoarm-split-rule">
          Deterministic 60% train / 20% validation / 20% test allocation.
        </p>
        <div class="myoarm-split-grid">
          <section data-role="split-train">
            <h4>Train</h4>
            <strong data-role="split-windows">0 windows</strong>
            <span data-role="split-groups">0 groups</span>
            <div data-role="split-labels"></div>
          </section>
          <section data-role="split-validation">
            <h4>Validation</h4>
            <strong data-role="split-windows">0 windows</strong>
            <span data-role="split-groups">0 groups</span>
            <div data-role="split-labels"></div>
          </section>
          <section data-role="split-test">
            <h4>Test</h4>
            <strong data-role="split-windows">0 windows</strong>
            <span data-role="split-groups">0 groups</span>
            <div data-role="split-labels"></div>
          </section>
        </div>
        <p class="myoarm-split-warning" data-role="split-warning">
          At least three independent groups per observed label are required.
        </p>
        <div class="myoarm-training-controls">
          <label>
            Epochs
            <input data-role="training-epochs" type="number" min="5" max="100" step="5" value="${BASELINE_ANN_CONFIG_V1.epochCount}">
          </label>
          <button type="button" data-role="train-ann" disabled>Train baseline ANN</button>
          <span class="myoarm-training-state" data-role="training-state">Waiting for a ready split</span>
        </div>
        <p class="myoarm-ann-architecture" data-role="ann-architecture">
          501 inputs → ${BASELINE_ANN_CONFIG_V1.hiddenUnitCount} ReLU units → observed labels · Adam optimizer
        </p>
        <div class="myoarm-loss-panel">
          <canvas data-role="loss-chart" aria-label="Training and validation loss chart"></canvas>
          <dl class="myoarm-training-metrics">
            <div><dt>Epoch</dt><dd data-role="metric-epoch">—</dd></div>
            <div><dt>Train loss</dt><dd data-role="metric-train-loss">—</dd></div>
            <div><dt>Validation loss</dt><dd data-role="metric-validation-loss">—</dd></div>
            <div><dt>Test accuracy</dt><dd data-role="metric-test-accuracy">—</dd></div>
          </dl>
        </div>
        <p class="myoarm-training-note" data-role="training-note">
          The model stays in memory after training. Fixture metrics verify execution only and are not evidence of real-world accuracy.
        </p>
      </article>
      <article class="myoarm-card myoarm-inference-card">
        <span class="myoarm-state" data-role="inference-state">No model</span>
        <h3>Inference</h3>
        <p>Play any built-in recording after training. A new prediction is produced every 100 ms from a 200 ms window.</p>
        <div class="myoarm-prediction">
          <strong data-role="prediction-label">—</strong>
          <span data-role="prediction-confidence">Train a model first</span>
        </div>
        <div class="myoarm-probabilities" data-role="prediction-scores"></div>
        <p class="myoarm-inference-meta" data-role="inference-meta">
          The live stream uses the same channel order, scaling, and window size as training.
        </p>
      </article>
      <article class="myoarm-card myoarm-hand-card">
        <span class="myoarm-state" data-role="mujoco-state">Not loaded</span>
        <h3>Hand visualization</h3>
        <p>
          MuJoCo is an optional simulation runtime. It is downloaded and initialized
          only after you request it, so data collection and ANN training stay lightweight.
        </p>
        <div class="myoarm-mujoco-controls">
          <button type="button" data-role="initialize-mujoco">Initialize MuJoCo</button>
          <span data-role="mujoco-message">No MuJoCo resources have been requested.</span>
        </div>
        <dl class="myoarm-mujoco-stats">
          <div><dt>Runtime</dt><dd>Official single-thread WebAssembly</dd></div>
          <div><dt>Engine</dt><dd data-role="mujoco-version">Not loaded</dd></div>
          <div><dt>Hand model</dt><dd data-role="shadow-hand-model">Not loaded</dd></div>
        </dl>
      </article>
    </div>
  `;

  const stateEl = root.querySelector<HTMLElement>("[data-role=source-state]")!;
  const summaryEl = root.querySelector<HTMLElement>("[data-role=source-summary]")!;
  const sourceNameEl = root.querySelector<HTMLElement>("[data-role=source-name]")!;
  const sampleCountEl = root.querySelector<HTMLElement>("[data-role=sample-count]")!;
  const fixtureSelect =
    root.querySelector<HTMLSelectElement>("[data-role=fixture-select]")!;
  const collectFixtureBtn =
    root.querySelector<HTMLButtonElement>("[data-role=collect-fixture]")!;
  const collectionSummaryEl =
    root.querySelector<HTMLElement>("[data-role=collection-summary]")!;
  const storageSummaryEl =
    root.querySelector<HTMLElement>("[data-role=storage-summary]")!;
  const exportDatasetBtn =
    root.querySelector<HTMLButtonElement>("[data-role=export-dataset]")!;
  const importDatasetBtn =
    root.querySelector<HTMLButtonElement>("[data-role=import-dataset]")!;
  const clearSegmentsBtn =
    root.querySelector<HTMLButtonElement>("[data-role=clear-segments]")!;
  const importFileInput =
    root.querySelector<HTMLInputElement>("[data-role=import-file]")!;
  const segmentListEl =
    root.querySelector<HTMLUListElement>("[data-role=segment-list]")!;
  const previewQualityEl =
    root.querySelector<HTMLElement>("[data-role=preview-quality]")!;
  const previewEmptyEl =
    root.querySelector<HTMLElement>("[data-role=preview-empty]")!;
  const previewContentEl =
    root.querySelector<HTMLElement>("[data-role=preview-content]")!;
  const previewGestureEl =
    root.querySelector<HTMLElement>("[data-role=preview-gesture]")!;
  const previewEffortEl =
    root.querySelector<HTMLElement>("[data-role=preview-effort]")!;
  const previewRepetitionEl =
    root.querySelector<HTMLElement>("[data-role=preview-repetition]")!;
  const previewDurationEl =
    root.querySelector<HTMLElement>("[data-role=preview-duration]")!;
  const previewSamplesEl =
    root.querySelector<HTMLElement>("[data-role=preview-samples]")!;
  const previewRateEl =
    root.querySelector<HTMLElement>("[data-role=preview-rate]")!;
  const previewFlagsEl =
    root.querySelector<HTMLElement>("[data-role=preview-flags]")!;
  const previewIdEl =
    root.querySelector<HTMLElement>("[data-role=preview-id]")!;
  const channelStatsEl =
    root.querySelector<HTMLElement>("[data-role=channel-stats]")!;
  const waveformCanvas =
    root.querySelector<HTMLCanvasElement>("[data-role=segment-waveform]")!;
  const segmentPlot = createMyoArmSegmentPlot(waveformCanvas);
  const preprocessingStateEl =
    root.querySelector<HTMLElement>("[data-role=preprocessing-state]")!;
  const preprocessingSegmentsEl =
    root.querySelector<HTMLElement>("[data-role=preprocessing-segments]")!;
  const preprocessingSkippedEl =
    root.querySelector<HTMLElement>("[data-role=preprocessing-skipped]")!;
  const preprocessingWindowsEl =
    root.querySelector<HTMLElement>("[data-role=preprocessing-windows]")!;
  const preprocessingInputsEl =
    root.querySelector<HTMLElement>("[data-role=preprocessing-inputs]")!;
  const preprocessingRuleEl =
    root.querySelector<HTMLElement>("[data-role=preprocessing-rule]")!;
  const labelDistributionEl =
    root.querySelector<HTMLElement>("[data-role=label-distribution]")!;
  const preprocessingWarningEl =
    root.querySelector<HTMLElement>("[data-role=preprocessing-warning]")!;
  const splitGroupingSelect =
    root.querySelector<HTMLSelectElement>("[data-role=split-grouping]")!;
  const splitStateEl =
    root.querySelector<HTMLElement>("[data-role=split-state]")!;
  const splitWarningEl =
    root.querySelector<HTMLElement>("[data-role=split-warning]")!;
  const trainingEpochsInput =
    root.querySelector<HTMLInputElement>("[data-role=training-epochs]")!;
  const trainAnnButton =
    root.querySelector<HTMLButtonElement>("[data-role=train-ann]")!;
  const trainingStateEl =
    root.querySelector<HTMLElement>("[data-role=training-state]")!;
  const annArchitectureEl =
    root.querySelector<HTMLElement>("[data-role=ann-architecture]")!;
  const lossCanvas =
    root.querySelector<HTMLCanvasElement>("[data-role=loss-chart]")!;
  const metricEpochEl =
    root.querySelector<HTMLElement>("[data-role=metric-epoch]")!;
  const metricTrainLossEl =
    root.querySelector<HTMLElement>("[data-role=metric-train-loss]")!;
  const metricValidationLossEl =
    root.querySelector<HTMLElement>("[data-role=metric-validation-loss]")!;
  const metricTestAccuracyEl =
    root.querySelector<HTMLElement>("[data-role=metric-test-accuracy]")!;
  const trainingNoteEl =
    root.querySelector<HTMLElement>("[data-role=training-note]")!;
  const inferenceStateEl =
    root.querySelector<HTMLElement>("[data-role=inference-state]")!;
  const predictionLabelEl =
    root.querySelector<HTMLElement>("[data-role=prediction-label]")!;
  const predictionConfidenceEl =
    root.querySelector<HTMLElement>("[data-role=prediction-confidence]")!;
  const predictionScoresEl =
    root.querySelector<HTMLElement>("[data-role=prediction-scores]")!;
  const inferenceMetaEl =
    root.querySelector<HTMLElement>("[data-role=inference-meta]")!;
  const mujocoStateEl =
    root.querySelector<HTMLElement>("[data-role=mujoco-state]")!;
  const initializeMujocoButton =
    root.querySelector<HTMLButtonElement>("[data-role=initialize-mujoco]")!;
  const mujocoMessageEl =
    root.querySelector<HTMLElement>("[data-role=mujoco-message]")!;
  const mujocoVersionEl =
    root.querySelector<HTMLElement>("[data-role=mujoco-version]")!;
  const shadowHandModelEl =
    root.querySelector<HTMLElement>("[data-role=shadow-hand-model]")!;
  let shadowHandModel: ShadowHandModel | null = null;

  initializeMujocoButton.addEventListener("click", async () => {
    initializeMujocoButton.disabled = true;
    initializeMujocoButton.textContent = "Initializing…";
    mujocoStateEl.textContent = "Loading";
    mujocoStateEl.classList.add("live");
    mujocoMessageEl.dataset.status = "loading";
    mujocoMessageEl.textContent =
      "Downloading the optional runtime and compiling its WebAssembly module…";

    try {
      const runtime = await loadMujocoRuntime();
      shadowHandModel = await loadShadowHandModel(runtime);
      mujocoStateEl.textContent = "Ready";
      initializeMujocoButton.textContent = "Shadow Hand loaded";
      mujocoVersionEl.textContent =
        `${runtime.version} (${runtime.versionNumber})`;
      const { stats } = shadowHandModel;
      shadowHandModelEl.textContent =
        `E3M5 right · ${stats.degreesOfFreedom} DoF · ` +
        `${stats.actuators} actuators · ${stats.meshes} meshes`;
      mujocoMessageEl.dataset.status = "ready";
      mujocoMessageEl.textContent =
        "Model compiled into MjModel/MjData. Rendering is the next milestone.";
    } catch (error) {
      shadowHandModel?.dispose();
      shadowHandModel = null;
      mujocoStateEl.textContent = "Load failed";
      mujocoStateEl.classList.remove("live");
      initializeMujocoButton.disabled = false;
      initializeMujocoButton.textContent = "Retry MuJoCo";
      mujocoVersionEl.textContent = "Not loaded";
      shadowHandModelEl.textContent = "Not loaded";
      mujocoMessageEl.dataset.status = "error";
      mujocoMessageEl.textContent =
        error instanceof Error ? error.message : "MuJoCo initialization failed.";
    }
  });

  const splitPartitionElements = new Map<
    DatasetPartitionName,
    {
      root: HTMLElement;
      windows: HTMLElement;
      groups: HTMLElement;
      labels: HTMLElement;
    }
  >();
  for (const partitionName of [
    "train",
    "validation",
    "test",
  ] as const) {
    const partitionRoot = root.querySelector<HTMLElement>(
      `[data-role=split-${partitionName}]`,
    )!;
    splitPartitionElements.set(partitionName, {
      root: partitionRoot,
      windows: partitionRoot.querySelector<HTMLElement>(
        "[data-role=split-windows]",
      )!,
      groups: partitionRoot.querySelector<HTMLElement>(
        "[data-role=split-groups]",
      )!,
      labels: partitionRoot.querySelector<HTMLElement>(
        "[data-role=split-labels]",
      )!,
    });
  }

  fixtureSelect.replaceChildren(
    ...options.fixtures.map((fixture) => {
      const option = document.createElement("option");
      option.value = fixture;
      option.textContent = fixture;
      return option;
    }),
  );
  collectFixtureBtn.addEventListener("click", () => {
    if (fixtureSelect.value) options.onCollectFixture(fixtureSelect.value);
  });
  exportDatasetBtn.addEventListener("click", options.onExportDataset);
  importDatasetBtn.addEventListener("click", () => importFileInput.click());
  importFileInput.addEventListener("change", () => {
    const file = importFileInput.files?.[0];
    importFileInput.value = "";
    if (file) options.onImportDataset(file);
  });
  clearSegmentsBtn.addEventListener("click", options.onClearSavedSegments);

  let activeSource: CaptureSource | null = null;
  let activeSourceName = "";
  let active = false;
  let sampleCount = 0;
  let renderRaf = 0;
  let collectorStatus: CollectorSnapshot["status"] = "idle";
  let persistenceStatus: PersistenceStatus = "loading";
  let storedSegmentCount = 0;
  let selectedSegmentId: string | null = null;
  let preparedDataset: PreparedMyoArmDataset | null = null;
  let currentSplit: GroupedDatasetSplit | null = null;
  let trainedModel: TrainedMyoArmClassifier | null = null;
  let training = false;
  let trainingAbortController: AbortController | null = null;
  let trainingHistory: TrainingEpochMetrics[] = [];
  let predictionHistory: ClassificationPrediction[] = [];
  let predictionWindowCount = 0;
  const liveWindowStream = createLiveWindowStream();

  const preprocessingIssueLabel: Record<PreprocessingIssueCode, string> = {
    "quality-rejected": "quality rejected",
    "sample-rate-mismatch": "sample rate mismatch",
    "channel-order-mismatch": "channel order mismatch",
    "sample-layout-mismatch": "sample layout mismatch",
    "too-short": "too short",
  };

  const renderPartition = (
    partition: DatasetPartition,
    groupBy: DatasetGroupBy,
  ) => {
    const elements = splitPartitionElements.get(partition.name)!;
    elements.windows.textContent =
      `${partition.summary.windowCount.toLocaleString()} windows`;
    elements.groups.textContent =
      `${partition.summary.groupCount.toLocaleString()} ${groupBy} group(s) · ` +
      `${partition.summary.segmentCount.toLocaleString()} segment(s)`;
    const labelCounts = Object.entries(
      partition.summary.windowsByLabel,
    ).filter(([, count]) => count > 0);
    elements.labels.replaceChildren(
      ...(labelCounts.length
        ? labelCounts.map(([label, count]) => {
            const item = document.createElement("span");
            item.textContent = `${label}: ${count.toLocaleString()}`;
            return item;
          })
        : [Object.assign(document.createElement("span"), {
            textContent: "No windows",
          })]),
    );
    elements.root.classList.toggle(
      "empty",
      partition.summary.windowCount === 0,
    );
  };

  const drawLossHistory = () => {
    const ratio = window.devicePixelRatio || 1;
    const width = Math.max(lossCanvas.clientWidth, 320);
    const height = Math.max(lossCanvas.clientHeight, 150);
    lossCanvas.width = Math.round(width * ratio);
    lossCanvas.height = Math.round(height * ratio);
    const context = lossCanvas.getContext("2d");
    if (!context) return;
    context.scale(ratio, ratio);
    context.clearRect(0, 0, width, height);
    context.fillStyle = "#fff";
    context.fillRect(0, 0, width, height);

    const left = 38;
    const right = 10;
    const top = 12;
    const bottom = 25;
    const plotWidth = width - left - right;
    const plotHeight = height - top - bottom;
    context.strokeStyle = "#d7d4ca";
    context.lineWidth = 1;
    context.beginPath();
    context.moveTo(left, top);
    context.lineTo(left, top + plotHeight);
    context.lineTo(left + plotWidth, top + plotHeight);
    context.stroke();
    context.fillStyle = "#8a877a";
    context.font = "10px system-ui";
    context.fillText("Loss", 6, 12);
    context.fillText("Epoch", width - 40, height - 6);

    if (!trainingHistory.length) {
      context.fillStyle = "#aaa69a";
      context.fillText("Loss appears here during training", left + 16, top + plotHeight / 2);
      return;
    }

    const losses = trainingHistory.flatMap((entry) => [
      entry.train.loss,
      entry.validation.loss,
    ]);
    const maximumLoss = Math.max(...losses, 0.01);
    context.fillStyle = "#8a877a";
    context.fillText(maximumLoss.toFixed(2), 4, top + 4);
    context.fillText("0", 24, top + plotHeight + 4);

    const drawSeries = (
      color: string,
      valueFor: (entry: TrainingEpochMetrics) => number,
    ) => {
      context.strokeStyle = color;
      context.lineWidth = 2;
      context.beginPath();
      trainingHistory.forEach((entry, index) => {
        const x =
          left +
          (trainingHistory.length === 1
            ? 0
            : (index / (trainingHistory.length - 1)) * plotWidth);
        const y =
          top + plotHeight - (valueFor(entry) / maximumLoss) * plotHeight;
        if (index === 0) context.moveTo(x, y);
        else context.lineTo(x, y);
      });
      context.stroke();
    };
    drawSeries("#c96442", (entry) => entry.train.loss);
    drawSeries("#214f88", (entry) => entry.validation.loss);
    context.fillStyle = "#c96442";
    context.fillText("Train", left + 6, height - 6);
    context.fillStyle = "#214f88";
    context.fillText("Validation", left + 44, height - 6);
  };

  const renderInferenceIdle = (message: string) => {
    inferenceStateEl.textContent = trainedModel ? "Model ready" : "No model";
    inferenceStateEl.classList.toggle("live", Boolean(trainedModel));
    predictionLabelEl.textContent = "—";
    predictionConfidenceEl.textContent = message;
    predictionScoresEl.replaceChildren();
    inferenceMetaEl.textContent = trainedModel
      ? `${trainedModel.id} · ${trainedModel.labels.join(", ")} · awaiting a new stream`
      : "Train the baseline ANN before playing a fixture for prediction.";
  };

  const updateTrainingControls = () => {
    const trainable =
      Boolean(currentSplit?.ready) &&
      (currentSplit?.observedLabels.length ?? 0) >= 2;
    trainAnnButton.disabled = training || active || !trainable;
    trainingEpochsInput.disabled = training;
    if (!training && !trainable) {
      trainingStateEl.textContent =
        currentSplit?.ready
          ? "At least two labels are required"
          : "Waiting for a ready split";
    } else if (!training && trainedModel) {
      trainingStateEl.textContent = "Model trained";
    } else if (!training) {
      trainingStateEl.textContent = "Ready to train";
    }
  };

  const invalidateTrainedModel = (reason: string) => {
    trainingAbortController?.abort();
    trainingAbortController = null;
    training = false;
    trainedModel = null;
    trainingHistory = [];
    predictionHistory = [];
    predictionWindowCount = 0;
    liveWindowStream.reset();
    metricEpochEl.textContent = "—";
    metricTrainLossEl.textContent = "—";
    metricValidationLossEl.textContent = "—";
    metricTestAccuracyEl.textContent = "—";
    trainingNoteEl.textContent = reason;
    drawLossHistory();
    renderInferenceIdle("Train a model first");
    updateTrainingControls();
  };

  const renderDatasetSplit = () => {
    const groupBy = splitGroupingSelect.value as DatasetGroupBy;
    if (!preparedDataset) return;
    const split = splitPreparedDataset(preparedDataset, {
      ...DATASET_SPLIT_CONFIG_V1,
      groupBy,
    });
    currentSplit = split;
    renderPartition(split.partitions.train, groupBy);
    renderPartition(split.partitions.validation, groupBy);
    renderPartition(split.partitions.test, groupBy);

    splitStateEl.textContent = split.ready ? "Ready" : "Not ready";
    splitStateEl.classList.toggle("ready", split.ready);
    updateTrainingControls();
    if (!split.observedLabels.length) {
      splitWarningEl.textContent =
        "Collect or import accepted segments before creating a split.";
      return;
    }
    if (split.ready) {
      splitWarningEl.textContent =
        `Ready: ${split.totalGroupCount.toLocaleString()} independent ${groupBy} groups, ` +
        "with every observed label represented in train, validation, and test.";
      return;
    }

    const missingCoverage = ([
      "train",
      "validation",
      "test",
    ] as const)
      .map((partitionName) => {
        const missing =
          split.partitions[partitionName].summary.missingLabels;
        return missing.length
          ? `${partitionName}: ${missing.join(", ")}`
          : null;
      })
      .filter((message): message is string => Boolean(message));
    const requirement =
      groupBy === "segment"
        ? "Collect at least three independent repetitions for every observed label; the protocol target is ten."
        : "Collect at least three independent sessions containing every observed label.";
    splitWarningEl.textContent =
      `Not ready for evaluation. Missing label coverage — ${missingCoverage.join("; ")}. ${requirement}`;
  };

  splitGroupingSelect.addEventListener("change", () => {
    invalidateTrainedModel(
      "The grouping mode changed. Train a new model before using live inference.",
    );
    renderDatasetSplit();
  });

  const formatMetrics = (metrics: ClassificationMetrics) =>
    `${(metrics.accuracy * 100).toFixed(1)}% accuracy · loss ${metrics.loss.toFixed(4)}`;

  const trainAnn = async () => {
    if (!preparedDataset || !currentSplit?.ready || training || active) return;
    const splitAtStart = currentSplit;
    const epochCount = Math.max(
      5,
      Math.min(100, Math.round(Number(trainingEpochsInput.value) || 30)),
    );
    trainingEpochsInput.value = String(epochCount);
    trainingAbortController?.abort();
    const controller = new AbortController();
    trainingAbortController = controller;
    training = true;
    trainedModel = null;
    trainingHistory = [];
    predictionHistory = [];
    predictionWindowCount = 0;
    trainingStateEl.textContent = "Training…";
    trainingNoteEl.textContent =
      "Training uses only Train windows; Validation is measured after each epoch. Test is evaluated once after training.";
    metricEpochEl.textContent = `0 / ${epochCount}`;
    metricTrainLossEl.textContent = "—";
    metricValidationLossEl.textContent = "—";
    metricTestAccuracyEl.textContent = "Held out";
    annArchitectureEl.textContent =
      `${preparedDataset.summary.inputValueCount} inputs → ` +
      `${BASELINE_ANN_CONFIG_V1.hiddenUnitCount} ReLU units → ` +
      `${splitAtStart.observedLabels.length} Softmax outputs · Adam optimizer`;
    drawLossHistory();
    renderInferenceIdle("Training in progress");
    updateTrainingControls();

    try {
      const result = await trainBaselineAnn({
        trainWindows: splitAtStart.partitions.train.windows,
        validationWindows: splitAtStart.partitions.validation.windows,
        labels: splitAtStart.observedLabels,
        inputValueCount: preparedDataset.summary.inputValueCount,
        config: { epochCount },
        signal: controller.signal,
        onEpoch(metrics) {
          if (controller.signal.aborted) return;
          trainingHistory.push(metrics);
          metricEpochEl.textContent = `${metrics.epoch} / ${epochCount}`;
          metricTrainLossEl.textContent = metrics.train.loss.toFixed(4);
          metricValidationLossEl.textContent =
            metrics.validation.loss.toFixed(4);
          trainingStateEl.textContent =
            `Epoch ${metrics.epoch}: train ${metrics.train.loss.toFixed(4)} · ` +
            `validation ${metrics.validation.loss.toFixed(4)}`;
          drawLossHistory();
        },
      });
      if (controller.signal.aborted || currentSplit !== splitAtStart) return;

      trainedModel = result.model;
      const finalMetrics = result.history[result.history.length - 1];
      const testMetrics = trainedModel.evaluate(
        splitAtStart.partitions.test.windows,
      );
      metricTestAccuracyEl.textContent =
        `${(testMetrics.accuracy * 100).toFixed(1)}%`;
      trainingStateEl.textContent = "Training complete";
      trainingNoteEl.textContent =
        `Final train: ${formatMetrics(finalMetrics.train)}. ` +
        `Validation: ${formatMetrics(finalMetrics.validation)}. ` +
        `Held-out test: ${formatMetrics(testMetrics)}. ` +
        "Fixture results are a pipeline check, not a real-world performance claim.";
      renderInferenceIdle("Play a fixture to start prediction");
    } catch (error) {
      if (!controller.signal.aborted) {
        trainingStateEl.textContent = "Training failed";
        trainingNoteEl.textContent = (error as Error).message;
        renderInferenceIdle("Training failed");
      }
    } finally {
      if (trainingAbortController === controller) {
        trainingAbortController = null;
        training = false;
        updateTrainingControls();
      }
    }
  };

  trainAnnButton.addEventListener("click", () => {
    void trainAnn();
  });

  const smoothPrediction = (
    prediction: ClassificationPrediction,
  ): ClassificationPrediction => {
    predictionHistory.push(prediction);
    if (predictionHistory.length > 5) predictionHistory.shift();
    const scores = prediction.scores.map(({ label }, index) => ({
      label,
      probability:
        predictionHistory.reduce(
          (total, item) => total + item.scores[index].probability,
          0,
        ) / predictionHistory.length,
    }));
    const best = scores.reduce((current, candidate) =>
      candidate.probability > current.probability ? candidate : current,
    );
    return {
      label: best.label,
      confidence: best.probability,
      scores,
    };
  };

  const renderPrediction = (
    prediction: ClassificationPrediction,
    endSampleExclusive: number,
  ) => {
    inferenceStateEl.textContent = "Predicting";
    inferenceStateEl.classList.add("live");
    predictionLabelEl.textContent = prediction.label;
    predictionConfidenceEl.textContent =
      `${(prediction.confidence * 100).toFixed(1)}% confidence`;
    predictionScoresEl.replaceChildren(
      ...[...prediction.scores]
        .sort((left, right) => right.probability - left.probability)
        .map((score) => {
          const row = document.createElement("div");
          const label = document.createElement("span");
          const track = document.createElement("div");
          const fill = document.createElement("i");
          const value = document.createElement("strong");
          label.textContent = score.label;
          fill.style.width = `${Math.max(0, Math.min(100, score.probability * 100))}%`;
          value.textContent = `${(score.probability * 100).toFixed(1)}%`;
          track.append(fill);
          row.append(label, track, value);
          return row;
        }),
    );
    inferenceMetaEl.textContent =
      `${activeSourceName || sourceLabel(activeSource)} · ` +
      `window ${predictionWindowCount.toLocaleString()} · ` +
      `through sample ${endSampleExclusive.toLocaleString()} · ` +
      "5-window probability average";
  };

  const renderPreprocessingSummary = (
    segments: readonly MyoArmSegment[],
  ) => {
    const prepared = prepareMyoArmDataset(segments);
    preparedDataset = prepared;
    const { config, summary } = prepared;
    preprocessingStateEl.textContent = summary.windowCount
      ? "Windows ready"
      : "No windows";
    preprocessingStateEl.classList.toggle(
      "live",
      summary.windowCount > 0,
    );
    preprocessingSegmentsEl.textContent =
      summary.usableSegmentCount.toLocaleString();
    preprocessingSkippedEl.textContent =
      summary.skippedSegmentCount.toLocaleString();
    preprocessingWindowsEl.textContent =
      summary.windowCount.toLocaleString();
    preprocessingInputsEl.textContent =
      summary.inputValueCount.toLocaleString();
    preprocessingRuleEl.textContent =
      `${config.windowDurationMs} ms window (${config.windowSampleCount} samples) · ` +
      `${config.strideDurationMs} ms stride (${config.strideSampleCount} samples) · ` +
      `sample-major ${config.channels.join(" / ")} · divide by 32,768`;

    const labelCounts = Object.entries(summary.windowsByLabel).filter(
      ([, count]) => count > 0,
    );
    labelDistributionEl.replaceChildren(
      ...(labelCounts.length
        ? labelCounts.map(([label, count]) => {
            const item = document.createElement("span");
            item.textContent = `${label}: ${count.toLocaleString()}`;
            return item;
          })
        : [Object.assign(document.createElement("span"), {
            textContent: "No labelled windows",
          })]),
    );

    if (!summary.windowCount) {
      preprocessingWarningEl.textContent =
        "Collect or import accepted segments to prepare model inputs.";
    } else if (summary.issues.length) {
      const reasons = Array.from(
        new Set(
          summary.issues.map((issue) => preprocessingIssueLabel[issue.code]),
        ),
      );
      preprocessingWarningEl.textContent =
        `${summary.skippedSegmentCount.toLocaleString()} segment(s) skipped: ${reasons.join(", ")}. ` +
        "Overlapping windows remain grouped by segment for future dataset splitting.";
    } else {
      preprocessingWarningEl.textContent =
        "Pipeline check only: overlapping windows remain grouped by segment. Fixtures must not be used to claim model accuracy.";
    }
    renderDatasetSplit();
  };

  const renderSegmentPreview = (segment: MyoArmSegment | null) => {
    selectedSegmentId = segment?.id ?? null;
    previewEmptyEl.hidden = Boolean(segment);
    previewContentEl.hidden = !segment;
    previewQualityEl.classList.toggle(
      "review",
      Boolean(segment && !segment.quality.accepted),
    );

    if (!segment) {
      previewQualityEl.textContent = "No selection";
      channelStatsEl.replaceChildren();
      segmentPlot.draw(null);
      return;
    }

    previewQualityEl.textContent = segment.quality.accepted
      ? "Accepted"
      : "Needs review";
    previewGestureEl.textContent = segment.label;
    previewEffortEl.textContent = segment.effort;
    previewRepetitionEl.textContent =
      segment.repetition.toLocaleString();
    previewDurationEl.textContent =
      `${(segment.durationMs / 1_000).toFixed(3)} s`;
    previewSamplesEl.textContent = segment.sampleCount.toLocaleString();
    previewRateEl.textContent = `${segment.sampleRateHz.toLocaleString()} Hz`;
    previewFlagsEl.textContent = segment.quality.flags.length
      ? segment.quality.flags.join(", ")
      : "None";
    previewIdEl.textContent = segment.id;

    channelStatsEl.replaceChildren(
      ...calculateChannelStatistics(segment).map((statistics) => {
        const card = document.createElement("section");
        card.className = "myoarm-channel-stat";
        const heading = document.createElement("h4");
        heading.textContent = statistics.channel;
        const details = document.createElement("dl");
        const rows: Array<[string, string]> = [
          ["Minimum", statistics.minimum.toLocaleString()],
          ["Maximum", statistics.maximum.toLocaleString()],
          ["Peak-to-peak", statistics.peakToPeak.toLocaleString()],
          [
            "RMS",
            statistics.rms.toLocaleString(undefined, {
              maximumFractionDigits: 1,
            }),
          ],
          ["Boundary hits", statistics.boundaryHits.toLocaleString()],
        ];
        details.replaceChildren(
          ...rows.map(([label, value]) => {
            const row = document.createElement("div");
            const term = document.createElement("dt");
            const description = document.createElement("dd");
            term.textContent = label;
            description.textContent = value;
            row.append(term, description);
            return row;
          }),
        );
        card.append(heading, details);
        return card;
      }),
    );
    segmentPlot.draw(segment);
  };

  const updateCollectionControls = () => {
    const collecting = collectorStatus === "collecting";
    const persistenceReady =
      persistenceStatus === "ready" ||
      persistenceStatus === "ready-error";
    fixtureSelect.disabled = collecting || !persistenceReady;
    collectFixtureBtn.disabled =
      collecting || !persistenceReady || options.fixtures.length === 0;
    exportDatasetBtn.disabled =
      collecting || !persistenceReady || storedSegmentCount === 0;
    importDatasetBtn.disabled = collecting || !persistenceReady;
    clearSegmentsBtn.disabled =
      collecting || !persistenceReady || storedSegmentCount === 0;
  };

  const renderStats = () => {
    renderRaf = 0;
    stateEl.textContent = active ? "Receiving samples" : "Waiting";
    stateEl.classList.toggle("live", active);
    sourceNameEl.textContent = activeSourceName || sourceLabel(activeSource);
    sampleCountEl.textContent = sampleCount.toLocaleString();

    if (active) {
      summaryEl.textContent = `${sourceLabel(activeSource)} is publishing decoded samples through the shared Sample Hub.`;
    } else if (sampleCount > 0) {
      summaryEl.textContent = `Last stream finished after ${sampleCount.toLocaleString()} decoded samples.`;
    } else {
      summaryEl.textContent =
        "Play a built-in fixture or connect a Mudra Link to verify the shared decoded-sample pipeline.";
    }
  };

  const scheduleRender = () => {
    if (!renderRaf) renderRaf = requestAnimationFrame(renderStats);
  };

  drawLossHistory();
  renderInferenceIdle("Train a model first");

  return {
    acceptSamples(chunk) {
      const channelCount = chunk.channels.length;
      if (!channelCount || chunk.samples.length % channelCount !== 0) return;
      if (!active || chunk.source !== activeSource) {
        active = true;
        activeSourceName = "";
        sampleCount = 0;
        liveWindowStream.reset();
        predictionHistory = [];
        predictionWindowCount = 0;
      }
      activeSource = chunk.source;
      sampleCount += Math.floor(chunk.samples.length / channelCount);
      if (trainedModel) {
        const windows = liveWindowStream.acceptSamples(chunk);
        for (const window of windows) {
          predictionWindowCount++;
          renderPrediction(
            smoothPrediction(trainedModel.predict(window.values)),
            window.endSampleExclusive,
          );
        }
      }
      scheduleRender();
    },

    dispose() {
      shadowHandModel?.dispose();
      shadowHandModel = null;
    },

    setCollectedSegments(segments) {
      if (trainedModel || training) {
        invalidateTrainedModel(
          "The saved dataset changed. Retrain the ANN so its weights and reported split remain consistent.",
        );
      }
      storedSegmentCount = segments.length;
      renderPreprocessingSummary(segments);
      if (!segments.length) {
        selectedSegmentId = null;
        segmentListEl.innerHTML =
          "<li>No saved segments in the local fixture dataset.</li>";
        renderSegmentPreview(null);
        updateCollectionControls();
        return;
      }

      const selectedSegment =
        segments.find((segment) => segment.id === selectedSegmentId) ??
        segments[segments.length - 1];
      segmentListEl.replaceChildren(
        ...segments.map((collected) => {
          const item = document.createElement("li");
          const button = document.createElement("button");
          const duration = (collected.durationMs / 1_000).toFixed(2);
          const quality = collected.quality.accepted ? "accepted" : "review";
          button.type = "button";
          button.className = "myoarm-segment-button";
          button.classList.toggle(
            "selected",
            collected.id === selectedSegment.id,
          );
          button.setAttribute(
            "aria-pressed",
            String(collected.id === selectedSegment.id),
          );
          button.textContent =
            `${collected.label} · ${collected.effort} · repetition ${collected.repetition} · ` +
            `${collected.sampleCount.toLocaleString()} samples · ${duration} s · ${quality}`;
          button.addEventListener("click", () => {
            renderSegmentPreview(collected);
            for (const candidate of segmentListEl.querySelectorAll(
              ".myoarm-segment-button",
            )) {
              const isSelected = candidate === button;
              candidate.classList.toggle("selected", isSelected);
              candidate.setAttribute("aria-pressed", String(isSelected));
            }
          });
          item.append(button);
          return item;
        }),
      );
      renderSegmentPreview(selectedSegment);
      updateCollectionControls();
    },

    setCollectionState(snapshot) {
      const collecting = snapshot.status === "collecting";
      collectorStatus = snapshot.status;
      collectFixtureBtn.textContent = collecting
        ? `Collecting ${snapshot.sampleCount.toLocaleString()} samples…`
        : "Collect fixture";
      collectionSummaryEl.textContent = snapshot.message;
      collectionSummaryEl.dataset.status = snapshot.status;
      updateCollectionControls();
    },

    setPersistenceState(status, message) {
      persistenceStatus = status;
      storageSummaryEl.textContent = message;
      storageSummaryEl.dataset.status = status;
      updateCollectionControls();
    },

    setSourceState(source, isActive, sourceName = "") {
      const startingNewStream = isActive && (!active || source !== activeSource);
      activeSource = source;
      activeSourceName = sourceName;
      active = isActive;
      if (startingNewStream) {
        sampleCount = 0;
        liveWindowStream.reset();
        predictionHistory = [];
        predictionWindowCount = 0;
        if (trainedModel) {
          renderInferenceIdle(
            `Buffering the first ${PREPROCESSING_CONFIG_V1.windowDurationMs} ms window…`,
          );
        }
      } else if (!isActive) {
        liveWindowStream.reset();
        predictionHistory = [];
        if (trainedModel) {
          inferenceStateEl.textContent = predictionWindowCount
            ? "Stream finished"
            : "Model ready";
          inferenceStateEl.classList.remove("live");
        }
      }
      updateTrainingControls();
      scheduleRender();
    },
  };
}
