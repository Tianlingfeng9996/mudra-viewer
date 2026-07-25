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

export interface MyoArmView {
  acceptSamples(chunk: SampleChunk): void;
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
}

type PersistenceStatus = "loading" | "ready" | "saving" | "error";

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
            <button type="button" data-role="clear-segments" disabled>Clear saved segments</button>
          </div>
          <ul class="myoarm-segment-list" data-role="segment-list">
            <li>No saved segments in the local fixture dataset.</li>
          </ul>
        </div>
      </article>
      <article class="myoarm-card">
        <h3>Model training</h3>
        <p>Dataset export, preprocessing, and browser-based training will build on the locally saved segments.</p>
      </article>
      <article class="myoarm-card">
        <h3>Inference</h3>
        <p>Real-time prediction output will appear here.</p>
      </article>
      <article class="myoarm-card">
        <h3>Hand visualization</h3>
        <p>The inferred hand pose visualization will appear here.</p>
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
  const clearSegmentsBtn =
    root.querySelector<HTMLButtonElement>("[data-role=clear-segments]")!;
  const segmentListEl =
    root.querySelector<HTMLUListElement>("[data-role=segment-list]")!;

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
  clearSegmentsBtn.addEventListener("click", options.onClearSavedSegments);

  let activeSource: CaptureSource | null = null;
  let activeSourceName = "";
  let active = false;
  let sampleCount = 0;
  let renderRaf = 0;
  let collectorStatus: CollectorSnapshot["status"] = "idle";
  let persistenceStatus: PersistenceStatus = "loading";
  let storedSegmentCount = 0;

  const updateCollectionControls = () => {
    const collecting = collectorStatus === "collecting";
    const persistenceReady = persistenceStatus === "ready";
    fixtureSelect.disabled = collecting || !persistenceReady;
    collectFixtureBtn.disabled =
      collecting || !persistenceReady || options.fixtures.length === 0;
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

  return {
    acceptSamples(chunk) {
      const channelCount = chunk.channels.length;
      if (!channelCount || chunk.samples.length % channelCount !== 0) return;
      if (!active || chunk.source !== activeSource) {
        active = true;
        activeSourceName = "";
        sampleCount = 0;
      }
      activeSource = chunk.source;
      sampleCount += Math.floor(chunk.samples.length / channelCount);
      scheduleRender();
    },

    setCollectedSegments(segments) {
      storedSegmentCount = segments.length;
      if (!segments.length) {
        segmentListEl.innerHTML =
          "<li>No saved segments in the local fixture dataset.</li>";
        updateCollectionControls();
        return;
      }

      segmentListEl.replaceChildren(
        ...segments.map((collected) => {
          const item = document.createElement("li");
          const duration = (collected.durationMs / 1_000).toFixed(2);
          const quality = collected.quality.accepted ? "accepted" : "review";
          item.textContent =
            `${collected.label} · ${collected.effort} · repetition ${collected.repetition} · ` +
            `${collected.sampleCount.toLocaleString()} samples · ${duration} s · ${quality}`;
          return item;
        }),
      );
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
      if (startingNewStream) sampleCount = 0;
      scheduleRender();
    },
  };
}
