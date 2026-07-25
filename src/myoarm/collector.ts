import type {
  CaptureSource,
  ChannelName,
  SampleChunk,
} from "../signal/types";
import type {
  EffortLevel,
  GestureLabel,
  MyoArmSegment,
  QualityFlag,
} from "./types";

export type CollectorStatus =
  | "idle"
  | "collecting"
  | "completed"
  | "cancelled"
  | "error";

export interface CollectionRequest {
  segmentId: string;
  sessionId: string;
  source: CaptureSource;
  label: GestureLabel;
  effort: EffortLevel;
  repetition: number;
  startOffsetMs: number;
  sampleRateHz: number;
  channels: readonly ChannelName[];
  minimumDurationMs: number;
}

export interface CollectorSnapshot {
  status: CollectorStatus;
  sampleCount: number;
  label: GestureLabel | null;
  effort: EffortLevel | null;
  repetition: number | null;
  message: string;
  segment: MyoArmSegment | null;
}

export interface SegmentCollector {
  readonly snapshot: CollectorSnapshot;
  start(request: CollectionRequest): void;
  acceptSamples(chunk: SampleChunk): void;
  complete(): MyoArmSegment | null;
  cancel(message?: string): void;
  reportError(message: string): void;
  subscribe(listener: (snapshot: CollectorSnapshot) => void): () => void;
}

const sameChannels = (
  left: readonly ChannelName[],
  right: readonly ChannelName[],
) =>
  left.length === right.length &&
  left.every((channel, index) => channel === right[index]);

export function createSegmentCollector(): SegmentCollector {
  const listeners = new Set<(snapshot: CollectorSnapshot) => void>();
  let status: CollectorStatus = "idle";
  let request: CollectionRequest | null = null;
  let chunks: Int32Array[] = [];
  let valueCount = 0;
  let message = "Ready to collect a fixture";
  let segment: MyoArmSegment | null = null;

  const getSnapshot = (): CollectorSnapshot => ({
    status,
    sampleCount:
      request && request.channels.length
        ? Math.floor(valueCount / request.channels.length)
        : segment?.sampleCount ?? 0,
    label: request?.label ?? segment?.label ?? null,
    effort: request?.effort ?? segment?.effort ?? null,
    repetition: request?.repetition ?? segment?.repetition ?? null,
    message,
    segment,
  });

  const notify = () => {
    const next = getSnapshot();
    for (const listener of [...listeners]) listener(next);
  };

  const fail = (reason: string) => {
    status = "error";
    message = reason;
    request = null;
    chunks = [];
    valueCount = 0;
    segment = null;
    notify();
  };

  return {
    get snapshot() {
      return getSnapshot();
    },

    start(nextRequest) {
      if (status === "collecting") {
        throw new Error("A segment collection is already in progress");
      }
      if (!nextRequest.channels.length) {
        throw new Error("A collection requires at least one signal channel");
      }

      request = {
        ...nextRequest,
        channels: [...nextRequest.channels],
      };
      chunks = [];
      valueCount = 0;
      segment = null;
      status = "collecting";
      message = `Collecting ${nextRequest.label} repetition ${nextRequest.repetition}`;
      notify();
    },

    acceptSamples(chunk) {
      if (status !== "collecting" || !request) return;
      if (chunk.source !== request.source) return;

      if (chunk.sampleRateHz !== request.sampleRateHz) {
        fail(
          `Sample rate changed from ${request.sampleRateHz} Hz to ${chunk.sampleRateHz} Hz`,
        );
        return;
      }
      if (!sameChannels(chunk.channels, request.channels)) {
        fail("Signal channel order changed during collection");
        return;
      }
      if (chunk.samples.length % request.channels.length !== 0) {
        fail("Received a malformed interleaved sample chunk");
        return;
      }

      const copy = chunk.samples.slice();
      chunks.push(copy);
      valueCount += copy.length;
      notify();
    },

    complete() {
      if (status !== "collecting" || !request) return null;
      if (!valueCount) {
        fail("The fixture finished without decoded samples");
        return null;
      }

      const samples = new Int32Array(valueCount);
      let offset = 0;
      for (const chunk of chunks) {
        samples.set(chunk, offset);
        offset += chunk.length;
      }

      const sampleCount = valueCount / request.channels.length;
      const durationMs = (sampleCount / request.sampleRateHz) * 1_000;
      const flags: QualityFlag[] =
        durationMs < request.minimumDurationMs ? ["too-short"] : [];

      segment = {
        id: request.segmentId,
        sessionId: request.sessionId,
        label: request.label,
        effort: request.effort,
        repetition: request.repetition,
        startOffsetMs: request.startOffsetMs,
        durationMs,
        sampleRateHz: request.sampleRateHz,
        channels: request.channels,
        sampleCount,
        samples,
        quality: {
          accepted: flags.length === 0,
          flags,
        },
      };

      status = "completed";
      message = flags.length
        ? `Collected ${sampleCount.toLocaleString()} samples, but the segment is too short`
        : `Collected ${sampleCount.toLocaleString()} samples`;
      request = null;
      chunks = [];
      valueCount = 0;
      notify();
      return segment;
    },

    cancel(reason = "Collection cancelled") {
      if (status !== "collecting") return;
      status = "cancelled";
      message = reason;
      request = null;
      chunks = [];
      valueCount = 0;
      segment = null;
      notify();
    },

    reportError(reason) {
      fail(reason);
    },

    subscribe(listener) {
      listeners.add(listener);
      listener(getSnapshot());
      return () => listeners.delete(listener);
    },
  };
}
