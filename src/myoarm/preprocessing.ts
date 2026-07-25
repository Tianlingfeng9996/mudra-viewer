import {
  EMG_CHANNELS,
  EMG_SAMPLE_RATE_HZ,
  type ChannelName,
} from "../signal/types";
import type {
  GestureLabel,
  MyoArmSegment,
} from "./types";

const SIGNED_INT16_FULL_SCALE = 32_768;

export type PreprocessingIssueCode =
  | "quality-rejected"
  | "sample-rate-mismatch"
  | "channel-order-mismatch"
  | "sample-layout-mismatch"
  | "too-short";

export interface PreprocessingConfig {
  id: string;
  sampleRateHz: number;
  channels: readonly ChannelName[];
  windowDurationMs: number;
  strideDurationMs: number;
  windowSampleCount: number;
  strideSampleCount: number;
  acceptedSegmentsOnly: boolean;
  scaling: "divide-by-int16-full-scale";
}

export interface PreprocessedWindow {
  segmentId: string;
  sessionId: string;
  groupId: string;
  label: GestureLabel;
  startSample: number;
  endSampleExclusive: number;
  sampleRateHz: number;
  channels: readonly ChannelName[];
  /** Sample-major interleaved Float32 values. */
  values: Float32Array;
}

export interface PreprocessingIssue {
  segmentId: string;
  code: PreprocessingIssueCode;
}

export interface PreprocessingSummary {
  segmentCount: number;
  usableSegmentCount: number;
  skippedSegmentCount: number;
  windowCount: number;
  inputValueCount: number;
  windowsByLabel: Record<GestureLabel, number>;
  issues: PreprocessingIssue[];
}

export interface PreparedMyoArmDataset {
  config: PreprocessingConfig;
  windows: PreprocessedWindow[];
  summary: PreprocessingSummary;
}

const windowSampleCount = Math.round(EMG_SAMPLE_RATE_HZ * 0.2);
const strideSampleCount = Math.round(EMG_SAMPLE_RATE_HZ * 0.1);

export const PREPROCESSING_CONFIG_V1: PreprocessingConfig = {
  id: "myoarm-raw-window-v1",
  sampleRateHz: EMG_SAMPLE_RATE_HZ,
  channels: EMG_CHANNELS,
  windowDurationMs: 200,
  strideDurationMs: 100,
  windowSampleCount,
  strideSampleCount,
  acceptedSegmentsOnly: true,
  scaling: "divide-by-int16-full-scale",
};

const emptyLabelCounts = (): Record<GestureLabel, number> => ({
  rest: 0,
  open: 0,
  grasp: 0,
  pinch: 0,
  pronation: 0,
  supination: 0,
});

const channelsMatch = (
  actual: readonly ChannelName[],
  expected: readonly ChannelName[],
) =>
  actual.length === expected.length &&
  actual.every((channel, index) => channel === expected[index]);

const segmentIssue = (
  segment: MyoArmSegment,
  config: PreprocessingConfig,
): PreprocessingIssueCode | null => {
  if (config.acceptedSegmentsOnly && !segment.quality.accepted) {
    return "quality-rejected";
  }
  if (segment.sampleRateHz !== config.sampleRateHz) {
    return "sample-rate-mismatch";
  }
  if (!channelsMatch(segment.channels, config.channels)) {
    return "channel-order-mismatch";
  }

  const channelCount = config.channels.length;
  if (
    !channelCount ||
    segment.samples.length !== segment.sampleCount * channelCount
  ) {
    return "sample-layout-mismatch";
  }
  if (segment.sampleCount < config.windowSampleCount) {
    return "too-short";
  }
  return null;
};

/**
 * Shared numeric transform for offline training and live inference.
 *
 * The caller must supply exactly one sample-major, interleaved window.
 * Values are scaled by the fixed device full scale without clipping, so this
 * transform has no train-fitted state and preserves out-of-range diagnostics.
 */
export function preprocessInterleavedWindow(
  samples: ArrayLike<number>,
  config: PreprocessingConfig = PREPROCESSING_CONFIG_V1,
): Float32Array {
  const expectedValueCount =
    config.windowSampleCount * config.channels.length;
  if (samples.length !== expectedValueCount) {
    throw new Error(
      `Expected ${expectedValueCount} interleaved values, received ${samples.length}`,
    );
  }

  const values = new Float32Array(expectedValueCount);
  for (let index = 0; index < samples.length; index++) {
    values[index] = samples[index] / SIGNED_INT16_FULL_SCALE;
  }
  return values;
}

/**
 * Converts accepted labelled segments into overlapping, fixed-size windows.
 *
 * Incomplete tails are dropped. Every window keeps its parent segment as a
 * group identifier so later train/validation/test splitting cannot mix
 * overlapping windows from the same repetition across partitions.
 */
export function prepareMyoArmDataset(
  segments: readonly MyoArmSegment[],
  config: PreprocessingConfig = PREPROCESSING_CONFIG_V1,
): PreparedMyoArmDataset {
  if (
    config.windowSampleCount <= 0 ||
    config.strideSampleCount <= 0 ||
    !config.channels.length
  ) {
    throw new Error("Preprocessing window, stride, and channels must be valid");
  }

  const windows: PreprocessedWindow[] = [];
  const issues: PreprocessingIssue[] = [];
  const windowsByLabel = emptyLabelCounts();
  let usableSegmentCount = 0;

  for (const segment of segments) {
    const issue = segmentIssue(segment, config);
    if (issue) {
      issues.push({ segmentId: segment.id, code: issue });
      continue;
    }
    usableSegmentCount++;

    const channelCount = config.channels.length;
    const windowValueCount = config.windowSampleCount * channelCount;
    for (
      let startSample = 0;
      startSample + config.windowSampleCount <= segment.sampleCount;
      startSample += config.strideSampleCount
    ) {
      const valueOffset = startSample * channelCount;
      const rawWindow = segment.samples.subarray(
        valueOffset,
        valueOffset + windowValueCount,
      );
      windows.push({
        segmentId: segment.id,
        sessionId: segment.sessionId,
        groupId: segment.id,
        label: segment.label,
        startSample,
        endSampleExclusive: startSample + config.windowSampleCount,
        sampleRateHz: config.sampleRateHz,
        channels: config.channels,
        values: preprocessInterleavedWindow(rawWindow, config),
      });
      windowsByLabel[segment.label]++;
    }
  }

  return {
    config,
    windows,
    summary: {
      segmentCount: segments.length,
      usableSegmentCount,
      skippedSegmentCount: issues.length,
      windowCount: windows.length,
      inputValueCount:
        config.windowSampleCount * config.channels.length,
      windowsByLabel,
      issues,
    },
  };
}
