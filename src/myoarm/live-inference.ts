import {
  preprocessInterleavedWindow,
  PREPROCESSING_CONFIG_V1,
  type PreprocessingConfig,
} from "./preprocessing";
import type { SampleChunk } from "../signal/types";

export interface LiveInferenceWindow {
  index: number;
  endSampleExclusive: number;
  values: Float32Array;
}

export interface LiveWindowStream {
  reset(): void;
  acceptSamples(chunk: SampleChunk): LiveInferenceWindow[];
}

const channelsMatch = (
  actual: readonly string[],
  expected: readonly string[],
) =>
  actual.length === expected.length &&
  actual.every((channel, index) => channel === expected[index]);

/**
 * Converts a continuous decoded stream into the exact same overlapping windows
 * used by offline preprocessing. Reset this object at every source boundary so
 * a model input can never combine samples from two recordings.
 */
export function createLiveWindowStream(
  config: PreprocessingConfig = PREPROCESSING_CONFIG_V1,
): LiveWindowStream {
  const channelCount = config.channels.length;
  let bufferedValues: number[] = [];
  let bufferStartSample = 0;
  let nextWindowStartSample = 0;
  let receivedSampleCount = 0;
  let windowIndex = 0;

  return {
    reset() {
      bufferedValues = [];
      bufferStartSample = 0;
      nextWindowStartSample = 0;
      receivedSampleCount = 0;
      windowIndex = 0;
    },

    acceptSamples(chunk) {
      if (
        chunk.sampleRateHz !== config.sampleRateHz ||
        !channelsMatch(chunk.channels, config.channels) ||
        chunk.samples.length % channelCount !== 0
      ) {
        return [];
      }

      for (const value of chunk.samples) bufferedValues.push(value);
      receivedSampleCount += chunk.samples.length / channelCount;
      const windows: LiveInferenceWindow[] = [];
      while (
        nextWindowStartSample + config.windowSampleCount <=
        receivedSampleCount
      ) {
        const valueOffset =
          (nextWindowStartSample - bufferStartSample) * channelCount;
        const rawWindow = bufferedValues.slice(
          valueOffset,
          valueOffset + config.windowSampleCount * channelCount,
        );
        windows.push({
          index: windowIndex++,
          endSampleExclusive:
            nextWindowStartSample + config.windowSampleCount,
          values: preprocessInterleavedWindow(rawWindow, config),
        });
        nextWindowStartSample += config.strideSampleCount;
      }

      const discardSampleCount =
        nextWindowStartSample - bufferStartSample;
      if (discardSampleCount > 0) {
        bufferedValues.splice(0, discardSampleCount * channelCount);
        bufferStartSample = nextWindowStartSample;
      }
      return windows;
    },
  };
}
