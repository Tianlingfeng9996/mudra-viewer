export const EMG_CHANNELS = ["ulnar", "median", "radial"] as const;
export const EMG_SAMPLE_RATE_HZ = 834;

export type ChannelName = (typeof EMG_CHANNELS)[number];
export type CaptureSource = "fixture" | "bluetooth" | "import";

export interface SampleChunk {
  source: CaptureSource;
  /** Monotonic time at which this chunk entered the shared sample pipeline. */
  receivedAtSec: number;
  sampleRateHz: number;
  channels: readonly ChannelName[];
  /** Sample-major interleaved values: s0c0, s0c1, s0c2, s1c0, ... */
  samples: Int32Array;
}
