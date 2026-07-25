import type {
  ChannelName,
  MyoArmSegment,
} from "../myoarm/types";

const FULL_SCALE = 32_768;
const DEFAULT_COLORS = ["#2563eb", "#0f9d58", "#c026d3"] as const;

export interface ChannelStatistics {
  channel: ChannelName;
  minimum: number;
  maximum: number;
  peakToPeak: number;
  rms: number;
  boundaryHits: number;
}

export interface MyoArmSegmentPlot {
  draw(segment: MyoArmSegment | null): void;
  destroy(): void;
}

export function calculateChannelStatistics(
  segment: MyoArmSegment,
): ChannelStatistics[] {
  const channelCount = segment.channels.length;
  if (!channelCount || segment.samples.length % channelCount !== 0) {
    return [];
  }

  return segment.channels.map((channel, channelIndex) => {
    let minimum = Number.POSITIVE_INFINITY;
    let maximum = Number.NEGATIVE_INFINITY;
    let squareSum = 0;
    let boundaryHits = 0;

    for (
      let sampleIndex = channelIndex;
      sampleIndex < segment.samples.length;
      sampleIndex += channelCount
    ) {
      const value = segment.samples[sampleIndex];
      minimum = Math.min(minimum, value);
      maximum = Math.max(maximum, value);
      squareSum += value * value;
      if (value <= -FULL_SCALE || value >= FULL_SCALE - 1) {
        boundaryHits++;
      }
    }

    const sampleCount = segment.samples.length / channelCount;
    return {
      channel,
      minimum,
      maximum,
      peakToPeak: maximum - minimum,
      rms: Math.sqrt(squareSum / sampleCount),
      boundaryHits,
    };
  });
}

export function createMyoArmSegmentPlot(
  canvas: HTMLCanvasElement,
): MyoArmSegmentPlot {
  let activeSegment: MyoArmSegment | null = null;
  let renderFrame = 0;

  const render = () => {
    renderFrame = 0;
    const width = canvas.clientWidth;
    const height = canvas.clientHeight;
    if (!width || !height) return;

    const dpr = window.devicePixelRatio || 1;
    const pixelWidth = Math.max(1, Math.floor(width * dpr));
    const pixelHeight = Math.max(1, Math.floor(height * dpr));
    if (canvas.width !== pixelWidth || canvas.height !== pixelHeight) {
      canvas.width = pixelWidth;
      canvas.height = pixelHeight;
    }

    const context = canvas.getContext("2d");
    if (!context) return;
    context.setTransform(dpr, 0, 0, dpr, 0, 0);
    context.clearRect(0, 0, width, height);
    context.fillStyle = "#ffffff";
    context.fillRect(0, 0, width, height);

    if (!activeSegment) return;
    const channelCount = activeSegment.channels.length;
    if (
      !channelCount ||
      activeSegment.samples.length % channelCount !== 0
    ) {
      return;
    }

    const sampleCount = activeSegment.samples.length / channelCount;
    if (!sampleCount) return;

    let maximumMagnitude = FULL_SCALE;
    for (const value of activeSegment.samples) {
      maximumMagnitude = Math.max(maximumMagnitude, Math.abs(value));
    }
    const amplitudeLimit = maximumMagnitude * 1.08;
    const plotLeft = 66;
    const plotRight = 16;
    const plotTop = 18;
    const plotBottom = 30;
    const plotWidth = Math.max(1, width - plotLeft - plotRight);
    const plotHeight = Math.max(1, height - plotTop - plotBottom);
    const laneHeight = plotHeight / channelCount;
    const durationSeconds = sampleCount / activeSegment.sampleRateHz;

    context.font = "11px system-ui, sans-serif";
    context.lineWidth = 1;
    context.textBaseline = "middle";

    const timeDivisions = Math.min(6, Math.max(2, Math.ceil(durationSeconds)));
    for (let division = 0; division <= timeDivisions; division++) {
      const ratio = division / timeDivisions;
      const x = plotLeft + ratio * plotWidth;
      context.strokeStyle = "#e8e6dd";
      context.beginPath();
      context.moveTo(x, plotTop);
      context.lineTo(x, plotTop + plotHeight);
      context.stroke();

      context.fillStyle = "#8a877a";
      context.textAlign = "center";
      context.textBaseline = "top";
      context.fillText(
        `${(ratio * durationSeconds).toFixed(1)} s`,
        x,
        plotTop + plotHeight + 8,
      );
    }

    for (let channelIndex = 0; channelIndex < channelCount; channelIndex++) {
      const laneTop = plotTop + channelIndex * laneHeight;
      const centerY = laneTop + laneHeight / 2;
      const channel = activeSegment.channels[channelIndex];
      const color = DEFAULT_COLORS[channelIndex] ?? "#6b6858";

      context.strokeStyle = "#d7d4ca";
      context.beginPath();
      context.moveTo(plotLeft, centerY);
      context.lineTo(plotLeft + plotWidth, centerY);
      context.stroke();

      if (channelIndex > 0) {
        context.strokeStyle = "#eeeCE5";
        context.beginPath();
        context.moveTo(plotLeft, laneTop);
        context.lineTo(plotLeft + plotWidth, laneTop);
        context.stroke();
      }

      context.fillStyle = color;
      context.textAlign = "right";
      context.textBaseline = "middle";
      context.font = "600 11px system-ui, sans-serif";
      context.fillText(channel, plotLeft - 10, centerY - 7);
      context.fillStyle = "#8a877a";
      context.font = "10px system-ui, sans-serif";
      context.fillText("raw count", plotLeft - 10, centerY + 8);

      const yForValue = (value: number) =>
        centerY - (value / amplitudeLimit) * (laneHeight * 0.42);

      context.save();
      context.beginPath();
      context.rect(plotLeft, laneTop, plotWidth, laneHeight);
      context.clip();
      context.strokeStyle = color;
      context.lineWidth = Math.max(1, dpr > 1 ? 0.75 : 1);
      context.beginPath();

      if (sampleCount <= plotWidth * 2) {
        for (let sampleIndex = 0; sampleIndex < sampleCount; sampleIndex++) {
          const value =
            activeSegment.samples[
              sampleIndex * channelCount + channelIndex
            ];
          const x =
            plotLeft +
            (sampleIndex / Math.max(1, sampleCount - 1)) * plotWidth;
          const y = yForValue(value);
          if (sampleIndex === 0) context.moveTo(x, y);
          else context.lineTo(x, y);
        }
      } else {
        const buckets = Math.max(1, Math.floor(plotWidth));
        for (let bucket = 0; bucket < buckets; bucket++) {
          const start = Math.floor((bucket / buckets) * sampleCount);
          const end = Math.max(
            start + 1,
            Math.floor(((bucket + 1) / buckets) * sampleCount),
          );
          let minimum = Number.POSITIVE_INFINITY;
          let maximum = Number.NEGATIVE_INFINITY;
          for (let sampleIndex = start; sampleIndex < end; sampleIndex++) {
            const value =
              activeSegment.samples[
                sampleIndex * channelCount + channelIndex
              ];
            minimum = Math.min(minimum, value);
            maximum = Math.max(maximum, value);
          }
          const x = plotLeft + ((bucket + 0.5) / buckets) * plotWidth;
          context.moveTo(x, yForValue(minimum));
          context.lineTo(x, yForValue(maximum));
        }
      }

      context.stroke();
      context.restore();
    }
  };

  const scheduleRender = () => {
    if (!renderFrame) renderFrame = requestAnimationFrame(render);
  };
  const resizeObserver = new ResizeObserver(scheduleRender);
  resizeObserver.observe(canvas);

  return {
    draw(segment) {
      activeSegment = segment;
      canvas.setAttribute(
        "aria-label",
        segment
          ? `Raw EMG waveform for ${segment.label}, ${segment.effort}, repetition ${segment.repetition}`
          : "No MyoArm segment selected",
      );
      scheduleRender();
    },

    destroy() {
      resizeObserver.disconnect();
      if (renderFrame) cancelAnimationFrame(renderFrame);
    },
  };
}
