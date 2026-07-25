import {
  GESTURE_LABELS,
  MYOARM_CHANNELS,
  MYOARM_DATASET_SCHEMA_VERSION,
  MYOARM_SAMPLE_RATE_HZ,
} from "./protocol";
import type {
  ArmSide,
  CaptureSource,
  EffortLevel,
  GestureLabel,
  MyoArmDataset,
  MyoArmSegment,
  MyoArmSession,
  QualityFlag,
} from "./types";
import { unzip, zip } from "../zip";

export const MYOARM_ARCHIVE_FORMAT = "mudra-viewer-myoarm-dataset";
export const MYOARM_ARCHIVE_VERSION = 1;

type ArchivedSegment = Omit<MyoArmSegment, "samples"> & {
  sampleEncoding: "int32-le-interleaved";
  samplesFile: string;
};

type ArchivedSession = Omit<MyoArmSession, "segments"> & {
  segments: ArchivedSegment[];
};

type ArchivedDataset = Omit<MyoArmDataset, "sessions"> & {
  sessions: ArchivedSession[];
};

interface MyoArmArchiveManifest {
  format: typeof MYOARM_ARCHIVE_FORMAT;
  archiveVersion: typeof MYOARM_ARCHIVE_VERSION;
  exportedAt: string;
  dataset: ArchivedDataset;
}

type JsonObject = Record<string, unknown>;

const textEncoder = new TextEncoder();
const textDecoder = new TextDecoder("utf-8", { fatal: true });
const sources = new Set<CaptureSource>(["fixture", "bluetooth", "import"]);
const armSides = new Set<ArmSide>(["left", "right", "unknown"]);
const efforts = new Set<EffortLevel>([
  "relaxed",
  "light",
  "medium",
  "strong",
  "unknown",
]);
const gestures = new Set<GestureLabel>(GESTURE_LABELS);
const qualityFlags = new Set<QualityFlag>([
  "clipping",
  "dropped-frames",
  "too-short",
  "contains-transition",
  "manual-reject",
]);

const fail = (path: string, message: string): never => {
  throw new Error(`Invalid MyoArm archive at ${path}: ${message}`);
};

const objectAt = (value: unknown, path: string): JsonObject => {
  if (!value || typeof value !== "object" || Array.isArray(value)) {
    return fail(path, "expected an object");
  }
  return value as JsonObject;
};

const arrayAt = (value: unknown, path: string): unknown[] => {
  if (!Array.isArray(value)) return fail(path, "expected an array");
  return value;
};

const stringAt = (value: unknown, path: string): string => {
  if (typeof value !== "string" || !value) {
    return fail(path, "expected a non-empty string");
  }
  return value;
};

const optionalStringAt = (
  value: unknown,
  path: string,
): string | undefined => {
  if (value === undefined) return undefined;
  return stringAt(value, path);
};

const numberAt = (value: unknown, path: string): number => {
  if (typeof value !== "number" || !Number.isFinite(value)) {
    return fail(path, "expected a finite number");
  }
  return value;
};

const integerAt = (value: unknown, path: string): number => {
  const number = numberAt(value, path);
  if (!Number.isInteger(number)) return fail(path, "expected an integer");
  return number;
};

const booleanAt = (value: unknown, path: string): boolean => {
  if (typeof value !== "boolean") return fail(path, "expected a boolean");
  return value;
};

const dateAt = (value: unknown, path: string): string => {
  const date = stringAt(value, path);
  if (Number.isNaN(Date.parse(date))) {
    return fail(path, "expected an ISO-compatible date");
  }
  return date;
};

const enumAt = <T extends string>(
  value: unknown,
  allowed: ReadonlySet<T>,
  path: string,
): T => {
  const item = stringAt(value, path);
  if (!allowed.has(item as T)) return fail(path, `unsupported value "${item}"`);
  return item as T;
};

const channelsAt = (value: unknown, path: string) => {
  const channels = arrayAt(value, path).map((channel, index) =>
    stringAt(channel, `${path}[${index}]`),
  );
  if (
    channels.length !== MYOARM_CHANNELS.length ||
    channels.some((channel, index) => channel !== MYOARM_CHANNELS[index])
  ) {
    return fail(
      path,
      `expected channels ${MYOARM_CHANNELS.join(", ")} in that order`,
    );
  }
  return [...MYOARM_CHANNELS];
};

const int32ToLittleEndian = (samples: Int32Array) => {
  const bytes = new Uint8Array(samples.length * Int32Array.BYTES_PER_ELEMENT);
  const view = new DataView(bytes.buffer);
  for (let index = 0; index < samples.length; index++) {
    view.setInt32(index * Int32Array.BYTES_PER_ELEMENT, samples[index], true);
  }
  return bytes;
};

const int32FromLittleEndian = (bytes: Uint8Array) => {
  if (bytes.byteLength % Int32Array.BYTES_PER_ELEMENT !== 0) {
    throw new Error("Int32 sample payload has a partial value");
  }
  const samples = new Int32Array(
    bytes.byteLength / Int32Array.BYTES_PER_ELEMENT,
  );
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  for (let index = 0; index < samples.length; index++) {
    samples[index] = view.getInt32(
      index * Int32Array.BYTES_PER_ELEMENT,
      true,
    );
  }
  return samples;
};

const assertExportableSegment = (
  segment: MyoArmSegment,
  session: MyoArmSession,
) => {
  if (segment.sessionId !== session.id) {
    throw new Error(
      `Segment "${segment.id}" does not belong to session "${session.id}"`,
    );
  }
  const expectedValues = segment.sampleCount * segment.channels.length;
  if (
    !Number.isSafeInteger(segment.sampleCount) ||
    segment.sampleCount <= 0 ||
    segment.samples.length !== expectedValues
  ) {
    throw new Error(`Segment "${segment.id}" has inconsistent sample counts`);
  }
};

export function exportMyoArmDatasetArchive(
  dataset: MyoArmDataset,
): Uint8Array {
  if (dataset.schemaVersion !== MYOARM_DATASET_SCHEMA_VERSION) {
    throw new Error(
      `Dataset schema ${dataset.schemaVersion} cannot be exported by schema ${MYOARM_DATASET_SCHEMA_VERSION}`,
    );
  }

  const files: { name: string; data: Uint8Array }[] = [];
  const segmentIds = new Set<string>();
  let sampleFileIndex = 0;

  const archivedSessions = dataset.sessions.map<ArchivedSession>((session) => {
    if (session.datasetId !== dataset.id) {
      throw new Error(
        `Session "${session.id}" does not belong to dataset "${dataset.id}"`,
      );
    }

    const archivedSegments = session.segments.map<ArchivedSegment>((segment) => {
      assertExportableSegment(segment, session);
      if (segmentIds.has(segment.id)) {
        throw new Error(`Duplicate segment identifier "${segment.id}"`);
      }
      segmentIds.add(segment.id);

      const samplesFile =
        `samples/${sampleFileIndex.toString().padStart(6, "0")}.i32le`;
      sampleFileIndex++;
      files.push({
        name: samplesFile,
        data: int32ToLittleEndian(segment.samples),
      });
      const { samples: _samples, ...metadata } = segment;
      void _samples;
      return {
        ...metadata,
        channels: [...metadata.channels],
        quality: {
          ...metadata.quality,
          flags: [...metadata.quality.flags],
        },
        sampleEncoding: "int32-le-interleaved",
        samplesFile,
      };
    });

    const { segments: _segments, ...metadata } = session;
    void _segments;
    return {
      ...metadata,
      channels: [...metadata.channels],
      segments: archivedSegments,
    };
  });

  const { sessions: _sessions, ...datasetMetadata } = dataset;
  void _sessions;
  const manifest: MyoArmArchiveManifest = {
    format: MYOARM_ARCHIVE_FORMAT,
    archiveVersion: MYOARM_ARCHIVE_VERSION,
    exportedAt: new Date().toISOString(),
    dataset: {
      ...datasetMetadata,
      labelSet: [...datasetMetadata.labelSet],
      sessions: archivedSessions,
    },
  };
  files.unshift({
    name: "manifest.json",
    data: textEncoder.encode(JSON.stringify(manifest, null, 2)),
  });
  return zip(files);
}

export function importMyoArmDatasetArchive(
  archive: Uint8Array,
): MyoArmDataset {
  const maximumArchiveBytes = 512 * 1024 * 1024;
  if (!archive.byteLength || archive.byteLength > maximumArchiveBytes) {
    throw new Error("MyoArm archive is empty or exceeds the 512 MiB limit");
  }

  const files = unzip(archive);
  const manifestBytes = files.get("manifest.json");
  if (!manifestBytes) throw new Error("MyoArm archive has no manifest.json");

  let parsed: unknown;
  try {
    parsed = JSON.parse(textDecoder.decode(manifestBytes));
  } catch (error) {
    throw new Error(`MyoArm manifest is not valid JSON: ${(error as Error).message}`);
  }

  const manifest = objectAt(parsed, "manifest");
  if (manifest.format !== MYOARM_ARCHIVE_FORMAT) {
    fail("manifest.format", "unsupported archive format");
  }
  if (manifest.archiveVersion !== MYOARM_ARCHIVE_VERSION) {
    fail(
      "manifest.archiveVersion",
      `expected version ${MYOARM_ARCHIVE_VERSION}`,
    );
  }
  dateAt(manifest.exportedAt, "manifest.exportedAt");

  const archivedDataset = objectAt(manifest.dataset, "manifest.dataset");
  const schemaVersion = integerAt(
    archivedDataset.schemaVersion,
    "manifest.dataset.schemaVersion",
  );
  if (schemaVersion !== MYOARM_DATASET_SCHEMA_VERSION) {
    fail(
      "manifest.dataset.schemaVersion",
      `expected schema ${MYOARM_DATASET_SCHEMA_VERSION}`,
    );
  }

  const datasetId = stringAt(
    archivedDataset.id,
    "manifest.dataset.id",
  );
  const labelSet = arrayAt(
    archivedDataset.labelSet,
    "manifest.dataset.labelSet",
  ).map((label, index) =>
    enumAt(label, gestures, `manifest.dataset.labelSet[${index}]`),
  );
  if (new Set(labelSet).size !== labelSet.length) {
    fail("manifest.dataset.labelSet", "contains duplicate labels");
  }
  if (!labelSet.length) {
    fail("manifest.dataset.labelSet", "must contain at least one label");
  }

  const sessionIds = new Set<string>();
  const segmentIds = new Set<string>();
  const samplesFiles = new Set<string>();
  const archivedSessions = arrayAt(
    archivedDataset.sessions,
    "manifest.dataset.sessions",
  );
  const sessions = archivedSessions.map<MyoArmSession>(
    (sessionValue, sessionIndex) => {
      const path = `manifest.dataset.sessions[${sessionIndex}]`;
      const archivedSession = objectAt(sessionValue, path);
      const sessionId = stringAt(archivedSession.id, `${path}.id`);
      if (sessionIds.has(sessionId)) {
        fail(`${path}.id`, `duplicate session "${sessionId}"`);
      }
      sessionIds.add(sessionId);
      if (stringAt(archivedSession.datasetId, `${path}.datasetId`) !== datasetId) {
        fail(`${path}.datasetId`, "does not match the dataset");
      }

      const sessionChannels = channelsAt(
        archivedSession.channels,
        `${path}.channels`,
      );
      const sessionRate = numberAt(
        archivedSession.sampleRateHz,
        `${path}.sampleRateHz`,
      );
      if (sessionRate !== MYOARM_SAMPLE_RATE_HZ) {
        fail(
          `${path}.sampleRateHz`,
          `expected ${MYOARM_SAMPLE_RATE_HZ} Hz`,
        );
      }

      const archivedSegments = arrayAt(
        archivedSession.segments,
        `${path}.segments`,
      );
      const segments = archivedSegments.map<MyoArmSegment>(
        (segmentValue, segmentIndex) => {
          const segmentPath = `${path}.segments[${segmentIndex}]`;
          const archivedSegment = objectAt(segmentValue, segmentPath);
          const segmentId = stringAt(
            archivedSegment.id,
            `${segmentPath}.id`,
          );
          if (segmentIds.has(segmentId)) {
            fail(`${segmentPath}.id`, `duplicate segment "${segmentId}"`);
          }
          segmentIds.add(segmentId);
          if (
            stringAt(
              archivedSegment.sessionId,
              `${segmentPath}.sessionId`,
            ) !== sessionId
          ) {
            fail(`${segmentPath}.sessionId`, "does not match the session");
          }

          const label = enumAt(
            archivedSegment.label,
            gestures,
            `${segmentPath}.label`,
          );
          if (!labelSet.includes(label)) {
            fail(`${segmentPath}.label`, "is not in the dataset label set");
          }
          const segmentChannels = channelsAt(
            archivedSegment.channels,
            `${segmentPath}.channels`,
          );
          const sampleRateHz = numberAt(
            archivedSegment.sampleRateHz,
            `${segmentPath}.sampleRateHz`,
          );
          if (sampleRateHz !== sessionRate) {
            fail(
              `${segmentPath}.sampleRateHz`,
              "does not match the session sample rate",
            );
          }

          const sampleCount = integerAt(
            archivedSegment.sampleCount,
            `${segmentPath}.sampleCount`,
          );
          if (!Number.isSafeInteger(sampleCount) || sampleCount <= 0) {
            fail(`${segmentPath}.sampleCount`, "must be a positive safe integer");
          }
          if (
            archivedSegment.sampleEncoding !== "int32-le-interleaved"
          ) {
            fail(`${segmentPath}.sampleEncoding`, "unsupported encoding");
          }
          const samplesFile = stringAt(
            archivedSegment.samplesFile,
            `${segmentPath}.samplesFile`,
          );
          if (samplesFiles.has(samplesFile)) {
            fail(
              `${segmentPath}.samplesFile`,
              `duplicate payload "${samplesFile}"`,
            );
          }
          samplesFiles.add(samplesFile);
          const sampleBytes =
            files.get(samplesFile) ??
            fail(
              `${segmentPath}.samplesFile`,
              `missing payload "${samplesFile}"`,
            );
          const expectedBytes =
            sampleCount *
            segmentChannels.length *
            Int32Array.BYTES_PER_ELEMENT;
          if (
            !Number.isSafeInteger(expectedBytes) ||
            sampleBytes.byteLength !== expectedBytes
          ) {
            fail(
              `${segmentPath}.samplesFile`,
              `expected ${expectedBytes} bytes, found ${sampleBytes.byteLength}`,
            );
          }

          const quality = objectAt(
            archivedSegment.quality,
            `${segmentPath}.quality`,
          );
          const flags = arrayAt(
            quality.flags,
            `${segmentPath}.quality.flags`,
          ).map((flag, flagIndex) =>
            enumAt(
              flag,
              qualityFlags,
              `${segmentPath}.quality.flags[${flagIndex}]`,
            ),
          );
          if (new Set(flags).size !== flags.length) {
            fail(`${segmentPath}.quality.flags`, "contains duplicate flags");
          }
          const notes = optionalStringAt(
            quality.notes,
            `${segmentPath}.quality.notes`,
          );

          const repetition = integerAt(
            archivedSegment.repetition,
            `${segmentPath}.repetition`,
          );
          if (repetition <= 0) {
            fail(`${segmentPath}.repetition`, "must be positive");
          }
          const startOffsetMs = numberAt(
            archivedSegment.startOffsetMs,
            `${segmentPath}.startOffsetMs`,
          );
          if (startOffsetMs < 0) {
            fail(`${segmentPath}.startOffsetMs`, "must not be negative");
          }
          const durationMs = numberAt(
            archivedSegment.durationMs,
            `${segmentPath}.durationMs`,
          );
          if (durationMs <= 0) {
            fail(`${segmentPath}.durationMs`, "must be positive");
          }

          return {
            id: segmentId,
            sessionId,
            label,
            effort: enumAt(
              archivedSegment.effort,
              efforts,
              `${segmentPath}.effort`,
            ),
            repetition,
            startOffsetMs,
            durationMs,
            sampleRateHz,
            channels: segmentChannels,
            sampleCount,
            samples: int32FromLittleEndian(sampleBytes),
            quality: {
              accepted: booleanAt(
                quality.accepted,
                `${segmentPath}.quality.accepted`,
              ),
              flags,
              ...(notes ? { notes } : {}),
            },
          };
        },
      );

      const sourceName = optionalStringAt(
        archivedSession.sourceName,
        `${path}.sourceName`,
      );
      return {
        id: sessionId,
        datasetId,
        participantId: stringAt(
          archivedSession.participantId,
          `${path}.participantId`,
        ),
        source: enumAt(
          archivedSession.source,
          sources,
          `${path}.source`,
        ),
        ...(sourceName ? { sourceName } : {}),
        armSide: enumAt(
          archivedSession.armSide,
          armSides,
          `${path}.armSide`,
        ),
        startedAt: dateAt(
          archivedSession.startedAt,
          `${path}.startedAt`,
        ),
        protocolId: stringAt(
          archivedSession.protocolId,
          `${path}.protocolId`,
        ),
        sampleRateHz: sessionRate,
        channels: sessionChannels,
        appVersion: stringAt(
          archivedSession.appVersion,
          `${path}.appVersion`,
        ),
        segments,
      };
    },
  );

  return {
    schemaVersion,
    id: datasetId,
    name: stringAt(archivedDataset.name, "manifest.dataset.name"),
    createdAt: dateAt(
      archivedDataset.createdAt,
      "manifest.dataset.createdAt",
    ),
    updatedAt: dateAt(
      archivedDataset.updatedAt,
      "manifest.dataset.updatedAt",
    ),
    labelSet,
    sessions,
  };
}
