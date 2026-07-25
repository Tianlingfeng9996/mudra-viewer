import type {
  PreparedMyoArmDataset,
  PreprocessedWindow,
} from "./preprocessing";
import type { GestureLabel } from "./types";

export type DatasetGroupBy = "segment" | "session";
export type DatasetPartitionName = "train" | "validation" | "test";

export interface DatasetSplitConfig {
  id: string;
  groupBy: DatasetGroupBy;
  trainRatio: number;
  validationRatio: number;
  testRatio: number;
  seed: string;
}

export interface DatasetPartitionSummary {
  groupCount: number;
  segmentCount: number;
  windowCount: number;
  windowsByLabel: Record<GestureLabel, number>;
  missingLabels: GestureLabel[];
}

export interface DatasetPartition {
  name: DatasetPartitionName;
  groupIds: string[];
  windows: PreprocessedWindow[];
  summary: DatasetPartitionSummary;
}

export interface GroupedDatasetSplit {
  config: DatasetSplitConfig;
  observedLabels: GestureLabel[];
  totalGroupCount: number;
  ready: boolean;
  partitions: Record<DatasetPartitionName, DatasetPartition>;
}

interface WindowGroup {
  id: string;
  windows: PreprocessedWindow[];
  labels: GestureLabel[];
}

const PARTITION_NAMES: readonly DatasetPartitionName[] = [
  "train",
  "validation",
  "test",
];

export const DATASET_SPLIT_CONFIG_V1: DatasetSplitConfig = {
  id: "myoarm-grouped-split-v1",
  groupBy: "segment",
  trainRatio: 0.6,
  validationRatio: 0.2,
  testRatio: 0.2,
  seed: "myoarm-split-v1",
};

const emptyLabelCounts = (): Record<GestureLabel, number> => ({
  rest: 0,
  open: 0,
  grasp: 0,
  pinch: 0,
  pronation: 0,
  supination: 0,
});

const hashString = (value: string) => {
  let hash = 0x811c9dc5;
  for (let index = 0; index < value.length; index++) {
    hash ^= value.charCodeAt(index);
    hash = Math.imul(hash, 0x01000193);
  }
  return hash >>> 0;
};

const deterministicOrder = (
  groups: readonly WindowGroup[],
  seed: string,
) =>
  [...groups].sort((left, right) => {
    const hashDifference =
      hashString(`${seed}:${left.id}`) - hashString(`${seed}:${right.id}`);
    return hashDifference || left.id.localeCompare(right.id);
  });

const validateConfig = (config: DatasetSplitConfig) => {
  const ratios = [
    config.trainRatio,
    config.validationRatio,
    config.testRatio,
  ];
  if (ratios.some((ratio) => !Number.isFinite(ratio) || ratio < 0)) {
    throw new Error("Dataset split ratios must be finite and non-negative");
  }
  const ratioTotal = ratios.reduce((total, ratio) => total + ratio, 0);
  if (Math.abs(ratioTotal - 1) > 1e-9) {
    throw new Error("Dataset split ratios must sum to 1");
  }
};

const groupCountsFor = (
  groupCount: number,
  config: DatasetSplitConfig,
): Record<DatasetPartitionName, number> => {
  if (groupCount <= 0) {
    return { train: 0, validation: 0, test: 0 };
  }
  if (groupCount === 1) {
    return { train: 1, validation: 0, test: 0 };
  }
  if (groupCount === 2) {
    return { train: 1, validation: 1, test: 0 };
  }

  let train = Math.max(1, Math.round(groupCount * config.trainRatio));
  let validation = Math.max(
    1,
    Math.round(groupCount * config.validationRatio),
  );
  let test = groupCount - train - validation;

  while (test < 1) {
    if (train > validation && train > 1) train--;
    else if (validation > 1) validation--;
    else train--;
    test++;
  }

  return { train, validation, test };
};

const buildGroups = (
  windows: readonly PreprocessedWindow[],
  groupBy: DatasetGroupBy,
) => {
  const grouped = new Map<string, PreprocessedWindow[]>();
  for (const window of windows) {
    const groupId =
      groupBy === "segment" ? window.segmentId : window.sessionId;
    const groupWindows = grouped.get(groupId) ?? [];
    groupWindows.push(window);
    grouped.set(groupId, groupWindows);
  }

  return Array.from(grouped, ([id, groupWindows]) => ({
    id,
    windows: groupWindows,
    labels: Array.from(
      new Set(groupWindows.map((window) => window.label)),
    ).sort(),
  }));
};

const emptyAssignments = (): Record<
  DatasetPartitionName,
  WindowGroup[]
> => ({
  train: [],
  validation: [],
  test: [],
});

const assignOrderedGroups = (
  groups: readonly WindowGroup[],
  assignments: Record<DatasetPartitionName, WindowGroup[]>,
  config: DatasetSplitConfig,
) => {
  const counts = groupCountsFor(groups.length, config);
  let cursor = 0;
  for (const partitionName of PARTITION_NAMES) {
    const end = cursor + counts[partitionName];
    assignments[partitionName].push(...groups.slice(cursor, end));
    cursor = end;
  }
};

const assignSegmentGroups = (
  groups: readonly WindowGroup[],
  config: DatasetSplitConfig,
) => {
  const assignments = emptyAssignments();
  const groupsByLabel = new Map<GestureLabel, WindowGroup[]>();

  for (const group of groups) {
    if (group.labels.length !== 1) {
      throw new Error(
        `Segment group "${group.id}" contains multiple gesture labels`,
      );
    }
    const label = group.labels[0];
    const labelGroups = groupsByLabel.get(label) ?? [];
    labelGroups.push(group);
    groupsByLabel.set(label, labelGroups);
  }

  for (const [label, labelGroups] of groupsByLabel) {
    assignOrderedGroups(
      deterministicOrder(labelGroups, `${config.seed}:${label}`),
      assignments,
      config,
    );
  }
  return assignments;
};

const assignSessionGroups = (
  groups: readonly WindowGroup[],
  config: DatasetSplitConfig,
) => {
  const assignments = emptyAssignments();
  assignOrderedGroups(
    deterministicOrder(groups, `${config.seed}:session`),
    assignments,
    config,
  );
  return assignments;
};

const createPartition = (
  name: DatasetPartitionName,
  groups: readonly WindowGroup[],
  observedLabels: readonly GestureLabel[],
): DatasetPartition => {
  const windows = groups.flatMap((group) => group.windows);
  const windowsByLabel = emptyLabelCounts();
  const segmentIds = new Set<string>();
  for (const window of windows) {
    windowsByLabel[window.label]++;
    segmentIds.add(window.segmentId);
  }

  return {
    name,
    groupIds: groups.map((group) => group.id),
    windows,
    summary: {
      groupCount: groups.length,
      segmentCount: segmentIds.size,
      windowCount: windows.length,
      windowsByLabel,
      missingLabels: observedLabels.filter(
        (label) => windowsByLabel[label] === 0,
      ),
    },
  };
};

const assertNoGroupLeakage = (
  partitions: Record<DatasetPartitionName, DatasetPartition>,
) => {
  const groupOwners = new Map<string, DatasetPartitionName>();
  for (const partitionName of PARTITION_NAMES) {
    for (const groupId of partitions[partitionName].groupIds) {
      const existingOwner = groupOwners.get(groupId);
      if (existingOwner) {
        throw new Error(
          `Group "${groupId}" appears in ${existingOwner} and ${partitionName}`,
        );
      }
      groupOwners.set(groupId, partitionName);
    }
  }
};

/**
 * Splits prepared windows without allowing a segment or session group to leak
 * across train, validation, and test partitions.
 *
 * Segment grouping is stratified per observed gesture label. Session grouping
 * keeps whole sessions intact and reports any resulting label gaps instead of
 * moving individual segments across partitions.
 */
export function splitPreparedDataset(
  dataset: PreparedMyoArmDataset,
  config: DatasetSplitConfig = DATASET_SPLIT_CONFIG_V1,
): GroupedDatasetSplit {
  validateConfig(config);
  const observedLabels = Object.entries(dataset.summary.windowsByLabel)
    .filter(([, count]) => count > 0)
    .map(([label]) => label as GestureLabel);
  const groups = buildGroups(dataset.windows, config.groupBy);
  const assignments =
    config.groupBy === "segment"
      ? assignSegmentGroups(groups, config)
      : assignSessionGroups(groups, config);
  const partitions = {
    train: createPartition("train", assignments.train, observedLabels),
    validation: createPartition(
      "validation",
      assignments.validation,
      observedLabels,
    ),
    test: createPartition("test", assignments.test, observedLabels),
  };
  assertNoGroupLeakage(partitions);

  return {
    config,
    observedLabels,
    totalGroupCount: groups.length,
    ready:
      observedLabels.length > 0 &&
      PARTITION_NAMES.every(
        (partitionName) =>
          partitions[partitionName].summary.windowCount > 0 &&
          partitions[partitionName].summary.missingLabels.length === 0,
      ),
    partitions,
  };
}
