import type {
  MyoArmDataset,
  MyoArmSegment,
  MyoArmSession,
} from "./types";

export const MYOARM_DATABASE_NAME = "mudra-viewer-myoarm";
export const MYOARM_DATABASE_VERSION = 1;

export type MyoArmDatasetMetadata = Omit<MyoArmDataset, "sessions">;
export type MyoArmSessionMetadata = Omit<MyoArmSession, "segments">;

interface StoredSegment extends MyoArmSegment {
  datasetId: string;
}

export interface MyoArmStorage {
  loadDataset(datasetId: string): Promise<MyoArmDataset | null>;
  saveSegment(
    dataset: MyoArmDatasetMetadata,
    session: MyoArmSessionMetadata,
    segment: MyoArmSegment,
  ): Promise<void>;
  replaceDataset(dataset: MyoArmDataset): Promise<void>;
  clearDataset(datasetId: string): Promise<void>;
  close(): Promise<void>;
}

const STORES = {
  datasets: "datasets",
  sessions: "sessions",
  segments: "segments",
} as const;

const INDEXES = {
  sessionDatasetId: "datasetId",
  segmentDatasetId: "datasetId",
  segmentSessionId: "sessionId",
} as const;

const requestResult = <T>(request: IDBRequest<T>) =>
  new Promise<T>((resolve, reject) => {
    request.addEventListener("success", () => resolve(request.result), {
      once: true,
    });
    request.addEventListener(
      "error",
      () => reject(request.error ?? new Error("IndexedDB request failed")),
      { once: true },
    );
  });

const transactionDone = (transaction: IDBTransaction) =>
  new Promise<void>((resolve, reject) => {
    transaction.addEventListener("complete", () => resolve(), { once: true });
    transaction.addEventListener(
      "abort",
      () =>
        reject(
          transaction.error ?? new Error("IndexedDB transaction was aborted"),
        ),
      { once: true },
    );
    transaction.addEventListener(
      "error",
      () =>
        reject(transaction.error ?? new Error("IndexedDB transaction failed")),
      { once: true },
    );
  });

const cloneSegment = (segment: MyoArmSegment): MyoArmSegment => ({
  ...segment,
  channels: [...segment.channels],
  samples: segment.samples.slice(),
  quality: {
    ...segment.quality,
    flags: [...segment.quality.flags],
  },
});

const fromStoredSegment = (stored: StoredSegment): MyoArmSegment => {
  const { datasetId, ...segment } = stored;
  if (!datasetId) {
    throw new Error(`Stored segment "${stored.id}" has no dataset identifier`);
  }
  return cloneSegment(segment);
};

export function createMyoArmStorage(
  factory: IDBFactory | undefined = globalThis.indexedDB,
): MyoArmStorage {
  let databasePromise: Promise<IDBDatabase> | null = null;

  const openDatabase = () => {
    if (databasePromise) return databasePromise;
    if (!factory) {
      return Promise.reject(
        new Error("This browser does not provide IndexedDB"),
      );
    }

    databasePromise = new Promise<IDBDatabase>((resolve, reject) => {
      let blocked = false;
      const request = factory.open(
        MYOARM_DATABASE_NAME,
        MYOARM_DATABASE_VERSION,
      );

      request.addEventListener("upgradeneeded", () => {
        const database = request.result;
        const transaction = request.transaction!;

        if (!database.objectStoreNames.contains(STORES.datasets)) {
          database.createObjectStore(STORES.datasets, { keyPath: "id" });
        }

        const sessions = database.objectStoreNames.contains(STORES.sessions)
          ? transaction.objectStore(STORES.sessions)
          : database.createObjectStore(STORES.sessions, { keyPath: "id" });
        if (!sessions.indexNames.contains(INDEXES.sessionDatasetId)) {
          sessions.createIndex(INDEXES.sessionDatasetId, "datasetId");
        }

        const segments = database.objectStoreNames.contains(STORES.segments)
          ? transaction.objectStore(STORES.segments)
          : database.createObjectStore(STORES.segments, { keyPath: "id" });
        if (!segments.indexNames.contains(INDEXES.segmentDatasetId)) {
          segments.createIndex(INDEXES.segmentDatasetId, "datasetId");
        }
        if (!segments.indexNames.contains(INDEXES.segmentSessionId)) {
          segments.createIndex(INDEXES.segmentSessionId, "sessionId");
        }
      });

      request.addEventListener(
        "success",
        () => {
          const database = request.result;
          if (blocked) {
            database.close();
            return;
          }
          database.addEventListener("versionchange", () => database.close());
          resolve(database);
        },
        { once: true },
      );
      request.addEventListener(
        "error",
        () =>
          reject(
            request.error ?? new Error("Unable to open the MyoArm database"),
          ),
        { once: true },
      );
      request.addEventListener(
        "blocked",
        () => {
          blocked = true;
          reject(
            new Error(
              "MyoArm storage upgrade is blocked by another open app tab",
            ),
          );
        },
        { once: true },
      );
    }).catch((error) => {
      databasePromise = null;
      throw error;
    });

    return databasePromise;
  };

  return {
    async loadDataset(datasetId) {
      const database = await openDatabase();
      const transaction = database.transaction(
        [STORES.datasets, STORES.sessions, STORES.segments],
        "readonly",
      );
      const done = transactionDone(transaction);

      const datasetRequest = transaction
        .objectStore(STORES.datasets)
        .get(datasetId);
      const sessionsRequest = transaction
        .objectStore(STORES.sessions)
        .index(INDEXES.sessionDatasetId)
        .getAll(IDBKeyRange.only(datasetId));
      const segmentsRequest = transaction
        .objectStore(STORES.segments)
        .index(INDEXES.segmentDatasetId)
        .getAll(IDBKeyRange.only(datasetId));

      const [dataset, sessions, segments] = await Promise.all([
        requestResult(datasetRequest) as Promise<
          MyoArmDatasetMetadata | undefined
        >,
        requestResult(sessionsRequest) as Promise<MyoArmSessionMetadata[]>,
        requestResult(segmentsRequest) as Promise<StoredSegment[]>,
      ]);
      await done;

      if (!dataset) return null;

      const segmentsBySession = new Map<string, MyoArmSegment[]>();
      for (const stored of segments) {
        const sessionSegments =
          segmentsBySession.get(stored.sessionId) ?? [];
        sessionSegments.push(fromStoredSegment(stored));
        segmentsBySession.set(stored.sessionId, sessionSegments);
      }

      const restoredSessions = sessions
        .sort((left, right) => left.startedAt.localeCompare(right.startedAt))
        .map<MyoArmSession>((session) => ({
          ...session,
          channels: [...session.channels],
          segments: (segmentsBySession.get(session.id) ?? []).sort(
            (left, right) => left.startOffsetMs - right.startOffsetMs,
          ),
        }));

      return {
        ...dataset,
        labelSet: [...dataset.labelSet],
        sessions: restoredSessions,
      };
    },

    async saveSegment(dataset, session, segment) {
      if (segment.sessionId !== session.id) {
        throw new Error("Segment and session identifiers do not match");
      }
      if (session.datasetId !== dataset.id) {
        throw new Error("Session and dataset identifiers do not match");
      }

      const database = await openDatabase();
      const transaction = database.transaction(
        [STORES.datasets, STORES.sessions, STORES.segments],
        "readwrite",
      );
      const done = transactionDone(transaction);
      const datasets = transaction.objectStore(STORES.datasets);
      const now = new Date().toISOString();
      const existingDatasetRequest = datasets.get(dataset.id);

      existingDatasetRequest.addEventListener(
        "success",
        () => {
          const existing = existingDatasetRequest.result as
            | MyoArmDatasetMetadata
            | undefined;
          datasets.put({
            ...dataset,
            labelSet: [...dataset.labelSet],
            createdAt: existing?.createdAt ?? dataset.createdAt,
            updatedAt: now,
          });
        },
        { once: true },
      );

      transaction.objectStore(STORES.sessions).put({
        ...session,
        channels: [...session.channels],
      });
      transaction.objectStore(STORES.segments).put({
        ...cloneSegment(segment),
        datasetId: dataset.id,
      } satisfies StoredSegment);

      await done;
    },

    async replaceDataset(dataset) {
      const sessionIds = new Set<string>();
      const segmentIds = new Set<string>();
      for (const session of dataset.sessions) {
        if (session.datasetId !== dataset.id) {
          throw new Error(
            `Session "${session.id}" does not belong to dataset "${dataset.id}"`,
          );
        }
        if (sessionIds.has(session.id)) {
          throw new Error(`Duplicate session identifier "${session.id}"`);
        }
        sessionIds.add(session.id);
        for (const segment of session.segments) {
          if (segment.sessionId !== session.id) {
            throw new Error(
              `Segment "${segment.id}" does not belong to session "${session.id}"`,
            );
          }
          if (segmentIds.has(segment.id)) {
            throw new Error(`Duplicate segment identifier "${segment.id}"`);
          }
          segmentIds.add(segment.id);
        }
      }

      const database = await openDatabase();
      const transaction = database.transaction(
        [STORES.datasets, STORES.sessions, STORES.segments],
        "readwrite",
      );
      const done = transactionDone(transaction);
      const sessionsStore = transaction.objectStore(STORES.sessions);
      const segmentsStore = transaction.objectStore(STORES.segments);
      const sessionKeysRequest = sessionsStore
        .index(INDEXES.sessionDatasetId)
        .getAllKeys(IDBKeyRange.only(dataset.id));
      const segmentKeysRequest = segmentsStore
        .index(INDEXES.segmentDatasetId)
        .getAllKeys(IDBKeyRange.only(dataset.id));
      let sessionKeys: IDBValidKey[] | null = null;
      let segmentKeys: IDBValidKey[] | null = null;
      let replacementWritten = false;

      const writeReplacement = () => {
        if (replacementWritten || !sessionKeys || !segmentKeys) return;
        replacementWritten = true;

        for (const key of sessionKeys) sessionsStore.delete(key);
        for (const key of segmentKeys) segmentsStore.delete(key);

        const { sessions, ...datasetMetadata } = dataset;
        transaction.objectStore(STORES.datasets).put({
          ...datasetMetadata,
          labelSet: [...datasetMetadata.labelSet],
        });
        for (const session of sessions) {
          const { segments, ...sessionMetadata } = session;
          sessionsStore.put({
            ...sessionMetadata,
            channels: [...sessionMetadata.channels],
          });
          for (const segment of segments) {
            segmentsStore.put({
              ...cloneSegment(segment),
              datasetId: dataset.id,
            } satisfies StoredSegment);
          }
        }
      };

      sessionKeysRequest.addEventListener(
        "success",
        () => {
          sessionKeys = sessionKeysRequest.result;
          writeReplacement();
        },
        { once: true },
      );
      segmentKeysRequest.addEventListener(
        "success",
        () => {
          segmentKeys = segmentKeysRequest.result;
          writeReplacement();
        },
        { once: true },
      );

      await done;
      if (!replacementWritten) {
        throw new Error("Dataset replacement did not write any records");
      }
    },

    async clearDataset(datasetId) {
      const database = await openDatabase();
      const lookup = database.transaction(
        [STORES.sessions, STORES.segments],
        "readonly",
      );
      const lookupDone = transactionDone(lookup);
      const sessionKeysRequest = lookup
        .objectStore(STORES.sessions)
        .index(INDEXES.sessionDatasetId)
        .getAllKeys(IDBKeyRange.only(datasetId));
      const segmentKeysRequest = lookup
        .objectStore(STORES.segments)
        .index(INDEXES.segmentDatasetId)
        .getAllKeys(IDBKeyRange.only(datasetId));

      const [sessionKeys, segmentKeys] = await Promise.all([
        requestResult(sessionKeysRequest),
        requestResult(segmentKeysRequest),
      ]);
      await lookupDone;

      const transaction = database.transaction(
        [STORES.datasets, STORES.sessions, STORES.segments],
        "readwrite",
      );
      const done = transactionDone(transaction);
      transaction.objectStore(STORES.datasets).delete(datasetId);
      for (const key of sessionKeys) {
        transaction.objectStore(STORES.sessions).delete(key);
      }
      for (const key of segmentKeys) {
        transaction.objectStore(STORES.segments).delete(key);
      }
      await done;
    },

    async close() {
      if (!databasePromise) return;
      const database = await databasePromise;
      database.close();
      databasePromise = null;
    },
  };
}
