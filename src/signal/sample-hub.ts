export type SampleSubscriber<T> = (chunk: T) => void;
export type Unsubscribe = () => void;

export interface SampleHub<T> {
  readonly subscriberCount: number;
  publish(chunk: T): void;
  subscribe(subscriber: SampleSubscriber<T>): Unsubscribe;
}
/**
 * Synchronous fan-out for decoded sample chunks.
 *
 * Sources publish once; display, collection, and inference consumers subscribe
 * independently. A broken consumer is isolated so it cannot stop the live
 * signal from reaching the remaining subscribers.
 */
export function createSampleHub<T>(): SampleHub<T> {
  const subscribers = new Set<SampleSubscriber<T>>();

  return {
    get subscriberCount() {
      return subscribers.size;
    },

    publish(chunk) {
      for (const subscriber of [...subscribers]) {
        try {
          subscriber(chunk);
        } catch (error) {
          console.error("Sample Hub subscriber failed", error);
        }
      }
    },

    subscribe(subscriber) {
      subscribers.add(subscriber);
      return () => subscribers.delete(subscriber);
    },
  };
}
