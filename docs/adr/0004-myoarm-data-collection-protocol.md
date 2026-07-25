# 0004. MyoArm data collection protocol and hardware-independent development

## Status

Accepted for the first data-collection vertical slice.

## Context

The application already decodes 3-channel Mudra Link sEMG at 834 Hz from both
Web Bluetooth and recorded fixtures. The MyoArm page does not yet have a
labelled dataset format, collection workflow, storage layer, or model contract.

The first implementation is being developed before a physical sensor is
available. The repository currently contains ten short fixtures: five gestures
at normal and strong effort. They are sufficient to test plumbing, but not to
measure physiological model performance.

## Decision

### Model task

Version 1 is a single-user, discrete gesture classifier with six labels:

1. `rest`
2. `open`
3. `grasp`
4. `pinch`
5. `pronation`
6. `supination`

Normal and strong effort are variations of one gesture, not separate classes.
Continuous finger or joint-angle regression is outside this version because it
requires synchronized ground-truth measurements from encoders, a glove, or a
vision system.

### Capture protocol

For every label, the default collection sequence is:

- 3 seconds of countdown;
- 3 seconds of stable labelled capture;
- 2 seconds of recovery;
- 10 repetitions.

The countdown and recovery intervals are not labelled as the active gesture.
`rest` is deliberately recorded from a relaxed wearer. Zeros or generated noise
must not be used as physiological `rest` training data.

### Data representation

The decoded contract is three channels in this fixed order:

1. `ulnar`
2. `median`
3. `radial`

Decoded values are retained as sample-major, interleaved `Int32Array` data at
834 Hz until preprocessing. Each segment records its label, effort, repetition,
duration, sample count, quality flags, session, source, arm side, and
pseudonymous participant identifier.

The schema version starts at `1`. Large sample arrays are stored in IndexedDB;
`localStorage` is reserved for small preferences. Dataset ZIP export includes a
versioned JSON manifest and one little-endian Int32 binary payload per segment.

### Hardware-independent development

The built-in fixtures act as a virtual source and must enter the same decoded
sample pipeline later used by Bluetooth. Folder names map as follows:

- `open` and `open (strong)` -> `open`
- `grasp` and `grasp (strong)` -> `grasp`
- `pinch` and `pinch (strong)` -> `pinch`
- `pronation` and `pronation (strong)` -> `pronation`
- `supination` and `supination (strong)` -> `supination`

The suffix controls effort metadata only. There is no fixture for `rest`.

Fixture data may verify routing, collection state, storage, preprocessing,
training execution, and inference execution. It must not be used to claim model
accuracy: there is only one recording per gesture/effort combination and no
independent participant or session.

### Evaluation boundary

Overlapping windows from one captured repetition must remain in the same
train, validation, or test partition. Splitting individual windows randomly
would leak nearly identical signal into multiple partitions and inflate
metrics.

## Consequences

- Data collection, storage, and training UI can be implemented without waiting
  for hardware.
- A real `rest` class and reliable performance measurements remain blocked on
  sensor access.
- The MyoArm presentation is isolated in `src/views/myoarm.ts`; data contracts
  and workflow logic remain under `src/myoarm/`.
- Bluetooth and fixture decoding publish sample-major `SampleChunk` values
  through `src/signal/`, which has no dependency on MyoArm. Display and MyoArm
  are independent subscribers.
- A fixture-fed collection state machine can capture a playback into a labelled
  in-memory `MyoArmSegment` without changing the original Play button's meaning.
- IndexedDB stores dataset, session, and segment records separately. A completed
  fixture segment is committed with its parent metadata in one transaction and
  restored when the page reloads.
- Dataset export and import use a validated, store-only ZIP archive. Import
  replaces the matching local fixture dataset atomically after user
  confirmation.
- The next implementation step is shared preprocessing for offline training
  and live inference.
