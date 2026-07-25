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

### Shared preprocessing v1

Offline training and live inference use the same versioned transform,
`myoarm-raw-window-v1`:

- accepted segments only;
- fixed `ulnar`, `median`, `radial` channel order at 834 Hz;
- 200 ms windows (167 samples) with a 100 ms stride (83 samples);
- sample-major interleaved Float32 model inputs;
- each raw value divided by the signed 16-bit full scale, 32,768;
- no clipping, per-window standardization, filtering, padding, or fitted
  dataset statistics;
- incomplete segment tails are dropped.

Fixed full-scale conversion makes the first transform deterministic and usable
before a representative training set exists. It also avoids leaking
validation/test statistics into preprocessing. Amplitude information remains
available for effort-sensitive models, and values beyond the nominal device
range remain visible instead of being silently clipped.

Every prepared window carries its parent segment as a grouping identifier.
Future dataset splitting must assign whole segment or session groups to one
partition before any ANN or TF2AngleNet performance is reported.

### Grouped dataset split v1

`myoarm-grouped-split-v1` creates deterministic 60% train, 20% validation,
and 20% test partitions. Ratios are applied to independent groups rather than
individual windows, so window totals may differ slightly when recordings have
different lengths.

Two explicit grouping modes are available:

- `segment`: all overlapping windows from one repetition stay together.
  Segment groups are stratified separately for each observed gesture label.
  With ten repetitions per label, each label contributes six segment groups
  to train, two to validation, and two to test.
- `session`: every segment and window recorded in one session stays together.
  Whole sessions are deterministically ordered and allocated, providing the
  stronger check against session-specific placement and recording conditions.

The assignment uses a fixed seed and stable group identifiers, so the same
dataset and configuration produce the same split. A split is marked ready only
when train, validation, and test are non-empty and every observed label is
represented in all three partitions. Missing coverage is reported instead of
moving windows or segments across group boundaries.

One segment per label, or one session containing all labels, is intentionally
not enough for evaluation. Such data remains useful for checking execution of
the preprocessing and model-training code.

### Baseline ANN and live inference v1

`myoarm-baseline-ann-v1` provides the first end-to-end browser training loop:

- the currently selected grouped split supplies train, validation, and test
  windows;
- only train windows update model weights;
- validation cross-entropy and accuracy are recalculated after every epoch;
- test metrics are calculated once after training completes;
- the model has 501 input values, 16 ReLU hidden units, and one Softmax output
  per observed gesture label;
- mini-batch Adam uses deterministic weight initialization and shuffle order;
- trained weights remain in memory and are invalidated when the saved dataset
  or grouping mode changes.

The runtime model contract is independent of the baseline implementation so a
future TF2AngleNet adapter can drive the same training and inference UI.

Live inference accumulates decoded `SampleChunk` values into the same 167-sample
windows used offline. It emits a prediction every 83 samples, presents a
five-window moving average of class probabilities, and resets at every source
boundary so samples from separate recordings cannot enter one model input.

Built-in fixtures can verify loss calculation, optimizer execution, live
windowing, and prediction display without hardware. Replaying one recording as
multiple segment identifiers does not create independent biological examples,
so fixture validation/test metrics must not be reported as model performance.

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
- Shared preprocessing now produces deterministic fixed-size inputs for both
  the baseline ANN and TF2AngleNet integration.
- Grouped splitting now prevents overlapping-window leakage and reports label
  coverage for both repetition-level and session-level evaluation.
- A baseline ANN can now be trained and evaluated in the browser, then reused
  immediately for fixture or Bluetooth stream predictions.
- The next implementation step is model export/import with preprocessing and
  label metadata, followed by a TF2AngleNet adapter implementing the shared
  classifier contract.
