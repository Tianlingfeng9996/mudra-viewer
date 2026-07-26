# ADR 0005: MuJoCo hand simulation architecture

- Status: Accepted
- Date: 2026-07-26

## Context

The MyoArm pipeline can now collect and persist labelled EMG segments, create
group-safe train/validation/test partitions, train a baseline ANN in the
browser, and run the trained classifier against fixture or Bluetooth sample
streams.

The next user-visible output must eventually support more than an animated
hand. The intended system should be able to represent articulated robot hands,
contact forces, collision, grasping, and interaction with virtual objects. A
future BIT HAND model may have different joints and actuators from the initial
demonstration model, and TF2AngleNet may provide continuous joint targets
instead of discrete gesture labels.

The application is built with Vite and deployed under a GitHub Pages subpath.
GitHub Pages does not provide project-controlled COOP and COEP response headers,
so browser features that require cross-origin isolation cannot be assumed.

## Decision

### Physics and browser runtime

Use the canonical Google DeepMind JavaScript bindings for MuJoCo,
`@mujoco/mujoco`, pinned initially to version `3.10.0`.

Use the default single-threaded WebAssembly build. The multi-threaded build
requires `SharedArrayBuffer` and cross-origin isolation headers that the current
GitHub Pages deployment cannot configure reliably.

MuJoCo is an optional MyoArm capability. It must be loaded with a dynamic
`import()` only after an explicit user action in the Hand Visualization card.
Opening the application, collecting data, training a model, and using the other
views must not download or initialize MuJoCo.

The initial runtime milestone only proves that the WASM module can be fetched
and initialized. Loading a robot model, creating `MjModel`/`MjData`, stepping
physics, and rendering are separate milestones.

### Control boundary

Neural-network code must not write model-specific actuator indices directly.
Predictions are converted into a semantic hand target:

```ts
interface HandTarget {
  thumbFlexion: number;
  thumbOpposition: number;
  indexFlexion: number;
  middleFlexion: number;
  ringFlexion: number;
  littleFlexion: number;
  forearmRotation: number;
  confidence: number;
}
```

The baseline ANN will map discrete gesture labels to target presets.
TF2AngleNet can later map continuous outputs to the same target. A model
adapter converts the semantic target to named MuJoCo actuator controls, clamps
them to each actuator's control range, and writes `MjData.ctrl` before physics
steps.

This boundary allows the robot model to change without modifying collection,
preprocessing, training, or live inference.

### Initial model and rendering

Use the right Shadow Hand E3M5 from MuJoCo Menagerie as the first articulated
model. Vendor the required MJCF, mesh assets, license, and source revision in
the repository so builds remain reproducible and do not fetch model files from
GitHub at runtime.

The model contains a forearm and dexterous hand but no shoulder or elbow.
Add an explicit forearm axial joint in a local wrapper or derived MJCF so
`pronation` and `supination` can be represented.

Use Three.js for browser rendering. MuJoCo remains the source of physical
state; Three.js renders geometry and camera interaction. Extend the canonical
MuJoCo example to support `mjGEOM_MESH`, because the Shadow Hand visual model
uses mesh geometry.

### Runtime ownership and cleanup

The simulation layer will live under `src/simulation/`, independently of
`src/views/` and `src/myoarm/`:

```text
src/simulation/
  hand-target.ts
  hand-model-adapter.ts
  mujoco-runtime.ts
  mujoco-three-renderer.ts
  shadow-hand-adapter.ts
```

The view owns only UI state and a simulation controller. Every Embind/WASM
object and Three.js GPU resource must have an explicit disposal path. Source
changes reset inference windows; model or simulation disposal must stop all
animation callbacks before releasing memory.

## Delivery sequence

1. Pin and lazily initialize the single-threaded MuJoCo WASM module.
2. Vendor and load the Shadow Hand assets through `MjVFS`.
3. Render the static model and provide orbit camera controls.
4. Verify named actuators with manual neutral/open/grasp/pinch controls.
5. Add forearm rotation and smooth actuator target interpolation.
6. Publish smoothed ANN predictions to the semantic hand-target controller.
7. Add a free object, contacts, reset controls, and grasp interaction.
8. Implement model export/import and a TF2AngleNet classifier adapter.
9. Convert and validate the BIT HAND model, then add a BIT HAND adapter without
   changing the neural-network pipeline.

## Consequences

- MuJoCo adds a comparatively large optional WASM download, but lazy loading
  keeps it out of the initial application path.
- The first GitHub Pages implementation is single-threaded. A future deployment
  with configurable security headers can evaluate the multi-threaded build.
- The first hand model is a robot hand rather than a biological hand, matching
  the intended interaction and later BIT HAND direction.
- Discrete ANN predictions can drive useful pose presets now, while the same
  semantic boundary remains usable by continuous angle models later.
- Mesh rendering and explicit WASM/GPU cleanup add implementation work, but
  they avoid coupling the product to a model-specific animation.
- Fixture-driven accuracy remains a pipeline check only; MuJoCo visualization
  does not change the need for independent real-sensor sessions.
