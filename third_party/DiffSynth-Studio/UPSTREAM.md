# Pinned DiffSynth backend

Upstream: https://github.com/modelscope/DiffSynth-Studio

Base commit: `bfaaf12bf41bb573336572e8c90cd0d5b3361f7c`

This is a source snapshot, not an unmodified upstream checkout. It includes the
local HandsOnWorld pipeline, adapter, data-loader and trainer changes built on
that commit. `backend_manifest.json` records every shipped file's SHA-256,
source SHA-256 and whether the original source differed from the base commit.
Do not replace this tree with an arbitrary `pip install diffsynth` version.

The selected surface is `diffsynth/**/*.py`, the Wan training entry point,
package metadata, and upstream license. Unrelated experiment launchers,
private data, model weights, binaries, logs and Git history are excluded.
Model modules needed by upstream dynamic registration are retained to avoid
breaking the backend's import and weight-loading graph.

Release changes made after taking the source snapshot:

- Hand-surface direction and Plücker construction call `handsonworld.geometry`.
- Channel composition calls `handsonworld.conditioning` (hand then camera).
- Mesh-face and joint-regressor resources use explicit environment paths;
  workstation paths have been removed.
- Package version identifies this backend as `2.0.0+handsonworld.1`.
- I2V training does not load the unrelated S2V audio processor by default;
  that optional resource now requires an explicit path.
- Hand-mesh rendering disables autocast for geometry and nvdiffrast operations,
  preserving float32 inputs when the training forward uses mixed precision.
- HandsOnWorld training resizes the complete source frame to the configured
  resolution and scales camera intrinsics per axis; decoded native dimensions
  are retained and checked against manifest metadata before rendering controls.

Apache-2.0 applies to the DiffSynth source. Upstream file-level notices remain
intact.
No public fork URL is claimed. This vendored tree is the sole backend source
for this local release; switching to a submodule later must retain its changes
and pass the same compatibility tests.

Release integration also disables implicit wandb tracking, declares the existing
video reader dependency (`decord`), and writes hash-bound control metadata at
each checkpoint save when a HandsOnWorld run manifest exists.
