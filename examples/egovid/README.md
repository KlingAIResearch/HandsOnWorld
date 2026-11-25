# EgoVid video preparation example

`taxi.mp4` is a 120-frame, 30 FPS, 720×540 EgoVid clip. `input.csv` pairs
it with its existing dataset caption. Paths in this CSV are relative to this
directory. `source.json` records the source clip identifier; frame 45 (1.5 s)
is the reference image used to select this example.

`train.csv` provides the prepared training sample, with relative paths to the
normalized video and the hand, camera, and intrinsics arrays under `annotations/`.
The public SFT and distillation configurations use this CSV by default.
These annotations come from a completed HaWoR + protagonist-filter run using
`chunk_first` sampling and shared beta per chunk; settings and filtering
decisions are included in `annotations/`. The clip has 120 frames at 720×540;
training resizes it and its camera intrinsics to the configured resolution.

To regenerate the annotations from the repository root, install upstream HaWoR
and its model assets following its own instructions, then use the built-in
adapter below. No edits or extra adapter files inside HaWoR are required:

```bash
python scripts/prepare_videos.py \
  --input-csv examples/egovid/input.csv \
  --hawor-root /path/to/HaWoR \
  --output-dir data/example
```

The video and caption are dataset material, separate from the repository's code license.
