# Curated FastGen runtime

This directory contains the Apache-2.0 FastGen components required by the
HandsOnWorld causal SFT, knowledge-distillation, and Self-Forcing recipes.
The source was curated from the local FastGen checkout used for the released
experiments. Model-specific Wan implementations are intentionally excluded:
HandsOnWorld supplies the shared Wan model and checkpoint bridge.

Only the reachable algorithm, schedule, and small utility closure is vendored.
S3 checkpoint I/O, unrelated architectures, callbacks, evaluators, and the
generic FastGen trainer are outside this release surface.
