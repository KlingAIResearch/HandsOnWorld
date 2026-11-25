#!/usr/bin/env bash
# Batch evaluation: one independent configured pipeline per visible GPU.
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
PYTHON="${PYTHON:-python}"
CONFIG="${CONFIG:-${REPO_ROOT}/configs/wan22_i2v14b_normal_plucker.json}"

args=(--config "$CONFIG")
[[ -z "${METADATA:-}" ]] || args+=(--metadata "$METADATA")
[[ -z "${DATA_ROOT:-}" ]] || args+=(--data-root "$DATA_ROOT")
[[ -z "${OUTPUT_DIR:-}" ]] || args+=(--output-dir "$OUTPUT_DIR")
[[ -z "${NUM_FRAMES:-}" ]] || args+=(--num-frames "$NUM_FRAMES")
[[ -z "${CLIPS_PER_VIDEO:-}" ]] || args+=(--clips-per-video "$CLIPS_PER_VIDEO")

# Plan and input checks run once and need no GPU or torchrun rendezvous.
for argument in "$@"; do
  case "$argument" in
    --dry-run|--check-inputs|--help|-h)
      exec "$PYTHON" "${SCRIPT_DIR}/infer_batch.py" "${args[@]}" "$@" ;;
  esac
done
NPROC_PER_NODE="${NPROC_PER_NODE:-$("$PYTHON" -c 'import torch; print(torch.cuda.device_count())')}"
if [[ ! "$NPROC_PER_NODE" =~ ^[1-9][0-9]*$ ]]; then
  echo 'No CUDA GPUs detected; set CUDA_VISIBLE_DEVICES or use --dry-run.' >&2
  exit 2
fi
echo "HandsOnWorld SFT inference: ${NPROC_PER_NODE} GPU workers"
exec "$PYTHON" -m torch.distributed.run --standalone --nnodes=1 \
  --nproc_per_node "$NPROC_PER_NODE" "${SCRIPT_DIR}/infer_batch.py" "${args[@]}" "$@"
