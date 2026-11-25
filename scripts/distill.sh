#!/usr/bin/env bash
# Reuse the SFT launcher's node coordination, NCCL and proxy configuration.
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
stage=${1:-dmd}
if (( $# )); then shift; fi
case "$stage" in
  prepare|causal_sft|ode|kd|dmd|infer) ;;
  *) echo 'Usage: bash scripts/distill.sh {prepare|causal_sft|ode|kd|dmd|infer} [options]' >&2; exit 2 ;;
esac
export HOW_TASK=distill DISTILL_STAGE="$stage"
export CONFIG="${CONFIG:-${SCRIPT_DIR}/../configs/distillation/wan21_normal.json}"
if [[ -n ${GPUS_PER_NODE:-} && -z ${GPUS_PER_MACHINE:-} ]]; then
  export GPUS_PER_MACHINE="$GPUS_PER_NODE"
fi
export RUN_ID="${RUN_ID:-distill-${stage}-$(basename -- "$CONFIG" .json)}"
exec bash "${SCRIPT_DIR}/train_cluster.sh" "$@"
