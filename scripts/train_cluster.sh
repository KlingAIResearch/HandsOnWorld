#!/usr/bin/env bash
# Launch one HandsOnWorld training wrapper per node. The wrapper starts the
# local Accelerate workers; this script only coordinates node-level settings.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
cluster_env="${CLUSTER_ENV_FILE:-${REPO_ROOT}/.local/cluster.env}"
if [[ -n "${CLUSTER_ENV_FILE+x}" && ! -f "$cluster_env" ]]; then
  echo "train_cluster.sh: CLUSTER_ENV_FILE does not exist: $cluster_env" >&2
  exit 2
fi
if [[ -f "$cluster_env" ]]; then
  # shellcheck disable=SC1090
  source "$cluster_env"
fi
PYTHON="${PYTHON:-python}"
CONFIG="${CONFIG:-${REPO_ROOT}/configs/wan22_i2v14b_normal_plucker.json}"
NOISE_LEVEL="${NOISE_LEVEL:-both}"
HOW_TASK="${HOW_TASK:-sft}"
case "$HOW_TASK" in
  sft) launcher_script="${SCRIPT_DIR}/train.py" ;;
  distill) launcher_script="${SCRIPT_DIR}/distill.py" ;;
  *) echo "Unknown HOW_TASK: $HOW_TASK" >&2; exit 2 ;;
esac
if [[ -n "${ACCELERATE_CONFIG:-}" ]]; then
  echo "train_cluster.sh: ACCELERATE_CONFIG conflicts with CLI cluster topology" >&2
  exit 2
fi

is_uint() { [[ "$1" =~ ^[0-9]+$ ]]; }
require_uint() {
  if ! is_uint "$2"; then
    echo "train_cluster.sh: $1 must be a non-negative integer, got: $2" >&2
    exit 2
  fi
}

GLOBAL_ID="${GLOBAL_ID:-${SLURM_NODEID:-0}}"
GLOBAL_SIZE="${GLOBAL_SIZE:-${SLURM_NNODES:-1}}"
require_uint GLOBAL_ID "$GLOBAL_ID"
require_uint GLOBAL_SIZE "$GLOBAL_SIZE"
if (( GLOBAL_SIZE < 1 || GLOBAL_ID >= GLOBAL_SIZE )); then
  echo "train_cluster.sh: require GLOBAL_SIZE >= 1 and GLOBAL_ID < GLOBAL_SIZE" >&2
  exit 2
fi

DRY_RUN=0
for argument in "$@"; do
  if [[ "$argument" == "--dry-run" ]]; then
    DRY_RUN=1
  fi
done

if [[ -n "${GPUS_PER_MACHINE:-}" ]]; then
  require_uint GPUS_PER_MACHINE "$GPUS_PER_MACHINE"
elif (( DRY_RUN )); then
  GPUS_PER_MACHINE=1
else
  GPUS_PER_MACHINE="$("$PYTHON" -c 'import torch; print(torch.cuda.device_count())')"
  require_uint GPUS_PER_MACHINE "$GPUS_PER_MACHINE"
fi
if (( GPUS_PER_MACHINE < 1 )); then
  echo "train_cluster.sh: GPUS_PER_MACHINE must be at least 1" >&2
  exit 2
fi
NUM_PROCESSES=$((GLOBAL_SIZE * GPUS_PER_MACHINE))

if [[ -z "${NCCL_LIB_DIR:-}" ]]; then
  vendored_nccl="${REPO_ROOT}/vendor/nccl/lib"
  if [[ -f "${vendored_nccl}/libnccl.so.2" ]]; then
    NCCL_LIB_DIR="$vendored_nccl"
  elif [[ -n "${CONDA_PREFIX:-}" ]]; then
    shopt -s nullglob
    conda_nccl=("${CONDA_PREFIX}"/lib/python*/site-packages/nvidia/nccl/lib)
    shopt -u nullglob
    if (( ${#conda_nccl[@]} > 0 )); then
      NCCL_LIB_DIR="${conda_nccl[0]}"
    fi
  fi
fi
if [[ -n "${NCCL_LIB_DIR:-}" ]]; then
  export NCCL_LIB_DIR
  export LD_LIBRARY_PATH="${NCCL_LIB_DIR}${CONDA_PREFIX:+:${CONDA_PREFIX}/lib}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
  if [[ -f "${NCCL_LIB_DIR}/libnccl.so.2" ]]; then
    export LD_PRELOAD="${NCCL_LIB_DIR}/libnccl.so.2${LD_PRELOAD:+:${LD_PRELOAD}}"
  fi
fi
export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-eth0}"
export GLOO_SOCKET_IFNAME="${GLOO_SOCKET_IFNAME:-eth0}"

if [[ -n "${PROXY_URL:-}" ]]; then
  export http_proxy="$PROXY_URL" https_proxy="$PROXY_URL"
  export HTTP_PROXY="$PROXY_URL" HTTPS_PROXY="$PROXY_URL"
fi

config_name="${CONFIG##*/}"
config_stem="${config_name%.*}"
config_stem="$(printf '%s' "$config_stem" | tr -c 'A-Za-z0-9_.-' '-')"
RUN_ID="${RUN_ID:-${SLURM_JOB_ID:-${JOB_ID:-handsonworld-${config_stem}}}}"
if [[ ! "$RUN_ID" =~ ^[A-Za-z0-9_.-]+$ ]]; then
  echo "train_cluster.sh: RUN_ID may contain only letters, digits, dot, underscore, and hyphen" >&2
  exit 2
fi
if [[ "$RUN_ID" == "." || "$RUN_ID" == ".." ]]; then
  echo "train_cluster.sh: RUN_ID cannot be dot or dot-dot" >&2
  exit 2
fi

MASTER_PORT="${MASTER_PORT:-29500}"
RENDEZVOUS_TIMEOUT="${RENDEZVOUS_TIMEOUT:-300}"
require_uint MASTER_PORT "$MASTER_PORT"
require_uint RENDEZVOUS_TIMEOUT "$RENDEZVOUS_TIMEOUT"
if (( MASTER_PORT < 1 || MASTER_PORT > 65535 || RENDEZVOUS_TIMEOUT < 1 )); then
  echo "train_cluster.sh: invalid MASTER_PORT or RENDEZVOUS_TIMEOUT" >&2
  exit 2
fi

MASTER_FILE="${MASTER_FILE:-${REPO_ROOT}/outputs/rendezvous/${RUN_ID}/master.json}"
if [[ "$MASTER_FILE" != /* ]]; then
  MASTER_FILE="${REPO_ROOT}/${MASTER_FILE}"
fi
MASTER_FILE="$(realpath -m -- "$MASTER_FILE")"
RENDEZVOUS_PYTHON="${RENDEZVOUS_PYTHON:-python3}"
if ! command -v "$RENDEZVOUS_PYTHON" >/dev/null 2>&1; then
  echo "train_cluster.sh: rendezvous helper not found: $RENDEZVOUS_PYTHON" >&2
  exit 2
fi

detect_master_addr() {
  if [[ -n "${MASTER_ADDR:-}" ]]; then
    printf '%s\n' "$MASTER_ADDR"
  elif [[ -n "${MY_NODE_IP:-}" ]]; then
    printf '%s\n' "$MY_NODE_IP"
  elif (( GLOBAL_SIZE == 1 )); then
    printf '127.0.0.1\n'
  else
    hostname -I 2>/dev/null | awk '{print $1}'
  fi
}

merge_no_proxy() {
  local value="${no_proxy:-${NO_PROXY:-}}" item
  for item in localhost 127.0.0.1 "$1"; do
    [[ -z "$item" ]] && continue
    if [[ ",${value}," != *",${item},"* ]]; then
      value="${value:+${value},}${item}"
    fi
  done
  export no_proxy="$value" NO_PROXY="$value"
}

child_pid=""
owned_launch_id=""
record_tmp=""
cleanup_record() {
  if [[ -n "$record_tmp" && -f "$record_tmp" ]]; then
    rm -f -- "$record_tmp"
  fi
  [[ "$GLOBAL_ID" == "0" && -n "$owned_launch_id" && -f "$MASTER_FILE" ]] || return 0
  if "$RENDEZVOUS_PYTHON" - "$MASTER_FILE" "$owned_launch_id" <<'PY'
import json, sys
try:
    record = json.load(open(sys.argv[1], encoding="utf-8"))
except (OSError, ValueError):
    raise SystemExit(1)
raise SystemExit(0 if record.get("launch_id") == sys.argv[2] else 1)
PY
  then
    rm -f -- "$MASTER_FILE"
  fi
}
on_signal() {
  local signal="$1" status="$2"
  # Background jobs can inherit SIGINT ignored; TERM reliably stops the group.
  if [[ "$signal" == "INT" ]]; then signal=TERM; fi
  if [[ -n "$child_pid" ]] && kill -0 "$child_pid" 2>/dev/null; then
    kill -s "$signal" -- "-$child_pid" 2>/dev/null || true
    wait "$child_pid" 2>/dev/null || true
  fi
  cleanup_record
  exit "$status"
}
trap cleanup_record EXIT
trap 'on_signal INT 130' INT
trap 'on_signal TERM 143' TERM

if (( DRY_RUN )); then
  master_addr="$(detect_master_addr || true)"
  master_addr="${master_addr:-127.0.0.1}"
  launch_id="dry-run-${RUN_ID}"
  rendezvous_dir="${RENDEZVOUS_DIR:-$(dirname -- "$MASTER_FILE")/session-${launch_id}}"
  if [[ "$rendezvous_dir" != /* ]]; then
    rendezvous_dir="${REPO_ROOT}/${rendezvous_dir}"
  fi
  rendezvous_dir="$(realpath -m -- "$rendezvous_dir")"
else
  mkdir -p -- "$(dirname -- "$MASTER_FILE")"
  if [[ "$GLOBAL_ID" == "0" ]]; then
    if [[ -e "$MASTER_FILE" ]]; then
      echo "train_cluster.sh: rendezvous record already exists; refusing to overwrite: $MASTER_FILE" >&2
      exit 2
    fi
    master_addr="$(detect_master_addr || true)"
    if [[ -z "$master_addr" ]]; then
      echo "train_cluster.sh: could not determine master address; set MASTER_ADDR or MY_NODE_IP" >&2
      exit 2
    fi
    launch_id="$(tr -d '-' < /proc/sys/kernel/random/uuid 2>/dev/null || true)"
    launch_id="${launch_id:-${RUN_ID}-$$-${RANDOM}}"
    rendezvous_dir="${RENDEZVOUS_DIR:-$(dirname -- "$MASTER_FILE")/session-${launch_id}}"
    if [[ "$rendezvous_dir" != /* ]]; then
      rendezvous_dir="${REPO_ROOT}/${rendezvous_dir}"
    fi
    rendezvous_dir="$(realpath -m -- "$rendezvous_dir")"
    record_tmp="${MASTER_FILE}.tmp.${launch_id}"
    "$RENDEZVOUS_PYTHON" - "$record_tmp" "$RUN_ID" "$GLOBAL_SIZE" "$master_addr" "$launch_id" "$rendezvous_dir" <<'PY'
import json, os, sys
path, run_id, world_size, address, launch_id, rendezvous_dir = sys.argv[1:]
payload = {"run_id": run_id, "world_size": int(world_size), "master_addr": address,
           "launch_id": launch_id, "rendezvous_dir": rendezvous_dir}
with open(path, "x", encoding="utf-8") as stream:
    json.dump(payload, stream, sort_keys=True)
    stream.write("\n")
    stream.flush()
    os.fsync(stream.fileno())
PY
    if ! ln -- "$record_tmp" "$MASTER_FILE" 2>/dev/null; then
      rm -f -- "$record_tmp"
      echo "train_cluster.sh: another launcher created rendezvous record: $MASTER_FILE" >&2
      exit 2
    fi
    rm -f -- "$record_tmp"
    record_tmp=""
    owned_launch_id="$launch_id"
  else
    deadline=$((SECONDS + RENDEZVOUS_TIMEOUT))
    while [[ ! -f "$MASTER_FILE" ]]; do
      if (( SECONDS >= deadline )); then
        echo "train_cluster.sh: timed out waiting for rendezvous record: $MASTER_FILE" >&2
        exit 2
      fi
      sleep 1
    done
    mapfile -t record_fields < <("$RENDEZVOUS_PYTHON" - "$MASTER_FILE" "$RUN_ID" "$GLOBAL_SIZE" <<'PY'
import json, sys
path, expected_run, expected_world = sys.argv[1:]
try:
    with open(path, encoding="utf-8") as stream:
        record = json.load(stream)
except (OSError, ValueError) as exc:
    raise SystemExit(f"invalid rendezvous record: {exc}")
if record.get("run_id") != expected_run or record.get("world_size") != int(expected_world):
    raise SystemExit("rendezvous record does not match RUN_ID/GLOBAL_SIZE")
for key in ("master_addr", "launch_id", "rendezvous_dir"):
    value = record.get(key)
    if not isinstance(value, str) or not value or "\n" in value:
        raise SystemExit(f"invalid rendezvous field: {key}")
    print(value)
PY
    )
    if (( ${#record_fields[@]} != 3 )); then
      echo "train_cluster.sh: invalid rendezvous record: $MASTER_FILE" >&2
      exit 2
    fi
    master_addr="${record_fields[0]}"
    launch_id="${record_fields[1]}"
    rendezvous_dir="${record_fields[2]}"
  fi
fi

if [[ "$rendezvous_dir" != /* ]]; then
  rendezvous_dir="${REPO_ROOT}/${rendezvous_dir}"
fi
rendezvous_dir="$(realpath -m -- "$rendezvous_dir")"

merge_no_proxy "$master_addr"

args=(
  --config "$CONFIG"
  --num-machines "$GLOBAL_SIZE"
  --machine-rank "$GLOBAL_ID"
  --num-processes "$NUM_PROCESSES"
  --main-process-ip "$master_addr"
  --main-process-port "$MASTER_PORT"
  --launch-id "$launch_id"
  --rendezvous-dir "$rendezvous_dir"
  --rendezvous-timeout "$RENDEZVOUS_TIMEOUT"
)
if [[ "$HOW_TASK" == "distill" ]]; then
  args+=(--stage "${DISTILL_STAGE:?Set DISTILL_STAGE}")
else
  args+=(--noise-level "$NOISE_LEVEL")
fi
if [[ "$HOW_TASK" == "sft" && "${ALLOW_MISSING_METADATA:-0}" == "1" ]]; then
  args+=(--allow-missing-metadata)
fi
if [[ "$HOW_TASK" == "sft" && -n "${MIXED_PRECISION:-}" ]]; then
  args+=(--mixed-precision "$MIXED_PRECISION")
fi

printf 'HandsOnWorld cluster launch: run_id=%s rank=%s/%s gpus_per_machine=%s master=%s:%s launch_id=%s\n' \
  "$RUN_ID" "$GLOBAL_ID" "$GLOBAL_SIZE" "$GPUS_PER_MACHINE" "$master_addr" "$MASTER_PORT" "$launch_id"
setsid "$PYTHON" "$launcher_script" "${args[@]}" "$@" &
child_pid=$!
if wait "$child_pid"; then
  child_status=0
else  
  child_status=$?
fi
child_pid=""
exit "$child_status"

python ../scripts/train.py
