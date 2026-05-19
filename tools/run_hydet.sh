#!/usr/bin/env bash
set -euo pipefail

ROOT="${HYDET_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
DATASET="${1:-hrsc}"
ACTION="${2:-train}"
GPUS="${HYDET_GPUS:-1}"
PORT="${PORT:-${HYDET_PORT:-29500}}"

case "$DATASET" in
  hrsc|fair1m)
    ;;
  *)
    echo "Unknown dataset: $DATASET (use hrsc or fair1m)" >&2
    exit 2
    ;;
esac

CONFIG="${HYDET_CONFIG:-projects/HyDet/configs/hydet_${DATASET}_r50.py}"
WORK_DIR="${HYDET_WORK_DIR:-$ROOT/work_dirs/${DATASET}_hydet_r50}"
EVAL_DIR="${HYDET_EVAL_DIR:-$WORK_DIR/eval}"
OUT_FILE="${HYDET_OUT_FILE:-$EVAL_DIR/preds.pkl}"

cd "$ROOT"

activate_conda() {
  local env_name="${HYDET_CONDA_ENV:-}"
  if [[ -z "$env_name" ]]; then
    return
  fi
  if command -v conda >/dev/null 2>&1; then
    eval "$(conda shell.bash hook)"
    conda activate "$env_name"
    return
  fi
  if [[ -n "${HYDET_CONDA_SH:-}" && -f "${HYDET_CONDA_SH}" ]]; then
    # shellcheck disable=SC1090
    source "${HYDET_CONDA_SH}"
    conda activate "$env_name"
    return
  fi
  echo "Unable to activate conda env: $env_name" >&2
  exit 2
}

split_opts() {
  local raw="$1"
  if [[ -z "$raw" ]]; then
    return
  fi
  read -r -a SPLIT_RESULT <<<"$raw"
  printf '%s\n' "${SPLIT_RESULT[@]}"
}

build_common_cfg_opts() {
  local opts=()
  opts+=("work_dir=${WORK_DIR}")
  opts+=("randomness.seed=${HYDET_SEED:-3407}")
  opts+=("randomness.deterministic=${HYDET_DETERMINISTIC:-True}")

  if [[ -n "${HYDET_ITERS:-}" ]]; then
    opts+=("train_cfg.max_iters=${HYDET_ITERS}")
  fi
  if [[ -n "${HYDET_VAL_INTERVAL:-}" ]]; then
    opts+=("train_cfg.val_interval=${HYDET_VAL_INTERVAL}")
    opts+=("default_hooks.checkpoint.interval=${HYDET_VAL_INTERVAL}")
  fi
  if [[ -n "${HYDET_BATCH_SIZE:-}" ]]; then
    opts+=("train_dataloader.batch_size=${HYDET_BATCH_SIZE}")
  fi
  if [[ -n "${HYDET_EVAL_BATCH_SIZE:-}" ]]; then
    opts+=("val_dataloader.batch_size=${HYDET_EVAL_BATCH_SIZE}")
    opts+=("test_dataloader.batch_size=${HYDET_EVAL_BATCH_SIZE}")
  fi
  if [[ -n "${HYDET_NUM_WORKERS:-}" ]]; then
    opts+=("train_dataloader.num_workers=${HYDET_NUM_WORKERS}")
    opts+=("val_dataloader.num_workers=${HYDET_NUM_WORKERS}")
    opts+=("test_dataloader.num_workers=${HYDET_NUM_WORKERS}")
  fi
  if [[ -n "${HYDET_CFG_OPTIONS:-}" ]]; then
    while IFS= read -r item; do
      [[ -n "$item" ]] && opts+=("$item")
    done < <(split_opts "${HYDET_CFG_OPTIONS}")
  fi
  printf '%s\n' "${opts[@]}"
}

resolve_checkpoint() {
  if [[ -n "${HYDET_CHECKPOINT:-}" ]]; then
    echo "${HYDET_CHECKPOINT}"
    return
  fi
  if [[ -f "${WORK_DIR}/latest.pth" ]]; then
    echo "${WORK_DIR}/latest.pth"
    return
  fi
  ls -1t "${WORK_DIR}"/iter_*.pth 2>/dev/null | head -n 1
}

run_launcher() {
  local mode="$1"
  shift
  local -a cmd=("$@")
  if [[ "${GPUS}" -gt 1 ]]; then
    python -m torch.distributed.run \
      --nproc_per_node="${GPUS}" \
      --master_port="${PORT}" \
      "${cmd[@]}" \
      --launcher pytorch
  else
    python "${cmd[@]}"
  fi
}

run_train() {
  mkdir -p "${WORK_DIR}"
  local -a cfg_opts=()
  while IFS= read -r item; do
    [[ -n "$item" ]] && cfg_opts+=("$item")
  done < <(build_common_cfg_opts)
  local -a cmd=("tools/train.py" "${CONFIG}")
  if [[ "${#cfg_opts[@]}" -gt 0 ]]; then
    cmd+=("--cfg-options" "${cfg_opts[@]}")
  fi
  run_launcher train "${cmd[@]}"
}

run_test() {
  mkdir -p "${EVAL_DIR}"
  local ckpt
  ckpt="$(resolve_checkpoint)"
  if [[ -z "${ckpt}" || ! -f "${ckpt}" ]]; then
    echo "Checkpoint not found. Set HYDET_CHECKPOINT or train first." >&2
    exit 2
  fi
  local -a cfg_opts=()
  while IFS= read -r item; do
    [[ -n "$item" ]] && cfg_opts+=("$item")
  done < <(build_common_cfg_opts)
  local -a cmd=("tools/test.py" "${CONFIG}" "${ckpt}" "--work-dir" "${EVAL_DIR}" "--out" "${OUT_FILE}")
  if [[ "${#cfg_opts[@]}" -gt 0 ]]; then
    cmd+=("--cfg-options" "${cfg_opts[@]}")
  fi
  run_launcher test "${cmd[@]}"
}

activate_conda
export PYTHONPATH="${ROOT}:${PYTHONPATH:-}"

case "$ACTION" in
  train)
    run_train
    ;;
  test|eval|predict)
    run_test
    ;;
  train-test|all)
    run_train
    run_test
    ;;
  *)
    echo "Unknown action: $ACTION (use train, test, or train-test)" >&2
    exit 2
    ;;
esac
