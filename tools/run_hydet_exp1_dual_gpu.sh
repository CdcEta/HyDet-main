#!/usr/bin/env bash
set -euo pipefail

ROOT="${HYDET_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
DATASET="${1:-hrsc}"
ACTION="${2:-all}"
PORT="${PORT:-29557}"
ITERS="${HYDET_ITERS:-10000}"
VARIANTS="${HYDET_VARIANTS:-base hydet}"

cd "$ROOT"
source /root/miniconda3/etc/profile.d/conda.sh
conda activate castdet
export PYTHONPATH="$ROOT:${PYTHONPATH:-}"

if [[ "$DATASET" == "hrsc" ]]; then
  BASE_CFG="projects/HyDet/configs/hrsc_exp1_castdet_main.py"
  HYDET_CFG="projects/HyDet/configs/hrsc_exp1_hydet_main.py"
  WORK_ROOT="${HYDET_EXP1_WORK_ROOT:-$ROOT/work_dirs/exp1_hrsc_main}"
  TREE="projects/HyDet/resources/hrsc_hier/tree_validated.json"
  LEAFS="projects/HyDet/resources/hrsc_hier/class_names_leaf.txt"
elif [[ "$DATASET" == "fair1m" ]]; then
  BASE_CFG="projects/HyDet/configs/fair1m_exp1_castdet_main.py"
  HYDET_CFG="projects/HyDet/configs/fair1m_exp1_hydet_main.py"
  WORK_ROOT="${HYDET_EXP1_WORK_ROOT:-$ROOT/work_dirs/exp1_fair1m_main}"
  TREE="projects/HyDet/resources/fair1m_hier/tree_validated.json"
  LEAFS="projects/HyDet/resources/fair1m_hier/class_names_leaf.txt"
else
  echo "Unknown dataset: $DATASET" >&2
  exit 2
fi

COMMON_OPTS=(
  train_dataloader.batch_size="${HYDET_BATCH_SIZE:-4}"
  train_dataloader.num_workers=4
  train_dataloader.persistent_workers=True
  val_dataloader.batch_size="${HYDET_EVAL_BATCH_SIZE:-4}"
  val_dataloader.num_workers=4
  val_dataloader.persistent_workers=True
  test_dataloader.batch_size="${HYDET_EVAL_BATCH_SIZE:-4}"
  test_dataloader.num_workers=4
  test_dataloader.persistent_workers=True
  randomness.seed=3407
  randomness.deterministic=True
)

variant_cfg() {
  if [[ "$1" == "base" ]]; then
    echo "$BASE_CFG"
  else
    echo "$HYDET_CFG"
  fi
}

variant_opts() {
  echo "work_dir=${WORK_ROOT}/$1"
}

latest_ckpt() {
  local work_dir="${WORK_ROOT}/$1"
  if [[ -f "${work_dir}/iter_${ITERS}.pth" ]]; then
    echo "${work_dir}/iter_${ITERS}.pth"
  elif [[ -f "${work_dir}/latest.pth" ]]; then
    echo "${work_dir}/latest.pth"
  else
    ls -1t "${work_dir}"/iter_*.pth 2>/dev/null | head -n 1
  fi
}

run_train() {
  local variant="$1"
  local cfg
  cfg="$(variant_cfg "$variant")"
  local work_dir="${WORK_ROOT}/${variant}"
  if [[ -f "${work_dir}/iter_${ITERS}.pth" ]]; then
    echo "Skip completed train: ${variant}"
    return
  fi
  read -r -a FLAGS <<<"$(variant_opts "$variant")"
  local resume_args=()
  local load_args=()
  if [[ -f "${work_dir}/latest.pth" ]]; then
    resume_args+=(--resume)
  else
    local newest
    newest="$(ls -1t "${work_dir}"/iter_*.pth 2>/dev/null | head -n 1 || true)"
    if [[ -n "$newest" && -f "$newest" ]]; then
      load_args+=(load_from="$newest")
    fi
  fi
  python -m torch.distributed.run \
    --nproc_per_node=2 \
    --master_port="$PORT" \
    tools/train.py "$cfg" \
    "${resume_args[@]}" \
    --launcher pytorch \
    --cfg-options \
    train_cfg.max_iters="$ITERS" \
    train_cfg.val_interval=2000 \
    default_hooks.checkpoint.interval=2000 \
    "${load_args[@]}" \
    "${COMMON_OPTS[@]}" \
    "${FLAGS[@]}"
}

run_test() {
  local variant="$1"
  local cfg ckpt
  cfg="$(variant_cfg "$variant")"
  ckpt="$(latest_ckpt "$variant")"
  if [[ -z "$ckpt" || ! -f "$ckpt" ]]; then
    echo "Checkpoint not found for ${variant}" >&2
    exit 2
  fi
  if [[ -f "${WORK_ROOT}/${variant}_eval/hierarchy_metrics.json" ]]; then
    echo "Skip completed test: ${variant}"
    return
  fi
  read -r -a FLAGS <<<"$(variant_opts "$variant")"
  python -m torch.distributed.run \
    --nproc_per_node=2 \
    --master_port="$PORT" \
    tools/test.py "$cfg" "$ckpt" \
    --launcher pytorch \
    --work-dir "${WORK_ROOT}/${variant}_eval" \
    --out "${WORK_ROOT}/${variant}_eval/preds.pkl" \
    --cfg-options \
    "${COMMON_OPTS[@]}" \
    "${FLAGS[@]}"
  if [[ -f "${WORK_ROOT}/${variant}_eval/preds.pkl" ]]; then
    python tools/compute_hierarchy_metrics.py \
      --preds "${WORK_ROOT}/${variant}_eval/preds.pkl" \
      --tree "$TREE" \
      --leaf-names "$LEAFS" \
      --out "${WORK_ROOT}/${variant}_eval/hierarchy_metrics.json" || true
  fi
}

run_table() {
  python tools/collect_exp1_tables.py \
    --work-dir "${WORK_ROOT}" \
    --variants ${VARIANTS} \
    --out-md "${WORK_ROOT}/reports/exp1_table.md" \
    --out-csv "${WORK_ROOT}/reports/exp1_table.csv"
}

mkdir -p "$WORK_ROOT"
for variant in ${VARIANTS}; do
  case "$ACTION" in
    train) run_train "$variant" ;;
    test) run_test "$variant" ;;
    all) run_train "$variant"; run_test "$variant"; run_table ;;
    table) run_table; break ;;
    *) echo "Unknown action: $ACTION" >&2; exit 2 ;;
  esac
done
