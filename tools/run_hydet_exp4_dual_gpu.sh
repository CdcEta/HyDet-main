#!/usr/bin/env bash
set -euo pipefail

ROOT="${HYDET_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
ACTION="${1:-train}"
PORT="${PORT:-29537}"
ITERS="${HYDET_ITERS:-10000}"
VARIANTS="${HYDET_VARIANTS:-base align rad sep sib}"
WORK_ROOT="${HYDET_EXP4_WORK_ROOT:-$ROOT/work_dirs/exp4_hrsc_hra_ablation}"
CFG="${HYDET_CFG:-projects/HyDet/configs/hrsc_exp4_hra_ablation.py}"
EXP3_ROOT="${HYDET_EXP3_WORK_ROOT:-$ROOT/work_dirs/exp3_hrsc_supervised}"

cd "$ROOT"
source /root/miniconda3/etc/profile.d/conda.sh
conda activate castdet
export PYTHONPATH="$ROOT:${PYTHONPATH:-}"
export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"

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

prepare_reuse() {
  mkdir -p "${WORK_ROOT}"
  if [[ ! -e "${WORK_ROOT}/align" && -d "${EXP3_ROOT}/hyp" ]]; then
    ln -s "${EXP3_ROOT}/hyp" "${WORK_ROOT}/align"
  fi
  if [[ ! -e "${WORK_ROOT}/align_eval" && -d "${EXP3_ROOT}/hyp_eval" ]]; then
    ln -s "${EXP3_ROOT}/hyp_eval" "${WORK_ROOT}/align_eval"
  fi
}

variant_opts() {
  local variant="$1"
  case "$variant" in
    base)
      echo "work_dir=${WORK_ROOT}/base model.roi_head.bbox_head.loss_profile=exp4_base model.roi_head.bbox_head.module_loss_cfg.hyp_contrast_w=0.0 model.roi_head.bbox_head.module_loss_cfg.hac_radius_w=0.0 model.roi_head.bbox_head.module_loss_cfg.hac_cross_w=0.0 model.roi_head.bbox_head.module_loss_cfg.hac_sibling_w=0.0 model.roi_head.bbox_head.module_loss_cfg.hyp_fg_ce_w=0.0 model.roi_head.bbox_head.module_loss_cfg.fused_fg_ce_w=0.0080"
      ;;
    align)
      echo "work_dir=${WORK_ROOT}/align model.roi_head.bbox_head.loss_profile=exp4_align model.roi_head.bbox_head.module_loss_cfg.hyp_contrast_w=0.0040 model.roi_head.bbox_head.module_loss_cfg.hac_radius_w=0.0 model.roi_head.bbox_head.module_loss_cfg.hac_cross_w=0.0 model.roi_head.bbox_head.module_loss_cfg.hac_sibling_w=0.0 model.roi_head.bbox_head.module_loss_cfg.hyp_fg_ce_w=0.0060 model.roi_head.bbox_head.module_loss_cfg.fused_fg_ce_w=0.0080"
      ;;
    rad)
      echo "work_dir=${WORK_ROOT}/rad model.roi_head.bbox_head.loss_profile=exp4_rad model.roi_head.bbox_head.module_loss_cfg.hyp_contrast_w=0.0040 model.roi_head.bbox_head.module_loss_cfg.hac_radius_w=0.0010 model.roi_head.bbox_head.module_loss_cfg.hac_cross_w=0.0 model.roi_head.bbox_head.module_loss_cfg.hac_sibling_w=0.0 model.roi_head.bbox_head.module_loss_cfg.hyp_fg_ce_w=0.0060 model.roi_head.bbox_head.module_loss_cfg.fused_fg_ce_w=0.0080"
      ;;
    sep)
      echo "work_dir=${WORK_ROOT}/sep model.roi_head.bbox_head.loss_profile=exp4_sep model.roi_head.bbox_head.module_loss_cfg.hyp_contrast_w=0.0040 model.roi_head.bbox_head.module_loss_cfg.hac_radius_w=0.0010 model.roi_head.bbox_head.module_loss_cfg.hac_cross_w=0.0010 model.roi_head.bbox_head.module_loss_cfg.hac_sibling_w=0.0 model.roi_head.bbox_head.module_loss_cfg.hyp_fg_ce_w=0.0060 model.roi_head.bbox_head.module_loss_cfg.fused_fg_ce_w=0.0080"
      ;;
    sib|hra)
      echo "work_dir=${WORK_ROOT}/sib model.roi_head.bbox_head.loss_profile=exp4_sib model.roi_head.bbox_head.module_loss_cfg.hyp_contrast_w=0.0040 model.roi_head.bbox_head.module_loss_cfg.hac_radius_w=0.0010 model.roi_head.bbox_head.module_loss_cfg.hac_cross_w=0.0010 model.roi_head.bbox_head.module_loss_cfg.hac_sibling_w=0.0008 model.roi_head.bbox_head.module_loss_cfg.hyp_fg_ce_w=0.0060 model.roi_head.bbox_head.module_loss_cfg.fused_fg_ce_w=0.0080"
      ;;
    *)
      echo "Unknown variant: $variant" >&2
      exit 2
      ;;
  esac
}

canonical_variant() {
  local variant="$1"
  if [[ "$variant" == "hra" ]]; then
    echo "sib"
  else
    echo "$variant"
  fi
}

ckpt_for() {
  local variant
  variant="$(canonical_variant "$1")"
  local work_dir="${WORK_ROOT}/${variant}"
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
  if [[ "$variant" == "align" && -e "${WORK_ROOT}/align" ]]; then
    echo "Reuse existing align result from ${WORK_ROOT}/align"
    return
  fi
  local canonical
  canonical="$(canonical_variant "$variant")"
  local work_dir="${WORK_ROOT}/${canonical}"
  if [[ -f "${work_dir}/iter_${ITERS}.pth" ]]; then
    echo "Skip completed train: ${canonical}"
    return
  fi
  read -r -a FLAGS <<<"$(variant_opts "$variant")"
  local resume_args=()
  local load_args=()
  if [[ -f "${work_dir}/latest.pth" || -f "${work_dir}/last_checkpoint" ]]; then
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
    tools/train.py "$CFG" \
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
  local canonical
  canonical="$(canonical_variant "$variant")"
  if [[ "$canonical" == "align" && -f "${WORK_ROOT}/align_eval/hierarchy_metrics.json" ]]; then
    echo "Reuse existing align_eval metrics from ${WORK_ROOT}/align_eval"
    return
  fi
  if [[ -f "${WORK_ROOT}/${canonical}_eval/hierarchy_metrics.json" ]]; then
    echo "Skip completed test: ${canonical}"
    return
  fi
  local ckpt
  ckpt="$(ckpt_for "$variant")"
  if [[ -z "$ckpt" || ! -f "$ckpt" ]]; then
    echo "Checkpoint not found for ${variant}" >&2
    exit 2
  fi
  read -r -a FLAGS <<<"$(variant_opts "$variant")"
  python -m torch.distributed.run \
    --nproc_per_node=2 \
    --master_port="$PORT" \
    tools/test.py "$CFG" "$ckpt" \
    --launcher pytorch \
    --work-dir "${WORK_ROOT}/${canonical}_eval" \
    --out "${WORK_ROOT}/${canonical}_eval/preds.pkl" \
    --cfg-options \
    "${COMMON_OPTS[@]}" \
    "${FLAGS[@]}"

  if [[ -f "${WORK_ROOT}/${canonical}_eval/preds.pkl" ]]; then
    python tools/compute_hierarchy_metrics.py \
      --preds "${WORK_ROOT}/${canonical}_eval/preds.pkl" \
      --tree "projects/HyDet/resources/hrsc_hier/tree_validated.json" \
      --leaf-names "projects/HyDet/resources/hrsc_hier/class_names_leaf.txt" \
      --out "${WORK_ROOT}/${canonical}_eval/hierarchy_metrics.json" || true
  fi
}

run_table() {
  python tools/collect_exp4_tables.py \
    --work-dir "${WORK_ROOT}" \
    --variants ${VARIANTS} \
    --out-md "${WORK_ROOT}/reports/exp4_table.md" \
    --out-csv "${WORK_ROOT}/reports/exp4_table.csv"
}

prepare_reuse
mkdir -p "${WORK_ROOT}"
for variant in ${VARIANTS}; do
  case "$ACTION" in
    train)
      run_train "$variant"
      ;;
    test)
      run_test "$variant"
      ;;
    all)
      run_train "$variant"
      run_test "$variant"
      run_table
      ;;
    table)
      run_table
      break
      ;;
    *)
      echo "Unknown action: $ACTION (use train, test, all, table)" >&2
      exit 2
      ;;
  esac
done
