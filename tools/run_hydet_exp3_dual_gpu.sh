#!/usr/bin/env bash
set -euo pipefail

ROOT="${HYDET_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
ACTION="${1:-train}"
PORT="${PORT:-29527}"
ITERS="${HYDET_ITERS:-10000}"
VARIANTS="${HYDET_VARIANTS:-base tax hyp hra cone hydet}"
WORK_ROOT="${HYDET_EXP3_WORK_ROOT:-$ROOT/work_dirs/exp3_hrsc_supervised}"

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

variant_cfg() {
  local variant="$1"
  if [[ "$variant" == "base" ]]; then
    echo "projects/HyDet/configs/hrsc_exp3_castdet_supervised.py"
    return
  fi
  echo "projects/HyDet/configs/hrsc_exp3_hydet_supervised.py"
}

variant_opts() {
  local variant="$1"
  case "$variant" in
    base)
      echo "work_dir=${WORK_ROOT}/base"
      ;;
    tax)
      echo "work_dir=${WORK_ROOT}/tax model.roi_head.bbox_head.loss_profile=tax model.roi_head.bbox_head.use_hyper_branch=True model.roi_head.bbox_head.use_logit_fusion=False model.roi_head.bbox_head.plugin_cfg.use_hyp_contrast=False model.roi_head.bbox_head.plugin_cfg.use_tp_projection=True model.roi_head.bbox_head.fc_cls.use_hyper_branch=True model.roi_head.bbox_head.fc_cls.use_logit_fusion=False model.roi_head.bbox_head.fc_cls.plugin_cfg.use_hyp_contrast=False model.roi_head.bbox_head.fc_cls.plugin_cfg.use_tp_projection=True model.roi_head.bbox_head.module_loss_cfg.tree_cone_w=0.0 model.roi_head.bbox_head.module_loss_cfg.tree_radial_w=0.0010 model.roi_head.bbox_head.module_loss_cfg.tree_text_image_w=0.0015 model.roi_head.bbox_head.module_loss_cfg.hyp_contrast_w=0.0 model.roi_head.bbox_head.module_loss_cfg.hac_radius_w=0.0 model.roi_head.bbox_head.module_loss_cfg.hac_cross_w=0.0 model.roi_head.bbox_head.module_loss_cfg.hac_sibling_w=0.0 model.roi_head.bbox_head.module_loss_cfg.tp_projection_w=0.0 model.roi_head.bbox_head.module_loss_cfg.joint_w=0.0 model.roi_head.bbox_head.module_loss_cfg.hyp_fg_ce_w=0.0 model.roi_head.bbox_head.module_loss_cfg.fused_fg_ce_w=0.0"
      ;;
    hyp)
      echo "work_dir=${WORK_ROOT}/hyp model.roi_head.bbox_head.loss_profile=hyp model.roi_head.bbox_head.use_hyper_branch=True model.roi_head.bbox_head.use_logit_fusion=True model.roi_head.bbox_head.plugin_cfg.use_hyp_contrast=True model.roi_head.bbox_head.plugin_cfg.use_tp_projection=False model.roi_head.bbox_head.fc_cls.use_hyper_branch=True model.roi_head.bbox_head.fc_cls.use_logit_fusion=True model.roi_head.bbox_head.fc_cls.plugin_cfg.use_hyp_contrast=True model.roi_head.bbox_head.fc_cls.plugin_cfg.use_tp_projection=False model.roi_head.bbox_head.module_loss_cfg.tree_cone_w=0.0 model.roi_head.bbox_head.module_loss_cfg.tree_radial_w=0.0 model.roi_head.bbox_head.module_loss_cfg.tree_text_image_w=0.0 model.roi_head.bbox_head.module_loss_cfg.hyp_contrast_w=0.0040 model.roi_head.bbox_head.module_loss_cfg.hac_radius_w=0.0 model.roi_head.bbox_head.module_loss_cfg.hac_cross_w=0.0 model.roi_head.bbox_head.module_loss_cfg.hac_sibling_w=0.0 model.roi_head.bbox_head.module_loss_cfg.tp_projection_w=0.0 model.roi_head.bbox_head.module_loss_cfg.joint_w=0.0 model.roi_head.bbox_head.module_loss_cfg.hyp_fg_ce_w=0.0060 model.roi_head.bbox_head.module_loss_cfg.fused_fg_ce_w=0.0080"
      ;;
    hra)
      echo "work_dir=${WORK_ROOT}/hra model.roi_head.bbox_head.loss_profile=hra model.roi_head.bbox_head.use_hyper_branch=True model.roi_head.bbox_head.use_logit_fusion=True model.roi_head.bbox_head.plugin_cfg.use_hyp_contrast=True model.roi_head.bbox_head.plugin_cfg.use_tp_projection=False model.roi_head.bbox_head.fc_cls.use_hyper_branch=True model.roi_head.bbox_head.fc_cls.use_logit_fusion=True model.roi_head.bbox_head.fc_cls.plugin_cfg.use_hyp_contrast=True model.roi_head.bbox_head.fc_cls.plugin_cfg.use_tp_projection=False model.roi_head.bbox_head.module_loss_cfg.tree_cone_w=0.0 model.roi_head.bbox_head.module_loss_cfg.tree_radial_w=0.0 model.roi_head.bbox_head.module_loss_cfg.tree_text_image_w=0.0 model.roi_head.bbox_head.module_loss_cfg.hyp_contrast_w=0.0030 model.roi_head.bbox_head.module_loss_cfg.hac_radius_w=0.0010 model.roi_head.bbox_head.module_loss_cfg.hac_cross_w=0.0010 model.roi_head.bbox_head.module_loss_cfg.hac_sibling_w=0.0008 model.roi_head.bbox_head.module_loss_cfg.tp_projection_w=0.0 model.roi_head.bbox_head.module_loss_cfg.joint_w=0.0 model.roi_head.bbox_head.module_loss_cfg.hyp_fg_ce_w=0.0060 model.roi_head.bbox_head.module_loss_cfg.fused_fg_ce_w=0.0080"
      ;;
    cone)
      echo "work_dir=${WORK_ROOT}/cone model.roi_head.bbox_head.loss_profile=cone model.roi_head.bbox_head.use_hyper_branch=True model.roi_head.bbox_head.use_logit_fusion=True model.roi_head.bbox_head.plugin_cfg.use_hyp_contrast=False model.roi_head.bbox_head.plugin_cfg.use_tp_projection=True model.roi_head.bbox_head.fc_cls.use_hyper_branch=True model.roi_head.bbox_head.fc_cls.use_logit_fusion=True model.roi_head.bbox_head.fc_cls.plugin_cfg.use_hyp_contrast=False model.roi_head.bbox_head.fc_cls.plugin_cfg.use_tp_projection=True model.roi_head.bbox_head.module_loss_cfg.tree_cone_w=0.0015 model.roi_head.bbox_head.module_loss_cfg.tree_radial_w=0.0 model.roi_head.bbox_head.module_loss_cfg.tree_text_image_w=0.0010 model.roi_head.bbox_head.module_loss_cfg.hyp_contrast_w=0.0 model.roi_head.bbox_head.module_loss_cfg.hac_radius_w=0.0 model.roi_head.bbox_head.module_loss_cfg.hac_cross_w=0.0 model.roi_head.bbox_head.module_loss_cfg.hac_sibling_w=0.0 model.roi_head.bbox_head.module_loss_cfg.tp_projection_w=0.0002 model.roi_head.bbox_head.module_loss_cfg.joint_w=0.0 model.roi_head.bbox_head.module_loss_cfg.hyp_fg_ce_w=0.0 model.roi_head.bbox_head.module_loss_cfg.fused_fg_ce_w=0.0060"
      ;;
    hydet)
      echo "work_dir=${WORK_ROOT}/hydet model.roi_head.bbox_head.loss_profile=hydet model.roi_head.bbox_head.use_hyper_branch=True model.roi_head.bbox_head.use_logit_fusion=True model.roi_head.bbox_head.plugin_cfg.use_hyp_contrast=True model.roi_head.bbox_head.plugin_cfg.use_tp_projection=True model.roi_head.bbox_head.fc_cls.use_hyper_branch=True model.roi_head.bbox_head.fc_cls.use_logit_fusion=True model.roi_head.bbox_head.fc_cls.plugin_cfg.use_hyp_contrast=True model.roi_head.bbox_head.fc_cls.plugin_cfg.use_tp_projection=True model.roi_head.bbox_head.module_loss_cfg.tree_cone_w=0.0015 model.roi_head.bbox_head.module_loss_cfg.tree_radial_w=0.0010 model.roi_head.bbox_head.module_loss_cfg.tree_text_image_w=0.0015 model.roi_head.bbox_head.module_loss_cfg.hyp_contrast_w=0.0040 model.roi_head.bbox_head.module_loss_cfg.hac_radius_w=0.0010 model.roi_head.bbox_head.module_loss_cfg.hac_cross_w=0.0010 model.roi_head.bbox_head.module_loss_cfg.hac_sibling_w=0.0008 model.roi_head.bbox_head.module_loss_cfg.tp_projection_w=0.0002 model.roi_head.bbox_head.module_loss_cfg.joint_w=0.0001 model.roi_head.bbox_head.module_loss_cfg.hyp_fg_ce_w=0.0060 model.roi_head.bbox_head.module_loss_cfg.fused_fg_ce_w=0.0080"
      ;;
    *)
      echo "Unknown variant: $variant" >&2
      exit 2
      ;;
  esac
}

ckpt_for() {
  local variant="$1"
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
  local cfg
  cfg="$(variant_cfg "$variant")"
  read -r -a FLAGS <<<"$(variant_opts "$variant")"
  python -m torch.distributed.run \
    --nproc_per_node=2 \
    --master_port="$PORT" \
    tools/train.py "$cfg" \
    --launcher pytorch \
    --cfg-options \
    train_cfg.max_iters="$ITERS" \
    train_cfg.val_interval=2000 \
    default_hooks.checkpoint.interval=2000 \
    "${COMMON_OPTS[@]}" \
    "${FLAGS[@]}"
}

run_test() {
  local variant="$1"
  local cfg
  cfg="$(variant_cfg "$variant")"
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
      --tree "projects/HyDet/resources/hrsc_hier/tree_validated.json" \
      --leaf-names "projects/HyDet/resources/hrsc_hier/class_names_leaf.txt" \
      --out "${WORK_ROOT}/${variant}_eval/hierarchy_metrics.json" || true
  fi
}

run_table() {
  python tools/collect_exp3_tables.py \
    --work-dir "${WORK_ROOT}" \
    --variants ${VARIANTS} \
    --out-md "${WORK_ROOT}/reports/exp3_table.md" \
    --out-csv "${WORK_ROOT}/reports/exp3_table.csv"
}

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
