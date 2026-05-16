#!/usr/bin/env bash
set -euo pipefail

ROOT="${HYDET_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
DATASET="${1:-hrsc}"
ACTION="${2:-all}"
PORT="${PORT:-29567}"
ITERS="${HYDET_ITERS:-10000}"
VARIANTS="${HYDET_VARIANTS:-base hra cone hydet}"

cd "$ROOT"
source /root/miniconda3/etc/profile.d/conda.sh
conda activate castdet
export PYTHONPATH="$ROOT:${PYTHONPATH:-}"

if [[ "$DATASET" == "hrsc" ]]; then
  BASE_CFG="projects/HyDet/configs/hrsc_exp2_castdet_openvocab.py"
  HYDET_CFG="projects/HyDet/configs/hrsc_exp2_hydet_openvocab.py"
  WORK_ROOT="${HYDET_EXP2_WORK_ROOT:-$ROOT/work_dirs/exp2_hrsc_openvocab}"
  SPLIT_ALL="ImageSets/Main/hrsc_test_all.txt"
  SPLIT_BASE="ImageSets/Main/hrsc_test_base_train.txt"
  SPLIT_NOVEL="ImageSets/Main/hrsc_test_novel.txt"
elif [[ "$DATASET" == "fair1m" ]]; then
  BASE_CFG="projects/HyDet/configs/fair1m_exp2_castdet_openvocab.py"
  HYDET_CFG="projects/HyDet/configs/fair1m_exp2_hydet_openvocab.py"
  WORK_ROOT="${HYDET_EXP2_WORK_ROOT:-$ROOT/work_dirs/exp2_fair1m_openvocab}"
  SPLIT_ALL="ImageSets/Main/fair1m_test_all.txt"
  SPLIT_BASE="ImageSets/Main/fair1m_test_base_train.txt"
  SPLIT_NOVEL="ImageSets/Main/fair1m_test_novel.txt"
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
  local variant="$1"
  case "$variant" in
    base)
      echo "work_dir=${WORK_ROOT}/base"
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

run_test_split() {
  local variant="$1"
  local split_name="$2"
  local split_ann="$3"
  local cfg ckpt
  cfg="$(variant_cfg "$variant")"
  ckpt="$(latest_ckpt "$variant")"
  if [[ -z "$ckpt" || ! -f "$ckpt" ]]; then
    echo "Checkpoint not found for ${variant}" >&2
    exit 2
  fi
  local out_dir="${WORK_ROOT}/${variant}_eval_${split_name}"
  if [[ -f "${out_dir}/preds.pkl" ]]; then
    echo "Skip completed test: ${variant}/${split_name}"
    return
  fi
  read -r -a FLAGS <<<"$(variant_opts "$variant")"
  python -m torch.distributed.run \
    --nproc_per_node=2 \
    --master_port="$PORT" \
    tools/test.py "$cfg" "$ckpt" \
    --launcher pytorch \
    --work-dir "$out_dir" \
    --out "${out_dir}/preds.pkl" \
    --cfg-options \
    "${COMMON_OPTS[@]}" \
    test_dataloader.dataset.ann_file="$split_ann" \
    "${FLAGS[@]}"
}

run_openvocab() {
  run_test_split "$1" all "$SPLIT_ALL"
  run_test_split "$1" base "$SPLIT_BASE"
  run_test_split "$1" novel "$SPLIT_NOVEL"
}

run_table() {
  python tools/collect_exp2_tables.py \
    --work-dir "${WORK_ROOT}" \
    --variants ${VARIANTS} \
    --out-md "${WORK_ROOT}/reports/exp2_table.md" \
    --out-csv "${WORK_ROOT}/reports/exp2_table.csv"
}

mkdir -p "$WORK_ROOT"
for variant in ${VARIANTS}; do
  case "$ACTION" in
    train) run_train "$variant" ;;
    openvocab|test) run_openvocab "$variant" ;;
    all) run_train "$variant"; run_openvocab "$variant"; run_table ;;
    table) run_table; break ;;
    *) echo "Unknown action: $ACTION" >&2; exit 2 ;;
  esac
done
