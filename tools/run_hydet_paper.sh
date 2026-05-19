#!/usr/bin/env bash
set -euo pipefail

ROOT="${HYDET_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
EXPERIMENT="${1:-exp1}"
DATASET="${2:-hrsc}"
ACTION="${3:-all}"
PORT="${PORT:-${HYDET_PORT:-29557}}"
ITERS="${HYDET_ITERS:-10000}"
GPUS="${HYDET_GPUS:-2}"

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

run_launcher() {
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

common_cfg_opts() {
  local opts=(
    "randomness.seed=${HYDET_SEED:-3407}"
    "randomness.deterministic=${HYDET_DETERMINISTIC:-True}"
  )
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
  printf '%s\n' "${opts[@]}"
}

setup_context() {
  case "$EXPERIMENT" in
    exp1)
      VARIANTS="${HYDET_VARIANTS:-base hydet}"
      TABLE_SCRIPT="tools/collect_exp1_tables.py"
      TABLE_NAME="exp1_table"
      case "$DATASET" in
        hrsc)
          BASE_CFG="projects/HyDet/configs/paper/hrsc_main_baseline.py"
          HYDET_CFG="projects/HyDet/configs/paper/hrsc_main_hydet.py"
          WORK_ROOT="${HYDET_WORK_ROOT:-$ROOT/work_dirs/paper/hrsc/main}"
          TREE="projects/HyDet/resources/hrsc_hier/tree_validated.json"
          LEAFS="projects/HyDet/resources/hrsc_hier/class_names_leaf.txt"
          ;;
        fair1m)
          BASE_CFG="projects/HyDet/configs/paper/fair1m_main_baseline.py"
          HYDET_CFG="projects/HyDet/configs/paper/fair1m_main_hydet.py"
          WORK_ROOT="${HYDET_WORK_ROOT:-$ROOT/work_dirs/paper/fair1m/main}"
          TREE="projects/HyDet/resources/fair1m_hier/tree_validated.json"
          LEAFS="projects/HyDet/resources/fair1m_hier/class_names_leaf.txt"
          ;;
        *)
          echo "Unknown dataset for exp1: $DATASET" >&2
          exit 2
          ;;
      esac
      ;;
    exp2)
      VARIANTS="${HYDET_VARIANTS:-base hra cone hydet}"
      TABLE_SCRIPT="tools/collect_exp2_tables.py"
      TABLE_NAME="exp2_table"
      case "$DATASET" in
        hrsc)
          BASE_CFG="projects/HyDet/configs/paper/hrsc_open_vocab_baseline.py"
          HYDET_CFG="projects/HyDet/configs/paper/hrsc_open_vocab_hydet.py"
          WORK_ROOT="${HYDET_WORK_ROOT:-$ROOT/work_dirs/paper/hrsc/open_vocab}"
          SPLIT_ALL="ImageSets/Main/hrsc_test_all.txt"
          SPLIT_BASE="ImageSets/Main/hrsc_test_base_train.txt"
          SPLIT_NOVEL="ImageSets/Main/hrsc_test_novel.txt"
          ;;
        fair1m)
          BASE_CFG="projects/HyDet/configs/paper/fair1m_open_vocab_baseline.py"
          HYDET_CFG="projects/HyDet/configs/paper/fair1m_open_vocab_hydet.py"
          WORK_ROOT="${HYDET_WORK_ROOT:-$ROOT/work_dirs/paper/fair1m/open_vocab}"
          SPLIT_ALL="ImageSets/Main/fair1m_test_all.txt"
          SPLIT_BASE="ImageSets/Main/fair1m_test_base_train.txt"
          SPLIT_NOVEL="ImageSets/Main/fair1m_test_novel.txt"
          ;;
        *)
          echo "Unknown dataset for exp2: $DATASET" >&2
          exit 2
          ;;
      esac
      ;;
    exp3)
      VARIANTS="${HYDET_VARIANTS:-base tax hyp hra cone hydet}"
      TABLE_SCRIPT="tools/collect_exp3_tables.py"
      TABLE_NAME="exp3_table"
      case "$DATASET" in
        hrsc)
          BASE_CFG="projects/HyDet/configs/paper/hrsc_module_ablation_baseline.py"
          HYDET_CFG="projects/HyDet/configs/paper/hrsc_module_ablation_hydet.py"
          WORK_ROOT="${HYDET_WORK_ROOT:-$ROOT/work_dirs/paper/hrsc/module_ablation}"
          TREE="projects/HyDet/resources/hrsc_hier/tree_validated.json"
          LEAFS="projects/HyDet/resources/hrsc_hier/class_names_leaf.txt"
          ;;
        fair1m)
          BASE_CFG="projects/HyDet/configs/paper/fair1m_module_ablation_baseline.py"
          HYDET_CFG="projects/HyDet/configs/paper/fair1m_module_ablation_hydet.py"
          WORK_ROOT="${HYDET_WORK_ROOT:-$ROOT/work_dirs/paper/fair1m/module_ablation}"
          TREE="projects/HyDet/resources/fair1m_hier/tree_validated.json"
          LEAFS="projects/HyDet/resources/fair1m_hier/class_names_leaf.txt"
          ;;
        *)
          echo "Unknown dataset for exp3: $DATASET" >&2
          exit 2
          ;;
      esac
      ;;
    exp4)
      VARIANTS="${HYDET_VARIANTS:-base align rad sep sib}"
      TABLE_SCRIPT="tools/collect_exp4_tables.py"
      TABLE_NAME="exp4_table"
      if [[ "$DATASET" != "hrsc" ]]; then
        echo "exp4 currently supports hrsc only" >&2
        exit 2
      fi
      CFG="projects/HyDet/configs/paper/hrsc_hra_ablation.py"
      WORK_ROOT="${HYDET_WORK_ROOT:-$ROOT/work_dirs/paper/hrsc/hra_ablation}"
      EXP3_ROOT="${HYDET_EXP3_WORK_ROOT:-$ROOT/work_dirs/paper/hrsc/module_ablation}"
      TREE="projects/HyDet/resources/hrsc_hier/tree_validated.json"
      LEAFS="projects/HyDet/resources/hrsc_hier/class_names_leaf.txt"
      ;;
    *)
      echo "Unknown experiment: $EXPERIMENT (use exp1, exp2, exp3, or exp4)" >&2
      exit 2
      ;;
  esac
}

variant_cfg() {
  local variant="$1"
  if [[ "$EXPERIMENT" == "exp4" ]]; then
    echo "$CFG"
  elif [[ "$variant" == "base" ]]; then
    echo "$BASE_CFG"
  else
    echo "$HYDET_CFG"
  fi
}

variant_opts() {
  local variant="$1"
  case "${EXPERIMENT}:${variant}" in
    exp1:base)
      echo "work_dir=${WORK_ROOT}/base"
      ;;
    exp1:hydet)
      echo "work_dir=${WORK_ROOT}/hydet"
      ;;
    exp2:base)
      echo "work_dir=${WORK_ROOT}/base"
      ;;
    exp2:hra)
      echo "work_dir=${WORK_ROOT}/hra model.roi_head.bbox_head.loss_profile=hra model.roi_head.bbox_head.use_hyper_branch=True model.roi_head.bbox_head.use_logit_fusion=True model.roi_head.bbox_head.plugin_cfg.use_hyp_contrast=True model.roi_head.bbox_head.plugin_cfg.use_tp_projection=False model.roi_head.bbox_head.fc_cls.use_hyper_branch=True model.roi_head.bbox_head.fc_cls.use_logit_fusion=True model.roi_head.bbox_head.fc_cls.plugin_cfg.use_hyp_contrast=True model.roi_head.bbox_head.fc_cls.plugin_cfg.use_tp_projection=False model.roi_head.bbox_head.module_loss_cfg.tree_cone_w=0.0 model.roi_head.bbox_head.module_loss_cfg.tree_radial_w=0.0 model.roi_head.bbox_head.module_loss_cfg.tree_text_image_w=0.0 model.roi_head.bbox_head.module_loss_cfg.hyp_contrast_w=0.0030 model.roi_head.bbox_head.module_loss_cfg.hac_radius_w=0.0010 model.roi_head.bbox_head.module_loss_cfg.hac_cross_w=0.0010 model.roi_head.bbox_head.module_loss_cfg.hac_sibling_w=0.0008 model.roi_head.bbox_head.module_loss_cfg.tp_projection_w=0.0 model.roi_head.bbox_head.module_loss_cfg.joint_w=0.0 model.roi_head.bbox_head.module_loss_cfg.hyp_fg_ce_w=0.0060 model.roi_head.bbox_head.module_loss_cfg.fused_fg_ce_w=0.0080"
      ;;
    exp2:cone)
      echo "work_dir=${WORK_ROOT}/cone model.roi_head.bbox_head.loss_profile=cone model.roi_head.bbox_head.use_hyper_branch=True model.roi_head.bbox_head.use_logit_fusion=True model.roi_head.bbox_head.plugin_cfg.use_hyp_contrast=False model.roi_head.bbox_head.plugin_cfg.use_tp_projection=True model.roi_head.bbox_head.fc_cls.use_hyper_branch=True model.roi_head.bbox_head.fc_cls.use_logit_fusion=True model.roi_head.bbox_head.fc_cls.plugin_cfg.use_hyp_contrast=False model.roi_head.bbox_head.fc_cls.plugin_cfg.use_tp_projection=True model.roi_head.bbox_head.module_loss_cfg.tree_cone_w=0.0015 model.roi_head.bbox_head.module_loss_cfg.tree_radial_w=0.0 model.roi_head.bbox_head.module_loss_cfg.tree_text_image_w=0.0010 model.roi_head.bbox_head.module_loss_cfg.hyp_contrast_w=0.0 model.roi_head.bbox_head.module_loss_cfg.hac_radius_w=0.0 model.roi_head.bbox_head.module_loss_cfg.hac_cross_w=0.0 model.roi_head.bbox_head.module_loss_cfg.hac_sibling_w=0.0 model.roi_head.bbox_head.module_loss_cfg.tp_projection_w=0.0002 model.roi_head.bbox_head.module_loss_cfg.joint_w=0.0 model.roi_head.bbox_head.module_loss_cfg.hyp_fg_ce_w=0.0 model.roi_head.bbox_head.module_loss_cfg.fused_fg_ce_w=0.0060"
      ;;
    exp2:hydet)
      echo "work_dir=${WORK_ROOT}/hydet model.roi_head.bbox_head.loss_profile=hydet model.roi_head.bbox_head.use_hyper_branch=True model.roi_head.bbox_head.use_logit_fusion=True model.roi_head.bbox_head.plugin_cfg.use_hyp_contrast=True model.roi_head.bbox_head.plugin_cfg.use_tp_projection=True model.roi_head.bbox_head.fc_cls.use_hyper_branch=True model.roi_head.bbox_head.fc_cls.use_logit_fusion=True model.roi_head.bbox_head.fc_cls.plugin_cfg.use_hyp_contrast=True model.roi_head.bbox_head.fc_cls.plugin_cfg.use_tp_projection=True model.roi_head.bbox_head.module_loss_cfg.tree_cone_w=0.0015 model.roi_head.bbox_head.module_loss_cfg.tree_radial_w=0.0010 model.roi_head.bbox_head.module_loss_cfg.tree_text_image_w=0.0015 model.roi_head.bbox_head.module_loss_cfg.hyp_contrast_w=0.0040 model.roi_head.bbox_head.module_loss_cfg.hac_radius_w=0.0010 model.roi_head.bbox_head.module_loss_cfg.hac_cross_w=0.0010 model.roi_head.bbox_head.module_loss_cfg.hac_sibling_w=0.0008 model.roi_head.bbox_head.module_loss_cfg.tp_projection_w=0.0002 model.roi_head.bbox_head.module_loss_cfg.joint_w=0.0001 model.roi_head.bbox_head.module_loss_cfg.hyp_fg_ce_w=0.0060 model.roi_head.bbox_head.module_loss_cfg.fused_fg_ce_w=0.0080"
      ;;
    exp3:base)
      echo "work_dir=${WORK_ROOT}/base"
      ;;
    exp3:tax)
      echo "work_dir=${WORK_ROOT}/tax model.roi_head.bbox_head.loss_profile=tax model.roi_head.bbox_head.use_hyper_branch=True model.roi_head.bbox_head.use_logit_fusion=False model.roi_head.bbox_head.plugin_cfg.use_hyp_contrast=False model.roi_head.bbox_head.plugin_cfg.use_tp_projection=True model.roi_head.bbox_head.fc_cls.use_hyper_branch=True model.roi_head.bbox_head.fc_cls.use_logit_fusion=False model.roi_head.bbox_head.fc_cls.plugin_cfg.use_hyp_contrast=False model.roi_head.bbox_head.fc_cls.plugin_cfg.use_tp_projection=True model.roi_head.bbox_head.module_loss_cfg.tree_cone_w=0.0 model.roi_head.bbox_head.module_loss_cfg.tree_radial_w=0.0010 model.roi_head.bbox_head.module_loss_cfg.tree_text_image_w=0.0015 model.roi_head.bbox_head.module_loss_cfg.hyp_contrast_w=0.0 model.roi_head.bbox_head.module_loss_cfg.hac_radius_w=0.0 model.roi_head.bbox_head.module_loss_cfg.hac_cross_w=0.0 model.roi_head.bbox_head.module_loss_cfg.hac_sibling_w=0.0 model.roi_head.bbox_head.module_loss_cfg.tp_projection_w=0.0 model.roi_head.bbox_head.module_loss_cfg.joint_w=0.0 model.roi_head.bbox_head.module_loss_cfg.hyp_fg_ce_w=0.0 model.roi_head.bbox_head.module_loss_cfg.fused_fg_ce_w=0.0"
      ;;
    exp3:hyp)
      echo "work_dir=${WORK_ROOT}/hyp model.roi_head.bbox_head.loss_profile=hyp model.roi_head.bbox_head.use_hyper_branch=True model.roi_head.bbox_head.use_logit_fusion=True model.roi_head.bbox_head.plugin_cfg.use_hyp_contrast=True model.roi_head.bbox_head.plugin_cfg.use_tp_projection=False model.roi_head.bbox_head.fc_cls.use_hyper_branch=True model.roi_head.bbox_head.fc_cls.use_logit_fusion=True model.roi_head.bbox_head.fc_cls.plugin_cfg.use_hyp_contrast=True model.roi_head.bbox_head.fc_cls.plugin_cfg.use_tp_projection=False model.roi_head.bbox_head.module_loss_cfg.tree_cone_w=0.0 model.roi_head.bbox_head.module_loss_cfg.tree_radial_w=0.0 model.roi_head.bbox_head.module_loss_cfg.tree_text_image_w=0.0 model.roi_head.bbox_head.module_loss_cfg.hyp_contrast_w=0.0040 model.roi_head.bbox_head.module_loss_cfg.hac_radius_w=0.0 model.roi_head.bbox_head.module_loss_cfg.hac_cross_w=0.0 model.roi_head.bbox_head.module_loss_cfg.hac_sibling_w=0.0 model.roi_head.bbox_head.module_loss_cfg.tp_projection_w=0.0 model.roi_head.bbox_head.module_loss_cfg.joint_w=0.0 model.roi_head.bbox_head.module_loss_cfg.hyp_fg_ce_w=0.0060 model.roi_head.bbox_head.module_loss_cfg.fused_fg_ce_w=0.0080"
      ;;
    exp3:hra)
      echo "work_dir=${WORK_ROOT}/hra model.roi_head.bbox_head.loss_profile=hra model.roi_head.bbox_head.use_hyper_branch=True model.roi_head.bbox_head.use_logit_fusion=True model.roi_head.bbox_head.plugin_cfg.use_hyp_contrast=True model.roi_head.bbox_head.plugin_cfg.use_tp_projection=False model.roi_head.bbox_head.fc_cls.use_hyper_branch=True model.roi_head.bbox_head.fc_cls.use_logit_fusion=True model.roi_head.bbox_head.fc_cls.plugin_cfg.use_hyp_contrast=True model.roi_head.bbox_head.fc_cls.plugin_cfg.use_tp_projection=False model.roi_head.bbox_head.module_loss_cfg.tree_cone_w=0.0 model.roi_head.bbox_head.module_loss_cfg.tree_radial_w=0.0 model.roi_head.bbox_head.module_loss_cfg.tree_text_image_w=0.0 model.roi_head.bbox_head.module_loss_cfg.hyp_contrast_w=0.0030 model.roi_head.bbox_head.module_loss_cfg.hac_radius_w=0.0010 model.roi_head.bbox_head.module_loss_cfg.hac_cross_w=0.0010 model.roi_head.bbox_head.module_loss_cfg.hac_sibling_w=0.0008 model.roi_head.bbox_head.module_loss_cfg.tp_projection_w=0.0 model.roi_head.bbox_head.module_loss_cfg.joint_w=0.0 model.roi_head.bbox_head.module_loss_cfg.hyp_fg_ce_w=0.0060 model.roi_head.bbox_head.module_loss_cfg.fused_fg_ce_w=0.0080"
      ;;
    exp3:cone)
      echo "work_dir=${WORK_ROOT}/cone model.roi_head.bbox_head.loss_profile=cone model.roi_head.bbox_head.use_hyper_branch=True model.roi_head.bbox_head.use_logit_fusion=True model.roi_head.bbox_head.plugin_cfg.use_hyp_contrast=False model.roi_head.bbox_head.plugin_cfg.use_tp_projection=True model.roi_head.bbox_head.fc_cls.use_hyper_branch=True model.roi_head.bbox_head.fc_cls.use_logit_fusion=True model.roi_head.bbox_head.fc_cls.plugin_cfg.use_hyp_contrast=False model.roi_head.bbox_head.fc_cls.plugin_cfg.use_tp_projection=True model.roi_head.bbox_head.module_loss_cfg.tree_cone_w=0.0015 model.roi_head.bbox_head.module_loss_cfg.tree_radial_w=0.0 model.roi_head.bbox_head.module_loss_cfg.tree_text_image_w=0.0010 model.roi_head.bbox_head.module_loss_cfg.hyp_contrast_w=0.0 model.roi_head.bbox_head.module_loss_cfg.hac_radius_w=0.0 model.roi_head.bbox_head.module_loss_cfg.hac_cross_w=0.0 model.roi_head.bbox_head.module_loss_cfg.hac_sibling_w=0.0 model.roi_head.bbox_head.module_loss_cfg.tp_projection_w=0.0002 model.roi_head.bbox_head.module_loss_cfg.joint_w=0.0 model.roi_head.bbox_head.module_loss_cfg.hyp_fg_ce_w=0.0 model.roi_head.bbox_head.module_loss_cfg.fused_fg_ce_w=0.0060"
      ;;
    exp3:hydet)
      echo "work_dir=${WORK_ROOT}/hydet model.roi_head.bbox_head.loss_profile=hydet model.roi_head.bbox_head.use_hyper_branch=True model.roi_head.bbox_head.use_logit_fusion=True model.roi_head.bbox_head.plugin_cfg.use_hyp_contrast=True model.roi_head.bbox_head.plugin_cfg.use_tp_projection=True model.roi_head.bbox_head.fc_cls.use_hyper_branch=True model.roi_head.bbox_head.fc_cls.use_logit_fusion=True model.roi_head.bbox_head.fc_cls.plugin_cfg.use_hyp_contrast=True model.roi_head.bbox_head.fc_cls.plugin_cfg.use_tp_projection=True model.roi_head.bbox_head.module_loss_cfg.tree_cone_w=0.0015 model.roi_head.bbox_head.module_loss_cfg.tree_radial_w=0.0010 model.roi_head.bbox_head.module_loss_cfg.tree_text_image_w=0.0015 model.roi_head.bbox_head.module_loss_cfg.hyp_contrast_w=0.0040 model.roi_head.bbox_head.module_loss_cfg.hac_radius_w=0.0010 model.roi_head.bbox_head.module_loss_cfg.hac_cross_w=0.0010 model.roi_head.bbox_head.module_loss_cfg.hac_sibling_w=0.0008 model.roi_head.bbox_head.module_loss_cfg.tp_projection_w=0.0002 model.roi_head.bbox_head.module_loss_cfg.joint_w=0.0001 model.roi_head.bbox_head.module_loss_cfg.hyp_fg_ce_w=0.0060 model.roi_head.bbox_head.module_loss_cfg.fused_fg_ce_w=0.0080"
      ;;
    exp4:base)
      echo "work_dir=${WORK_ROOT}/base model.roi_head.bbox_head.loss_profile=exp4_base model.roi_head.bbox_head.module_loss_cfg.hyp_contrast_w=0.0 model.roi_head.bbox_head.module_loss_cfg.hac_radius_w=0.0 model.roi_head.bbox_head.module_loss_cfg.hac_cross_w=0.0 model.roi_head.bbox_head.module_loss_cfg.hac_sibling_w=0.0 model.roi_head.bbox_head.module_loss_cfg.hyp_fg_ce_w=0.0 model.roi_head.bbox_head.module_loss_cfg.fused_fg_ce_w=0.0080"
      ;;
    exp4:align)
      echo "work_dir=${WORK_ROOT}/align model.roi_head.bbox_head.loss_profile=exp4_align model.roi_head.bbox_head.module_loss_cfg.hyp_contrast_w=0.0040 model.roi_head.bbox_head.module_loss_cfg.hac_radius_w=0.0 model.roi_head.bbox_head.module_loss_cfg.hac_cross_w=0.0 model.roi_head.bbox_head.module_loss_cfg.hac_sibling_w=0.0 model.roi_head.bbox_head.module_loss_cfg.hyp_fg_ce_w=0.0060 model.roi_head.bbox_head.module_loss_cfg.fused_fg_ce_w=0.0080"
      ;;
    exp4:rad)
      echo "work_dir=${WORK_ROOT}/rad model.roi_head.bbox_head.loss_profile=exp4_rad model.roi_head.bbox_head.module_loss_cfg.hyp_contrast_w=0.0040 model.roi_head.bbox_head.module_loss_cfg.hac_radius_w=0.0010 model.roi_head.bbox_head.module_loss_cfg.hac_cross_w=0.0 model.roi_head.bbox_head.module_loss_cfg.hac_sibling_w=0.0 model.roi_head.bbox_head.module_loss_cfg.hyp_fg_ce_w=0.0060 model.roi_head.bbox_head.module_loss_cfg.fused_fg_ce_w=0.0080"
      ;;
    exp4:sep)
      echo "work_dir=${WORK_ROOT}/sep model.roi_head.bbox_head.loss_profile=exp4_sep model.roi_head.bbox_head.module_loss_cfg.hyp_contrast_w=0.0040 model.roi_head.bbox_head.module_loss_cfg.hac_radius_w=0.0010 model.roi_head.bbox_head.module_loss_cfg.hac_cross_w=0.0010 model.roi_head.bbox_head.module_loss_cfg.hac_sibling_w=0.0 model.roi_head.bbox_head.module_loss_cfg.hyp_fg_ce_w=0.0060 model.roi_head.bbox_head.module_loss_cfg.fused_fg_ce_w=0.0080"
      ;;
    exp4:sib|exp4:hra)
      echo "work_dir=${WORK_ROOT}/sib model.roi_head.bbox_head.loss_profile=exp4_sib model.roi_head.bbox_head.module_loss_cfg.hyp_contrast_w=0.0040 model.roi_head.bbox_head.module_loss_cfg.hac_radius_w=0.0010 model.roi_head.bbox_head.module_loss_cfg.hac_cross_w=0.0010 model.roi_head.bbox_head.module_loss_cfg.hac_sibling_w=0.0008 model.roi_head.bbox_head.module_loss_cfg.hyp_fg_ce_w=0.0060 model.roi_head.bbox_head.module_loss_cfg.fused_fg_ce_w=0.0080"
      ;;
    *)
      echo "Unknown variant for ${EXPERIMENT}: ${variant}" >&2
      exit 2
      ;;
  esac
}

canonical_variant() {
  if [[ "$EXPERIMENT" == "exp4" && "$1" == "hra" ]]; then
    echo "sib"
  else
    echo "$1"
  fi
}

latest_ckpt() {
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

prepare_exp4_reuse() {
  if [[ "$EXPERIMENT" != "exp4" ]]; then
    return
  fi
  mkdir -p "${WORK_ROOT}"
  if [[ ! -e "${WORK_ROOT}/align" && -d "${EXP3_ROOT}/hyp" ]]; then
    ln -s "${EXP3_ROOT}/hyp" "${WORK_ROOT}/align"
  fi
  if [[ ! -e "${WORK_ROOT}/align_eval" && -d "${EXP3_ROOT}/hyp_eval" ]]; then
    ln -s "${EXP3_ROOT}/hyp_eval" "${WORK_ROOT}/align_eval"
  fi
}

run_train() {
  local variant="$1"
  local canonical
  canonical="$(canonical_variant "$variant")"
  local cfg
  cfg="$(variant_cfg "$variant")"
  if [[ "$EXPERIMENT" == "exp4" && "$canonical" == "align" && -e "${WORK_ROOT}/align" ]]; then
    echo "Reuse existing align result from ${WORK_ROOT}/align"
    return
  fi
  local work_dir="${WORK_ROOT}/${canonical}"
  if [[ -f "${work_dir}/iter_${ITERS}.pth" ]]; then
    echo "Skip completed train: ${canonical}"
    return
  fi
  read -r -a flags <<<"$(variant_opts "$variant")"
  local -a cfg_opts=(
    "train_cfg.max_iters=${ITERS}"
    "train_cfg.val_interval=${HYDET_VAL_INTERVAL:-2000}"
    "default_hooks.checkpoint.interval=${HYDET_VAL_INTERVAL:-2000}"
  )
  while IFS= read -r item; do
    [[ -n "$item" ]] && cfg_opts+=("$item")
  done < <(common_cfg_opts)
  cfg_opts+=("${flags[@]}")
  local -a cmd=("tools/train.py" "${cfg}")
  if [[ "${#cfg_opts[@]}" -gt 0 ]]; then
    cmd+=("--cfg-options" "${cfg_opts[@]}")
  fi
  run_launcher "${cmd[@]}"
}

run_test_metrics() {
  local out_dir="$1"
  if [[ -f "${out_dir}/preds.pkl" && -n "${TREE:-}" && -n "${LEAFS:-}" ]]; then
    python tools/compute_hierarchy_metrics.py \
      --preds "${out_dir}/preds.pkl" \
      --tree "${TREE}" \
      --leaf-names "${LEAFS}" \
      --out "${out_dir}/hierarchy_metrics.json" || true
  fi
}

run_test() {
  local variant="$1"
  local canonical
  canonical="$(canonical_variant "$variant")"
  local cfg
  cfg="$(variant_cfg "$variant")"
  local ckpt
  ckpt="$(latest_ckpt "$variant")"
  if [[ -z "${ckpt}" || ! -f "${ckpt}" ]]; then
    echo "Checkpoint not found for ${variant}" >&2
    exit 2
  fi
  local out_dir="${WORK_ROOT}/${canonical}_eval"
  if [[ "$EXPERIMENT" == "exp4" && "$canonical" == "align" && -f "${out_dir}/hierarchy_metrics.json" ]]; then
    echo "Reuse existing align_eval metrics from ${out_dir}"
    return
  fi
  read -r -a flags <<<"$(variant_opts "$variant")"
  local -a cfg_opts=()
  while IFS= read -r item; do
    [[ -n "$item" ]] && cfg_opts+=("$item")
  done < <(common_cfg_opts)
  cfg_opts+=("${flags[@]}")
  local -a cmd=("tools/test.py" "${cfg}" "${ckpt}" "--work-dir" "${out_dir}" "--out" "${out_dir}/preds.pkl")
  if [[ "${#cfg_opts[@]}" -gt 0 ]]; then
    cmd+=("--cfg-options" "${cfg_opts[@]}")
  fi
  run_launcher "${cmd[@]}"
  run_test_metrics "${out_dir}"
}

run_test_split() {
  local variant="$1"
  local split_name="$2"
  local split_ann="$3"
  local cfg
  cfg="$(variant_cfg "$variant")"
  local ckpt
  ckpt="$(latest_ckpt "$variant")"
  if [[ -z "${ckpt}" || ! -f "${ckpt}" ]]; then
    echo "Checkpoint not found for ${variant}" >&2
    exit 2
  fi
  local out_dir="${WORK_ROOT}/${variant}_eval_${split_name}"
  read -r -a flags <<<"$(variant_opts "$variant")"
  local -a cfg_opts=("test_dataloader.dataset.ann_file=${split_ann}")
  while IFS= read -r item; do
    [[ -n "$item" ]] && cfg_opts+=("$item")
  done < <(common_cfg_opts)
  cfg_opts+=("${flags[@]}")
  local -a cmd=("tools/test.py" "${cfg}" "${ckpt}" "--work-dir" "${out_dir}" "--out" "${out_dir}/preds.pkl")
  if [[ "${#cfg_opts[@]}" -gt 0 ]]; then
    cmd+=("--cfg-options" "${cfg_opts[@]}")
  fi
  run_launcher "${cmd[@]}"
}

run_open_vocab() {
  run_test_split "$1" all "${SPLIT_ALL}"
  run_test_split "$1" base "${SPLIT_BASE}"
  run_test_split "$1" novel "${SPLIT_NOVEL}"
}

run_table() {
  mkdir -p "${WORK_ROOT}/reports"
  python "${TABLE_SCRIPT}" \
    --work-dir "${WORK_ROOT}" \
    --variants ${VARIANTS} \
    --out-md "${WORK_ROOT}/reports/${TABLE_NAME}.md" \
    --out-csv "${WORK_ROOT}/reports/${TABLE_NAME}.csv"
}

activate_conda
export PYTHONPATH="${ROOT}:${PYTHONPATH:-}"
setup_context
prepare_exp4_reuse
mkdir -p "${WORK_ROOT}"

for variant in ${VARIANTS}; do
  case "$ACTION" in
    train)
      run_train "${variant}"
      ;;
    test)
      if [[ "$EXPERIMENT" == "exp2" ]]; then
        run_open_vocab "${variant}"
      else
        run_test "${variant}"
      fi
      ;;
    all)
      run_train "${variant}"
      if [[ "$EXPERIMENT" == "exp2" ]]; then
        run_open_vocab "${variant}"
      else
        run_test "${variant}"
      fi
      run_table
      ;;
    table)
      run_table
      break
      ;;
    *)
      echo "Unknown action: $ACTION (use train, test, all, or table)" >&2
      exit 2
      ;;
  esac
done
