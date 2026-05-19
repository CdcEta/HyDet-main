# Reproduction Guide

Paper-specific configs are organized under `projects/HyDet/configs/paper/`.

The unified reproduction runner is:

```bash
bash tools/run_hydet_paper.sh <exp> <dataset> <action>
```

Examples:

```bash
bash tools/run_hydet_paper.sh exp1 hrsc all
bash tools/run_hydet_paper.sh exp1 fair1m all
bash tools/run_hydet_paper.sh exp2 hrsc all
bash tools/run_hydet_paper.sh exp2 fair1m all
bash tools/run_hydet_paper.sh exp3 hrsc all
bash tools/run_hydet_paper.sh exp4 hrsc all
```

Experiment groups:

1. `exp1`: main comparison (`base`, `hydet`)
2. `exp2`: open-vocabulary evaluation (`base`, `hra`, `cone`, `hydet`)
3. `exp3`: module ablation (`base`, `tax`, `hyp`, `hra`, `cone`, `hydet`)
4. `exp4`: HRA ablation (`base`, `align`, `rad`, `sep`, `sib`)

Recommended order:

```bash
bash tools/run_hydet_paper.sh exp3 hrsc all
bash tools/run_hydet_paper.sh exp4 hrsc all
bash tools/run_hydet_paper.sh exp2 hrsc all
bash tools/run_hydet_paper.sh exp1 hrsc all
```

Compatibility wrappers are still available:

```bash
bash tools/run_hydet_exp1_dual_gpu.sh hrsc all
bash tools/run_hydet_exp2_dual_gpu.sh fair1m all
bash tools/run_hydet_exp3_dual_gpu.sh all
bash tools/run_hydet_exp4_dual_gpu.sh all
```

Use the corresponding `collect_exp*_tables.py` script to export Markdown and CSV tables from the generated `work_dirs/paper/...` directories.
