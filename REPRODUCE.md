# Reproduction Guide

This project follows four paper experiment groups.

1. Main results: `run_hydet_exp1_dual_gpu.sh`
2. Base/novel open-vocabulary evaluation: `run_hydet_exp2_dual_gpu.sh`
3. Main module ablation: `run_hydet_exp3_dual_gpu.sh`
4. HRA loss ablation: `run_hydet_exp4_dual_gpu.sh`

Recommended order:

```bash
bash tools/run_hydet_exp3_dual_gpu.sh all
bash tools/run_hydet_exp4_dual_gpu.sh all
bash tools/run_hydet_exp2_dual_gpu.sh hrsc all
bash tools/run_hydet_exp1_dual_gpu.sh hrsc all
```

Use the corresponding `collect_exp*_tables.py` script after each experiment to export Markdown and CSV tables.
