# HyDet

HyDet is a hierarchical open-vocabulary detector for remote-sensing images. It builds on the MMRotate training stack and adds taxonomy-aware text priors, Lorentz hyperbolic region-text alignment, HRA losses, and an entailment-cone hierarchy constraint for fine-grained remote-sensing categories.

![](./HyDet.png)

![](./HRA.png)

## Features

- Open-vocabulary oriented object detection for HRSC2016 and FAIR1M.
- LLM-generated category taxonomy support with fixed training-time hierarchy metadata.
- Hyperbolic region-text projection and logit fusion.
- HRA ablations for Align, Radial, Separation, and Sibling losses.
- Experiment runners and table collectors for the paper experiments.

## Repository Layout

```text
configs/                  MMRotate base dataset/runtime configs
mmrotate/                 bundled MMRotate framework code
projects/HyDet/hydet/     HyDet model heads, losses, hooks, queues, and utilities
projects/HyDet/configs/   paper experiment configs
projects/HyDet/resources/ taxonomy and split metadata
tools/                    training, testing, metrics, and experiment launch scripts
requirements/             dependency lists
```

## Installation

```bash
conda create -n hydet python=3.8 -y
conda activate hydet
pip install -U openmim
mim install mmengine mmcv mmdet
pip install -r requirements.txt
pip install -v -e .
```

Set the repository root on `PYTHONPATH` when running scripts:

```bash
export PYTHONPATH=$PWD:${PYTHONPATH:-}
```

## Data Preparation

Organize datasets using MMRotate-style annotation files. The expected symbolic layout is:

```text
data/
  HRSC2016/
    ImageSets/Main/*.txt
    FullDataSet/Annotations/*.xml
    FullDataSet/AllImages/*
  FAIR1M/
    ImageSets/Main/*.txt
    images/*
    annotations/*
```

The taxonomy metadata is stored in `projects/HyDet/resources/<dataset>_hier/`. Text embeddings can be regenerated with the preparation utilities in `projects/HyDet/tools/` after placing the required text encoder assets in your local environment.

## Training And Evaluation

Run standard training with MMRotate tools:

```bash
python tools/train.py projects/HyDet/configs/hrsc_exp3_hydet_supervised.py
python tools/test.py projects/HyDet/configs/hrsc_exp3_hydet_supervised.py TRAINED_MODEL
```

Paper experiment helpers:

```bash
bash tools/run_hydet_exp1_dual_gpu.sh hrsc all
bash tools/run_hydet_exp2_dual_gpu.sh hrsc all
bash tools/run_hydet_exp3_dual_gpu.sh all
bash tools/run_hydet_exp4_dual_gpu.sh all
```

Collect tables:

```bash
python tools/collect_exp1_tables.py --work-dir work_dirs/exp1_hrsc_main --variants base hydet --out-md reports/exp1.md --out-csv reports/exp1.csv
python tools/collect_exp2_tables.py --work-dir work_dirs/exp2_hrsc_openvocab --variants base hra cone hydet --out-md reports/exp2.md --out-csv reports/exp2.csv
python tools/collect_exp3_tables.py --work-dir work_dirs/exp3_hrsc_supervised --variants base tax hyp hra cone hydet --out-md reports/exp3.md --out-csv reports/exp3.csv
python tools/collect_exp4_tables.py --work-dir work_dirs/exp4_hrsc_hra_ablation --variants base align rad sep sib hra --out-md reports/exp4.md --out-csv reports/exp4.csv
```

## Notes

- Internal taxonomy nodes are used only for training constraints and analysis; final detections are leaf categories.
- Runtime outputs, datasets, generated embeddings, and result files are intentionally kept outside the source package.
