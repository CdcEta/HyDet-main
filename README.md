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

## Notes

- Internal taxonomy nodes are used only for training constraints and analysis; final detections are leaf categories.
- Runtime outputs, datasets, generated embeddings, and result files are intentionally kept outside the source package.
