# HyDet

HyDet is a hierarchical open-vocabulary detector for remote-sensing images built on top of MMRotate. It extends the standard rotated detection pipeline with taxonomy-aware text priors, hyperbolic region-text alignment, HRA losses, and hierarchy-constrained optimization for fine-grained categories.

![](./HyDet.png)

![](./HRA.png)

## Repository Layout

```text
configs/                         MMRotate base dataset/runtime configs
mmrotate/                        bundled MMRotate framework code
projects/HyDet/hydet/            HyDet model heads, losses, hooks, queues, and utilities
projects/HyDet/configs/          public configs and paper configs
projects/HyDet/configs/paper/    paper-only experiment wrappers
projects/HyDet/resources/        hierarchy metadata templates and split files
projects/HyDet/tools/            data preparation and resource generation scripts
tools/                           train, test, evaluation, and runner scripts
requirements/                    dependency lists
```

## Installation

```bash
conda create -n hydet python=3.8 -y
conda activate hydet
pip install -U openmim
mim install mmengine mmcv mmdet
pip install -r requirements.txt
pip install -v -e .
export PYTHONPATH=$PWD:${PYTHONPATH:-}
```

If you want the helper scripts to activate conda automatically, set:

```bash
export HYDET_CONDA_ENV=hydet
```

## Downloads

The following links are placeholders and currently all point to the same Baidu Netdisk page. Replace them with your final links before public release.

- Re-split [HRSC2016]( https://pan.baidu.com/s/13hiyO7EMM7xXU8Qux7R0ew?pwd=62ax) data prepared for HyDet.
- Re-split [FAIR1M]( ) data prepared for HyDet.
- [Released checkpoints](https://pan.baidu.com/s/1id8SHZhDRX3_YHFc8kPfLA?pwd=16i4) .
- [Resource files](https://pan.baidu.com/s/1E4FMfWnIMsxULzHyPXzw?pwd=jede) required by the configs.

## Data And Resource Layout

The repo expects datasets outside the source tree and reads them through relative paths or environment variables.

Recommended layout:

```text
data/
  HRSC2016/
    ImageSets/Main/*.txt
    FullDataSet/Annotations/*.xml
    FullDataSet/AllImages/*
  FAIR1M/
    ImageSets/Main/*.txt
    annfiles/*.xml
    images/*
```

Resource layout:

```text
projects/HyDet/resources/
  hrsc_hier/
    class_names_leaf.txt
    tree_validated.json
    parent_map.json
    leaf_text_embeddings_with_bg_euc.npy
    all_nodes_text_embeddings_euc.npy
  fair1m_hier/
    class_names_leaf.txt
    tree_validated.json
    parent_map.json
    leaf_text_embeddings_with_bg_euc.npy
    all_nodes_text_embeddings_euc.npy
```

## Dataset Preparation

### HRSC2016

If you already provide the processed split package, unpack it to `data/HRSC2016/`.If you only have the raw HRSC dataset, you can rebuild the open-vocabulary split:

```bash
python projects/HyDet/tools/prepare_hrsc_open_vocab.py \
  --source-root data/HRSC2016_raw \
  --target-root data/HRSC2016 \
  --repo-data-link data/HRSC2016
```

### FAIR1M

If you already provide the processed split package, unpack it to `data/FAIR1M/`.If you only have the raw FAIR1M archives, rebuild the split and annotation conversion:

```bash
python projects/HyDet/tools/prepare_fair1m_open_vocab.py \
  --source-root data/FAIR1M_raw \
  --target-root data/FAIR1M \
  --resource-root projects/HyDet/resources/fair1m_hier \
  --repo-data-link data/FAIR1M
```

## Training

Train on `HRSC2016`:

```bash
bash tools/run_hydet.sh hrsc train
```

Train on `FAIR1M`:

```bash
bash tools/run_hydet.sh fair1m train
```

Or launch MMRotate directly:

```bash
python tools/train.py projects/HyDet/configs/hydet_hrsc_r50.py
python tools/train.py projects/HyDet/configs/hydet_fair1m_r50.py
```

## Evaluation And Inference

The default inference path in this repository is dataset-level inference through `tools/test.py`.

Evaluate a trained model on `HRSC2016`:

```bash
HYDET_CHECKPOINT=work_dirs/hrsc_hydet_r50/latest.pth \
bash tools/run_hydet.sh hrsc test
```

Evaluate a trained model on `FAIR1M`:

```bash
HYDET_CHECKPOINT=work_dirs/fair1m_hydet_r50/latest.pth \
bash tools/run_hydet.sh fair1m test
```

Run training and evaluation in one go:

```bash
bash tools/run_hydet.sh hrsc train-test
```

Direct MMRotate evaluation:

```bash
python tools/test.py projects/HyDet/configs/hydet_hrsc_r50.py /path/to/checkpoint.pth
python tools/test.py projects/HyDet/configs/hydet_fair1m_r50.py /path/to/checkpoint.pth
```
