import os
import importlib.util
from pathlib import Path

# mmengine's config evaluator may not define __file__ in some contexts.
_THIS_FILE = Path(__file__).resolve() if '__file__' in globals() else Path.cwd() / 'configs/_base_/datasets/hrsc_hier.py'
_REPO_ROOT = _THIS_FILE.parents[3]
_BUILDER_PATH = _REPO_ROOT / 'projects' / 'HyDet' / 'tools' / 'hrsc_open_vocab_builder.py'
_SPEC = importlib.util.spec_from_file_location('hrsc_open_vocab_builder', _BUILDER_PATH)
_MOD = importlib.util.module_from_spec(_SPEC)
assert _SPEC is not None and _SPEC.loader is not None
_SPEC.loader.exec_module(_MOD)
ensure_hrsc_open_vocab_layout = _MOD.ensure_hrsc_open_vocab_layout

_DEFAULT_SOURCE_ROOT = os.getenv('HYDET_HRSC_SOURCE_ROOT', 'data/HRSC2016_raw')
_DEFAULT_TARGET_ROOT = os.getenv('HYDET_HRSC_TARGET_ROOT', 'data/HRSC2016')
_DEFAULT_LINK_ROOT = str(_REPO_ROOT / 'data' / 'HRSC2016')

_SOURCE_ROOT = os.getenv('HRSC_SOURCE_ROOT', _DEFAULT_SOURCE_ROOT)
_TARGET_ROOT = os.getenv('HRSC_OPENVOCAB_ROOT', _DEFAULT_TARGET_ROOT)
_LINK_ROOT = os.getenv('HRSC_REPO_DATA_LINK', _DEFAULT_LINK_ROOT)
if os.path.exists(_SOURCE_ROOT) or os.path.exists(_TARGET_ROOT):
    ensure_hrsc_open_vocab_layout(
        source_root=_SOURCE_ROOT,
        target_root=_TARGET_ROOT,
        repo_data_link=_LINK_ROOT,
        seed=int(os.getenv('HRSC_OPENVOCAB_SEED', '3407')),
        unseen_class_ratio=float(os.getenv('HRSC_OPENVOCAB_UNSEEN_CLASS_RATIO', '0.2')),
        unseen_instance_ratio=float(os.getenv('HRSC_OPENVOCAB_UNSEEN_INSTANCE_RATIO', '0.15')),
        force_rebuild=bool(int(os.getenv('HRSC_OPENVOCAB_FORCE_REBUILD', '0'))),
    )

# Keep config namespace serializable for mmengine pretty_text/yapf.
del importlib
del Path
del _THIS_FILE
del _REPO_ROOT
del _BUILDER_PATH
del _SPEC
del _MOD
del _LINK_ROOT
del _TARGET_ROOT
del _SOURCE_ROOT

dataset_type = 'HRSCDataset'
data_root = 'data/HRSC2016/'
backend_args = None

split_root = 'ImageSets/Main'
base_train_file = f'{split_root}/hrsc_base_train.txt'
train_unlabeled_file = f'{split_root}/hrsc_train_unlabeled.txt'
test_all = f'{split_root}/hrsc_test_all.txt'
test_base_train = f'{split_root}/hrsc_test_base_train.txt'
test_base_test_only = f'{split_root}/hrsc_test_base_test_only.txt'
test_novel = f'{split_root}/hrsc_test_novel.txt'

all_leaf_classes = (
    'class_000001', 'class_100000001', 'class_100000002', 'class_100000003',
    'class_100000004', 'class_100000005', 'class_100000006', 'class_100000007',
    'class_100000008', 'class_100000009', 'class_100000010', 'class_100000011',
    'class_100000012', 'class_100000013', 'class_100000015', 'class_100000016',
    'class_100000017', 'class_100000018', 'class_100000019', 'class_100000020',
    'class_100000022', 'class_100000024', 'class_100000025', 'class_100000026',
    'class_100000027', 'class_100000028', 'class_100000029', 'class_100000030',
    'class_100000032')
base_train_classes = (
    'class_000001', 'class_100000001', 'class_100000002', 'class_100000004',
    'class_100000005', 'class_100000007', 'class_100000008', 'class_100000009',
    'class_100000011', 'class_100000012', 'class_100000013', 'class_100000017',
    'class_100000024', 'class_100000025', 'class_100000026', 'class_100000027',
    'class_100000028', 'class_100000029', 'class_100000032')

train_pipeline = [
    dict(type='mmdet.LoadImageFromFile', backend_args=backend_args),
    dict(type='mmdet.LoadAnnotations', with_bbox=True, box_type='qbox'),
    dict(type='ConvertBoxType', box_type_mapping=dict(gt_bboxes='rbox')),
    dict(type='mmdet.Resize', scale=(800, 512), keep_ratio=True),
    dict(type='mmdet.RandomFlip', prob=0.75, direction=['horizontal', 'vertical', 'diagonal']),
    dict(type='mmdet.FilterAnnotations', min_gt_bbox_wh=(1e-2, 1e-2)),
    dict(type='mmdet.PackDetInputs')
]
val_pipeline = [
    dict(type='mmdet.LoadImageFromFile', backend_args=backend_args),
    dict(type='mmdet.Resize', scale=(800, 512), keep_ratio=True),
    dict(type='mmdet.LoadAnnotations', with_bbox=True, box_type='qbox'),
    dict(type='ConvertBoxType', box_type_mapping=dict(gt_bboxes='rbox')),
    dict(type='mmdet.PackDetInputs', meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape', 'scale_factor'))
]
test_pipeline = [
    dict(type='mmdet.LoadImageFromFile', backend_args=backend_args),
    dict(type='mmdet.Resize', scale=(800, 512), keep_ratio=True),
    dict(type='mmdet.LoadAnnotations', with_bbox=True, box_type='qbox'),
    dict(type='ConvertBoxType', box_type_mapping=dict(gt_bboxes='rbox')),
    dict(type='mmdet.PackDetInputs', meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape', 'scale_factor'))
]

train_dataloader = dict(
    batch_size=2,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    batch_sampler=None,
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=base_train_file,
        data_prefix=dict(sub_data_root='FullDataSet/'),
        ann_subdir='Annotations',
        img_subdir='AllImages',
        classwise=True,
        metainfo=dict(classes=base_train_classes),
        filter_cfg=dict(filter_empty_gt=True),
        pipeline=train_pipeline,
        backend_args=backend_args))
val_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=test_base_train,
        data_prefix=dict(sub_data_root='FullDataSet/'),
        ann_subdir='Annotations',
        img_subdir='AllImages',
        classwise=True,
        metainfo=dict(classes=base_train_classes),
        test_mode=True,
        pipeline=val_pipeline,
        backend_args=backend_args))
test_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=test_all,
        data_prefix=dict(sub_data_root='FullDataSet/'),
        ann_subdir='Annotations',
        img_subdir='AllImages',
        classwise=True,
        metainfo=dict(classes=all_leaf_classes),
        test_mode=True,
        pipeline=test_pipeline,
        backend_args=backend_args))

val_evaluator = dict(type='DOTAMetric', metric='mAP')
test_evaluator = val_evaluator
