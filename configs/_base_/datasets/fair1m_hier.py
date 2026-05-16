dataset_type = 'DIORDataset'
# Use the split-consistent FAIR1M hierarchy data prepared for CastDet.
data_root = '/root/autodl-tmp/data/fair1m_hier/castdet_compat/'
backend_args = None

split_root = 'ImageSets/Main'
base_train_file = f'{split_root}/fair1m_base_train.txt'
train_unlabeled_file = f'{split_root}/fair1m_train_unlabeled.txt'
test_all = f'{split_root}/fair1m_test_all.txt'
test_base_train = f'{split_root}/fair1m_test_base_train.txt'
test_base_test_only = f'{split_root}/fair1m_test_base_test_only.txt'
test_novel = f'{split_root}/fair1m_test_novel.txt'

all_leaf_classes = (
    'a220', 'a321', 'a330', 'a350', 'arj21', 'baseball field',
    'basketball court', 'boeing737', 'boeing747', 'boeing777', 'boeing787',
    'bridge', 'bus', 'c919', 'cargo truck', 'dry cargo ship', 'dump truck',
    'engineering ship', 'excavator', 'fishing boat', 'football field',
    'intersection', 'liquid cargo ship', 'motorboat', 'passenger ship',
    'roundabout', 'small car', 'tennis court', 'tractor', 'trailer',
    'truck tractor', 'tugboat', 'van', 'warship', 'other-airplane',
    'other-ship', 'other-vehicle')
fair_palette = [
    (220, 20, 60), (119, 11, 32), (0, 0, 142), (0, 0, 230), (106, 0, 228),
    (0, 60, 100), (0, 80, 100), (0, 0, 70), (0, 0, 192), (250, 170, 30),
    (100, 170, 30), (220, 220, 0), (175, 116, 175), (250, 0, 30), (165, 42, 42),
    (255, 77, 255), (0, 226, 252), (182, 182, 255), (0, 82, 0), (120, 166, 157),
    (110, 76, 0), (174, 57, 255), (199, 100, 0), (72, 0, 118), (255, 179, 240),
    (0, 125, 92), (209, 0, 151), (188, 208, 182), (0, 220, 176), (255, 99, 164),
    (92, 0, 73), (133, 129, 255), (78, 180, 255), (0, 228, 0), (174, 255, 243),
    (45, 89, 255), (134, 134, 103)
]
base_train_classes = (
    'a220', 'a321', 'basketball court', 'boeing737', 'boeing747',
    'cargo truck', 'dry cargo ship', 'dump truck', 'excavator',
    'fishing boat', 'intersection', 'liquid cargo ship', 'motorboat',
    'small car', 'tennis court', 'warship', 'other-airplane',
    'other-ship', 'other-vehicle')

train_pipeline = [
    dict(type='mmdet.LoadImageFromFile', backend_args=backend_args),
    dict(type='mmdet.LoadAnnotations', with_bbox=True, box_type='qbox'),
    dict(type='ConvertBoxType', box_type_mapping=dict(gt_bboxes='rbox')),
    dict(type='mmdet.Resize', scale=(1024, 1024), keep_ratio=True),
    dict(type='mmdet.RandomFlip', prob=0.75, direction=['horizontal', 'vertical', 'diagonal']),
    dict(type='mmdet.FilterAnnotations', min_gt_bbox_wh=(1e-2, 1e-2)),
    dict(type='mmdet.PackDetInputs')
]
val_pipeline = [
    dict(type='mmdet.LoadImageFromFile', backend_args=backend_args),
    dict(type='mmdet.Resize', scale=(1024, 1024), keep_ratio=True),
    dict(type='mmdet.LoadAnnotations', with_bbox=True, box_type='qbox'),
    dict(type='ConvertBoxType', box_type_mapping=dict(gt_bboxes='rbox')),
    dict(type='mmdet.PackDetInputs', meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape', 'scale_factor'))
]
test_pipeline = [
    dict(type='mmdet.LoadImageFromFile', backend_args=backend_args),
    dict(type='mmdet.Resize', scale=(1024, 1024), keep_ratio=True),
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
        data_prefix=dict(img_path='images'),
        ann_subdir='annfiles',
        metainfo=dict(classes=base_train_classes, palette=fair_palette),
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
        data_prefix=dict(img_path='images'),
        ann_subdir='annfiles',
        metainfo=dict(classes=base_train_classes, palette=fair_palette),
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
        data_prefix=dict(img_path='images'),
        ann_subdir='annfiles',
        metainfo=dict(classes=all_leaf_classes, palette=fair_palette),
        test_mode=True,
        pipeline=test_pipeline,
        backend_args=backend_args))

val_evaluator = dict(type='DOTAMetric', metric='mAP')
test_evaluator = val_evaluator
