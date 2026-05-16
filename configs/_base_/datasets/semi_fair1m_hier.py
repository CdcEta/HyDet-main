dataset_type = 'DIORDataset'
data_root = 'data/FAIR1M/'
backend_args = None
batch_size = 4
num_workers = 4

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

branch_field = ['sup', 'unsup_teacher', 'unsup_student']
sup_pipeline = [
    dict(type='mmdet.LoadImageFromFile', backend_args=backend_args),
    dict(type='mmdet.LoadAnnotations', with_bbox=True, box_type='qbox'),
    dict(type='ConvertBoxType', box_type_mapping=dict(gt_bboxes='rbox')),
    dict(type='mmdet.Resize', scale=(1024, 1024), keep_ratio=True),
    dict(type='mmdet.RandomFlip', prob=0.5, direction=['horizontal', 'vertical', 'diagonal']),
    dict(type='mmdet.FilterAnnotations', min_gt_bbox_wh=(1e-2, 1e-2)),
    dict(type='mmdet.MultiBranch', branch_field=branch_field, sup=dict(type='mmdet.PackDetInputs'))
]
weak_pipeline = [
    dict(type='mmdet.PackDetInputs', meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape', 'scale_factor', 'flip', 'flip_direction', 'homography_matrix')),
]
strong_pipeline = [
    dict(type='mmdet.RandAugment', aug_space=[[dict(type='mmdet.ColorTransform')], [dict(type='mmdet.AutoContrast')], [dict(type='mmdet.Color')], [dict(type='mmdet.Contrast')], [dict(type='mmdet.Brightness')]], aug_num=1),
    dict(type='RandomRotate', prob=0.5, angle_range=180),
    dict(type='mmdet.PackDetInputs', meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape', 'scale_factor', 'flip', 'flip_direction', 'homography_matrix')),
]
unsup_pipeline = [
    dict(type='mmdet.LoadImageFromFile', backend_args=backend_args),
    dict(type='mmdet.LoadEmptyAnnotations'),
    dict(type='mmdet.Resize', scale=(1024, 1024), keep_ratio=True),
    dict(type='mmdet.RandomFlip', prob=0.5, direction=['horizontal', 'vertical', 'diagonal']),
    dict(type='mmdet.MultiBranch', branch_field=branch_field, unsup_teacher=weak_pipeline, unsup_student=strong_pipeline),
]
val_pipeline = [
    dict(type='mmdet.LoadImageFromFile', backend_args=backend_args),
    dict(type='mmdet.Resize', scale=(1024, 1024), keep_ratio=True),
    dict(type='mmdet.LoadAnnotations', with_bbox=True, box_type='qbox'),
    dict(type='ConvertBoxType', box_type_mapping=dict(gt_bboxes='rbox')),
    dict(type='mmdet.PackDetInputs', meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape', 'scale_factor'))
]

labeled_dataset = dict(
    type=dataset_type,
    data_root=data_root,
    ann_file=base_train_file,
    data_prefix=dict(img_path='images'),
    ann_subdir='annfiles',
    metainfo=dict(classes=all_leaf_classes),
    filter_cfg=dict(filter_empty_gt=True, min_size=16),
    pipeline=sup_pipeline,
    backend_args=backend_args)
unlabeled_dataset = dict(
    type=dataset_type,
    data_root=data_root,
    ann_file=train_unlabeled_file,
    data_prefix=dict(img_path='images'),
    ann_subdir='annfiles',
    metainfo=dict(classes=all_leaf_classes),
    filter_cfg=dict(filter_empty_gt=False),
    pipeline=unsup_pipeline,
    backend_args=backend_args)
val_dataset = dict(
    type=dataset_type,
    data_root=data_root,
    ann_file=test_all,
    data_prefix=dict(img_path='images'),
    ann_subdir='annfiles',
    metainfo=dict(classes=all_leaf_classes),
    test_mode=True,
    pipeline=val_pipeline,
    backend_args=backend_args)

train_dataloader = dict(
    batch_size=batch_size,
    num_workers=num_workers,
    persistent_workers=True,
    sampler=dict(type='mmdet.MultiSourceSampler', batch_size=batch_size, source_ratio=[1, 3]),
    dataset=dict(type='ConcatDataset', datasets=[labeled_dataset, unlabeled_dataset]))
val_dataloader = dict(
    batch_size=batch_size,
    num_workers=num_workers,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=val_dataset)
test_dataloader = val_dataloader

val_evaluator = dict(type='DOTAMetric', metric='mAP')
test_evaluator = val_evaluator
