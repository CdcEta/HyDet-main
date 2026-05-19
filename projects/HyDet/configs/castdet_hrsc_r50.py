import os

_base_ = [
    'mmrotate::_base_/models/oriented-rcnn-le90_r50_fpn.py',
    'mmrotate::_base_/default_runtime.py',
    '../../../configs/_base_/datasets/hrsc_hier.py',
]


def _read_lines(path):
    if not os.path.exists(path):
        return []
    with open(path, 'r', encoding='utf-8') as fp:
        return [x.strip() for x in fp.readlines() if x.strip()]


custom_imports = dict(
    imports=['projects.HyDet.hydet', 'projects.HyDet.hydet.hyper'],
    allow_failed_imports=False)

id_root = os.getenv('HYDET_HRSC_RESOURCE_ROOT', 'projects/HyDet/resources/hrsc_hier')
all_leaf_classes_path = f'{id_root}/class_names_leaf.txt'
all_leaf_classes = tuple(_read_lines(all_leaf_classes_path)) or ('class_000001',)
leaf_text_vector_path = f'{id_root}/leaf_text_embeddings_with_bg_euc.npy'

split_root = 'ImageSets/Main'
test_all = f'{split_root}/hrsc_test_all.txt'

batch_size = 4
num_workers = 4
all_class_metainfo = dict(classes=all_leaf_classes)

train_dataloader = dict(
    batch_size=batch_size,
    num_workers=num_workers,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        ann_file='ImageSets/Main/hrsc_base_train.txt',
        metainfo=all_class_metainfo,
        classwise=True,
        filter_cfg=dict(filter_empty_gt=True)))
val_dataloader = dict(
    batch_size=batch_size,
    num_workers=num_workers,
    dataset=dict(
        ann_file=test_all,
        metainfo=all_class_metainfo,
        classwise=True,
        test_mode=True))
test_dataloader = dict(
    batch_size=batch_size,
    num_workers=num_workers,
    dataset=dict(
        ann_file=test_all,
        metainfo=all_class_metainfo,
        classwise=True,
        test_mode=True))

detector = _base_.model
detector.data_preprocessor = dict(
    type='mmdet.DetDataPreprocessor',
    mean=[122.7709383, 116.7460125, 104.09373615],
    std=[68.5005327, 66.6321579, 70.32316305],
    bgr_to_rgb=True,
    pad_size_divisor=32,
    boxtype2tensor=False)
detector.roi_head.bbox_head.type = 'Shared2FCBBoxHeadZSD'
detector.roi_head.bbox_head.num_classes = len(all_leaf_classes)
detector.roi_head.bbox_head.reg_class_agnostic = True
detector.roi_head.bbox_head.fc_cls = dict(
    type='Projection2',
    vector_path=leaf_text_vector_path,
    is_scale=True,
    is_grad=False,
    is_grad_bg=True)
detector.test_cfg.rcnn.score_thr = 0.01
detector.test_cfg.rcnn.nms.iou_threshold = 0.1
detector.test_cfg.rcnn.max_per_img = 2000

model = detector
model_wrapper_cfg = dict(
    type='MMDistributedDataParallel',
    find_unused_parameters=False)

load_from = None
resume = False
train_cfg = dict(type='IterBasedTrainLoop', max_iters=10000, val_interval=2000)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

param_scheduler = [
    dict(type='LinearLR', start_factor=0.001, by_epoch=False, begin=0, end=500),
    dict(type='MultiStepLR', begin=0, end=10000, by_epoch=False, milestones=[8000, 9500], gamma=0.1),
]
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='SGD', lr=0.005, momentum=0.9, weight_decay=0.0001),
    clip_grad=dict(max_norm=35, norm_type=2))
default_hooks = dict(
    logger=dict(type='LoggerHook', interval=20),
    checkpoint=dict(by_epoch=False, interval=2000, max_keep_ckpts=3))
log_processor = dict(by_epoch=False)
visualizer = dict(
    vis_backends=[dict(type='LocalVisBackend'), dict(type='TensorboardVisBackend')])
work_dir = 'work_dirs/hrsc_castdet_r50'
