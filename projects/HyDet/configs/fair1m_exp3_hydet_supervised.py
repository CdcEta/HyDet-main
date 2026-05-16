import os

_base_ = [
    'mmrotate::_base_/models/oriented-rcnn-le90_r50_fpn.py',
    'mmrotate::_base_/default_runtime.py',
    '../../../configs/_base_/datasets/fair1m_hier.py',
]


def _read_lines(path):
    if not os.path.exists(path):
        return []
    with open(path, 'r', encoding='utf-8') as fp:
        return [x.strip() for x in fp.readlines() if x.strip()]


custom_imports = dict(
    imports=['projects.HyDet.hydet', 'projects.HyDet.hydet.hyper'],
    allow_failed_imports=False)

id_root = 'projects/HyDet/resources/fair1m_hier'
all_leaf_classes_path = f'{id_root}/class_names_leaf.txt'
leaf_text_vector_path = f'{id_root}/leaf_text_embeddings_with_bg_euc.npy'
all_nodes_text_vector_path = f'{id_root}/all_nodes_text_embeddings_euc.npy'
all_leaf_classes = tuple(_read_lines(all_leaf_classes_path)) or ('ship', 'airplane', 'vehicle')

split_root = 'ImageSets/Main'
test_all = f'{split_root}/fair1m_test_all.txt'

batch_size = 2
num_workers = 2
all_class_metainfo = dict(classes=all_leaf_classes)

train_dataloader = dict(
    batch_size=batch_size,
    num_workers=num_workers,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        ann_file='ImageSets/Main/fair1m_base_train.txt',
        metainfo=all_class_metainfo,
        filter_cfg=dict(filter_empty_gt=True)))
val_dataloader = dict(
    batch_size=1,
    num_workers=num_workers,
    dataset=dict(
        ann_file=test_all,
        metainfo=all_class_metainfo,
        test_mode=True))
test_dataloader = dict(
    batch_size=1,
    num_workers=num_workers,
    dataset=dict(
        ann_file=test_all,
        metainfo=all_class_metainfo,
        test_mode=True))

detector = _base_.model
detector.data_preprocessor = dict(
    type='mmdet.DetDataPreprocessor',
    mean=[122.7709383, 116.7460125, 104.09373615],
    std=[68.5005327, 66.6321579, 70.32316305],
    bgr_to_rgb=True,
    pad_size_divisor=32,
    boxtype2tensor=False)
detector.roi_head.bbox_head.type = 'Shared2FCBBoxHeadHyperZSD'
detector.roi_head.bbox_head.num_classes = len(all_leaf_classes)
detector.roi_head.bbox_head.reg_class_agnostic = True
detector.roi_head.bbox_head.return_aux_features = True
detector.roi_head.bbox_head.use_hyper_branch = True
detector.roi_head.bbox_head.use_logit_fusion = True
detector.roi_head.bbox_head.loss_profile = 'hydet'
detector.roi_head.bbox_head.plugin_cfg = dict(
    use_hyper_branch=True,
    use_logit_fusion=True,
    use_hyp_contrast=True,
    use_tp_projection=True)
detector.roi_head.bbox_head.module_loss_cfg = dict(
    tree_cone_margin=0.02,
    tree_radial_margin=0.10,
    tree_text_image_margin=0.08,
    tree_cone_w=0.0015,
    tree_radial_w=0.0010,
    tree_text_image_w=0.0015,
    hyp_contrast_w=0.0040,
    hac_radius_w=0.0010,
    hac_cross_w=0.0010,
    hac_sibling_w=0.0008,
    tp_projection_w=0.0002,
    joint_w=0.0001,
    agreement_w=0.0,
    nerve_w=0.0,
    hyp_fg_ce_w=0.0060,
    fused_fg_ce_w=0.0080,
    fast_fg_bg_w=0.0,
    fast_fused_adv_w=0.0,
    pseudo_w=0.0)
detector.roi_head.bbox_head.fc_cls = dict(
    type='HyperProjection',
    leaf_vector_path=leaf_text_vector_path,
    all_nodes_vector_path=all_nodes_text_vector_path,
    feature_dim=1024,
    hyper_dim=1024,
    curvature=1.0,
    is_scale=True,
    use_hyper_branch=True,
    use_logit_fusion=True,
    lambda_logit_fusion=0.25,
    euc_temperature=1.0,
    hyp_temperature=1.2,
    bg_logit_shift=-0.20,
    fg_logit_boost=0.05,
    plugin_cfg=dict(
        use_hyper_branch=True,
        use_logit_fusion=True,
        use_hyp_contrast=True,
        use_tp_projection=True))
detector.test_cfg.rcnn.score_thr = 0.01
detector.test_cfg.rcnn.nms.iou_threshold = 0.1
detector.test_cfg.rcnn.max_per_img = 2000

model = detector
model_wrapper_cfg = dict(
    type='MMDistributedDataParallel',
    find_unused_parameters=True)

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
    clip_grad=dict(max_norm=20, norm_type=2))
default_hooks = dict(
    logger=dict(type='LoggerHook', interval=20),
    checkpoint=dict(by_epoch=False, interval=2000, max_keep_ckpts=3))
log_processor = dict(by_epoch=False)
visualizer = dict(
    vis_backends=[dict(type='LocalVisBackend'), dict(type='TensorboardVisBackend')])
work_dir = 'work_dirs/exp3_fair1m_supervised/hydet'
