import copy
from typing import Any, Dict, Mapping, Optional, Tuple

import mmcv
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmdet.structures import SampleList
from mmdet.structures.bbox import bbox2roi
from torch import Tensor
from tqdm import tqdm

from mmrotate.models.utils import _filter_rpn_results_by_score
from mmrotate.registry import MODELS
from mmrotate.structures.bbox import rbox_project

from ...castdet import RotatedCastDet
from ..utils import HyperPluginSwitch


@MODELS.register_module()
class HyperRotatedCastDet(RotatedCastDet):
    def __init__(
        self,
        tree_path: Optional[str] = None,
        parent_map_path: Optional[str] = None,
        leaf_text_vector_path: Optional[str] = None,
        all_nodes_text_vector_path: Optional[str] = None,
        use_hier_tree: bool = False,
        use_hyper_branch: bool = False,
        use_hier_losses: bool = False,
        use_hyp_contrast: bool = False,
        use_logit_fusion: bool = False,
        use_hier_queue_filter: bool = False,
        use_tp_projection: bool = False,
        hierarchy_loss_cfg: Optional[Mapping[str, Any]] = None,
        queue_filter_cfg: Optional[Mapping[str, Any]] = None,
        plugin_cfg: Optional[Mapping[str, Any]] = None,
        pseudo_queue_cfg: Optional[Mapping[str, Any]] = None,
        **kwargs
    ) -> None:
        super().__init__(pseudo_queue_cfg=pseudo_queue_cfg, **kwargs)
        self.hyper_plugin_switch = HyperPluginSwitch.parse(plugin_cfg or kwargs)
        self.tree_path = tree_path
        self.parent_map_path = parent_map_path
        self.leaf_text_vector_path = leaf_text_vector_path
        self.all_nodes_text_vector_path = all_nodes_text_vector_path

        self.use_hier_tree = bool(use_hier_tree) and bool(self.hyper_plugin_switch.use_hier_tree)
        self.use_hyper_branch = bool(use_hyper_branch) and bool(self.hyper_plugin_switch.use_hyper_branch)
        self.use_hier_losses = bool(use_hier_losses) and bool(self.hyper_plugin_switch.use_hier_losses)
        self.use_hyp_contrast = bool(use_hyp_contrast) and bool(self.hyper_plugin_switch.use_hyp_contrast)
        self.use_logit_fusion = bool(use_logit_fusion) and bool(self.hyper_plugin_switch.use_logit_fusion)
        self.use_hier_queue_filter = bool(use_hier_queue_filter) and bool(self.hyper_plugin_switch.use_hier_queue_filter)
        self.use_tp_projection = bool(use_tp_projection) and bool(self.hyper_plugin_switch.use_tp_projection)

        self.queue_filter_cfg = dict(queue_filter_cfg or {})
        if self.use_hier_queue_filter and pseudo_queue_cfg is not None:
            pq_cfg = dict(pseudo_queue_cfg)
            pq_cfg.update(self.queue_filter_cfg)
            pq_cfg['type'] = 'HierPseudoQueue'
            pq_cfg['use_hier_queue_filter'] = True
            pq_cfg['plugin_cfg'] = dict(
                use_hier_queue_filter=True,
                use_hyper_branch=self.use_hyper_branch,
                use_logit_fusion=self.use_logit_fusion
            )
            self.pseudo_queue = MODELS.build(pq_cfg)

        if leaf_text_vector_path is not None:
            self.words = nn.Parameter(torch.tensor(np.load(leaf_text_vector_path)), requires_grad=False)

        self.hierarchy_loss_cfg = dict(hierarchy_loss_cfg or {})
        if self.use_hier_losses:
            if 'type' not in self.hierarchy_loss_cfg:
                self.hierarchy_loss_cfg['type'] = 'HyperbolicHierarchyLoss'
            self.hierarchy_losses = MODELS.build(self.hierarchy_loss_cfg)
        else:
            self.hierarchy_losses = None

        self._sync_bbox_head_switches()

    def _sync_bbox_head_switches(self) -> None:
        switch_values = {
            'use_hier_tree': self.use_hier_tree,
            'use_hyper_branch': self.use_hyper_branch,
            'use_hier_losses': self.use_hier_losses,
            'use_hyp_contrast': self.use_hyp_contrast,
            'use_logit_fusion': self.use_logit_fusion,
            'use_hier_queue_filter': self.use_hier_queue_filter,
            'use_tp_projection': self.use_tp_projection,
        }

        def apply_switches(obj) -> None:
            if obj is None:
                return
            if hasattr(obj, 'hyper_plugin_switch'):
                for key, value in switch_values.items():
                    if hasattr(obj.hyper_plugin_switch, key):
                        setattr(obj.hyper_plugin_switch, key, bool(value))
            if hasattr(obj, 'plugin_switch'):
                for key, value in switch_values.items():
                    if hasattr(obj.plugin_switch, key):
                        setattr(obj.plugin_switch, key, bool(value))
            if hasattr(obj, 'use_hyper_branch'):
                obj.use_hyper_branch = bool(self.use_hyper_branch)
            if hasattr(obj, 'use_logit_fusion'):
                obj.use_logit_fusion = bool(self.use_logit_fusion)

        for model in [getattr(self, 'student', None), getattr(self, 'teacher', None)]:
            if model is None:
                continue
            roi_head = getattr(model, 'roi_head', None)
            if roi_head is None:
                continue
            bbox_head = getattr(roi_head, 'bbox_head', None)
            if bbox_head is None:
                continue
            apply_switches(bbox_head)
            apply_switches(getattr(bbox_head, 'fc_cls', None))

    def _fuse_logits(
        self,
        cls_score_euc: Optional[Tensor],
        cls_score_hyp: Optional[Tensor],
        lambda_logit_fusion: float = 0.5
    ) -> Optional[Tensor]:
        if cls_score_euc is None:
            return cls_score_hyp
        if cls_score_hyp is None:
            return cls_score_euc
        if not self.use_logit_fusion:
            return cls_score_euc
        lam = max(0.0, min(1.0, float(lambda_logit_fusion)))
        return (1.0 - lam) * cls_score_euc + lam * cls_score_hyp

    def _compute_hyper_losses(
        self,
        multi_batch_inputs: Dict[str, Tensor],
        multi_batch_data_samples: Dict[str, SampleList],
        base_losses: Dict[str, Tensor]
    ) -> Dict[str, Tensor]:
        if not self.use_hier_losses or self.hierarchy_losses is None:
            return {}
        batch_dict = None
        if isinstance(multi_batch_inputs.get('hyper_batch_dict', None), Mapping):
            batch_dict = multi_batch_inputs['hyper_batch_dict']
        elif isinstance(multi_batch_inputs.get('batch_dict', None), Mapping):
            batch_dict = multi_batch_inputs['batch_dict']
        elif isinstance(base_losses.get('hyper_batch_dict', None), Mapping):
            batch_dict = base_losses['hyper_batch_dict']
        if batch_dict is None:
            batch_dict = {}
        else:
            batch_dict = dict(batch_dict)

        if 'supervised' not in batch_dict:
            # Build hierarchy batch from real ROI/text features instead of scalar fallback.
            roi_head = getattr(self.student, 'roi_head', None)
            bbox_head = getattr(roi_head, 'bbox_head', None)
            aux = getattr(bbox_head, '_last_hyper_aux', None) if bbox_head is not None else None
            if isinstance(aux, Mapping):
                proj_hyp = aux.get('proj_feature_hyp', None)
                proj_euc = aux.get('proj_feature_euc', None)
                mapped_euc = aux.get('roi_feat_hyp_euc', None)
                cls_fused = aux.get('cls_score_fused', None)
                fc_cls = getattr(bbox_head, 'fc_cls', None)
                if proj_hyp is not None and fc_cls is not None and hasattr(fc_cls, '_leaf_bank_with_bg_euc'):
                    words_euc = fc_cls._leaf_bank_with_bg_euc()
                    words_hyp = fc_cls.hyper_mapper.map_to_hyperbolic(
                        fc_cls.hyper_mapper.project_euclidean(words_euc))
                    if cls_fused is not None and cls_fused.numel() > 0:
                        top_idx = torch.argmax(cls_fused, dim=-1).clamp(max=words_hyp.shape[0] - 1)
                        pos_text = words_hyp[top_idx]
                    else:
                        pos_text = words_hyp[:proj_hyp.shape[0]]
                    batch_dict['supervised'] = {
                        'roi_feat': proj_hyp,
                        'class_feat': pos_text,
                        'all_text_feat': words_hyp,
                        'mapped_feat': mapped_euc if mapped_euc is not None else proj_euc,
                        'expmapped_feat': proj_hyp,
                    }
                elif proj_euc is not None:
                    batch_dict['supervised'] = {'roi_feat': proj_euc}
        if 'supervised' not in batch_dict:
            if isinstance(base_losses.get('loss_bbox', None), torch.Tensor):
                batch_dict['supervised'] = {'roi_feat': base_losses['loss_bbox'].reshape(-1, 1)}
            else:
                batch_dict['supervised'] = {'roi_feat': multi_batch_inputs['sup'].new_zeros((0, 1))}
        # Runtime-sync hierarchy switches with current ablation flags, instead of
        # relying on static config defaults.
        if hasattr(self.hierarchy_losses, 'use_hyp_contrast_flag'):
            self.hierarchy_losses.use_hyp_contrast_flag = bool(self.use_hyp_contrast)
        if hasattr(self.hierarchy_losses, 'use_tp_reg_flag'):
            self.hierarchy_losses.use_tp_reg_flag = bool(self.use_tp_projection)
        if hasattr(self.hierarchy_losses, 'hyper_plugin_switch'):
            self.hierarchy_losses.hyper_plugin_switch.use_hyp_contrast = bool(self.use_hyp_contrast)
            self.hierarchy_losses.hyper_plugin_switch.use_tp_projection = bool(self.use_tp_projection)
        losses = self.hierarchy_losses(batch_dict)
        if not self.use_hyp_contrast and 'loss_hyp_contrast' in losses:
            losses['loss_hyp_contrast'] = losses['loss_hyp_contrast'] * 0.0
        if not self.use_tp_projection and 'loss_tp_reg' in losses:
            losses['loss_tp_reg'] = losses['loss_tp_reg'] * 0.0
        if not (self.use_hyp_contrast and self.use_tp_projection) and 'loss_joint_synergy' in losses:
            losses['loss_joint_synergy'] = losses['loss_joint_synergy'] * 0.0
        return losses

    def _attach_module_monitors(self, losses: Dict[str, Tensor]) -> Dict[str, Tensor]:
        """Attach non-optimization monitor keys for newly added modules.

        These keys are logged together with the real losses, but they do not
        contribute to the summed training objective because their names do not
        contain the `loss` keyword.
        """
        monitor_map = {
            'loss_txt_txt_hier': 'mod_tree_text_text',
            'loss_txt_img_hier': 'mod_tree_text_image',
            'loss_hyp_contrast': 'mod_hyperbolic_contrast',
            'loss_tp_reg': 'mod_tp_projection',
            'loss_joint_synergy': 'mod_joint_synergy',
        }
        for loss_key, monitor_key in monitor_map.items():
            value = losses.get(loss_key, None)
            if isinstance(value, torch.Tensor):
                losses[monitor_key] = value.detach()
        return losses

    def loss(self, multi_batch_inputs: Dict[str, Tensor], multi_batch_data_samples: Dict[str, SampleList]) -> Dict[str, Tensor]:
        self._sync_bbox_head_switches()
        losses = super().loss(multi_batch_inputs, multi_batch_data_samples)
        losses.update(self._compute_hyper_losses(multi_batch_inputs, multi_batch_data_samples, losses))
        return self._attach_module_monitors(losses)

    @torch.no_grad()
    def get_pseudo_instances(self, batch_inputs: Tensor, batch_data_samples: SampleList) -> Tuple[SampleList, Optional[dict]]:
        if not self.use_hier_queue_filter:
            return super().get_pseudo_instances(batch_inputs, batch_data_samples)

        assert self.teacher.with_bbox, 'Bbox head must be implemented.'
        x = self.teacher.extract_feat(batch_inputs)
        if batch_data_samples[0].get('proposals', None) is None:
            rpn_results_list = self.teacher.rpn_head.predict(x, batch_data_samples, rescale=False)
        else:
            rpn_results_list = [data_sample.proposals for data_sample in batch_data_samples]

        if self.pseudo_queue.cur_iter == 0:
            self.initialize_pseudo_queue()
            self.pseudo_queue.save_queue()

        if self.pseudo_queue.start_update():
            num_classes = self.teacher.roi_head.bbox_head.num_classes
            words = self._get_teacher_words()
            split_per_img = tuple([len(res.labels) for res in rpn_results_list])
            for idx, (inputs, rpn_result, data_samples) in enumerate(zip(batch_inputs, rpn_results_list, batch_data_samples)):
                save_proposals = []
                for _ in range(self.semi_reg_iter):
                    if self.rpn_bbox_type == 'xywh':
                        rois_ = bbox2roi([res.bboxes.convert_to('hbox') for res in rpn_results_list])
                    elif self.rpn_bbox_type == 'xywha':
                        rois_ = bbox2roi([res.bboxes.convert_to('rbox') for res in rpn_results_list])
                    else:
                        raise NotImplementedError
                    rois = torch.split(rois_, split_per_img)[idx]
                    bbox_feats = self.teacher.roi_head.bbox_roi_extractor(
                        x[:self.teacher.roi_head.bbox_roi_extractor.num_inputs], rois)
                    if self.teacher.roi_head.with_shared_head:
                        bbox_feats = self.teacher.roi_head.shared_head(bbox_feats)
                    reg_bboxes = self.teacher.roi_head.bbox_head.forward_reg(bbox_feats)
                    img_shape = batch_data_samples[idx].img_shape
                    rpn_result.bboxes = self.teacher.roi_head.bbox_head.bbox_coder.decode(
                        rois[:, 1:], reg_bboxes, max_shape=img_shape)
                    b = copy.deepcopy(rpn_result.bboxes)
                    if data_samples.get('homography_matrix', None) is not None:
                        b.project_(torch.from_numpy(data_samples.homography_matrix).inverse().to(self.data_preprocessor.device))
                    save_proposals.append(b)

                data_samples_ = copy.deepcopy(data_samples)
                data_samples_.gt_instances = copy.deepcopy(rpn_result)
                data_samples_.gt_instances.bboxes = data_samples_.gt_instances.bboxes.tensor
                reg_uncs_list, angle_uncs_list = self.compute_uncertainty_with_aug(x, [data_samples_], with_angle_uncs=True)
                rpn_result = self.filter_bboxes(rpn_result, save_proposals, reg_uncs_list[0], angle_uncs_list[0])
                if len(rpn_result) == 0:
                    continue

                crop_batch = self.crop_images(inputs.unsqueeze(0), [rpn_result])
                clip_feature = F.normalize(self.get_clip_features(crop_batch), dim=-1)
                logits = (self.clip_logit_scale * clip_feature @ words.T).softmax(dim=-1)
                scores, labels = logits.max(dim=-1)
                labels[labels >= num_classes] = num_classes
                ids = scores > self.semi_cls_score

                img_path = batch_data_samples[idx].img_path
                bboxes = copy.deepcopy(rpn_result.bboxes)
                bboxes.project_(torch.from_numpy(data_samples.homography_matrix).inverse().to(self.data_preprocessor.device))
                selected_probs = logits[ids].detach().cpu().numpy()
                selected_labels = labels[ids].detach().cpu().numpy()
                ancestor_ids = [self._get_ancestor_ids(int(lb)) for lb in selected_labels]
                self.pseudo_queue.update_pseudo_queue(
                    img_path,
                    bboxes[ids].detach().cpu().numpy(),
                    selected_labels,
                    scores[ids].detach().cpu().numpy(),
                    cls_probs=selected_probs,
                    ancestor_ids=ancestor_ids,
                    source='external_teacher'
                )

        if self.rpn_bbox_type == 'xywh':
            for rpn_result in rpn_results_list:
                rpn_result.bboxes = rpn_result.bboxes.convert_to('hbox')

        rpn_results_list = _filter_rpn_results_by_score(rpn_results_list, self.initial_rpn_score_thr)
        if self.semi_train_cfg.get('semi_mask_loss', False):
            results_list = self.teacher.roi_head.predict(x, rpn_results_list, batch_data_samples, rescale=False, with_mask=False)
        else:
            results_list = self.teacher.roi_head.predict(x, rpn_results_list, batch_data_samples, rescale=False)

        for data_samples, results in zip(batch_data_samples, results_list):
            data_samples.gt_instances = results

        reg_uncs_list = self.compute_uncertainty_with_aug(x, batch_data_samples)
        for data_samples, reg_uncs in zip(batch_data_samples, reg_uncs_list):
            data_samples.gt_instances['reg_uncs'] = reg_uncs
            data_samples.gt_instances.bboxes = rbox_project(
                data_samples.gt_instances.bboxes,
                torch.from_numpy(data_samples.homography_matrix).inverse().to(self.data_preprocessor.device),
                data_samples.ori_shape
            )
        batch_info = {'feat': x, 'img_shape': [], 'homography_matrix': [], 'metainfo': []}
        for data_samples in batch_data_samples:
            batch_info['img_shape'].append(data_samples.img_shape)
            batch_info['homography_matrix'].append(torch.from_numpy(data_samples.homography_matrix).to(self.data_preprocessor.device))
            batch_info['metainfo'].append(data_samples.metainfo)
        return batch_data_samples, batch_info

    def initialize_pseudo_queue(self):
        if not self.use_hier_queue_filter:
            return super().initialize_pseudo_queue()
        words = self._get_teacher_words()
        for img_path in tqdm(self.pseudo_queue.init_imgs):
            results = self.pseudo_queue.transform({'img': mmcv.imread(img_path)})
            batch_inputs = results['img'].unsqueeze(0).to(self.device)
            data_samples = self.pseudo_queue.bbox2detSample(img_path=img_path, **results)
            x = self.teacher.extract_feat(batch_inputs)
            rpn_result = self.teacher.rpn_head.predict(x, [data_samples], rescale=False)[0]
            save_proposals = []
            for _ in range(self.semi_reg_iter):
                if self.rpn_bbox_type == 'xywh':
                    rois = bbox2roi([rpn_result.bboxes.convert_to('hbox')])
                elif self.rpn_bbox_type == 'xywha':
                    rois = bbox2roi([rpn_result.bboxes.convert_to('rbox')])
                else:
                    raise NotImplementedError
                bbox_feats = self.teacher.roi_head.bbox_roi_extractor(
                    x[:self.teacher.roi_head.bbox_roi_extractor.num_inputs], rois)
                if self.teacher.roi_head.with_shared_head:
                    bbox_feats = self.teacher.roi_head.shared_head(bbox_feats)
                bbox_preds = self.teacher.roi_head.bbox_head.forward_reg(bbox_feats)
                img_shape = data_samples.img_shape
                rpn_result.bboxes = self.teacher.roi_head.bbox_head.bbox_coder.decode(
                    rois[:, 1:], bbox_preds, max_shape=img_shape)
                b = copy.deepcopy(rpn_result.bboxes)
                if data_samples.get('homography_matrix', None) is not None:
                    b.project_(torch.from_numpy(data_samples.homography_matrix).inverse().to(self.data_preprocessor.device))
                save_proposals.append(b)
            data_samples.gt_instances = copy.deepcopy(rpn_result)
            data_samples.gt_instances.bboxes = data_samples.gt_instances.bboxes.tensor
            reg_uncs_list, angle_uncs_list = self.compute_uncertainty_with_aug(x, [data_samples], with_angle_uncs=True)
            rpn_result.reg_uncs = reg_uncs_list[0]
            rpn_result.angle_uncs = angle_uncs_list[0]
            rpn_result = self.filter_bboxes(rpn_result, save_proposals, reg_uncs_list[0], angle_uncs_list[0])
            if len(rpn_result) == 0:
                continue
            num_classes = self.teacher.roi_head.bbox_head.num_classes
            crop_batch = self.crop_images(batch_inputs, [rpn_result])
            clip_feature = F.normalize(self.get_clip_features(crop_batch), dim=-1)
            logits = (self.clip_logit_scale * clip_feature @ words.T).softmax(dim=-1)
            scores, labels = logits.max(dim=-1)
            labels[labels >= num_classes] = num_classes
            ids = scores > self.semi_cls_score
            bboxes = copy.deepcopy(rpn_result.bboxes)
            if data_samples.get('homography_matrix', None) is not None:
                bboxes.project_(torch.from_numpy(data_samples.homography_matrix).inverse().to(self.data_preprocessor.device))
            selected_labels = labels[ids].detach().cpu().numpy()
            self.pseudo_queue.update_pseudo_queue(
                data_samples.img_path,
                bboxes[ids].detach().cpu().numpy(),
                selected_labels,
                scores[ids].detach().cpu().numpy(),
                cls_probs=logits[ids].detach().cpu().numpy(),
                ancestor_ids=[self._get_ancestor_ids(int(lb)) for lb in selected_labels],
                source='initialize'
            )

    def predict(self, batch_inputs: Tensor, batch_data_samples: SampleList, rescale: bool = True):
        self._sync_bbox_head_switches()
        return super().predict(batch_inputs, batch_data_samples)

    def _get_teacher_words(self) -> Tensor:
        num_classes = self.teacher.roi_head.bbox_head.num_classes
        if self.words is not None:
            return F.normalize(self.words, dim=-1)
        fc_cls = self.teacher.roi_head.bbox_head.fc_cls
        if hasattr(fc_cls, 'get_words'):
            return F.normalize(fc_cls.get_words(with_bg=True), dim=-1)
        if hasattr(fc_cls, '_leaf_bank_with_bg_euc'):
            words = fc_cls._leaf_bank_with_bg_euc()
            if words.shape[0] == num_classes + 1:
                return F.normalize(words, dim=-1)
        raise RuntimeError('can not fetch teacher text words for pseudo queue update')

    def _get_ancestor_ids(self, label: int):
        if hasattr(self.pseudo_queue, 'class_to_ancestors'):
            return list(self.pseudo_queue.class_to_ancestors.get(int(label), []))
        return []
