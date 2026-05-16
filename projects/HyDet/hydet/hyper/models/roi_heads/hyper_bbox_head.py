import os
from typing import Any, Mapping, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor
from mmrotate.registry import MODELS

from ....ovd_bbox_head import Shared2FCBBoxHeadZSD
from ...utils import HyperPluginSwitch
from ...utils.hyperbolic_ops import entailment_aperture, entailment_exterior_angle, hyp_distance, logmap0, pairwise_hyp_distance


class BBoxForwardOutput(dict):
    def __iter__(self):
        yield self['cls_score']
        yield self['bbox_pred']


@MODELS.register_module()
class Shared2FCBBoxHeadHyperZSD(Shared2FCBBoxHeadZSD):
    def __init__(
        self,
        plugin_cfg: Optional[Mapping[str, Any]] = None,
        use_hyper_branch: bool = False,
        use_logit_fusion: bool = False,
        return_aux_features: bool = False,
        module_loss_cfg: Optional[Mapping[str, Any]] = None,
        loss_profile: str = 'default',
        **kwargs
    ) -> None:
        super().__init__(**kwargs)
        self.hyper_plugin_switch = HyperPluginSwitch.parse(plugin_cfg or kwargs)
        self.use_hyper_branch = bool(use_hyper_branch) and bool(self.hyper_plugin_switch.use_hyper_branch)
        self.use_logit_fusion = bool(use_logit_fusion)
        self.return_aux_features = bool(return_aux_features)
        self.module_loss_cfg = dict(module_loss_cfg or {})
        self.loss_profile = str(loss_profile)
        self.prototype_momentum = self._envf('HAC_PROTO_MOMENTUM', 0.90)
        self.cross_proto_thr = self._envf('HAC_CROSS_PROTO_THR', 0.55)
        self.sibling_proto_thr = self._envf('HAC_SIB_PROTO_THR', 0.35)
        self.cross_margin = self._envf('HAC_CROSS_MARGIN', 0.45)
        self.sibling_dir_margin = self._envf('HAC_SIB_DIR_MARGIN', 0.75)
        self.radius_gap = self._envf('HAC_RADIUS_GAP', 0.35)
        self.register_buffer('class_prototypes', torch.zeros(self.num_classes, 1024), persistent=False)
        self.register_buffer('class_proto_valid', torch.zeros(self.num_classes, dtype=torch.bool), persistent=False)

    def _forward_shared(self, x: Tensor) -> Tensor:
        if self.num_shared_convs > 0:
            for conv in self.shared_convs:
                x = conv(x)
        if self.num_shared_fcs > 0:
            if self.with_avg_pool:
                x = self.avg_pool(x)
            x = x.flatten(1)
            for fc in self.shared_fcs:
                x = self.relu(fc(x))
        return x

    def _forward_cls_feat(self, x: Tensor) -> Tensor:
        x_cls = x
        for conv in self.cls_convs:
            x_cls = conv(x_cls)
        if x_cls.dim() > 2:
            if self.with_avg_pool:
                x_cls = self.avg_pool(x_cls)
            x_cls = x_cls.flatten(1)
        for fc in self.cls_fcs:
            x_cls = self.relu(fc(x_cls))
        return x_cls

    def _forward_reg_feat(self, x: Tensor) -> Tensor:
        x_reg = x
        for conv in self.reg_convs:
            x_reg = conv(x_reg)
        if x_reg.dim() > 2:
            if self.with_avg_pool:
                x_reg = self.avg_pool(x_reg)
            x_reg = x_reg.flatten(1)
        for fc in self.reg_fcs:
            x_reg = self.relu(fc(x_reg))
        return x_reg

    def _pick_default_cls_score(self, cls_score_euc: Tensor, cls_score_hyp: Optional[Tensor], cls_score_fused: Optional[Tensor]) -> Tensor:
        if self.use_hyper_branch and self.use_logit_fusion and cls_score_fused is not None:
            return cls_score_fused
        return cls_score_euc

    @staticmethod
    def _envf(name: str, default: float) -> float:
        val = os.getenv(name, '')
        if val == '':
            return float(default)
        try:
            return float(val)
        except ValueError:
            return float(default)

    def _cfgf(self, key: str, env_name: str, default: float) -> float:
        if key in self.module_loss_cfg:
            try:
                return float(self.module_loss_cfg[key])
            except (TypeError, ValueError):
                return float(default)
        return self._envf(env_name, default)

    @staticmethod
    def _finite(x: Tensor, clamp: float = 1e4) -> Tensor:
        return torch.clamp(torch.nan_to_num(x, nan=0.0, posinf=float(clamp), neginf=-float(clamp)), -float(clamp), float(clamp))

    def _strict_ablation_logit_adjust(self, cls_score: Tensor) -> Tensor:
        """Compatibility hook kept neutral; module gains must come from losses."""
        return cls_score

    def forward(self, x: Tensor) -> BBoxForwardOutput:
        x = self._finite(self._forward_shared(x), clamp=100.0)
        x_cls = self._finite(self._forward_cls_feat(x), clamp=100.0)
        x_reg = self._finite(self._forward_reg_feat(x), clamp=100.0)

        bbox_pred = self._finite(self.fc_reg(x_reg), clamp=1e4) if self.with_reg else None
        proj_feat_euc: Optional[Tensor] = None
        roi_feat_hyp: Optional[Tensor] = None
        cls_score_hyp: Optional[Tensor] = None
        cls_score_fused: Optional[Tensor] = None

        if self.with_cls:
            try:
                cls_out = self.fc_cls(x_cls, return_dict=True)
            except TypeError:
                cls_out = self.fc_cls(x_cls)
            if isinstance(cls_out, Mapping):
                cls_score_euc = cls_out.get('cls_score_euc')
                if cls_score_euc is None:
                    cls_score_euc = cls_out.get('cls_score')
                if cls_score_euc is None:
                    raise ValueError('hyper projection output missing cls_score_euc')
                cls_score_euc = self._finite(cls_score_euc, clamp=100.0)
                proj_feat_euc = cls_out.get('proj_feature_euc', x_cls)
                roi_feat_hyp = cls_out.get('proj_feature_hyp', None)
                cls_score_hyp = cls_out.get('cls_score_hyp', None)
                cls_score_fused = cls_out.get('cls_score_fused', None)
                if isinstance(cls_score_hyp, Tensor):
                    cls_score_hyp = self._finite(cls_score_hyp, clamp=100.0)
                if isinstance(cls_score_fused, Tensor):
                    cls_score_fused = self._finite(cls_score_fused, clamp=100.0)
                self._last_hyper_aux = {
                    'proj_feature_euc': cls_out.get('proj_feature_euc', None),
                    'roi_feat_hyp_euc': cls_out.get('roi_feat_hyp_euc', None),
                    'proj_feature_hyp': cls_out.get('proj_feature_hyp', None),
                    'cls_score_euc': cls_score_euc,
                    'cls_score_hyp': cls_score_hyp,
                    'cls_score_fused': cls_out.get('cls_score_fused', None),
                    'node_conf': cls_out.get('node_conf', None),
                    'all_nodes_hyp': cls_out.get('all_nodes_hyp', None),
                    'parent_child_index': cls_out.get('parent_child_index', None),
                    'leaf_ancestor_index': cls_out.get('leaf_ancestor_index', None),
                    'all_node_depth': cls_out.get('all_node_depth', None),
                    'leaf_depth': cls_out.get('leaf_depth', None),
                    'leaf_parent_index': cls_out.get('leaf_parent_index', None),
                    'all_text_hyp': cls_out.get('all_text_hyp', None),
                }
            else:
                cls_score_euc = cls_out
                proj_feat_euc = x_cls
                self._last_hyper_aux = None
        else:
            cls_score_euc = None

        if cls_score_euc is None:
            raise ValueError('cls_score_euc can not be None when with_cls=True')

        cls_score = self._pick_default_cls_score(cls_score_euc, cls_score_hyp, cls_score_fused)
        cls_score = self._finite(cls_score, clamp=100.0)
        # Keep strict-ablation calibration in inference only, avoid switch-driven
        # optimization bias during training.
        if not self.training:
            cls_score = self._strict_ablation_logit_adjust(cls_score)
        if cls_score_fused is None:
            cls_score_fused = cls_score_euc

        output = BBoxForwardOutput(
            cls_score=cls_score,
            cls_score_euc=cls_score_euc,
            cls_score_hyp=cls_score_hyp,
            cls_score_fused=cls_score_fused,
            bbox_pred=bbox_pred,
            roi_feat_euc=x_cls if self.return_aux_features else None,
            roi_feat_hyp=roi_feat_hyp if self.return_aux_features else None,
            proj_feat_euc=proj_feat_euc if self.return_aux_features else None
        )
        return output

    def _select_score_tensor(self, score_item: Any) -> Tensor:
        if isinstance(score_item, Mapping):
            if self.use_hyper_branch and self.use_logit_fusion and score_item.get('cls_score_fused') is not None:
                return score_item['cls_score_fused']
            if score_item.get('cls_score_euc') is not None:
                return score_item['cls_score_euc']
            if score_item.get('cls_score') is not None:
                return score_item['cls_score']
            raise ValueError('cls score dict has no usable key')
        if isinstance(score_item, Tensor):
            return score_item
        raise TypeError('unsupported cls score item type')

    def _filter_prediction_instances(self, results):
        """Drop auxiliary background labels and non-finite predictions.

        The hyper projection head keeps an extra background prototype, whose
        label index equals `num_classes`. Some checkpoints may still surface
        that index in prediction results and visualization, producing noisy
        `class19:100.0` boxes for a 19-class setup.
        """
        if results is None or len(results) == 0:
            return results
        labels = getattr(results, 'labels', None)
        if labels is None:
            return results
        keep = torch.ones(len(results), dtype=torch.bool, device=labels.device)
        if labels is not None:
            keep = keep & (results.labels >= 0) & (results.labels < self.num_classes)
        if hasattr(results, 'scores') and results.scores is not None:
            keep = keep & torch.isfinite(results.scores)
        if hasattr(results, 'bboxes') and results.bboxes is not None:
            box_tensor = results.bboxes.tensor if hasattr(results.bboxes, 'tensor') else results.bboxes
            if isinstance(box_tensor, Tensor):
                keep = keep & torch.isfinite(box_tensor).all(dim=-1)
        return results[keep]

    def _ensure_proto_shape(self, feat_dim: int, device: torch.device) -> None:
        if self.class_prototypes.shape != (self.num_classes, feat_dim):
            self.class_prototypes = torch.zeros(self.num_classes, feat_dim, device=device)
            self.class_proto_valid = torch.zeros(self.num_classes, dtype=torch.bool, device=device)
        elif self.class_prototypes.device != device:
            self.class_prototypes = self.class_prototypes.to(device)
            self.class_proto_valid = self.class_proto_valid.to(device)

    @torch.no_grad()
    def _update_class_prototypes(self, roi_euc: Tensor, labels: Tensor) -> None:
        if roi_euc.numel() == 0 or labels.numel() == 0:
            return
        self._ensure_proto_shape(int(roi_euc.shape[-1]), roi_euc.device)
        momentum = max(0.0, min(0.999, float(self.prototype_momentum)))
        for cls_id in labels.unique():
            cls_int = int(cls_id.item())
            if cls_int < 0 or cls_int >= self.num_classes:
                continue
            cls_feat = roi_euc[labels == cls_id]
            if cls_feat.numel() == 0:
                continue
            proto = F.normalize(cls_feat.detach().mean(dim=0), dim=0)
            if bool(self.class_proto_valid[cls_int]):
                self.class_prototypes[cls_int] = F.normalize(
                    momentum * self.class_prototypes[cls_int] + (1.0 - momentum) * proto,
                    dim=0)
            else:
                self.class_prototypes[cls_int] = proto
                self.class_proto_valid[cls_int] = True

    def _compute_depth_radius_loss(
        self,
        text_hyp: Tensor,
        depths: Tensor,
    ) -> Tensor:
        if text_hyp.numel() == 0 or not isinstance(depths, Tensor) or depths.numel() == 0:
            return text_hyp.new_zeros(())
        n = min(text_hyp.shape[0], depths.shape[0])
        text_hyp = text_hyp[:n]
        depths = depths[:n].to(device=text_hyp.device, dtype=text_hyp.dtype)
        radius = torch.linalg.vector_norm(logmap0(text_hyp, c=1.0), dim=-1)
        max_depth = torch.clamp(depths.max(), min=1.0)
        target = (depths / max_depth) * float(self.radius_gap) * max_depth
        return F.smooth_l1_loss(radius, target)

    def _compute_hac_losses(
        self,
        roi_euc: Tensor,
        pos_labels: Tensor,
        all_text_hyp: Tensor,
        all_nodes_hyp: Optional[Tensor],
        all_node_depth: Optional[Tensor],
        leaf_depth: Optional[Tensor],
        leaf_parent_index: Optional[Tensor],
    ) -> Mapping[str, Tensor]:
        zero = roi_euc.new_zeros(())
        use_hyp = bool(getattr(self.hyper_plugin_switch, 'use_hyp_contrast', False))
        if not use_hyp or all_text_hyp.numel() == 0:
            return {
                'mod_hac_radius': zero,
                'mod_hac_cross_parent': zero,
                'mod_hac_sibling': zero,
                'hac_cross_pairs': zero,
                'hac_sibling_pairs': zero,
            }

        self._update_class_prototypes(roi_euc, pos_labels)
        leaf_text_hyp = all_text_hyp[:self.num_classes]
        radius_loss = zero
        if isinstance(leaf_depth, Tensor) and leaf_depth.numel() > 0:
            radius_loss = radius_loss + self._compute_depth_radius_loss(leaf_text_hyp, leaf_depth)
        if isinstance(all_nodes_hyp, Tensor) and isinstance(all_node_depth, Tensor) and all_nodes_hyp.numel() > 0:
            radius_loss = radius_loss + self._compute_depth_radius_loss(all_nodes_hyp, all_node_depth)

        valid = self.class_proto_valid[:self.num_classes]
        if valid.sum() < 2 or not isinstance(leaf_parent_index, Tensor) or leaf_parent_index.numel() < self.num_classes:
            return {
                'mod_hac_radius': radius_loss,
                'mod_hac_cross_parent': zero,
                'mod_hac_sibling': zero,
                'hac_cross_pairs': zero,
                'hac_sibling_pairs': zero,
            }

        proto = F.normalize(self.class_prototypes[:self.num_classes], dim=-1)
        sim = torch.mm(proto, proto.t())
        parents = leaf_parent_index[:self.num_classes].to(device=sim.device)
        eye = torch.eye(self.num_classes, dtype=torch.bool, device=sim.device)
        valid_pair = valid[:, None] & valid[None, :] & (~eye)
        diff_parent = parents[:, None] != parents[None, :]
        same_parent = (parents[:, None] == parents[None, :]) & (parents[:, None] >= 0)
        upper = torch.triu(torch.ones_like(valid_pair, dtype=torch.bool), diagonal=1)

        cross_mask = valid_pair & upper & diff_parent & (sim >= float(self.cross_proto_thr))
        if cross_mask.any():
            ij = cross_mask.nonzero(as_tuple=False)
            d = hyp_distance(leaf_text_hyp[ij[:, 0]], leaf_text_hyp[ij[:, 1]], c=1.0)
            cross_loss = F.relu(float(self.cross_margin) - d).mean()
            cross_pairs = cross_mask.float().sum()
        else:
            cross_loss = zero
            cross_pairs = zero

        sibling_mask = valid_pair & upper & same_parent & (sim <= float(self.sibling_proto_thr))
        if sibling_mask.any():
            ij = sibling_mask.nonzero(as_tuple=False)
            ti = leaf_text_hyp[ij[:, 0]]
            tj = leaf_text_hyp[ij[:, 1]]
            ri = torch.linalg.vector_norm(logmap0(ti, c=1.0), dim=-1)
            rj = torch.linalg.vector_norm(logmap0(tj, c=1.0), dim=-1)
            radius_same = F.smooth_l1_loss(ri, rj)
            di = F.normalize(logmap0(ti, c=1.0), dim=-1)
            dj = F.normalize(logmap0(tj, c=1.0), dim=-1)
            dir_cos = (di * dj).sum(dim=-1)
            dir_decouple = F.relu(dir_cos - float(self.sibling_dir_margin)).mean()
            sibling_loss = radius_same + dir_decouple
            sibling_pairs = sibling_mask.float().sum()
        else:
            sibling_loss = zero
            sibling_pairs = zero

        return {
            'mod_hac_radius': torch.nan_to_num(radius_loss),
            'mod_hac_cross_parent': torch.nan_to_num(cross_loss),
            'mod_hac_sibling': torch.nan_to_num(sibling_loss),
            'hac_cross_pairs': cross_pairs.detach(),
            'hac_sibling_pairs': sibling_pairs.detach(),
        }

    def _compute_module_monitors(self, labels: Tensor) -> Mapping[str, Tensor]:
        aux = getattr(self, '_last_hyper_aux', None)
        zero = labels.new_zeros((), dtype=torch.float32)
        if not isinstance(aux, Mapping):
            return {}
        roi_hyp = aux.get('proj_feature_hyp', None)
        roi_euc = aux.get('roi_feat_hyp_euc', None)
        all_text_hyp = aux.get('all_text_hyp', None)
        node_conf = aux.get('node_conf', None)
        all_nodes_hyp = aux.get('all_nodes_hyp', None)
        parent_child_index = aux.get('parent_child_index', None)
        leaf_ancestor_index = aux.get('leaf_ancestor_index', None)
        all_node_depth = aux.get('all_node_depth', None)
        leaf_depth = aux.get('leaf_depth', None)
        leaf_parent_index = aux.get('leaf_parent_index', None)
        cls_score_euc = aux.get('cls_score_euc', None)
        cls_score_hyp = aux.get('cls_score_hyp', None)
        cls_score_fused = aux.get('cls_score_fused', None)
        if not isinstance(roi_hyp, Tensor) or not isinstance(roi_euc, Tensor) or not isinstance(all_text_hyp, Tensor):
            return {
                'mod_tree_text_text': zero,
                'mod_tree_text_image': zero,
                'mod_hyperbolic_contrast': zero,
                'mod_tp_projection': zero,
                'mod_joint_synergy': zero,
            }

        pos_inds = (labels >= 0) & (labels < min(self.num_classes, all_text_hyp.shape[0] - 1))
        if not pos_inds.any():
            return {
                'mod_tree_text_text': zero,
                'mod_tree_text_image': zero,
                'mod_hyperbolic_contrast': zero,
                'mod_tp_projection': zero,
                'mod_joint_synergy': zero,
            }

        roi_hyp = roi_hyp[pos_inds]
        roi_hyp = self._finite(roi_hyp, clamp=1.0)
        roi_euc = self._finite(roi_euc[pos_inds], clamp=100.0)
        all_text_hyp = self._finite(all_text_hyp, clamp=1.0)
        if isinstance(all_nodes_hyp, Tensor):
            all_nodes_hyp = self._finite(all_nodes_hyp, clamp=1.0)
        if isinstance(node_conf, Tensor):
            node_conf = node_conf[pos_inds]
        if isinstance(cls_score_euc, Tensor):
            cls_score_euc = self._finite(cls_score_euc[pos_inds], clamp=100.0)
        if isinstance(cls_score_hyp, Tensor):
            cls_score_hyp = self._finite(cls_score_hyp[pos_inds], clamp=100.0)
        if isinstance(cls_score_fused, Tensor):
            cls_score_fused = self._finite(cls_score_fused[pos_inds], clamp=100.0)
        pos_labels = labels[pos_inds].long()
        class_hyp = all_text_hyp[pos_labels]
        logits = self._finite(-pairwise_hyp_distance(roi_hyp, all_text_hyp[:-1], c=1.0), clamp=100.0)
        txt_img = self._finite(hyp_distance(roi_hyp, class_hyp, c=1.0), clamp=100.0).mean()
        hyp_contrast = self._finite(F.cross_entropy(logits, pos_labels))
        tp_reg_raw = self._finite(F.mse_loss(self._finite(logmap0(roi_hyp, c=1.0), clamp=100.0), roi_euc))
        if isinstance(node_conf, Tensor):
            node_prob = torch.sigmoid(4.0 * (node_conf - 0.10))
            tp_reg = torch.relu(0.65 - node_prob).mean()
        else:
            tp_reg = tp_reg_raw

        d_hyp = self._finite(hyp_distance(roi_hyp, class_hyp, c=1.0), clamp=100.0)
        d_tp = F.mse_loss(self._finite(logmap0(roi_hyp, c=1.0), clamp=100.0), roi_euc, reduction='none')
        d_tp = self._finite(d_tp, clamp=100.0)
        if d_tp.dim() > 1:
            d_tp = d_tp.mean(dim=-1)
        d_h = (d_hyp - d_hyp.mean()) / (d_hyp.std(unbiased=False) + 1e-6)
        d_t = (d_tp - d_tp.mean()) / (d_tp.std(unbiased=False) + 1e-6)
        joint = (d_h - d_t).pow(2).mean()
        tree_cone = zero
        tree_radial = zero
        if isinstance(all_nodes_hyp, Tensor) and isinstance(parent_child_index, Tensor) and parent_child_index.numel() > 0:
            pc = parent_child_index.to(device=all_nodes_hyp.device)
            parent_feat = all_nodes_hyp[pc[:, 0]]
            child_feat = all_nodes_hyp[pc[:, 1]]
            margin = self._cfgf('tree_cone_margin', 'ABL_TREE_TXTTXT_M', 0.02)
            angle = entailment_exterior_angle(parent_feat, child_feat, c=1.0)
            aperture = entailment_aperture(parent_feat, c=1.0)
            tree_cone = F.relu(angle - aperture + float(margin)).mean()
            parent_radius = torch.linalg.vector_norm(logmap0(parent_feat, c=1.0), dim=-1)
            child_radius = torch.linalg.vector_norm(logmap0(child_feat, c=1.0), dim=-1)
            radial_margin = self._cfgf('tree_radial_margin', 'ABL_TREE_RADIAL_M', 0.10)
            tree_radial = F.relu(parent_radius + float(radial_margin) - child_radius).mean()
        tree_txt_img = txt_img
        if isinstance(all_nodes_hyp, Tensor) and isinstance(leaf_ancestor_index, Tensor) and leaf_ancestor_index.numel() > 0:
            anc_index = leaf_ancestor_index.to(device=pos_labels.device)[pos_labels]
            anc_mask = anc_index >= 0
            if anc_mask.any():
                anc_feat = all_nodes_hyp[anc_index.clamp(min=0)]
                roi_expand = roi_hyp.unsqueeze(1).expand_as(anc_feat)
                d_anc = hyp_distance(roi_expand, anc_feat, c=1.0)
                d_anc = (d_anc * anc_mask.float()).sum(dim=1) / anc_mask.float().sum(dim=1).clamp(min=1.0)
                hier_margin = self._cfgf('tree_text_image_margin', 'ABL_TREE_TXTIMG_M', 0.08)
                hier_penalty = F.relu(d_anc - (d_hyp + float(hier_margin)))
                if isinstance(node_conf, Tensor):
                    tree_gate = torch.sigmoid(4.0 * (node_conf - 0.05))
                    tree_txt_img = ((d_hyp + hier_penalty) * (0.5 + 0.5 * tree_gate)).mean()
                else:
                    tree_txt_img = (d_hyp + hier_penalty).mean()
        tree_cone = self._finite(tree_cone)
        tree_radial = self._finite(tree_radial)
        tree_txt_txt = self._finite(tree_cone + tree_radial)
        tree_txt_img = self._finite(tree_txt_img)
        hyp_contrast = self._finite(hyp_contrast)
        tp_reg = self._finite(tp_reg)
        joint = self._finite(joint)

        out = {
            'mod_tree_cone': tree_cone.detach(),
            'mod_tree_radial': tree_radial.detach(),
            'mod_tree_text_text': tree_txt_txt.detach(),
            'mod_tree_text_image': tree_txt_img.detach(),
            'mod_hyperbolic_contrast': hyp_contrast.detach(),
            'mod_tp_projection': tp_reg.detach(),
            'mod_joint_synergy': joint.detach(),
        }
        hac_losses = self._compute_hac_losses(
            roi_euc=roi_euc,
            pos_labels=pos_labels,
            all_text_hyp=all_text_hyp,
            all_nodes_hyp=all_nodes_hyp,
            all_node_depth=all_node_depth,
            leaf_depth=leaf_depth,
            leaf_parent_index=leaf_parent_index,
        )
        out.update({k: v.detach() for k, v in hac_losses.items() if not k.startswith('loss_')})
        agree_loss = zero
        if isinstance(cls_score_euc, Tensor) and isinstance(cls_score_hyp, Tensor) and cls_score_euc.shape[-1] > 1:
            t = self._envf('ABL_AGR_T', 2.0)
            p_e = F.log_softmax(cls_score_euc[..., :-1] / float(t), dim=-1)
            q_h = F.softmax(cls_score_hyp[..., :-1] / float(t), dim=-1)
            p_h = F.log_softmax(cls_score_hyp[..., :-1] / float(t), dim=-1)
            q_e = F.softmax(cls_score_euc[..., :-1] / float(t), dim=-1)
            agree_loss = self._finite(0.5 * (F.kl_div(p_e, q_h, reduction='batchmean') + F.kl_div(p_h, q_e, reduction='batchmean')))
            out['mod_agreement'] = agree_loss.detach()
        # NERVE-style dual-teacher consistency:
        # use euc/hyp as two teachers to supervise fused student logits on confident samples.
        nerve_loss = zero
        if isinstance(cls_score_fused, Tensor) and isinstance(cls_score_euc, Tensor) and isinstance(cls_score_hyp, Tensor):
            t_n = self._envf('ABL_NERVE_T', 1.8)
            teacher_logits = 0.5 * cls_score_euc + 0.5 * cls_score_hyp
            with torch.no_grad():
                teacher_prob = F.softmax(teacher_logits / float(t_n), dim=-1)
                conf, _ = teacher_prob[..., :-1].max(dim=-1)
                thr = self._envf('ABL_NERVE_THR', 0.35)
                mask = conf >= float(thr)
            if mask.any():
                stu_log = F.log_softmax(cls_score_fused[mask] / float(t_n), dim=-1)
                tea_prob = F.softmax(teacher_logits[mask].detach() / float(t_n), dim=-1)
                nerve_loss = self._finite(F.kl_div(stu_log, tea_prob, reduction='batchmean'))
                out['mod_nerve_consistency'] = nerve_loss.detach()
        # Direct task supervision for module branches.
        hyp_fg_ce = zero
        if isinstance(cls_score_hyp, Tensor) and cls_score_hyp.shape[-1] > 1:
            hyp_fg_ce = self._finite(F.cross_entropy(cls_score_hyp[..., :-1], pos_labels))
            out['mod_hyp_fg_ce'] = hyp_fg_ce.detach()
        fg_supervise = zero
        if isinstance(cls_score_fused, Tensor) and cls_score_fused.shape[-1] > 1:
            fg_supervise = self._finite(F.cross_entropy(cls_score_fused[..., :-1], pos_labels))
            out['mod_fused_fg_ce'] = fg_supervise.detach()
        # Fast gain constraints:
        # (1) enforce foreground-background margin on positive samples;
        # (2) for m123, require fused branch to outperform single branches.
        fg_bg_margin = zero
        fused_adv_margin = zero
        if isinstance(cls_score_fused, Tensor) and cls_score_fused.shape[-1] > 1:
            fg_pos = cls_score_fused[..., :-1].gather(dim=1, index=pos_labels.unsqueeze(1)).squeeze(1)
            bg_pos = cls_score_fused[..., -1]
            m_fg_bg = self._envf('ABL_FAST_M_FGBG', 1.00)
            fg_bg_margin = self._finite(F.relu(float(m_fg_bg) - (fg_pos - bg_pos)).mean())
            out['mod_fast_fg_bg_margin'] = fg_bg_margin.detach()
        if (
            isinstance(cls_score_fused, Tensor)
            and isinstance(cls_score_euc, Tensor)
            and isinstance(cls_score_hyp, Tensor)
            and cls_score_fused.shape[-1] > 1
            and cls_score_euc.shape[-1] > 1
            and cls_score_hyp.shape[-1] > 1
        ):
            fused_pos = cls_score_fused[..., :-1].gather(dim=1, index=pos_labels.unsqueeze(1)).squeeze(1)
            euc_pos = cls_score_euc[..., :-1].gather(dim=1, index=pos_labels.unsqueeze(1)).squeeze(1)
            hyp_pos = cls_score_hyp[..., :-1].gather(dim=1, index=pos_labels.unsqueeze(1)).squeeze(1)
            m_adv = self._envf('ABL_FAST_M_ADV', 0.20)
            fused_adv_margin = self._finite(F.relu(float(m_adv) - (fused_pos - torch.maximum(euc_pos, hyp_pos))).mean())
            out['mod_fast_fused_adv_margin'] = fused_adv_margin.detach()
        pseudo_loss = zero
        if isinstance(cls_score_euc, Tensor) and isinstance(cls_score_hyp, Tensor) and cls_score_euc.shape[-1] > 1:
            with torch.no_grad():
                euc_prob = F.softmax(cls_score_euc[..., :-1], dim=-1)
                euc_conf, euc_pl = euc_prob.max(dim=-1)
                thr = self._envf('ABL_PSEUDO_THR', 0.60)
                mask = euc_conf >= float(thr)
            if mask.any():
                hyp_logits = cls_score_hyp[mask, :-1]
                pseudo_loss = self._finite(F.cross_entropy(hyp_logits, euc_pl[mask].long()))
                out['mod_pseudo'] = pseudo_loss.detach()
        sw = getattr(self, 'hyper_plugin_switch', None)
        use_hyp = bool(getattr(sw, 'use_hyp_contrast', False))
        use_tp = bool(getattr(sw, 'use_tp_projection', False))
        w_h = self._cfgf('hyp_contrast_w', 'HYP_CONTRAST_W', 0.010) if use_hyp else 0.0
        w_x = self._cfgf('tree_text_image_w', 'TREE_TEXT_IMAGE_W', 0.005) if (use_hyp or use_tp) else 0.0
        w_cone = self._cfgf('tree_cone_w', 'CONE_W_TEXT', 0.010) if use_tp else 0.0
        w_tree_rad = self._cfgf('tree_radial_w', 'TREE_RADIAL_W', 0.005) if use_tp else 0.0
        w_rad = self._cfgf('hac_radius_w', 'HAC_W_RADIUS', 0.010) if use_hyp else 0.0
        w_cross = self._cfgf('hac_cross_w', 'HAC_W_CROSS', 0.010) if use_hyp else 0.0
        w_sib = self._cfgf('hac_sibling_w', 'HAC_W_SIBLING', 0.010) if use_hyp else 0.0
        w_t = self._cfgf('tp_projection_w', 'TP_PROJ_W', 0.0) if use_tp else 0.0
        w_j = self._cfgf('joint_w', 'JOINT_W', 0.0) if (use_hyp and use_tp) else 0.0
        w_a = self._cfgf('agreement_w', 'AGREEMENT_W', 0.0)
        w_n = self._cfgf('nerve_w', 'NERVE_W', 0.0)
        w_hc = self._cfgf('hyp_fg_ce_w', 'HYP_FG_CE_W', 0.0) if use_hyp else 0.0
        w_f = self._cfgf('fused_fg_ce_w', 'FUSED_FG_CE_W', 0.0) if self.use_logit_fusion else 0.0
        w_fast_fg = self._cfgf('fast_fg_bg_w', 'FAST_FG_BG_W', 0.0)
        w_fast_adv = self._cfgf('fast_fused_adv_w', 'FAST_FUSED_ADV_W', 0.0)
        w_p = self._cfgf('pseudo_w', 'PSEUDO_W', 0.0)
        out.update({
            'loss_mod_tree_cone': float(w_cone) * tree_cone,
            'loss_mod_tree_radial': float(w_tree_rad) * tree_radial,
            'loss_mod_tree_text_text': float(w_cone) * tree_txt_txt,
            'loss_mod_hyp_contrast': float(w_h) * hyp_contrast,
            'loss_hac_radius': float(w_rad) * hac_losses['mod_hac_radius'],
            'loss_hac_cross_parent': float(w_cross) * hac_losses['mod_hac_cross_parent'],
            'loss_hac_sibling': float(w_sib) * hac_losses['mod_hac_sibling'],
            'loss_mod_hyp_fg_ce': float(w_hc) * hyp_fg_ce,
            'loss_mod_tp_projection': float(w_t) * tp_reg,
            'loss_mod_joint_synergy': float(w_j) * joint,
            'loss_mod_text_image': float(w_x) * tree_txt_img,
            'loss_mod_agreement': float(w_a) * agree_loss,
            'loss_mod_nerve_consistency': float(w_n) * nerve_loss,
            'loss_mod_fused_fg_ce': float(w_f) * fg_supervise,
            'loss_mod_fast_fg_bg_margin': float(w_fast_fg) * fg_bg_margin,
            'loss_mod_fast_fused_adv_margin': float(w_fast_adv) * fused_adv_margin,
            'loss_mod_pseudo': float(w_p) * pseudo_loss,
        })
        return out

    def loss(
        self,
        cls_score: Tensor,
        bbox_pred: Tensor,
        rois: Tensor,
        labels: Tensor,
        label_weights: Tensor,
        bbox_targets: Tensor,
        bbox_weights: Tensor,
        reduction_override: Optional[str] = None,
    ) -> dict:
        losses = super().loss(
            cls_score,
            bbox_pred,
            rois,
            labels,
            label_weights,
            bbox_targets,
            bbox_weights,
            reduction_override=reduction_override)
        losses.update(self._compute_module_monitors(labels))
        return losses

    def predict_by_feat(
        self,
        rois: Tuple[Tensor],
        cls_scores: Tuple[Any],
        bbox_preds: Tuple[Tensor],
        batch_img_metas: Sequence[dict],
        rcnn_test_cfg: Optional[dict] = None,
        rescale: bool = False
    ):
        cls_scores_tensor = tuple(self._select_score_tensor(item) for item in cls_scores)
        results_list = super().predict_by_feat(
            rois=rois,
            cls_scores=cls_scores_tensor,
            bbox_preds=bbox_preds,
            batch_img_metas=batch_img_metas,
            rcnn_test_cfg=rcnn_test_cfg,
            rescale=rescale
        )
        return [self._filter_prediction_instances(results) for results in results_list]


@MODELS.register_module()
class HyperBBoxHead(Shared2FCBBoxHeadHyperZSD):
    pass
