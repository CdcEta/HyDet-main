"""HyDet 插件化层级损失模块。"""

from typing import Any, Dict, List, Mapping, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmrotate.registry import MODELS

from ..utils import HyperPluginSwitch
from ..utils.hyperbolic_ops import entailment_aperture, entailment_exterior_angle, hyp_distance, logmap0, pairwise_hyp_distance


def _safe_tensor(x: torch.Tensor) -> torch.Tensor:
    return torch.nan_to_num(x, nan=0.0, posinf=1e6, neginf=-1e6)


@MODELS.register_module()
class HyperbolicHierarchyLoss(nn.Module):
    """HyDet 层级损失组合器。

    支持四类损失，均可独立开关：
    - 文本-文本层级约束
    - 文本-图像层级约束
    - 双曲对比损失
    - TP 正则损失
    """

    def __init__(
        self,
        lambda_txt_txt: float = 1.0,
        lambda_txt_img: float = 1.0,
        lambda_hyp_contrast: float = 1.0,
        lambda_tp_reg: float = 1.0,
        delta_margin: float = 0.1,
        curvature: float = 1.0,
        kappa: float = 1.0,
        contrast_temperature: float = 0.07,
        use_txt_txt_loss: bool = True,
        use_txt_img_loss: bool = True,
        use_hyp_contrast: bool = True,
        use_tp_reg: bool = True,
        use_joint_synergy: bool = True,
        lambda_joint_synergy: float = 0.4,
        plugin_cfg: Optional[Mapping[str, Any]] = None,
        **kwargs
    ) -> None:
        super().__init__()
        self.hyper_plugin_switch = HyperPluginSwitch.parse(plugin_cfg or kwargs)
        self.lambda_txt_txt = float(lambda_txt_txt)
        self.lambda_txt_img = float(lambda_txt_img)
        self.lambda_hyp_contrast = float(lambda_hyp_contrast)
        self.lambda_tp_reg = float(lambda_tp_reg)
        self.delta_margin = float(delta_margin)
        self.curvature = float(curvature)
        self.kappa = float(kappa)
        self.contrast_temperature = float(contrast_temperature)
        self.use_txt_txt_loss = bool(use_txt_txt_loss)
        self.use_txt_img_loss = bool(use_txt_img_loss)
        self.use_hyp_contrast_flag = bool(use_hyp_contrast)
        self.use_tp_reg_flag = bool(use_tp_reg)
        self.use_joint_synergy_flag = bool(use_joint_synergy)
        self.lambda_joint_synergy = float(lambda_joint_synergy)

    def _zero_from_any(self, batch_dict: Mapping[str, Any]) -> torch.Tensor:
        for value in batch_dict.values():
            if isinstance(value, torch.Tensor):
                return value.new_zeros(())
            if isinstance(value, Mapping):
                z = self._zero_from_any(value)
                if isinstance(z, torch.Tensor):
                    return z
            if isinstance(value, (list, tuple)):
                for item in value:
                    if isinstance(item, torch.Tensor):
                        return item.new_zeros(())
                    if isinstance(item, Mapping):
                        z = self._zero_from_any(item)
                        if isinstance(z, torch.Tensor):
                            return z
        return torch.tensor(0.0)

    def _extract_tensor(self, source: Mapping[str, Any], keys: Sequence[str]) -> Optional[torch.Tensor]:
        for key in keys:
            if key in source and isinstance(source[key], torch.Tensor):
                return source[key]
        return None

    def _extract_sources(self, batch_dict: Mapping[str, Any]) -> List[Mapping[str, Any]]:
        sources: List[Mapping[str, Any]] = []
        for key in ('supervised', 'unsupervised', 'queue'):
            src = batch_dict.get(key, None)
            if isinstance(src, Mapping):
                sources.append(src)
        if sources:
            return sources
        return [batch_dict]

    def _cat_valid(self, tensors: Sequence[Optional[torch.Tensor]]) -> Optional[torch.Tensor]:
        valid: List[torch.Tensor] = []
        for x in tensors:
            if isinstance(x, torch.Tensor) and x.numel() > 0:
                valid.append(x)
        if not valid:
            return None
        try:
            return torch.cat(valid, dim=0)
        except RuntimeError:
            return valid[0]

    def loss_text_text_hier(self, parent_feat: torch.Tensor, child_feat: torch.Tensor) -> torch.Tensor:
        if parent_feat.numel() == 0 or child_feat.numel() == 0:
            return parent_feat.new_zeros(())
        if parent_feat.shape[0] != child_feat.shape[0]:
            n = min(parent_feat.shape[0], child_feat.shape[0])
            parent_feat = parent_feat[:n]
            child_feat = child_feat[:n]
        angle = entailment_exterior_angle(parent_feat, child_feat, c=self.curvature)
        aperture = entailment_aperture(parent_feat, c=self.curvature)
        raw = F.relu(angle - aperture + self.delta_margin)
        loss = self.kappa * _safe_tensor(raw).mean()
        return _safe_tensor(loss)

    def loss_text_image_hier(
        self,
        class_feat: torch.Tensor,
        roi_feat: torch.Tensor,
        ancestor_feats: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        if class_feat.numel() == 0 or roi_feat.numel() == 0:
            return roi_feat.new_zeros(())
        n = min(class_feat.shape[0], roi_feat.shape[0])
        class_feat = class_feat[:n]
        roi_feat = roi_feat[:n]
        d_pos = hyp_distance(roi_feat, class_feat, c=self.curvature)
        pull_loss = _safe_tensor(d_pos).mean()
        if ancestor_feats is None or ancestor_feats.numel() == 0:
            return pull_loss
        if ancestor_feats.dim() == 2:
            anc = ancestor_feats[:n]
            d_anc = hyp_distance(roi_feat, anc, c=self.curvature)
        elif ancestor_feats.dim() == 3:
            anc = ancestor_feats[:n]
            roi_expand = roi_feat[:n].unsqueeze(1).expand_as(anc)
            d_anc = hyp_distance(roi_expand, anc, c=self.curvature)
            d_anc = _safe_tensor(d_anc).mean(dim=1)
        else:
            return pull_loss
        hier_term = F.relu(_safe_tensor(d_anc) - (_safe_tensor(d_pos) + self.delta_margin))
        return _safe_tensor(pull_loss + hier_term.mean())

    def loss_hyp_contrast(self, roi_feat: torch.Tensor, pos_text_feat: torch.Tensor, all_text_feat: torch.Tensor) -> torch.Tensor:
        if roi_feat.numel() == 0 or pos_text_feat.numel() == 0 or all_text_feat.numel() == 0:
            return roi_feat.new_zeros(())
        n = min(roi_feat.shape[0], pos_text_feat.shape[0])
        roi = roi_feat[:n]
        pos = pos_text_feat[:n]
        all_bank = all_text_feat
        logits = -pairwise_hyp_distance(roi, all_bank, c=self.curvature)
        logits = _safe_tensor(logits) / max(self.contrast_temperature, 1e-6)
        with torch.no_grad():
            dist_pos_to_all = pairwise_hyp_distance(pos, all_bank, c=self.curvature)
            target = torch.argmin(dist_pos_to_all, dim=1)
        loss = F.cross_entropy(logits, target)
        return _safe_tensor(loss)

    def loss_tp_reg(self, mapped_feat: torch.Tensor, expmapped_feat: torch.Tensor) -> torch.Tensor:
        if mapped_feat.numel() == 0 or expmapped_feat.numel() == 0:
            return mapped_feat.new_zeros(())
        n = min(mapped_feat.shape[0], expmapped_feat.shape[0])
        mapped = mapped_feat[:n]
        expmapped = expmapped_feat[:n]
        recon = logmap0(expmapped, c=self.curvature)
        loss = F.mse_loss(_safe_tensor(recon), _safe_tensor(mapped))
        return _safe_tensor(loss)

    def loss_joint_synergy(
        self,
        roi_feat: torch.Tensor,
        class_feat: torch.Tensor,
        mapped_feat: torch.Tensor,
        expmapped_feat: torch.Tensor
    ) -> torch.Tensor:
        """Only used when both m12 and m13 are enabled (m123).

        Encourage hyper-space alignment to agree with TP projection alignment.
        """
        if roi_feat.numel() == 0 or class_feat.numel() == 0 or mapped_feat.numel() == 0 or expmapped_feat.numel() == 0:
            return roi_feat.new_zeros(())
        n = min(roi_feat.shape[0], class_feat.shape[0], mapped_feat.shape[0], expmapped_feat.shape[0])
        roi = roi_feat[:n]
        cls = class_feat[:n]
        mapped = mapped_feat[:n]
        expmapped = expmapped_feat[:n]
        d_hyp = hyp_distance(roi, cls, c=self.curvature)
        d_tp = F.mse_loss(_safe_tensor(logmap0(expmapped, c=self.curvature)), _safe_tensor(mapped), reduction='none')
        if d_tp.dim() > 1:
            d_tp = d_tp.mean(dim=-1)
        d_hyp = _safe_tensor(d_hyp)
        d_tp = _safe_tensor(d_tp)
        # Correlation-style agreement: lower is better.
        d_hyp_n = (d_hyp - d_hyp.mean()) / (d_hyp.std(unbiased=False) + 1e-6)
        d_tp_n = (d_tp - d_tp.mean()) / (d_tp.std(unbiased=False) + 1e-6)
        return _safe_tensor((d_hyp_n - d_tp_n).pow(2).mean())

    def forward(self, batch_dict: Mapping[str, Any]) -> Dict[str, torch.Tensor]:
        zero = self._zero_from_any(batch_dict)
        sources = self._extract_sources(batch_dict)

        parent_feats = self._cat_valid([
            self._extract_tensor(src, ('parent_feat', 'parent_text_feat', 'parent_text_feats'))
            for src in sources
        ])
        child_feats = self._cat_valid([
            self._extract_tensor(src, ('child_feat', 'child_text_feat', 'child_text_feats'))
            for src in sources
        ])
        class_feats = self._cat_valid([
            self._extract_tensor(src, ('class_feat', 'class_text_feat', 'pos_text_feat'))
            for src in sources
        ])
        roi_feats = self._cat_valid([
            self._extract_tensor(src, ('roi_feat', 'roi_feats', 'pos_roi_feat', 'pos_roi_feats'))
            for src in sources
        ])
        anc_feats = self._cat_valid([
            self._extract_tensor(src, ('ancestor_feats', 'ancestor_feat'))
            for src in sources
        ])
        all_text_feat = self._extract_tensor(batch_dict, ('all_text_feat', 'all_text_feats', 'text_bank_all'))
        if all_text_feat is None:
            all_text_feat = self._cat_valid([
                self._extract_tensor(src, ('all_text_feat', 'all_text_feats', 'text_bank_all'))
                for src in sources
            ])
        mapped_feat = self._cat_valid([
            self._extract_tensor(src, ('mapped_feat', 'mapped_feats'))
            for src in sources
        ])
        expmapped_feat = self._cat_valid([
            self._extract_tensor(src, ('expmapped_feat', 'expmapped_feats', 'roi_feat_hyp'))
            for src in sources
        ])

        if self.use_txt_txt_loss and parent_feats is not None and child_feats is not None:
            l_txt_txt = self.lambda_txt_txt * self.loss_text_text_hier(parent_feats, child_feats)
        else:
            l_txt_txt = zero.clone()

        if self.use_txt_img_loss and class_feats is not None and roi_feats is not None:
            l_txt_img = self.lambda_txt_img * self.loss_text_image_hier(class_feats, roi_feats, anc_feats)
        else:
            l_txt_img = zero.clone()

        if self.use_hyp_contrast_flag and roi_feats is not None and class_feats is not None and all_text_feat is not None:
            l_contrast = self.lambda_hyp_contrast * self.loss_hyp_contrast(roi_feats, class_feats, all_text_feat)
        else:
            l_contrast = zero.clone()

        if self.use_tp_reg_flag and mapped_feat is not None and expmapped_feat is not None:
            l_tp = self.lambda_tp_reg * self.loss_tp_reg(mapped_feat, expmapped_feat)
        else:
            l_tp = zero.clone()

        use_joint = (
            self.use_joint_synergy_flag
            and bool(self.hyper_plugin_switch.use_hyp_contrast)
            and bool(self.hyper_plugin_switch.use_tp_projection)
            and roi_feats is not None
            and class_feats is not None
            and mapped_feat is not None
            and expmapped_feat is not None
        )
        if use_joint:
            l_joint = self.lambda_joint_synergy * self.loss_joint_synergy(
                roi_feats, class_feats, mapped_feat, expmapped_feat)
        else:
            l_joint = zero.clone()

        return {
            'loss_txt_txt_hier': _safe_tensor(l_txt_txt),
            'loss_txt_img_hier': _safe_tensor(l_txt_img),
            'loss_hyp_contrast': _safe_tensor(l_contrast),
            'loss_tp_reg': _safe_tensor(l_tp),
            'loss_joint_synergy': _safe_tensor(l_joint),
        }


@MODELS.register_module()
class HierLosses(HyperbolicHierarchyLoss):
    pass
