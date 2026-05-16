import json
import os
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmrotate.registry import MODELS

from ..utils import HyperPluginSwitch
from .hyper_mapper import SharedHyperbolicMapper
from ..utils.hyperbolic_ops import pairwise_hyp_distance


@MODELS.register_module()
class HyperProjection(nn.Module):
    def __init__(
        self,
        leaf_vector_path: str,
        all_nodes_vector_path: str,
        feature_dim: int,
        hyper_dim: int,
        curvature: float = 1.0,
        is_scale: bool = True,
        use_hyper_branch: bool = False,
        use_logit_fusion: bool = False,
        lambda_logit_fusion: float = 0.5,
        euc_temperature: Optional[float] = None,
        hyp_temperature: Optional[float] = None,
        bg_logit_shift: Optional[float] = None,
        fg_logit_boost: Optional[float] = None,
        plugin_cfg: Optional[Mapping[str, Any]] = None,
        **kwargs
    ) -> None:
        super().__init__()
        self.plugin_switch = HyperPluginSwitch.parse(plugin_cfg or kwargs)
        self.use_hyper_branch = bool(use_hyper_branch) and bool(self.plugin_switch.use_hyper_branch)
        self.use_logit_fusion = bool(use_logit_fusion)
        self.lambda_logit_fusion = float(lambda_logit_fusion)
        self.curvature = float(curvature)
        self.is_scale = bool(is_scale)
        self.euc_temperature = None if euc_temperature is None else float(euc_temperature)
        self.hyp_temperature = None if hyp_temperature is None else float(hyp_temperature)
        self.bg_logit_shift = None if bg_logit_shift is None else float(bg_logit_shift)
        self.fg_logit_boost = None if fg_logit_boost is None else float(fg_logit_boost)

        leaf_np = np.load(leaf_vector_path)
        all_np = np.load(all_nodes_vector_path)
        if leaf_np.ndim != 2 or all_np.ndim != 2:
            raise ValueError('text vectors must be 2D arrays')
        if leaf_np.shape[0] < 2:
            raise ValueError('leaf vectors must include foreground and bg rows')

        self.words = nn.Parameter(torch.from_numpy(leaf_np[:-1]).float(), requires_grad=False)
        self.bg = nn.Parameter(torch.from_numpy(leaf_np[-1:]).float(), requires_grad=True)
        self.all_nodes_euc = nn.Parameter(torch.from_numpy(all_np).float(), requires_grad=False)

        text_dim = int(leaf_np.shape[1])
        self.feature_dim = int(feature_dim)
        self.hyper_dim = int(hyper_dim)
        self.fc_proj = nn.Linear(self.feature_dim, self.hyper_dim)
        if text_dim == self.hyper_dim:
            self.text_align = nn.Identity()
            self.node_text_align = nn.Identity()
        else:
            self.text_align = nn.Linear(text_dim, self.hyper_dim, bias=False)
            self.node_text_align = nn.Linear(text_dim, self.hyper_dim, bias=False)

        self.hyper_mapper = SharedHyperbolicMapper(
            in_dim=self.hyper_dim,
            out_dim=self.hyper_dim,
            curvature=self.curvature,
            dropout=0.0,
            use_residual=False,
            enabled=self.use_hyper_branch,
            plugin_cfg={'use_hyper_branch': self.use_hyper_branch}
        )
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07), requires_grad=True) if self.is_scale else None
        self._init_hierarchy_metadata(leaf_vector_path, all_nodes_vector_path)

    @staticmethod
    def _envf(name: str, default: float) -> float:
        val = os.getenv(name, '')
        if val == '':
            return float(default)
        try:
            return float(val)
        except ValueError:
            return float(default)

    def _module_fusion_ratio(self) -> float:
        """Weight for combining Euclidean and hyperbolic class logits."""
        if bool(self.plugin_switch.use_hyper_branch):
            return self._envf('HYP_LOGIT_FUSION_LAM', float(self.lambda_logit_fusion))
        return 0.0

    def _module_hyp_temperature(self) -> float:
        if self.hyp_temperature is not None:
            return max(float(self.hyp_temperature), 1e-3)
        return self._envf('HYP_HYPT', 1.0)

    def _module_euc_temperature(self) -> float:
        if self.euc_temperature is not None:
            return max(float(self.euc_temperature), 1e-3)
        return self._envf('HYP_EUCT', 1.0)

    def _module_bg_logit_shift(self) -> float:
        if self.bg_logit_shift is not None:
            return float(self.bg_logit_shift)
        return self._envf('HYP_BGSHIFT', 0.0)

    def _module_fg_logit_boost(self) -> float:
        if self.fg_logit_boost is not None:
            return float(self.fg_logit_boost)
        return self._envf('HYP_FGBOOST', 0.0)

    @staticmethod
    def _finite(x: torch.Tensor, clamp: Optional[float] = None) -> torch.Tensor:
        x = torch.nan_to_num(x, nan=0.0, posinf=1e4, neginf=-1e4)
        if clamp is not None:
            x = torch.clamp(x, min=-float(clamp), max=float(clamp))
        return x

    def _scale_value(self) -> torch.Tensor:
        if self.logit_scale is None:
            return torch.ones((), device=self.words.device)
        scale_log = torch.nan_to_num(self.logit_scale, nan=float(np.log(1 / 0.07)), posinf=float(np.log(100.0)), neginf=0.0)
        return torch.clamp(scale_log.exp(), min=1.0, max=100.0)

    def _module_single_dynamic_gain(self, fg_conf: torch.Tensor) -> torch.Tensor:
        """Dynamic gain for single-module state based on foreground confidence."""
        scale = torch.sigmoid(1.8 * (fg_conf - 0.8))
        return scale

    def _consistency_gate(self, cls_score_euc: torch.Tensor, cls_score_hyp: torch.Tensor) -> torch.Tensor:
        """Per-sample gate from branch agreement.

        If euc/hyp predictions disagree, reduce hyp contribution for stability.
        """
        with torch.no_grad():
            pred_e = torch.argmax(cls_score_euc, dim=-1)
            pred_h = torch.argmax(cls_score_hyp, dim=-1)
            agree = (pred_e == pred_h).float().unsqueeze(-1)
            # Disagree -> 0.75, Agree -> 1.0
            gate = 0.75 + 0.25 * agree
        return gate

    def _confidence_gate(
        self,
        cls_score_euc: torch.Tensor,
        cls_score_hyp: torch.Tensor,
        node_conf: Optional[torch.Tensor]
    ) -> torch.Tensor:
        """Dual-branch confidence gate for m123.

        Hyperbolic fusion should dominate only when:
        - euc/hyp predictions are consistent,
        - foreground confidence is sufficiently high,
        - ROI is close to hierarchy nodes.
        """
        fg_e = cls_score_euc[..., :-1].max(dim=-1).values
        fg_h = cls_score_hyp[..., :-1].max(dim=-1).values
        conf = torch.sigmoid(1.8 * (torch.minimum(fg_e, fg_h) - 0.8))
        if isinstance(node_conf, torch.Tensor):
            tree = torch.sigmoid(3.0 * (node_conf - 0.10))
            conf = conf * tree
        return conf.unsqueeze(-1)

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        alias_map = {
            f'{prefix}leaf_words_euc': f'{prefix}words',
            f'{prefix}bg_euc': f'{prefix}bg',
            f'{prefix}fc_proj_euc.weight': f'{prefix}fc_proj.weight',
            f'{prefix}fc_proj_euc.bias': f'{prefix}fc_proj.bias',
        }
        for old_key, new_key in alias_map.items():
            if old_key in state_dict and new_key not in state_dict:
                state_dict[new_key] = state_dict[old_key]
        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

    def _init_hierarchy_metadata(self, leaf_vector_path: str, all_nodes_vector_path: str) -> None:
        num_leaf = int(self.words.shape[0])
        parent_child = torch.empty((0, 2), dtype=torch.long)
        leaf_ancestors = torch.full((num_leaf, 0), -1, dtype=torch.long)
        all_node_depth = torch.empty((0,), dtype=torch.float32)
        leaf_depth = torch.zeros((num_leaf,), dtype=torch.float32)
        leaf_parent_index = torch.full((num_leaf,), -1, dtype=torch.long)

        root_dir = Path(all_nodes_vector_path).resolve().parent
        tree_path = root_dir / 'tree_validated.json'
        leaf_name_path = root_dir / 'class_names_leaf.txt'
        if not tree_path.is_file() or not leaf_name_path.is_file():
            self.register_buffer('parent_child_index', parent_child, persistent=False)
            self.register_buffer('leaf_ancestor_index', leaf_ancestors, persistent=False)
            return

        try:
            tree = json.loads(tree_path.read_text())
            leaf_names = [line.strip() for line in leaf_name_path.read_text().splitlines() if line.strip()]
            nodes = tree.get('nodes', [])
            node_names = [str(node.get('name', '')) for node in nodes]
            node_to_idx = {name: idx for idx, name in enumerate(node_names)}
            parent_of = {str(node.get('name', '')): node.get('parent', None) for node in nodes}
            depth_of = {str(node.get('name', '')): float(node.get('depth', 0)) for node in nodes}
            all_node_depth = torch.tensor([depth_of.get(name, 0.0) for name in node_names], dtype=torch.float32)

            parent_child_pairs: List[List[int]] = []
            for node in nodes:
                child = str(node.get('name', ''))
                parent = node.get('parent', None)
                if parent is not None and parent in node_to_idx and child in node_to_idx:
                    parent_child_pairs.append([node_to_idx[parent], node_to_idx[child]])

            ancestor_lists: List[List[int]] = []
            leaf_depth_values: List[float] = []
            leaf_parent_values: List[int] = []
            for leaf_name in leaf_names[:num_leaf]:
                ancestors: List[int] = []
                cur = parent_of.get(leaf_name, None)
                leaf_parent_values.append(node_to_idx.get(cur, -1) if cur is not None else -1)
                while cur is not None and cur in node_to_idx:
                    ancestors.append(node_to_idx[cur])
                    cur = parent_of.get(cur, None)
                ancestor_lists.append(ancestors)
                leaf_depth_values.append(depth_of.get(leaf_name, float(len(ancestors))))

            max_anc = max((len(x) for x in ancestor_lists), default=0)
            if max_anc > 0:
                leaf_ancestors = torch.full((num_leaf, max_anc), -1, dtype=torch.long)
                for i, anc in enumerate(ancestor_lists):
                    if anc:
                        leaf_ancestors[i, :len(anc)] = torch.tensor(anc, dtype=torch.long)
            if parent_child_pairs:
                parent_child = torch.tensor(parent_child_pairs, dtype=torch.long)
            if leaf_depth_values:
                leaf_depth = torch.tensor(leaf_depth_values[:num_leaf], dtype=torch.float32)
            if leaf_parent_values:
                leaf_parent_index = torch.tensor(leaf_parent_values[:num_leaf], dtype=torch.long)
        except Exception:
            parent_child = torch.empty((0, 2), dtype=torch.long)
            leaf_ancestors = torch.full((num_leaf, 0), -1, dtype=torch.long)
            all_node_depth = torch.empty((0,), dtype=torch.float32)
            leaf_depth = torch.zeros((num_leaf,), dtype=torch.float32)
            leaf_parent_index = torch.full((num_leaf,), -1, dtype=torch.long)

        self.register_buffer('parent_child_index', parent_child, persistent=False)
        self.register_buffer('leaf_ancestor_index', leaf_ancestors, persistent=False)
        self.register_buffer('all_node_depth', all_node_depth, persistent=False)
        self.register_buffer('leaf_depth', leaf_depth, persistent=False)
        self.register_buffer('leaf_parent_index', leaf_parent_index, persistent=False)

    def _leaf_bank_with_bg_euc(self) -> torch.Tensor:
        leaf = self.text_align(self.words)
        bg = self.text_align(self.bg)
        return self._finite(torch.cat([leaf, bg], dim=0), clamp=100.0)

    def _all_nodes_bank_euc(self) -> torch.Tensor:
        return self._finite(self.node_text_align(self.all_nodes_euc), clamp=100.0)

    def _compute_euc_logits(self, proj_feature_euc: torch.Tensor) -> torch.Tensor:
        words = self._leaf_bank_with_bg_euc()
        if self.is_scale:
            logits = torch.einsum(
                'bd,cd->bc',
                F.normalize(proj_feature_euc, dim=-1),
                F.normalize(words, dim=-1)
            )
            logits = self._scale_value().to(device=logits.device, dtype=logits.dtype) * logits
        else:
            logits = torch.einsum('bd,cd->bc', proj_feature_euc, words)
        return self._finite(logits, clamp=100.0)

    def _compute_hyp_logits(self, proj_feature_hyp: torch.Tensor) -> torch.Tensor:
        words_euc = self._leaf_bank_with_bg_euc()
        words_hyp = self.hyper_mapper.map_to_hyperbolic(self.hyper_mapper.project_euclidean(words_euc))
        dist = pairwise_hyp_distance(proj_feature_hyp, words_hyp, c=self.curvature)
        logits = -dist
        if self.is_scale:
            logits = self._scale_value().to(device=logits.device, dtype=logits.dtype) * logits
        return self._finite(logits, clamp=100.0)

    def forward(self, feature: torch.Tensor, return_dict: bool = True):
        feature = self._finite(feature, clamp=100.0)
        proj_feature_euc = self._finite(self.fc_proj(feature), clamp=100.0)
        cls_score_euc = self._compute_euc_logits(proj_feature_euc)
        cls_score_euc = self._finite(cls_score_euc / self._module_euc_temperature(), clamp=100.0)

        proj_feature_hyp: Optional[torch.Tensor] = None
        roi_feat_hyp_euc: Optional[torch.Tensor] = None
        cls_score_hyp: Optional[torch.Tensor] = None
        node_conf: Optional[torch.Tensor] = None
        all_nodes_hyp: Optional[torch.Tensor] = None
        if self.use_hyper_branch:
            roi_feat_hyp_euc = self._finite(self.hyper_mapper.project_euclidean(proj_feature_euc), clamp=100.0)
            proj_feature_hyp = self._finite(self.hyper_mapper.map_to_hyperbolic(roi_feat_hyp_euc), clamp=1.0)
            cls_score_hyp = self._compute_hyp_logits(proj_feature_hyp)
            cls_score_hyp = self._finite(cls_score_hyp / self._module_hyp_temperature(), clamp=100.0)
            all_nodes_hyp = self._finite(self.hyper_mapper.map_to_hyperbolic(
                self.hyper_mapper.project_euclidean(self._all_nodes_bank_euc())), clamp=1.0)

        if self.use_hyper_branch and self.use_logit_fusion and cls_score_hyp is not None:
            lam = max(0.0, min(1.0, self._module_fusion_ratio()))
            gate = self._consistency_gate(cls_score_euc, cls_score_hyp)
            node_bank = self._all_nodes_bank_euc()
            node_logits = torch.einsum(
                'bd,nd->bn',
                F.normalize(proj_feature_euc, dim=-1),
                F.normalize(node_bank, dim=-1))
            node_conf = node_logits.max(dim=-1).values
            conf_gate = self._confidence_gate(cls_score_euc, cls_score_hyp, node_conf)
            cls_score_fused = cls_score_euc + lam * gate * conf_gate * (cls_score_hyp - cls_score_euc)
        else:
            cls_score_fused = cls_score_euc
        cls_score_fused = self._finite(cls_score_fused, clamp=100.0)

        # Calibrate foreground/background balance by module state.
        bg_shift = self._module_bg_logit_shift()
        if bg_shift != 0.0 and cls_score_fused.shape[-1] > 1:
            cls_score_fused = cls_score_fused.clone()
            cls_score_fused[..., -1] = cls_score_fused[..., -1] + bg_shift
        fg_boost = self._module_fg_logit_boost()
        if fg_boost != 0.0 and cls_score_fused.shape[-1] > 1:
            cls_score_fused = cls_score_fused.clone()
            cls_score_fused[..., :-1] = cls_score_fused[..., :-1] + fg_boost
        cls_score_fused = self._finite(cls_score_fused, clamp=100.0)

        output = {
            'proj_feature_euc': proj_feature_euc,
            'cls_score_euc': cls_score_euc,
            'roi_feat_hyp_euc': roi_feat_hyp_euc,
            'proj_feature_hyp': proj_feature_hyp,
            'cls_score_hyp': cls_score_hyp,
            'cls_score_fused': cls_score_fused,
            'node_conf': node_conf,
            'all_nodes_hyp': all_nodes_hyp,
            'parent_child_index': self.parent_child_index,
            'leaf_ancestor_index': self.leaf_ancestor_index,
            'all_node_depth': self.all_node_depth,
            'leaf_depth': self.leaf_depth,
            'leaf_parent_index': self.leaf_parent_index,
            'all_text_hyp': self.hyper_mapper.map_to_hyperbolic(
                self.hyper_mapper.project_euclidean(self._leaf_bank_with_bg_euc()))
            if self.use_hyper_branch else None,
        }
        if return_dict:
            return output
        return cls_score_fused
