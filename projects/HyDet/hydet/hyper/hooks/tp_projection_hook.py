from typing import Any, List, Mapping, Optional

import torch
from mmengine.hooks import Hook
from mmrotate.registry import HOOKS

from ..utils import HyperPluginSwitch


@HOOKS.register_module()
class TPProjectionHook(Hook):
    """近似 TP 投影 Hook。

    该实现仅更新 SharedHyperbolicMapper 参数，不触碰 backbone/RPN/bbox reg。
    通过能量阈值控制更新触发，按固定间隔执行轻量投影更新。
    """

    def __init__(
        self,
        plugin_cfg: Optional[Mapping[str, Any]] = None,
        energy_thr: float = 1.0,
        update_interval: int = 50,
        feature_key: str = 'proj_feat_euc',
        priority: str = 'NORMAL'
    ) -> None:
        super().__init__()
        self.priority = priority
        self.hyper_plugin_switch = HyperPluginSwitch.parse(plugin_cfg)
        self.energy_thr = float(energy_thr)
        self.update_interval = int(update_interval)
        self.feature_key = str(feature_key)
        self._iter = 0

    def _unwrap_model(self, model):
        return model.module if hasattr(model, 'module') else model

    def _collect_hyper_mappers(self, model) -> List[torch.nn.Module]:
        out: List[torch.nn.Module] = []
        for _, mod in model.named_modules():
            if mod.__class__.__name__ == 'SharedHyperbolicMapper':
                out.append(mod)
        return out

    def _feature_energy(self, outputs: Optional[Mapping[str, Any]]) -> Optional[torch.Tensor]:
        if not isinstance(outputs, Mapping):
            return None
        feat = outputs.get(self.feature_key, None)
        if not isinstance(feat, torch.Tensor):
            return None
        if feat.numel() == 0:
            return None
        energy = (feat.float() * feat.float()).mean()
        return torch.nan_to_num(energy, nan=0.0, posinf=0.0, neginf=0.0)

    def _apply_projection(self, mapper: torch.nn.Module, scale: float) -> None:
        with torch.no_grad():
            for name, p in mapper.named_parameters(recurse=False):
                if not p.requires_grad:
                    continue
                if ('linear' in name) or ('proj' in name) or ('residual_scale' in name):
                    p.mul_(scale)

    def after_train_iter(self, runner, batch_idx: int, data_batch=None, outputs=None) -> None:
        if not self.hyper_plugin_switch.use_tp_projection:
            return
        self._iter += 1
        if self.update_interval <= 0 or (self._iter % self.update_interval != 0):
            return
        model = self._unwrap_model(runner.model)
        if not getattr(model, 'use_tp_projection', False):
            return
        mappers = self._collect_hyper_mappers(model)
        if not mappers:
            return

        energy = self._feature_energy(outputs)
        if energy is None:
            return
        if float(energy) < self.energy_thr:
            return

        shrink = float(self.energy_thr / (float(energy) + 1e-6))
        shrink = max(0.5, min(1.0, shrink))
        for mapper in mappers:
            self._apply_projection(mapper, shrink)
