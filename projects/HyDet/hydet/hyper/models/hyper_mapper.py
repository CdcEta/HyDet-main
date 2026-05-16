from typing import Any, Mapping, Optional

import torch
import torch.nn as nn
from mmrotate.registry import MODELS

from ..utils import HyperPluginSwitch
from ..utils.hyperbolic_ops import expmap0, logmap0, proj


@MODELS.register_module()
class SharedHyperbolicMapper(nn.Module):
    def __init__(
        self,
        in_dim: int = 1024,
        out_dim: int = 1024,
        curvature: float = 1.0,
        dropout: float = 0.0,
        use_residual: bool = True,
        enabled: bool = True,
        plugin_cfg: Optional[Mapping[str, Any]] = None
    ) -> None:
        super().__init__()
        self.plugin_switch = HyperPluginSwitch.parse(plugin_cfg)
        self.use_hyper_branch = bool(self.plugin_switch.use_hyper_branch)
        self.enabled = bool(enabled) and self.use_hyper_branch
        self.in_dim = int(in_dim)
        self.out_dim = int(out_dim)
        self.curvature = float(curvature)
        self.use_residual = bool(use_residual)
        self.dropout_layer = nn.Dropout(float(dropout))
        self.linear = nn.Linear(self.in_dim, self.out_dim)
        self.norm = nn.LayerNorm(self.out_dim)
        self.act = nn.GELU()
        self.identity_projection: Optional[nn.Module] = None
        if self.in_dim != self.out_dim:
            self.identity_projection = nn.Linear(self.in_dim, self.out_dim, bias=False)
        self.residual_scale = nn.Parameter(torch.tensor(1.0))

    def project_euclidean(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-1] != self.in_dim:
            raise ValueError(f'expected input dim {self.in_dim}, got {x.shape[-1]}')
        h = self.linear(x)
        h = self.act(h)
        h = self.dropout_layer(h)
        if self.use_residual:
            if self.identity_projection is None:
                res = x
            else:
                res = self.identity_projection(x)
            h = h + self.residual_scale * res
        h = self.norm(h)
        return torch.nan_to_num(h, nan=0.0, posinf=1e6, neginf=-1e6)

    def map_to_hyperbolic(self, x_euc: torch.Tensor) -> torch.Tensor:
        if not self.enabled:
            return x_euc
        hyp = expmap0(x_euc, c=self.curvature)
        hyp = proj(hyp, c=self.curvature)
        return torch.nan_to_num(hyp, nan=0.0, posinf=1.0, neginf=-1.0)

    def tp_regularization_loss(self, x_input: torch.Tensor, mapped_hyp: torch.Tensor) -> torch.Tensor:
        if not self.enabled:
            return x_input.new_zeros(())
        x_recon = logmap0(mapped_hyp, c=self.curvature)
        loss = torch.mean((x_recon - self.project_euclidean(x_input)) ** 2)
        return torch.nan_to_num(loss, nan=0.0, posinf=1e6, neginf=0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.enabled:
            if self.in_dim == self.out_dim:
                return x
            return self.project_euclidean(x)
        x_euc = self.project_euclidean(x)
        return self.map_to_hyperbolic(x_euc)


@MODELS.register_module()
class HyperMapper(SharedHyperbolicMapper):
    pass
