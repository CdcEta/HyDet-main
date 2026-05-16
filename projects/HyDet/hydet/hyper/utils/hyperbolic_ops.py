from typing import Any, Mapping, Optional

import torch
from mmrotate.registry import TASK_UTILS

from .split_utils import HyperPluginSwitch


def safe_acosh(x: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    x_safe = torch.clamp(x, min=1.0 + eps)
    out = torch.log(x_safe + torch.sqrt(torch.clamp(x_safe * x_safe - 1.0, min=eps)))
    return torch.nan_to_num(out, nan=0.0, posinf=1e6, neginf=0.0)


def safe_artanh(x: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    x_safe = torch.clamp(x, min=-1.0 + eps, max=1.0 - eps)
    out = 0.5 * (torch.log1p(x_safe) - torch.log1p(-x_safe))
    return torch.nan_to_num(out, nan=0.0, posinf=1e6, neginf=-1e6)


def proj(x: torch.Tensor, c: float = 1.0, eps: float = 1e-5) -> torch.Tensor:
    if c <= 0:
        raise ValueError(f'curvature c must be positive, got {c}')
    maxnorm = (1.0 - eps) / (c ** 0.5)
    norm = torch.linalg.vector_norm(x, dim=-1, keepdim=True)
    scale = torch.where(norm > maxnorm, maxnorm / torch.clamp(norm, min=eps), torch.ones_like(norm))
    out = x * scale
    return torch.nan_to_num(out, nan=0.0, posinf=maxnorm, neginf=-maxnorm)


def expmap0(u: torch.Tensor, c: float = 1.0, eps: float = 1e-8) -> torch.Tensor:
    if c <= 0:
        raise ValueError(f'curvature c must be positive, got {c}')
    sqrt_c = c ** 0.5
    norm_u = torch.linalg.vector_norm(u, dim=-1, keepdim=True)
    scaled = sqrt_c * norm_u
    coef = torch.where(
        norm_u > eps,
        torch.tanh(torch.clamp(scaled, max=15.0)) / torch.clamp(scaled, min=eps),
        torch.ones_like(norm_u),
    )
    out = coef * u
    return proj(out, c=c, eps=1e-5)


def logmap0(y: torch.Tensor, c: float = 1.0, eps: float = 1e-8) -> torch.Tensor:
    if c <= 0:
        raise ValueError(f'curvature c must be positive, got {c}')
    y = proj(y, c=c, eps=1e-5)
    sqrt_c = c ** 0.5
    norm_y = torch.linalg.vector_norm(y, dim=-1, keepdim=True)
    scaled = torch.clamp(sqrt_c * norm_y, max=1.0 - 1e-7)
    coef = torch.where(
        norm_y > eps,
        safe_artanh(scaled) / torch.clamp(scaled, min=eps),
        torch.ones_like(norm_y),
    )
    out = coef * y
    return torch.nan_to_num(out, nan=0.0, posinf=1e6, neginf=-1e6)


def _mobius_add(x: torch.Tensor, y: torch.Tensor, c: float, eps: float = 1e-8) -> torch.Tensor:
    x2 = (x * x).sum(dim=-1, keepdim=True)
    y2 = (y * y).sum(dim=-1, keepdim=True)
    xy = (x * y).sum(dim=-1, keepdim=True)
    num = (1 + 2 * c * xy + c * y2) * x + (1 - c * x2) * y
    den = 1 + 2 * c * xy + (c ** 2) * x2 * y2
    out = num / torch.clamp(den, min=eps)
    return proj(out, c=c, eps=1e-5)


def hyp_distance(x: torch.Tensor, y: torch.Tensor, c: float = 1.0, eps: float = 1e-8) -> torch.Tensor:
    if c <= 0:
        raise ValueError(f'curvature c must be positive, got {c}')
    x = proj(x, c=c, eps=1e-5)
    y = proj(y, c=c, eps=1e-5)
    minus_x = -x
    diff = _mobius_add(minus_x, y, c=c, eps=eps)
    norm_diff = torch.linalg.vector_norm(diff, dim=-1, keepdim=False)
    dist = 2.0 / (c ** 0.5) * safe_artanh((c ** 0.5) * torch.clamp(norm_diff, max=(1.0 - 1e-7) / (c ** 0.5)))
    return torch.nan_to_num(dist, nan=0.0, posinf=1e6, neginf=0.0)


def pairwise_hyp_distance(x: torch.Tensor, y: torch.Tensor, c: float = 1.0) -> torch.Tensor:
    if x.ndim != 2 or y.ndim != 2:
        raise ValueError('pairwise_hyp_distance expects 2D tensors')
    x_exp = x.unsqueeze(1)
    y_exp = y.unsqueeze(0)
    d = hyp_distance(x_exp, y_exp, c=c)
    return d


def entailment_aperture(center: torch.Tensor, c: float = 1.0, min_value: float = 1e-4) -> torch.Tensor:
    center = proj(center, c=c, eps=1e-5)
    norm = torch.linalg.vector_norm(center, dim=-1)
    radius = (1.0 / (c ** 0.5)) * safe_artanh((c ** 0.5) * torch.clamp(norm, max=(1.0 - 1e-7) / (c ** 0.5)))
    aperture = torch.atan2(torch.ones_like(radius), torch.clamp(radius, min=min_value))
    return torch.nan_to_num(aperture, nan=0.0, posinf=1.57, neginf=0.0)


def entailment_exterior_angle(
    parent_center: torch.Tensor,
    child_center: torch.Tensor,
    c: float = 1.0
) -> torch.Tensor:
    parent_center = proj(parent_center, c=c, eps=1e-5)
    child_center = proj(child_center, c=c, eps=1e-5)
    p = logmap0(parent_center, c=c)
    ch = logmap0(child_center, c=c)
    pn = torch.linalg.vector_norm(p, dim=-1)
    cn = torch.linalg.vector_norm(ch, dim=-1)
    dot = (p * ch).sum(dim=-1)
    cosv = dot / torch.clamp(pn * cn, min=1e-8)
    cosv = torch.clamp(cosv, min=-1.0 + 1e-7, max=1.0 - 1e-7)
    angle = torch.acos(cosv)
    return torch.nan_to_num(angle, nan=0.0, posinf=3.14159, neginf=0.0)


@TASK_UTILS.register_module()
class HyperbolicOps:
    def __init__(
        self,
        plugin_cfg: Optional[Mapping[str, Any]] = None,
        c: float = 1.0,
        eps: float = 1e-7
    ) -> None:
        self.plugin_switch = HyperPluginSwitch.parse(plugin_cfg)
        self.c = float(c)
        self.eps = float(eps)
        self.use_hyper_branch = bool(self.plugin_switch.use_hyper_branch)

    def proj(self, x: torch.Tensor) -> torch.Tensor:
        if not self.use_hyper_branch:
            return x
        return proj(x, c=self.c, eps=max(self.eps, 1e-8))

    def expmap0(self, x: torch.Tensor) -> torch.Tensor:
        if not self.use_hyper_branch:
            return x
        return expmap0(x, c=self.c, eps=max(self.eps, 1e-8))

    def logmap0(self, x: torch.Tensor) -> torch.Tensor:
        if not self.use_hyper_branch:
            return x
        return logmap0(x, c=self.c, eps=max(self.eps, 1e-8))

    def hyp_distance(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        if not self.use_hyper_branch:
            return torch.linalg.vector_norm(x - y, dim=-1)
        return hyp_distance(x, y, c=self.c, eps=max(self.eps, 1e-8))

    def pairwise_hyp_distance(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        if not self.use_hyper_branch:
            return torch.cdist(x, y, p=2)
        return pairwise_hyp_distance(x, y, c=self.c)

    def entailment_aperture(self, center: torch.Tensor) -> torch.Tensor:
        if not self.use_hyper_branch:
            n = torch.linalg.vector_norm(center, dim=-1)
            return torch.atan2(torch.ones_like(n), torch.clamp(n, min=1e-4))
        return entailment_aperture(center, c=self.c)

    def entailment_exterior_angle(self, parent_center: torch.Tensor, child_center: torch.Tensor) -> torch.Tensor:
        if not self.use_hyper_branch:
            p = parent_center
            ch = child_center
            pn = torch.linalg.vector_norm(p, dim=-1)
            cn = torch.linalg.vector_norm(ch, dim=-1)
            cosv = (p * ch).sum(dim=-1) / torch.clamp(pn * cn, min=1e-8)
            return torch.acos(torch.clamp(cosv, min=-1.0 + 1e-7, max=1.0 - 1e-7))
        return entailment_exterior_angle(parent_center, child_center, c=self.c)

    def safe_acosh(self, x: torch.Tensor) -> torch.Tensor:
        return safe_acosh(x, eps=max(self.eps, 1e-8))

    def safe_artanh(self, x: torch.Tensor) -> torch.Tensor:
        return safe_artanh(x, eps=max(self.eps, 1e-8))
