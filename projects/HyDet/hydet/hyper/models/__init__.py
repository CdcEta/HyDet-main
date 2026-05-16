"""HyDet 模型模块导出。

职责：
- 导出 HyDet 主体与子模块占位实现。

TODO:
- 接入真实映射层、投影层、检测器与 ROI Head 逻辑。
"""

from .hyper_mapper import HyperMapper, SharedHyperbolicMapper
from .hyper_projection import HyperProjection
from .hyper_rotated_castdet import HyperRotatedCastDet
from .roi_heads import HyperBBoxHead, Shared2FCBBoxHeadHyperZSD

__all__ = [
    'HyperMapper',
    'SharedHyperbolicMapper',
    'HyperProjection',
    'HyperRotatedCastDet',
    'Shared2FCBBoxHeadHyperZSD',
    'HyperBBoxHead',
]
