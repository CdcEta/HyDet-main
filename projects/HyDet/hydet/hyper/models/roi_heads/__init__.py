"""HyDet ROI Heads 导出。

职责：
- 导出层次化 ROI 头占位实现。

TODO:
- 接入层次分类、logit 融合和双曲对比学习分支。
"""

from .hyper_bbox_head import HyperBBoxHead, Shared2FCBBoxHeadHyperZSD

__all__ = [
    'Shared2FCBBoxHeadHyperZSD',
    'HyperBBoxHead',
]
