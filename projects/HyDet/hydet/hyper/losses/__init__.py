"""HyDet 损失模块导出。

职责：
- 导出层次化损失占位实现。

TODO:
- 支持层次监督、对比学习与多分支权重调度。
"""

from .hier_losses import HierLosses

__all__ = [
    'HierLosses',
]
