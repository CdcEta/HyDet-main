"""HyDet Hook 模块导出。

职责：
- 导出 TP 投影过程的训练 Hook 占位实现。

TODO:
- 接入训练阶段投影时机控制与统计日志。
"""

from .tp_projection_hook import TPProjectionHook

__all__ = [
    'TPProjectionHook',
]
