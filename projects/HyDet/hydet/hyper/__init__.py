"""HyDet 插件扩展入口。

职责：
- 统一导出 HyDet 扩展包中的可注册组件。
- 统一维护插件化开关字段约定，供配置文件与模块复用。

TODO:
- 将各占位模块逐步替换为真实算法实现。
- 将插件开关接入完整训练/推理路径与可视化日志。
- 补齐单元测试与配置回归测试。
"""

from .utils import HYPER_PLUGIN_SWITCH_KEYS, HyperPluginConfig, HyperPluginSwitch
from .utils import HyperbolicOps, SplitUtils, TreeUtils
from .modules import (
    MODULE_1_TREE,
    MODULE_2_HYPERBOLIC,
    MODULE_3_ANCHOR,
    MODULE_SWITCH_KEYS,
    MODULE_TO_COMPONENTS,
    expand_module_switches,
    infer_module_switches,
)
from .structures import HyperTextBank, SemanticTree, TextBank
from .models import HyperMapper, HyperProjection, HyperRotatedCastDet, HyperBBoxHead, Shared2FCBBoxHeadHyperZSD, SharedHyperbolicMapper
from .losses import HierLosses
from .queues import HierPseudoQueue
from .hooks import TPProjectionHook

__all__ = [
    'HYPER_PLUGIN_SWITCH_KEYS',
    'HyperPluginConfig',
    'HyperPluginSwitch',
    'MODULE_1_TREE',
    'MODULE_2_HYPERBOLIC',
    'MODULE_3_ANCHOR',
    'MODULE_SWITCH_KEYS',
    'MODULE_TO_COMPONENTS',
    'expand_module_switches',
    'infer_module_switches',
    'HyperbolicOps',
    'SplitUtils',
    'TreeUtils',
    'SemanticTree',
    'HyperTextBank',
    'TextBank',
    'HyperMapper',
    'SharedHyperbolicMapper',
    'HyperProjection',
    'HyperRotatedCastDet',
    'Shared2FCBBoxHeadHyperZSD',
    'HyperBBoxHead',
    'HierLosses',
    'HierPseudoQueue',
    'TPProjectionHook',
]
