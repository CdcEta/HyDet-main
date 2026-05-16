from typing import Any, Dict, Mapping, Optional

MODULE_1_TREE = 'module_tree_builder'
MODULE_2_HYPERBOLIC = 'module_hyperbolic_contrast'
MODULE_3_ANCHOR = 'module_hierarchy_anchor'

MODULE_SWITCH_KEYS = (
    MODULE_1_TREE,
    MODULE_2_HYPERBOLIC,
    MODULE_3_ANCHOR,
)

MODULE_TO_COMPONENTS = {
    MODULE_1_TREE: ('use_hier_tree',),
    MODULE_2_HYPERBOLIC: (
        'use_hyper_branch',
        'cache_hyp_bank',
        'use_hier_losses',
        'use_hyp_contrast',
        'use_logit_fusion',
    ),
    MODULE_3_ANCHOR: (
        'use_hier_queue_filter',
        'use_tp_projection',
    ),
}


def expand_module_switches(
    cfg: Optional[Mapping[str, Any]],
    component_keys: tuple,
) -> Dict[str, bool]:
    """将三模块总开关展开到细粒度组件开关。

    规则:
    - 若组件开关在 cfg 中显式给出，优先使用显式值；
    - 否则由对应模块开关决定；
    - 若模块开关也未给出，则默认 False。
    """
    cfg = cfg or {}
    module_values = {k: bool(cfg.get(k, False)) for k in MODULE_SWITCH_KEYS}
    explicit_components = {k: bool(cfg[k]) for k in component_keys if k in cfg}
    out: Dict[str, bool] = {}
    for key in component_keys:
        if key in explicit_components:
            out[key] = explicit_components[key]
            continue
        val = False
        for module_key, members in MODULE_TO_COMPONENTS.items():
            if key in members:
                val = val or module_values[module_key]
        out[key] = bool(val)
    return out


def infer_module_switches(component_values: Mapping[str, bool]) -> Dict[str, bool]:
    out: Dict[str, bool] = {}
    for module_key, members in MODULE_TO_COMPONENTS.items():
        out[module_key] = any(bool(component_values.get(k, False)) for k in members)
    return out
