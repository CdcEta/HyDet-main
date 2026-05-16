import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from mmrotate.registry import TASK_UTILS

from .split_utils import HyperPluginSwitch


def _normalize_name(name: str) -> str:
    return str(name).strip()


def _norm_key(name: str) -> str:
    return _normalize_name(name).lower()


def _dedup_keep_order(items: Iterable[str]) -> List[str]:
    out: List[str] = []
    seen = set()
    for item in items:
        v = _normalize_name(item)
        if not v:
            continue
        k = _norm_key(v)
        if k in seen:
            continue
        seen.add(k)
        out.append(v)
    return out


@dataclass
class ParsedTree:
    root: str
    children_map: Dict[str, List[str]]
    node_explanations: Dict[str, str]


@TASK_UTILS.register_module()
class TreeUtils:
    def __init__(self, plugin_cfg: Optional[Mapping[str, Any]] = None) -> None:
        self.plugin_switch = HyperPluginSwitch.parse(plugin_cfg)

    @staticmethod
    def read_leaf_names(path: str) -> List[str]:
        file_path = Path(path)
        if not file_path.exists():
            raise FileNotFoundError(f'leaf class file not found: {path}')
        lines = file_path.read_text(encoding='utf-8').splitlines()
        leaves = _dedup_keep_order(lines)
        if not leaves:
            raise ValueError('leaf class file is empty')
        return leaves

    @staticmethod
    def load_json(path: str) -> Any:
        file_path = Path(path)
        if not file_path.exists():
            raise FileNotFoundError(f'json file not found: {path}')
        return json.loads(file_path.read_text(encoding='utf-8'))

    @staticmethod
    def save_json(data: Any, path: str) -> None:
        file_path = Path(path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf-8')

    @staticmethod
    def normalize_tree(tree: Mapping[str, Any]) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        for parent, children in dict(tree).items():
            p = _normalize_name(parent)
            c = _dedup_keep_order(children or [])
            out[p] = c
            for node in c:
                out.setdefault(node, [])
        return out

    @staticmethod
    def parse_raw_tree(raw: Mapping[str, Any]) -> ParsedTree:
        raw_dict = dict(raw)
        node_explanations: Dict[str, str] = {}
        children_map: Dict[str, List[str]] = {}
        root = _normalize_name(raw_dict.get('root', ''))

        if 'nodes' in raw_dict:
            nodes = raw_dict['nodes']
            if not isinstance(nodes, list):
                raise ValueError('raw tree field "nodes" must be a list')
            for entry in nodes:
                if not isinstance(entry, dict):
                    raise ValueError('each item in "nodes" must be an object')
                name = _normalize_name(entry.get('name', ''))
                if not name:
                    raise ValueError('node name can not be empty')
                if name in node_explanations:
                    raise ValueError(f'duplicated node name: {name}')
                explanation = _normalize_name(entry.get('explanation', ''))
                if explanation:
                    node_explanations[name] = explanation
                children_field = entry.get('children', [])
                if not isinstance(children_field, list):
                    raise ValueError(f'children must be list for node: {name}')
                children: List[str] = []
                for child in children_field:
                    if isinstance(child, str):
                        child_name = _normalize_name(child)
                    elif isinstance(child, dict):
                        child_name = _normalize_name(child.get('name', ''))
                        cexp = _normalize_name(child.get('explanation', ''))
                        if cexp:
                            node_explanations[child_name] = cexp
                    else:
                        raise ValueError(f'invalid child type for node: {name}')
                    if not child_name:
                        raise ValueError(f'empty child name under node: {name}')
                    children.append(child_name)
                children_map[name] = _dedup_keep_order(children)
            if not root:
                roots = TreeUtils.infer_roots(children_map)
                if len(roots) == 1:
                    root = roots[0]
        elif 'tree' in raw_dict:
            tree = raw_dict['tree']
            root, children_map, node_explanations = TreeUtils._parse_nested_tree(tree)
        elif 'parent_map' in raw_dict:
            parent_map = raw_dict['parent_map']
            if not isinstance(parent_map, dict):
                raise ValueError('raw tree field "parent_map" must be an object')
            children_map = TreeUtils.parent_map_to_children_map(parent_map)
            roots = TreeUtils.infer_roots(children_map)
            root = root or (roots[0] if len(roots) == 1 else '')
        else:
            raise ValueError('raw tree must include one of: nodes / tree / parent_map')

        children_map = TreeUtils.normalize_tree(children_map)
        if not root:
            roots = TreeUtils.infer_roots(children_map)
            if len(roots) == 1:
                root = roots[0]
        if not root:
            raise ValueError('can not infer a unique root from raw tree')
        if root not in children_map:
            children_map[root] = []
        return ParsedTree(root=root, children_map=children_map, node_explanations=node_explanations)

    @staticmethod
    def infer_roots(children_map: Mapping[str, Sequence[str]]) -> List[str]:
        indegree: Dict[str, int] = {}
        for parent, children in children_map.items():
            indegree.setdefault(parent, 0)
            for child in children:
                indegree[child] = indegree.get(child, 0) + 1
        return [node for node, deg in indegree.items() if deg == 0]

    @staticmethod
    def parent_map_to_children_map(parent_map: Mapping[str, Any]) -> Dict[str, List[str]]:
        children_map: Dict[str, List[str]] = {}
        for child, parent in parent_map.items():
            c = _normalize_name(child)
            p = _normalize_name(parent) if parent is not None else ''
            if not c:
                raise ValueError('child name in parent_map can not be empty')
            children_map.setdefault(c, [])
            if p:
                children_map.setdefault(p, [])
                children_map[p].append(c)
        return TreeUtils.normalize_tree(children_map)

    @staticmethod
    def children_map_to_parent_map(children_map: Mapping[str, Sequence[str]]) -> Dict[str, Optional[str]]:
        parent_map: Dict[str, Optional[str]] = {node: None for node in children_map.keys()}
        for parent, children in children_map.items():
            for child in children:
                if child in parent_map and parent_map[child] is not None and parent_map[child] != parent:
                    raise ValueError(f'node has multiple parents: {child}')
                parent_map[child] = parent
        return parent_map

    @staticmethod
    def topological_nodes(children_map: Mapping[str, Sequence[str]], root: str) -> List[str]:
        indegree: Dict[str, int] = {}
        for parent, children in children_map.items():
            indegree.setdefault(parent, 0)
            for child in children:
                indegree[child] = indegree.get(child, 0) + 1
        queue: List[str] = [node for node, deg in indegree.items() if deg == 0]
        out: List[str] = []
        idx = 0
        while idx < len(queue):
            node = queue[idx]
            idx += 1
            out.append(node)
            for child in children_map.get(node, []):
                indegree[child] -= 1
                if indegree[child] == 0:
                    queue.append(child)
        if len(out) != len(indegree):
            raise ValueError('cycle detected in tree')
        if root not in out:
            raise ValueError(f'root "{root}" is not in graph')
        return out

    @staticmethod
    def descendants(children_map: Mapping[str, Sequence[str]], node: str) -> List[str]:
        out: List[str] = []
        stack = list(children_map.get(node, []))
        seen = set()
        while stack:
            cur = stack.pop()
            if cur in seen:
                continue
            seen.add(cur)
            out.append(cur)
            stack.extend(children_map.get(cur, []))
        return out

    @staticmethod
    def dataset_ontology_hints(dataset_name: str) -> Dict[str, List[str]]:
        name = _norm_key(dataset_name)
        if name == 'hrsc2016':
            return {
                '舰船': ['ship', 'vessel', 'boat', 'craft'],
                '军舰': ['warship', 'destroyer', 'frigate', 'carrier', 'submarine', 'naval'],
                '民船': ['cargo', 'container', 'tanker', 'fishing', 'passenger', 'merchant'],
            }
        if name == 'fair1m':
            return {
                '舰船': ['ship', 'vessel', 'boat', 'craft'],
                '航空器': ['aircraft', 'airplane', 'plane', 'jet', 'helicopter', 'uav'],
                '地面车辆': ['vehicle', 'truck', 'car', 'bus', 'tractor', 'tank'],
                '场地设施': ['field', 'stadium', 'court', 'port', 'harbor', 'airport', 'industrial'],
            }
        return {
            '舰船': ['ship', 'vessel', 'boat', 'craft'],
            '航空器': ['aircraft', 'airplane', 'plane', 'jet', 'helicopter'],
            '地面车辆': ['vehicle', 'truck', 'car', 'bus', 'tank'],
            '场地设施': ['field', 'stadium', 'court', 'port', 'harbor', 'airport', 'industrial'],
        }

    @staticmethod
    def _parse_nested_tree(tree_obj: Any) -> Tuple[str, Dict[str, List[str]], Dict[str, str]]:
        if not isinstance(tree_obj, dict):
            raise ValueError('raw tree field "tree" must be an object')
        children_map: Dict[str, List[str]] = {}
        explanations: Dict[str, str] = {}

        def walk(node: Dict[str, Any]) -> str:
            if not isinstance(node, dict):
                raise ValueError('nested tree node must be an object')
            name = _normalize_name(node.get('name', ''))
            if not name:
                raise ValueError('nested tree node name can not be empty')
            if name in children_map:
                raise ValueError(f'duplicated node in nested tree: {name}')
            explanation = _normalize_name(node.get('explanation', ''))
            if explanation:
                explanations[name] = explanation
            raw_children = node.get('children', [])
            if not isinstance(raw_children, list):
                raise ValueError(f'children must be list for nested node: {name}')
            out_children: List[str] = []
            for child in raw_children:
                if isinstance(child, str):
                    cname = _normalize_name(child)
                    out_children.append(cname)
                    children_map.setdefault(cname, [])
                elif isinstance(child, dict):
                    cname = walk(child)
                    out_children.append(cname)
                else:
                    raise ValueError(f'invalid child type in nested node: {name}')
            children_map[name] = _dedup_keep_order(out_children)
            return name

        root = walk(tree_obj)
        children_map = TreeUtils.normalize_tree(children_map)
        return root, children_map, explanations
