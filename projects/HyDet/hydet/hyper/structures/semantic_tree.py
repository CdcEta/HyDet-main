from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from mmrotate.registry import TASK_UTILS

from ..utils import HyperPluginSwitch, TreeUtils


@dataclass
class SemanticTreeStats:
    node_count: int
    edge_count: int
    leaf_count: int
    internal_count: int


@TASK_UTILS.register_module()
class SemanticTree:
    def __init__(
        self,
        tree_data: Optional[Mapping[str, Any]] = None,
        root: Optional[str] = None,
        parent_map: Optional[Mapping[str, Optional[str]]] = None,
        plugin_cfg: Optional[Mapping[str, Any]] = None
    ) -> None:
        self.plugin_switch = HyperPluginSwitch.parse(plugin_cfg)

        if parent_map is not None:
            children_map = TreeUtils.parent_map_to_children_map(parent_map)
            inferred_roots = TreeUtils.infer_roots(children_map)
            if root is None:
                if len(inferred_roots) != 1:
                    raise ValueError('can not infer unique root from parent_map')
                root = inferred_roots[0]
        else:
            children_map = TreeUtils.normalize_tree(dict(tree_data or {}))
            inferred_roots = TreeUtils.infer_roots(children_map)
            if root is None:
                if len(inferred_roots) != 1:
                    raise ValueError('can not infer unique root from tree_data')
                root = inferred_roots[0]

        self._root = str(root)
        self._children_map: Dict[str, List[str]] = {
            node: list(children) for node, children in children_map.items()
        }
        if self._root not in self._children_map:
            self._children_map[self._root] = []

        self._parent_map: Dict[str, Optional[str]] = TreeUtils.children_map_to_parent_map(self._children_map)
        self._topological_nodes: List[str] = TreeUtils.topological_nodes(self._children_map, self._root)
        self._depth_map: Dict[str, int] = self._build_depth_map()

    @property
    def root(self) -> str:
        return self._root

    def parent(self, node: str) -> Optional[str]:
        self._assert_node_exists(node)
        return self._parent_map[node]

    def children(self, node: str) -> List[str]:
        self._assert_node_exists(node)
        return list(self._children_map.get(node, []))

    def get_children(self, node: str) -> List[str]:
        return self.children(node)

    def ancestors(self, node: str) -> List[str]:
        self._assert_node_exists(node)
        out: List[str] = []
        cur = self._parent_map[node]
        while cur is not None:
            out.append(cur)
            cur = self._parent_map[cur]
        return out

    def siblings(self, node: str) -> List[str]:
        self._assert_node_exists(node)
        p = self._parent_map[node]
        if p is None:
            return []
        return [x for x in self._children_map.get(p, []) if x != node]

    def descendants(self, node: str) -> List[str]:
        self._assert_node_exists(node)
        return TreeUtils.descendants(self._children_map, node)

    def is_leaf(self, node: str) -> bool:
        self._assert_node_exists(node)
        return len(self._children_map.get(node, [])) == 0

    def leaf_nodes(self) -> List[str]:
        return [n for n in self._topological_nodes if self.is_leaf(n)]

    def internal_nodes(self) -> List[str]:
        return [n for n in self._topological_nodes if not self.is_leaf(n)]

    def all_nodes(self) -> List[str]:
        return list(self._topological_nodes)

    def edge_list(self) -> List[Tuple[str, str]]:
        out: List[Tuple[str, str]] = []
        for parent in self._topological_nodes:
            for child in self._children_map.get(parent, []):
                out.append((parent, child))
        return out

    def topological_nodes(self) -> List[str]:
        return list(self._topological_nodes)

    def depth(self, node: str) -> int:
        self._assert_node_exists(node)
        return self._depth_map[node]

    def path_from_root(self, node: str) -> List[str]:
        self._assert_node_exists(node)
        path = [node]
        cur = node
        while self._parent_map[cur] is not None:
            cur = self._parent_map[cur]  # type: ignore[assignment]
            path.append(cur)
        path.reverse()
        return path

    def as_parent_map(self) -> Dict[str, Optional[str]]:
        return dict(self._parent_map)

    def as_children_map(self) -> Dict[str, List[str]]:
        return {k: list(v) for k, v in self._children_map.items()}

    def stats(self) -> SemanticTreeStats:
        nodes = len(self._topological_nodes)
        edges = len(self.edge_list())
        leaves = len(self.leaf_nodes())
        internals = nodes - leaves
        return SemanticTreeStats(
            node_count=nodes,
            edge_count=edges,
            leaf_count=leaves,
            internal_count=internals
        )

    def _build_depth_map(self) -> Dict[str, int]:
        depth_map: Dict[str, int] = {self._root: 0}
        queue = [self._root]
        idx = 0
        while idx < len(queue):
            parent = queue[idx]
            idx += 1
            d = depth_map[parent]
            for child in self._children_map.get(parent, []):
                depth_map[child] = d + 1
                queue.append(child)
        return depth_map

    def _assert_node_exists(self, node: str) -> None:
        if node not in self._children_map:
            raise KeyError(f'node not found: {node}')
