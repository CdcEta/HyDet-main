from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

import numpy as np
import torch
import json
from mmrotate.registry import TASK_UTILS

from ..utils import HyperPluginSwitch


def _read_name_list(path: str) -> List[str]:
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f'name list not found: {path}')
    names = [line.strip() for line in file_path.read_text(encoding='utf-8').splitlines() if line.strip()]
    if not names:
        raise ValueError(f'empty name list: {path}')
    return names


@TASK_UTILS.register_module()
class HyperTextBank:
    def __init__(
        self,
        leaf_text_embeddings_euc_path: str,
        all_nodes_text_embeddings_euc_path: str,
        tree_validated_path: str,
        parent_map_path: str,
        class_names_leaf_path: str,
        class_names_all_nodes_path: str,
        plugin_cfg: Optional[Mapping[str, Any]] = None,
        device: str = 'cpu'
    ) -> None:
        self.plugin_switch = HyperPluginSwitch.parse(plugin_cfg)
        self.use_hyper_branch = bool(self.plugin_switch.use_hyper_branch)
        self.cache_hyp_bank = bool(self.plugin_switch.cache_hyp_bank)
        self.device = torch.device(device)

        self.leaf_names = _read_name_list(class_names_leaf_path)
        self.all_node_names = _read_name_list(class_names_all_nodes_path)
        self.leaf_name_to_id = {name: idx for idx, name in enumerate(self.leaf_names)}
        self.node_name_to_id = {name: idx for idx, name in enumerate(self.all_node_names)}

        leaf_euc_np = np.load(leaf_text_embeddings_euc_path)
        all_euc_np = np.load(all_nodes_text_embeddings_euc_path)
        if leaf_euc_np.shape[0] != len(self.leaf_names):
            raise ValueError('leaf embedding row count mismatch class_names_leaf.txt')
        if all_euc_np.shape[0] != len(self.all_node_names):
            raise ValueError('all-node embedding row count mismatch class_names_all_nodes.txt')

        self.leaf_bank_euc = torch.from_numpy(leaf_euc_np).float().to(self.device)
        self.all_bank_euc = torch.from_numpy(all_euc_np).float().to(self.device)
        self.leaf_bank_hyp: Optional[torch.Tensor] = None
        self.all_bank_hyp: Optional[torch.Tensor] = None

        tree_obj = self._load_json(tree_validated_path)
        parent_obj = self._load_json(parent_map_path)
        if not isinstance(tree_obj, dict):
            raise ValueError('tree_validated.json must be object')
        if not isinstance(parent_obj, dict):
            raise ValueError('parent_map.json must be object')
        self.root = str(tree_obj.get('root', 'entity'))
        self.parent_map: Dict[str, Optional[str]] = {}
        for node, parent in parent_obj.items():
            key = str(node).strip()
            self.parent_map[key] = str(parent).strip() if parent is not None else None

        self.children_map: Dict[str, List[str]] = {}
        nodes = tree_obj.get('nodes', [])
        if not isinstance(nodes, list):
            raise ValueError('tree_validated.json nodes must be list')
        for item in nodes:
            if not isinstance(item, dict):
                continue
            name = str(item.get('name', '')).strip()
            if not name:
                continue
            children = item.get('children', [])
            if not isinstance(children, list):
                children = []
            self.children_map[name] = [str(x).strip() for x in children if str(x).strip()]
        for parent, children in list(self.children_map.items()):
            for child in children:
                self.children_map.setdefault(child, [])
        self.children_map.setdefault(self.root, [])

        for leaf in self.leaf_names:
            if leaf not in self.parent_map:
                raise ValueError(f'leaf missing in parent_map: {leaf}')
            if self.parent_map[leaf] is None:
                raise ValueError(f'leaf has null parent: {leaf}')

    @staticmethod
    def _load_json(path: str) -> Any:
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f'json file not found: {path}')
        return json.loads(p.read_text(encoding='utf-8'))

    def get_leaf_bank_euc(self) -> torch.Tensor:
        return self.leaf_bank_euc

    def get_all_bank_euc(self) -> torch.Tensor:
        return self.all_bank_euc

    def build_leaf_bank_hyp(self, mapper: torch.nn.Module) -> torch.Tensor:
        with torch.no_grad():
            out = mapper(self.leaf_bank_euc)
        self.leaf_bank_hyp = out.detach()
        return self.leaf_bank_hyp

    def build_all_bank_hyp(self, mapper: torch.nn.Module) -> torch.Tensor:
        with torch.no_grad():
            out = mapper(self.all_bank_euc)
        self.all_bank_hyp = out.detach()
        return self.all_bank_hyp

    def get_leaf_name(self, leaf_id: int) -> str:
        idx = int(leaf_id)
        if idx < 0 or idx >= len(self.leaf_names):
            raise IndexError(f'invalid leaf id: {leaf_id}')
        return self.leaf_names[idx]

    def get_node_name(self, node_id: int) -> str:
        idx = int(node_id)
        if idx < 0 or idx >= len(self.all_node_names):
            raise IndexError(f'invalid node id: {node_id}')
        return self.all_node_names[idx]

    def _ancestors_by_name(self, name: str) -> List[str]:
        out: List[str] = []
        cur = self.parent_map.get(name, None)
        while cur is not None:
            out.append(cur)
            cur = self.parent_map.get(cur, None)
        return out

    def get_ancestors_by_leaf_id(self, leaf_id: int) -> List[str]:
        leaf_name = self.get_leaf_name(leaf_id)
        return self._ancestors_by_name(leaf_name)

    def get_siblings_by_leaf_id(self, leaf_id: int) -> List[str]:
        leaf_name = self.get_leaf_name(leaf_id)
        parent = self.parent_map.get(leaf_name, None)
        if parent is None:
            return []
        siblings = [x for x in self.children_map.get(parent, []) if x != leaf_name]
        return siblings

    def get_leaf_to_node_path(self, leaf_id: int) -> List[str]:
        leaf_name = self.get_leaf_name(leaf_id)
        path = [leaf_name]
        cur = leaf_name
        while self.parent_map.get(cur, None) is not None:
            cur = str(self.parent_map[cur])
            path.append(cur)
        path.reverse()
        return path

    def export_cache(
        self,
        out_dir: str,
        leaf_file_name: str = 'leaf_text_embeddings_hyp.pt',
        all_file_name: str = 'all_nodes_text_embeddings_hyp.pt'
    ) -> Dict[str, str]:
        if self.leaf_bank_hyp is None or self.all_bank_hyp is None:
            raise ValueError('hyperbolic banks are not built')
        out_path = Path(out_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        leaf_path = out_path / leaf_file_name
        all_path = out_path / all_file_name
        torch.save(self.leaf_bank_hyp.detach().cpu(), leaf_path)
        torch.save(self.all_bank_hyp.detach().cpu(), all_path)
        return {'leaf': str(leaf_path), 'all_nodes': str(all_path)}


TextBank = HyperTextBank
