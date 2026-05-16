import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Set, Tuple

from mmrotate.registry import TASK_UTILS
from ..modules import MODULE_SWITCH_KEYS, expand_module_switches, infer_module_switches

HYPER_PLUGIN_SWITCH_KEYS = (
    'use_hier_tree',
    'use_hyper_branch',
    'cache_hyp_bank',
    'use_hier_losses',
    'use_hyp_contrast',
    'use_logit_fusion',
    'use_hier_queue_filter',
    'use_tp_projection',
)


@dataclass
class HyperPluginConfig:
    module_tree_builder: bool = False
    module_hyperbolic_contrast: bool = False
    module_hierarchy_anchor: bool = False
    use_hier_tree: bool = False
    use_hyper_branch: bool = False
    cache_hyp_bank: bool = False
    use_hier_losses: bool = False
    use_hyp_contrast: bool = False
    use_logit_fusion: bool = False
    use_hier_queue_filter: bool = False
    use_tp_projection: bool = False

    def to_dict(self) -> Dict[str, bool]:
        module_dict = {key: getattr(self, key) for key in MODULE_SWITCH_KEYS}
        component_dict = {key: getattr(self, key) for key in HYPER_PLUGIN_SWITCH_KEYS}
        out = {}
        out.update(module_dict)
        out.update(component_dict)
        return out


@dataclass
class ImageRecord:
    image_id: str
    split: str
    labels: List[str]
    image_path: Optional[str] = None


@TASK_UTILS.register_module()
class HyperPluginSwitch:
    def __init__(self, **kwargs) -> None:
        self.config = self.parse(kwargs)

    @staticmethod
    def parse(cfg: Optional[Mapping[str, Any]] = None) -> HyperPluginConfig:
        cfg = cfg or {}
        parsed = expand_module_switches(cfg, HYPER_PLUGIN_SWITCH_KEYS)
        module_flags = infer_module_switches(parsed)
        for key in MODULE_SWITCH_KEYS:
            if key in cfg:
                module_flags[key] = bool(cfg[key])
        parsed.update(module_flags)
        return HyperPluginConfig(**parsed)

    def as_dict(self) -> Dict[str, bool]:
        return self.config.to_dict()


@TASK_UTILS.register_module()
class SplitUtils:
    def __init__(self, plugin_cfg: Optional[Mapping[str, Any]] = None) -> None:
        self.plugin_switch = HyperPluginSwitch.parse(plugin_cfg)

    @staticmethod
    def load_json(path: str) -> Any:
        file_path = Path(path)
        if not file_path.exists():
            raise FileNotFoundError(f'file not found: {path}')
        return json.loads(file_path.read_text(encoding='utf-8'))

    @staticmethod
    def save_json(path: str, data: Any) -> None:
        file_path = Path(path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf-8')

    @staticmethod
    def save_txt(path: str, lines: Sequence[str]) -> None:
        file_path = Path(path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text('\n'.join(lines) + ('\n' if lines else ''), encoding='utf-8')

    @staticmethod
    def normalize_name(name: str) -> str:
        return str(name).strip()

    @staticmethod
    def norm_key(name: str) -> str:
        return SplitUtils.normalize_name(name).lower()

    @staticmethod
    def load_tree_and_parent(
        tree_validated_path: str,
        parent_map_path: str
    ) -> Tuple[str, Dict[str, List[str]], Dict[str, Optional[str]], Set[str]]:
        tree_obj = SplitUtils.load_json(tree_validated_path)
        parent_obj = SplitUtils.load_json(parent_map_path)

        if not isinstance(tree_obj, dict):
            raise ValueError('tree_validated.json must be an object')
        if not isinstance(parent_obj, dict):
            raise ValueError('parent_map.json must be an object')

        root = SplitUtils.normalize_name(tree_obj.get('root', ''))
        if not root:
            raise ValueError('tree_validated.json missing root')
        nodes = tree_obj.get('nodes', [])
        if not isinstance(nodes, list):
            raise ValueError('tree_validated.json nodes must be a list')

        children_map: Dict[str, List[str]] = {}
        leaf_nodes: Set[str] = set()
        for entry in nodes:
            if not isinstance(entry, dict):
                raise ValueError('each node entry must be object')
            name = SplitUtils.normalize_name(entry.get('name', ''))
            if not name:
                raise ValueError('node name can not be empty')
            children = entry.get('children', [])
            if not isinstance(children, list):
                raise ValueError(f'children must be list for node: {name}')
            children_map[name] = [SplitUtils.normalize_name(x) for x in children]
            if bool(entry.get('is_leaf', False)):
                leaf_nodes.add(name)
        for parent, children in list(children_map.items()):
            dedup: List[str] = []
            seen = set()
            for child in children:
                if not child:
                    continue
                if child.lower() in seen:
                    continue
                seen.add(child.lower())
                dedup.append(child)
            children_map[parent] = dedup
            for child in dedup:
                children_map.setdefault(child, [])
        if root not in children_map:
            children_map[root] = []

        parent_map: Dict[str, Optional[str]] = {}
        for node, parent in parent_obj.items():
            n = SplitUtils.normalize_name(node)
            p = SplitUtils.normalize_name(parent) if parent is not None else None
            parent_map[n] = p

        for child, parent in parent_map.items():
            if parent is not None and child not in children_map.get(parent, []):
                raise ValueError(f'inconsistent tree and parent_map: {parent} -> {child} missing in tree')
        return root, children_map, parent_map, leaf_nodes

    @staticmethod
    def load_class_freq(
        class_freq_path: str
    ) -> Tuple[Dict[str, int], Dict[str, int], Dict[int, str]]:
        raw = SplitUtils.load_json(class_freq_path)
        name_to_count: Dict[str, int] = {}
        name_to_id: Dict[str, int] = {}
        id_to_name: Dict[int, str] = {}

        def _put(name: str, count: int, class_id: Optional[int]) -> None:
            n = SplitUtils.normalize_name(name)
            if not n:
                return
            c = int(count)
            name_to_count[n] = c
            if class_id is not None:
                cid = int(class_id)
                name_to_id[n] = cid
                id_to_name[cid] = n

        if isinstance(raw, dict) and all(isinstance(v, int) for v in raw.values()):
            for name, count in raw.items():
                _put(name, int(count), None)
        elif isinstance(raw, dict) and 'class_freq' in raw:
            class_freq = raw['class_freq']
            if isinstance(class_freq, dict):
                for name, count in class_freq.items():
                    _put(name, int(count), None)
            elif isinstance(class_freq, list):
                for item in class_freq:
                    if not isinstance(item, dict):
                        continue
                    _put(
                        item.get('class_name', item.get('name', '')),
                        int(item.get('count', 0)),
                        item.get('class_id', item.get('id', None))
                    )
            id2name = raw.get('id_to_name', {})
            if isinstance(id2name, dict):
                for k, v in id2name.items():
                    cid = int(k)
                    name = SplitUtils.normalize_name(v)
                    id_to_name[cid] = name
                    name_to_id[name] = cid
        elif isinstance(raw, list):
            for item in raw:
                if not isinstance(item, dict):
                    continue
                _put(
                    item.get('class_name', item.get('name', '')),
                    int(item.get('count', 0)),
                    item.get('class_id', item.get('id', None))
                )
        else:
            raise ValueError('unsupported class frequency format')

        if not name_to_count:
            raise ValueError('empty class frequency data')
        if not name_to_id:
            names_sorted = sorted(name_to_count.keys())
            name_to_id = {n: i for i, n in enumerate(names_sorted)}
            id_to_name = {i: n for n, i in name_to_id.items()}
        return name_to_count, name_to_id, id_to_name

    @staticmethod
    def load_image_index(
        image_index_path: str,
        id_to_name: Mapping[int, str]
    ) -> List[ImageRecord]:
        path = Path(image_index_path)
        if not path.exists():
            raise FileNotFoundError(f'image index not found: {image_index_path}')
        text = path.read_text(encoding='utf-8')
        raw: Any
        try:
            raw = json.loads(text)
        except json.JSONDecodeError:
            raw = []
            for line in text.splitlines():
                line = line.strip()
                if not line:
                    continue
                raw.append(json.loads(line))

        if isinstance(raw, dict):
            items = raw.get('images', [])
        elif isinstance(raw, list):
            items = raw
        else:
            raise ValueError('unsupported image index format')
        if not isinstance(items, list):
            raise ValueError('image index "images" must be list')

        records: List[ImageRecord] = []
        for item in items:
            if not isinstance(item, dict):
                continue
            image_id = SplitUtils.normalize_name(
                item.get('image_id', item.get('id', item.get('img_id', '')))
            )
            if not image_id:
                continue
            split = SplitUtils.normalize_name(item.get('split', 'train')).lower()
            if split not in {'train', 'test'}:
                split = 'train'

            raw_labels = item.get('labels', item.get('leaf_labels', []))
            labels: List[str] = []
            if isinstance(raw_labels, list):
                for label in raw_labels:
                    if isinstance(label, int):
                        if label in id_to_name:
                            labels.append(id_to_name[label])
                    elif isinstance(label, str):
                        labels.append(SplitUtils.normalize_name(label))
                    elif isinstance(label, dict):
                        if 'class_name' in label or 'name' in label:
                            labels.append(SplitUtils.normalize_name(label.get('class_name', label.get('name', ''))))
                        elif 'class_id' in label or 'id' in label:
                            lid = int(label.get('class_id', label.get('id')))
                            if lid in id_to_name:
                                labels.append(id_to_name[lid])
            labels = [x for x in labels if x]
            image_path = item.get('image_path', item.get('img_path', None))
            records.append(
                ImageRecord(
                    image_id=image_id,
                    split=split,
                    labels=labels,
                    image_path=SplitUtils.normalize_name(image_path) if image_path else None
                ))
        return records

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
    def build_leaf_groups(
        dataset_name: str,
        leaf_nodes: Sequence[str],
        parent_map: Mapping[str, Optional[str]],
        children_map: Mapping[str, Sequence[str]],
        class_freq: Mapping[str, int],
        min_base_train_per_parent: int,
        rare_leaf_threshold: int,
        seed: int
    ) -> Dict[str, Set[str]]:
        ds = SplitUtils.norm_key(dataset_name)
        if ds not in {'hrsc2016', 'fair1m'}:
            raise ValueError('dataset_name must be HRSC2016 or FAIR1M')
        rng = random.Random(seed)

        leaves = sorted(set(leaf_nodes))
        leaf_set = set(leaves)
        for leaf in leaves:
            if leaf not in parent_map or parent_map[leaf] is None:
                raise ValueError(f'leaf has no parent: {leaf}')

        parent_to_leaves: Dict[str, List[str]] = {}
        for leaf in leaves:
            parent = str(parent_map[leaf])
            parent_to_leaves.setdefault(parent, []).append(leaf)
        for parent in parent_to_leaves:
            parent_to_leaves[parent].sort()

        base_train: Set[str] = set()
        base_test_only: Set[str] = set()
        novel: Set[str] = set()

        rare_force_base = {leaf for leaf in leaves if int(class_freq.get(leaf, 0)) <= rare_leaf_threshold}
        base_train.update(rare_force_base)

        for parent in sorted(parent_to_leaves.keys()):
            p_leaves = parent_to_leaves[parent]
            n = len(p_leaves)
            if n == 0:
                continue

            forced_base = set([x for x in p_leaves if x in rare_force_base])
            if n == 1:
                base_train.add(p_leaves[0])
                continue

            base_min = max(1, min_base_train_per_parent)
            base_min = min(base_min, n)

            candidates = [x for x in p_leaves if x not in forced_base]
            candidates.sort(key=lambda x: (int(class_freq.get(x, 0)), x))
            grouped: Dict[int, List[str]] = {}
            for leaf in candidates:
                grouped.setdefault(int(class_freq.get(leaf, 0)), []).append(leaf)
            ranked: List[str] = []
            for freq in sorted(grouped.keys()):
                block = grouped[freq]
                rng.shuffle(block)
                ranked.extend(block)
            candidates = ranked

            max_bto = 0
            max_novel = 0
            if n == 2:
                max_bto = 1
                max_novel = 0
            elif 3 <= n <= 4:
                max_bto = 1
                max_novel = 1
            else:
                max_bto = int(math.floor(n * 0.2))
                if ds == 'hrsc2016':
                    max_novel = int(math.floor(n * 0.3))
                else:
                    max_novel = int(math.floor(n * 0.4))

            base_slots = max(base_min, len(forced_base))
            free_slots = max(0, n - base_slots)
            max_bto = min(max_bto, free_slots)
            free_slots_after_bto = max(0, free_slots - max_bto)
            max_novel = min(max_novel, free_slots_after_bto)

            chosen_bto = set(candidates[:max_bto])
            remain = [x for x in candidates if x not in chosen_bto]
            chosen_novel = set(remain[:max_novel])
            chosen_base = set(p_leaves) - chosen_bto - chosen_novel
            chosen_base.update(forced_base)

            if len(chosen_base) < base_min:
                need = base_min - len(chosen_base)
                pool = [x for x in p_leaves if x not in chosen_base]
                pool.sort(key=lambda x: (int(class_freq.get(x, 0)), x), reverse=True)
                for leaf in pool[:need]:
                    if leaf in chosen_bto:
                        chosen_bto.remove(leaf)
                    if leaf in chosen_novel:
                        chosen_novel.remove(leaf)
                    chosen_base.add(leaf)

            base_train.update(chosen_base)
            base_test_only.update(chosen_bto)
            novel.update(chosen_novel)

        if not base_test_only:
            candidate_parents = sorted(parent_to_leaves.keys(), key=lambda p: len(parent_to_leaves[p]), reverse=True)
            moved = False
            for parent in candidate_parents:
                p_leaves = parent_to_leaves[parent]
                if len(p_leaves) < 2:
                    continue
                movable = [x for x in p_leaves if x in base_train and x not in rare_force_base]
                if not movable:
                    continue
                movable.sort(key=lambda x: (int(class_freq.get(x, 0)), x))
                pick = movable[0]
                remain_base = [x for x in p_leaves if x in base_train and x != pick]
                if len(remain_base) < max(1, min_base_train_per_parent):
                    continue
                base_train.remove(pick)
                base_test_only.add(pick)
                moved = True
                break
            if not moved:
                raise ValueError('can not create base_test_only class under current constraints')

        top_level = list(children_map.get('entity', []))
        for top in top_level:
            sub_nodes = set(SplitUtils.descendants(children_map, top) + [top])
            sub_leaves = sorted(sub_nodes & leaf_set)
            if not sub_leaves:
                continue
            if any(x in base_train for x in sub_leaves):
                continue
            candidates = [x for x in sub_leaves if x in base_test_only or x in novel]
            if not candidates:
                continue
            candidates.sort(key=lambda x: (int(class_freq.get(x, 0)), x), reverse=True)
            pick = candidates[0]
            if pick in base_test_only:
                base_test_only.remove(pick)
            if pick in novel:
                novel.remove(pick)
            base_train.add(pick)

        SplitUtils.validate_leaf_groups(parent_to_leaves, set(leaves), base_train, base_test_only, novel)
        return {
            'base_train': base_train,
            'base_test_only': base_test_only,
            'novel': novel
        }

    @staticmethod
    def validate_leaf_groups(
        parent_to_leaves: Mapping[str, Sequence[str]],
        all_leaves: Set[str],
        base_train: Set[str],
        base_test_only: Set[str],
        novel: Set[str]
    ) -> None:
        if base_train & base_test_only:
            raise ValueError('base_train and base_test_only overlap')
        if base_train & novel:
            raise ValueError('base_train and novel overlap')
        if base_test_only & novel:
            raise ValueError('base_test_only and novel overlap')
        union = set(base_train) | set(base_test_only) | set(novel)
        if union != set(all_leaves):
            missing = sorted(set(all_leaves) - union)
            extra = sorted(union - set(all_leaves))
            raise ValueError(f'leaf partition mismatch, missing={missing}, extra={extra}')
        for parent, leaves in parent_to_leaves.items():
            leaves_set = set(leaves)
            if leaves_set and leaves_set.issubset(novel):
                raise ValueError(f'parent subtree fully novel is forbidden: {parent}')

    @staticmethod
    def parent_stats(
        parent_to_leaves: Mapping[str, Sequence[str]],
        base_train: Set[str],
        base_test_only: Set[str],
        novel: Set[str]
    ) -> Dict[str, Dict[str, Any]]:
        out: Dict[str, Dict[str, Any]] = {}
        for parent in sorted(parent_to_leaves.keys()):
            leaves = list(parent_to_leaves[parent])
            bt = [x for x in leaves if x in base_train]
            bto = [x for x in leaves if x in base_test_only]
            nv = [x for x in leaves if x in novel]
            out[parent] = {
                'total': len(leaves),
                'base_train': {'count': len(bt), 'leaves': bt},
                'base_test_only': {'count': len(bto), 'leaves': bto},
                'novel': {'count': len(nv), 'leaves': nv},
            }
        return out

    @staticmethod
    def split_images(
        records: Sequence[ImageRecord],
        base_train: Set[str],
        base_test_only: Set[str],
        novel: Set[str]
    ) -> Dict[str, List[ImageRecord]]:
        train_records = [r for r in records if r.split == 'train']
        test_records = [r for r in records if r.split == 'test']

        def _labels_set(r: ImageRecord) -> Set[str]:
            return set(r.labels)

        labeled_train: List[ImageRecord] = []
        train_unlabeled: List[ImageRecord] = []
        for rec in train_records:
            labels = _labels_set(rec)
            if not labels:
                train_unlabeled.append(rec)
                continue
            if labels.issubset(base_train):
                labeled_train.append(rec)
            else:
                train_unlabeled.append(rec)

        test_all = list(test_records)
        test_base_train = [r for r in test_records if _labels_set(r) & base_train]
        test_base_test_only = [r for r in test_records if _labels_set(r) & base_test_only]
        test_novel = [r for r in test_records if _labels_set(r) & novel]

        return {
            'train_labeled': labeled_train,
            'train_unlabeled': train_unlabeled,
            'test_all': test_all,
            'test_base_train': test_base_train,
            'test_base_test_only': test_base_test_only,
            'test_novel': test_novel,
        }

    @staticmethod
    def ids_from_records(records: Sequence[ImageRecord]) -> List[str]:
        return sorted({r.image_id for r in records})

    @staticmethod
    def initialize_lines(
        records: Sequence[ImageRecord],
        use_image_path: bool
    ) -> List[str]:
        out: List[str] = []
        for rec in records:
            if use_image_path and rec.image_path:
                out.append(rec.image_path)
            else:
                out.append(rec.image_id)
        return sorted(set(out))
