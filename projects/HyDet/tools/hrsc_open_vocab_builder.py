import itertools
import json
import os
import random
import xml.etree.ElementTree as ET
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, Tuple


@dataclass(frozen=True)
class HRSCClassInfo:
    class_id: str
    parent_id: Optional[str]
    eng_name: str
    short_name: str


def _norm_id(raw: str) -> str:
    return str(raw).strip()


def _class_name(class_id: str) -> str:
    return f'class_{_norm_id(class_id)}'


def _prompt_name(class_id: str, info: Optional[HRSCClassInfo]) -> str:
    """Return a stable human-readable HRSC category name for text prompts."""
    raw = ''
    if info is not None:
        raw = (info.eng_name or info.short_name or '').strip()
    if raw:
        return raw.replace('_', ' ').strip()
    # HRSC xml ids in this project are long raw ids while public mmrotate
    # ids are compact. The fallback keeps prompts meaningful if sysdata lacks
    # English names.
    fallback = {
        '000001': 'ship',
        '100000001': 'aircraft carrier',
        '100000002': 'warcraft',
        '100000003': 'merchant ship',
        '100000004': 'Nimitz aircraft carrier',
        '100000005': 'Enterprise aircraft carrier',
        '100000006': 'Arleigh Burke destroyer',
        '100000007': 'Whidbey Island amphibious ship',
        '100000008': 'Oliver Hazard Perry frigate',
        '100000009': 'San Antonio amphibious transport dock',
        '100000010': 'Ticonderoga cruiser',
        '100000011': 'Kitty Hawk aircraft carrier',
        '100000012': 'Kuznetsov aircraft carrier',
        '100000013': 'Abukuma escort ship',
        '100000015': 'Austin amphibious transport dock',
        '100000016': 'Tarawa amphibious assault ship',
        '100000017': 'Blue Ridge command ship',
        '100000018': 'container ship',
        '100000019': 'Oxo merchant ship',
        '100000020': 'car carrier',
        '100000022': 'hovercraft',
        '100000024': 'yacht',
        '100000025': 'container cargo ship',
        '100000026': 'cruise ship',
        '100000027': 'submarine',
        '100000028': 'lute vessel',
        '100000029': 'medical ship',
        '100000030': 'large car carrier',
        '100000032': 'Midway aircraft carrier',
    }
    return fallback.get(_norm_id(class_id), _class_name(class_id).replace('_', ' '))


def _node_prompt(display_name: str) -> str:
    article = 'an' if display_name[:1].lower() in {'a', 'e', 'i', 'o', 'u'} else 'a'
    return f'a remote sensing image of {article} {display_name}'


def _read_ids(txt_path: Path) -> List[str]:
    if not txt_path.exists():
        return []
    out = []
    for line in txt_path.read_text(encoding='utf-8').splitlines():
        line = line.strip()
        if line:
            out.append(line)
    return out


def _parse_sysdata(sysdata_xml: Path) -> Dict[str, HRSCClassInfo]:
    root = ET.parse(str(sysdata_xml)).getroot()
    out: Dict[str, HRSCClassInfo] = {}
    for cls_node in root.findall('./HRSC_Classes/HRSC_Class'):
        class_id = _norm_id(cls_node.findtext('Class_ID', default=''))
        if not class_id:
            continue
        parent_id = _norm_id(cls_node.findtext('HRS_Class_ID', default=''))
        parent_id = parent_id if parent_id else None
        eng_name = cls_node.findtext('Class_EngName', default='').strip()
        short_name = cls_node.findtext('Class_ShortName', default='').strip()
        out[class_id] = HRSCClassInfo(
            class_id=class_id,
            parent_id=parent_id,
            eng_name=eng_name,
            short_name=short_name,
        )
    if not out:
        raise ValueError(f'no class definitions found in {sysdata_xml}')
    return out


def _parse_annotation_labels(ann_path: Path) -> List[str]:
    labels: List[str] = []
    root = ET.parse(str(ann_path)).getroot()
    for obj in root.findall('./HRSC_Objects/HRSC_Object'):
        cid = _norm_id(obj.findtext('Class_ID', default=''))
        if cid:
            labels.append(cid)
    return labels


def _build_image_index(
    ann_dir: Path,
    train_ids: Set[str],
    test_ids: Set[str],
) -> Tuple[List[Dict], Dict[str, int], Dict[str, int], Set[str]]:
    images: List[Dict] = []
    class_count_total: Dict[str, int] = defaultdict(int)
    class_count_train: Dict[str, int] = defaultdict(int)
    used_class_ids: Set[str] = set()

    for ann_file in sorted(ann_dir.glob('*.xml')):
        image_id = ann_file.stem
        if not image_id.isdigit():
            continue
        if image_id in test_ids:
            split = 'test'
        elif image_id in train_ids:
            split = 'train'
        else:
            split = 'train'
        labels = _parse_annotation_labels(ann_file)
        for cid in labels:
            class_count_total[cid] += 1
            if split == 'train':
                class_count_train[cid] += 1
            used_class_ids.add(cid)
        images.append({
            'image_id': image_id,
            'split': split,
            'labels': [_class_name(cid) for cid in labels],
        })
    return images, dict(class_count_total), dict(class_count_train), used_class_ids


def _choose_unseen_classes(
    leaf_ids: Sequence[str],
    train_count_by_class: Dict[str, int],
    unseen_class_ratio: float,
    unseen_instance_ratio: float,
    seed: int,
) -> Set[str]:
    rng = random.Random(seed)
    leaf_ids = sorted(set(leaf_ids))
    total_cls = len(leaf_ids)
    unseen_n = max(1, int(round(total_cls * unseen_class_ratio)))
    unseen_n = min(unseen_n, max(1, total_cls - 1))
    total_train_instances = sum(int(train_count_by_class.get(cid, 0)) for cid in leaf_ids)
    if total_train_instances <= 0:
        rng.shuffle(leaf_ids)
        return set(leaf_ids[:unseen_n])

    best_combo: Optional[Tuple[str, ...]] = None
    best_gap: Optional[float] = None
    best_unseen_instances = -1
    for combo in itertools.combinations(leaf_ids, unseen_n):
        unseen_instances = sum(int(train_count_by_class.get(cid, 0)) for cid in combo)
        ratio = unseen_instances / float(total_train_instances)
        gap = abs(ratio - unseen_instance_ratio)
        if best_gap is None or gap < best_gap - 1e-12:
            best_gap = gap
            best_combo = combo
            best_unseen_instances = unseen_instances
        elif abs(gap - best_gap) <= 1e-12 and unseen_instances > best_unseen_instances:
            best_combo = combo
            best_unseen_instances = unseen_instances
    if best_combo is None:
        rng.shuffle(leaf_ids)
        best_combo = tuple(leaf_ids[:unseen_n])
    return set(best_combo)


def _save_txt(path: Path, lines: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = '\n'.join(lines)
    if text:
        text += '\n'
    path.write_text(text, encoding='utf-8')


def _save_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf-8')


def _resolve_image_path(all_images_dir: Path, image_id: str) -> str:
    for ext in ('.bmp', '.jpg', '.jpeg', '.png', '.tif', '.tiff'):
        candidate = all_images_dir / f'{image_id}{ext}'
        if candidate.exists():
            return str(candidate)
    return str(all_images_dir / f'{image_id}.bmp')


def _build_tree_artifacts(
    class_defs: Dict[str, HRSCClassInfo],
    leaf_ids: Sequence[str],
    seen_ids: Set[str],
    unseen_ids: Set[str],
) -> Dict[str, object]:
    leaf_ids = sorted(set(leaf_ids))
    kept_ids: Set[str] = set()
    for cid in leaf_ids:
        cur = cid
        while cur and cur not in kept_ids:
            kept_ids.add(cur)
            parent = class_defs.get(cur).parent_id if cur in class_defs else None
            cur = parent if parent in class_defs else None

    children_map: Dict[str, List[str]] = defaultdict(list)
    children_map['entity'] = []
    parent_map: Dict[str, Optional[str]] = {'entity': None}
    node_expl: Dict[str, str] = {'entity': 'all remote sensing objects'}
    node_display: Dict[str, str] = {'entity': 'remote sensing object'}
    for cid in sorted(kept_ids):
        node = _class_name(cid)
        info = class_defs.get(cid)
        parent_cid = info.parent_id if info is not None else None
        if parent_cid in kept_ids:
            parent_node = _class_name(parent_cid)
        else:
            parent_node = 'entity'
        parent_map[node] = parent_node
        children_map[parent_node].append(node)
        children_map.setdefault(node, [])
        expl = info.eng_name if info and info.eng_name else (info.short_name if info else '')
        node_expl[node] = expl
        node_display[node] = _prompt_name(cid, info)

    for p in list(children_map.keys()):
        children_map[p] = sorted(set(children_map[p]))

    depth: Dict[str, int] = {'entity': 0}
    queue = ['entity']
    while queue:
        cur = queue.pop(0)
        for ch in children_map.get(cur, []):
            if ch not in depth:
                depth[ch] = depth[cur] + 1
                queue.append(ch)

    all_nodes_topo = sorted(depth.keys(), key=lambda x: (depth[x], x))
    leaf_nodes = {_class_name(cid) for cid in leaf_ids}

    tree_validated_nodes = []
    node_meta = {}
    for node in all_nodes_topo:
        # 语义树中的叶节点以“是否无子节点”为准；即便某类在检测标签中出现，
        # 若它存在下位类，也应被视为内部节点，避免父子语义冲突。
        is_leaf = len(children_map.get(node, [])) == 0
        meta = {
            'parent': parent_map.get(node),
            'children': children_map.get(node, []),
            'depth': depth.get(node, 0),
            'is_leaf': is_leaf,
            'is_real_class': node in leaf_nodes,
            'path': [],
            'explanation': node_expl.get(node, ''),
            'display_name': node_display.get(node, node.replace('_', ' ')),
            'prompt': _node_prompt(node_display.get(node, node.replace('_', ' '))),
        }
        path = []
        cur = node
        while cur is not None:
            path.append(cur)
            cur = parent_map.get(cur)
        meta['path'] = list(reversed(path))
        node_meta[node] = meta
        tree_validated_nodes.append({
            'name': node,
            'parent': meta['parent'],
            'children': meta['children'],
            'is_leaf': meta['is_leaf'],
            'is_real_class': node in leaf_nodes,
            'depth': meta['depth'],
            'explanation': meta['explanation'],
            'display_name': node_display.get(node, node.replace('_', ' ')),
            'prompt': _node_prompt(node_display.get(node, node.replace('_', ' '))),
        })

    tree_validated = {'root': 'entity', 'nodes': tree_validated_nodes}
    tree_with_split_roles = {'root': 'entity', 'nodes': []}
    for n in tree_validated_nodes:
        role = 'internal'
        if n['is_leaf']:
            cid = n['name'][len('class_'):]
            role = 'seen' if cid in seen_ids else ('unseen' if cid in unseen_ids else 'leaf')
        item = dict(n)
        item['split_role'] = role
        tree_with_split_roles['nodes'].append(item)

    return {
        'tree_validated': tree_validated,
        'parent_map': parent_map,
        'node_meta': node_meta,
        'tree_with_split_roles': tree_with_split_roles,
        'class_names_all_nodes': all_nodes_topo,
    }


def _ensure_symlink(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.is_symlink():
        cur = os.readlink(str(dst))
        if os.path.abspath(cur) == str(src.resolve()):
            return
        dst.unlink()
    elif dst.exists():
        return
    os.symlink(str(src.resolve()), str(dst))


def ensure_hrsc_open_vocab_layout(
    source_root: str = '/root/autodl-tmp/HRSC2016-master/HRSC2016-master',
    target_root: str = '/root/autodl-tmp/data/hrsc_hier',
    repo_data_link: Optional[str] = None,
    seed: int = 3407,
    unseen_class_ratio: float = 0.2,
    unseen_instance_ratio: float = 0.15,
    force_rebuild: bool = False,
) -> Dict[str, object]:
    src = Path(source_root)
    dst = Path(target_root)
    if not src.exists():
        raise FileNotFoundError(f'HRSC source root not found: {source_root}')

    split_main = dst / 'ImageSets' / 'Main'
    meta_dir = dst / 'meta'
    marker = meta_dir / 'open_vocab_split_report.json'
    if marker.exists() and not force_rebuild:
        report = json.loads(marker.read_text(encoding='utf-8'))
        if repo_data_link:
            _ensure_symlink(dst, Path(repo_data_link))
        return report

    full_data_src = src / 'FullDataSet'
    ann_dir = full_data_src / 'Annotations'
    sysdata_xml = full_data_src / 'sysdata.xml'
    if not ann_dir.exists():
        raise FileNotFoundError(f'annotation dir not found: {ann_dir}')
    if not sysdata_xml.exists():
        raise FileNotFoundError(f'sysdata.xml not found: {sysdata_xml}')

    train_ids = set(_read_ids(src / 'ImageSets' / 'trainval.txt'))
    test_ids = set(_read_ids(src / 'ImageSets' / 'test.txt'))

    class_defs = _parse_sysdata(sysdata_xml)
    images, class_count_total, class_count_train, used_ids = _build_image_index(ann_dir, train_ids, test_ids)
    leaf_ids = sorted(used_ids)
    unseen_ids = _choose_unseen_classes(
        leaf_ids=leaf_ids,
        train_count_by_class=class_count_train,
        unseen_class_ratio=unseen_class_ratio,
        unseen_instance_ratio=unseen_instance_ratio,
        seed=seed,
    )
    seen_ids = set(leaf_ids) - unseen_ids

    class_names_leaf = [_class_name(cid) for cid in leaf_ids]
    name_to_idx = {name: idx for idx, name in enumerate(class_names_leaf)}
    seen_leaf_ids = sorted([name_to_idx[_class_name(cid)] for cid in seen_ids])
    unseen_leaf_ids = sorted([name_to_idx[_class_name(cid)] for cid in unseen_ids])

    train_labeled: List[str] = []
    train_unlabeled: List[str] = []
    test_all: List[str] = []
    test_seen: List[str] = []
    test_unseen: List[str] = []
    seen_name_set = {_class_name(x) for x in seen_ids}
    unseen_name_set = {_class_name(x) for x in unseen_ids}
    for rec in images:
        image_id = rec['image_id']
        labels = set(rec['labels'])
        split = rec['split']
        if split == 'train':
            if labels and labels.issubset(seen_name_set):
                train_labeled.append(image_id)
            else:
                train_unlabeled.append(image_id)
        else:
            test_all.append(image_id)
            if labels & seen_name_set:
                test_seen.append(image_id)
            if labels & unseen_name_set:
                test_unseen.append(image_id)

    tree_artifacts = _build_tree_artifacts(
        class_defs=class_defs,
        leaf_ids=leaf_ids,
        seen_ids=seen_ids,
        unseen_ids=unseen_ids,
    )

    _save_txt(split_main / 'hrsc_base_train.txt', sorted(set(train_labeled)))
    _save_txt(split_main / 'hrsc_train_unlabeled.txt', sorted(set(train_unlabeled)))
    _save_txt(split_main / 'hrsc_test_all.txt', sorted(set(test_all)))
    _save_txt(split_main / 'hrsc_test_base_train.txt', sorted(set(test_seen)))
    _save_txt(split_main / 'hrsc_test_base_test_only.txt', [])
    _save_txt(split_main / 'hrsc_test_novel.txt', sorted(set(test_unseen)))
    all_images_dir = full_data_src / 'AllImages'
    init_image_paths = [_resolve_image_path(all_images_dir, iid) for iid in sorted(set(train_unlabeled))]
    _save_txt(split_main / 'hrsc_initialize.txt', init_image_paths)

    _save_txt(meta_dir / 'class_names_leaf.txt', class_names_leaf)
    _save_txt(meta_dir / 'class_names_all_nodes.txt', tree_artifacts['class_names_all_nodes'])
    _save_json(meta_dir / 'base_train_leaf_ids.json', seen_leaf_ids)
    _save_json(meta_dir / 'base_test_only_leaf_ids.json', [])
    _save_json(meta_dir / 'novel_leaf_ids.json', unseen_leaf_ids)
    _save_json(meta_dir / 'class_freq_total.json', class_count_total)
    _save_json(meta_dir / 'class_freq_train.json', class_count_train)
    _save_json(meta_dir / 'image_index.json', {'images': images})
    _save_json(meta_dir / 'tree_validated.json', tree_artifacts['tree_validated'])
    _save_json(meta_dir / 'parent_map.json', tree_artifacts['parent_map'])
    _save_json(meta_dir / 'node_meta.json', tree_artifacts['node_meta'])
    _save_json(meta_dir / 'tree_with_split_roles.json', tree_artifacts['tree_with_split_roles'])

    total_train_instances = sum(class_count_train.get(cid, 0) for cid in leaf_ids)
    unseen_train_instances = sum(class_count_train.get(cid, 0) for cid in unseen_ids)
    report = {
        'seed': seed,
        'source_root': str(src),
        'target_root': str(dst),
        'leaf_class_count': len(leaf_ids),
        'seen_class_count': len(seen_ids),
        'unseen_class_count': len(unseen_ids),
        'seen_unseen_class_ratio': (
            float(len(seen_ids)) / max(float(len(unseen_ids)), 1.0)
        ),
        'train_instance_total': int(total_train_instances),
        'train_instance_unseen': int(unseen_train_instances),
        'train_instance_unseen_ratio': (
            float(unseen_train_instances) / max(float(total_train_instances), 1.0)
        ),
        'train_instance_seen_ratio': (
            float(total_train_instances - unseen_train_instances) / max(float(total_train_instances), 1.0)
        ),
        'seen_classes': sorted([_class_name(x) for x in seen_ids]),
        'unseen_classes': sorted([_class_name(x) for x in unseen_ids]),
        'split_counts': {
            'train_labeled': len(set(train_labeled)),
            'train_unlabeled': len(set(train_unlabeled)),
            'test_all': len(set(test_all)),
            'test_base_train': len(set(test_seen)),
            'test_novel': len(set(test_unseen)),
        },
    }
    _save_json(marker, report)

    _ensure_symlink(full_data_src, dst / 'FullDataSet')
    if repo_data_link:
        _ensure_symlink(dst, Path(repo_data_link))
    return report
