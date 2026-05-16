import argparse
import hashlib
import json
import shutil
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Sequence, Set, Tuple
from zipfile import ZipFile

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from projects.HyDet.hydet.hyper.utils.split_utils import ImageRecord, SplitUtils


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Prepare FAIR1M open-vocabulary data (annfiles + 8:2 parent-aware split).')
    parser.add_argument('--source-root', default='data/FAIR1M_raw')
    parser.add_argument('--target-root', default='data/FAIR1M')
    parser.add_argument('--resource-root', default='projects/HyDet/resources/fair1m_hier')
    parser.add_argument('--seed', type=int, default=3407)
    parser.add_argument('--test-ratio', type=float, default=0.25)
    parser.add_argument('--extract-images', action='store_true')
    parser.add_argument('--force-rebuild', action='store_true')
    parser.add_argument('--repo-data-link', default='data/FAIR1M')
    return parser.parse_args()


def _norm_name(x: str) -> str:
    return ' '.join(str(x).strip().lower().split())


def _load_leaf_classes(resource_root: Path) -> Tuple[List[str], Dict[str, str], Dict[str, int]]:
    names = [x.strip() for x in (resource_root / 'class_names_leaf.txt').read_text(encoding='utf-8').splitlines() if x.strip()]
    norm_to_name = {_norm_name(x): x for x in names}
    name_to_id = {x: i for i, x in enumerate(names)}
    return names, norm_to_name, name_to_id


def _write_dior_obb_xml(xml_path: Path, image_id: str, width: int, height: int, objects: Sequence[Tuple[str, List[float]]]) -> None:
    root = ET.Element('annotation')
    ET.SubElement(root, 'folder').text = 'images'
    ET.SubElement(root, 'filename').text = f'{image_id}.jpg'
    size = ET.SubElement(root, 'size')
    ET.SubElement(size, 'width').text = str(int(width))
    ET.SubElement(size, 'height').text = str(int(height))
    ET.SubElement(size, 'depth').text = '3'
    for cls, pts in objects:
        obj = ET.SubElement(root, 'object')
        ET.SubElement(obj, 'name').text = cls.lower()
        ET.SubElement(obj, 'difficult').text = '0'
        rb = ET.SubElement(obj, 'robndbox')
        ET.SubElement(rb, 'x_left_top').text = f'{pts[0]:.6f}'
        ET.SubElement(rb, 'y_left_top').text = f'{pts[1]:.6f}'
        ET.SubElement(rb, 'x_right_top').text = f'{pts[2]:.6f}'
        ET.SubElement(rb, 'y_right_top').text = f'{pts[3]:.6f}'
        ET.SubElement(rb, 'x_right_bottom').text = f'{pts[4]:.6f}'
        ET.SubElement(rb, 'y_right_bottom').text = f'{pts[5]:.6f}'
        ET.SubElement(rb, 'x_left_bottom').text = f'{pts[6]:.6f}'
        ET.SubElement(rb, 'y_left_bottom').text = f'{pts[7]:.6f}'
    tree = ET.ElementTree(root)
    tree.write(str(xml_path), encoding='utf-8', xml_declaration=False)


def _parse_label_zip(
    label_zip: Path,
    ann_dir: Path,
    norm_to_name: Dict[str, str]
) -> Tuple[List[Tuple[str, List[str]]], Dict[str, int]]:
    records: List[Tuple[str, List[str]]] = []
    class_freq: Dict[str, int] = {}
    with ZipFile(label_zip, 'r') as zf:
        xml_names = [n for n in zf.namelist() if n.lower().endswith('.xml')]
        for name in xml_names:
            xml_text = zf.read(name).decode('utf-8', 'ignore')
            root = ET.fromstring(xml_text)
            filename = root.findtext('./source/filename', default='').strip()
            image_id = Path(filename).stem if filename else Path(name).stem
            width = int(float(root.findtext('./size/width', default='1000')))
            height = int(float(root.findtext('./size/height', default='1000')))
            labels: List[str] = []
            objects: List[Tuple[str, List[float]]] = []
            for obj in root.findall('./objects/object'):
                raw_cls = obj.findtext('./possibleresult/name', default='').strip()
                cls = norm_to_name.get(_norm_name(raw_cls))
                if not cls:
                    continue
                pts = [p.text.strip() for p in obj.findall('./points/point') if p.text]
                # FAIR1M xml usually repeats first point as the 5th point.
                if len(pts) < 4:
                    continue
                coords: List[str] = []
                for p in pts[:4]:
                    xy = p.split(',')
                    if len(xy) != 2:
                        coords = []
                        break
                    coords.extend([float(xy[0]), float(xy[1])])
                if len(coords) != 8:
                    continue
                objects.append((cls, coords))
                labels.append(cls)
                class_freq[cls] = class_freq.get(cls, 0) + 1
            _write_dior_obb_xml(ann_dir / f'{image_id}.xml', image_id=image_id, width=width, height=height, objects=objects)
            records.append((image_id, labels))
    return records, class_freq


def _extract_images(image_zip: Path, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    with ZipFile(image_zip, 'r') as zf:
        for member in zf.namelist():
            if member.lower().endswith(('.tif', '.tiff', '.png', '.jpg', '.jpeg')):
                zf.extract(member, path=out_dir)
    # flatten one nested level if needed
    for sub in list(out_dir.iterdir()):
        if sub.is_dir():
            for f in sub.iterdir():
                target = out_dir / f.name
                if not target.exists():
                    f.replace(target)
            sub.rmdir()
    # DIORDataset in this repo uses `<img_id>.jpg`; create lightweight aliases.
    for tif in out_dir.glob('*.tif'):
        jpg = out_dir / f'{tif.stem}.jpg'
        if jpg.exists():
            continue
        try:
            jpg.symlink_to(tif.name)
        except OSError:
            shutil.copy2(tif, jpg)


def _pick_first_existing(base_dir: Path, candidates: Sequence[str]) -> Path:
    for name in candidates:
        p = base_dir / name
        if p.exists():
            return p
    raise FileNotFoundError(f'none of {candidates} exists under {base_dir}')


def _stable_test_split(image_id: str, seed: int, ratio: float) -> str:
    h = hashlib.md5(f'{seed}:{image_id}'.encode('utf-8')).hexdigest()
    v = int(h[:8], 16) / 0xFFFFFFFF
    return 'test' if v < ratio else 'train'


def _rebalance_novel_ratio(
    leaf_nodes: Sequence[str],
    parent_map: Dict[str, str],
    class_freq: Dict[str, int],
    base_train: Set[str],
    base_test_only: Set[str],
    novel: Set[str],
    target_ratio: float = 0.2
) -> Tuple[Set[str], Set[str], Set[str]]:
    total = len(leaf_nodes)
    target_novel = max(1, int(round(total * target_ratio)))
    base_train = set(base_train)
    base_test_only = set(base_test_only)
    novel = set(novel)
    if len(novel) > target_novel:
        # move high-frequency novel leaves to base_test_only first.
        movable = sorted(novel, key=lambda x: class_freq.get(x, 0), reverse=True)
        for name in movable:
            if len(novel) <= target_novel:
                break
            novel.remove(name)
            base_test_only.add(name)
    elif len(novel) < target_novel:
        # move low-frequency base_test_only leaves to novel when parent still has seen leaves.
        movable = sorted(base_test_only, key=lambda x: class_freq.get(x, 0))
        for name in movable:
            if len(novel) >= target_novel:
                break
            parent = parent_map.get(name)
            if parent is None:
                continue
            siblings = [x for x in leaf_nodes if parent_map.get(x) == parent]
            seen_cnt = sum(1 for x in siblings if x in base_train or x in base_test_only)
            if seen_cnt <= 1:
                continue
            base_test_only.remove(name)
            novel.add(name)
    return base_train, base_test_only, novel


def main() -> None:
    args = parse_args()
    source_root = Path(args.source_root)
    target_root = Path(args.target_root)
    resource_root = Path(args.resource_root)
    meta_dir = target_root / 'meta'
    split_main = target_root / 'ImageSets' / 'Main'
    ann_dir = target_root / 'annfiles'
    img_dir = target_root / 'images'
    compat_root = target_root / 'castdet_compat'

    if args.force_rebuild and target_root.exists():
        shutil.rmtree(target_root)
    meta_dir.mkdir(parents=True, exist_ok=True)
    split_main.mkdir(parents=True, exist_ok=True)
    ann_dir.mkdir(parents=True, exist_ok=True)
    img_dir.mkdir(parents=True, exist_ok=True)

    names, norm_to_name, name_to_id = _load_leaf_classes(resource_root)
    all_records: List[Tuple[str, List[str]]] = []
    class_freq_train: Dict[str, int] = {k: 0 for k in names}

    for part in ('part1', 'part2'):
        part_dir = source_root / 'train' / part
        label_zip = _pick_first_existing(part_dir, ('labelXml.zip', 'labelXmls.zip'))
        part_records, part_freq = _parse_label_zip(label_zip, ann_dir, norm_to_name)
        all_records.extend(part_records)
        for k, v in part_freq.items():
            class_freq_train[k] = class_freq_train.get(k, 0) + int(v)
        if args.extract_images:
            image_zips = [p for p in part_dir.glob('images*.zip')]
            if not image_zips:
                image_zips = [part_dir / 'images.zip']
            for image_zip in image_zips:
                if not image_zip.exists():
                    continue
                _extract_images(image_zip, img_dir)

    image_records: List[ImageRecord] = []
    for image_id, labels in all_records:
        image_records.append(ImageRecord(
            image_id=image_id,
            split=_stable_test_split(image_id, args.seed, args.test_ratio),
            labels=labels,
            image_path=str((img_dir / f'{image_id}.tif').resolve()) if args.extract_images else None
        ))

    id_to_name = {i: n for n, i in name_to_id.items()}
    SplitUtils.save_json(str(meta_dir / 'class_freq_train.json'), {
        'class_freq': [{'class_id': i, 'class_name': id_to_name[i], 'count': int(class_freq_train.get(id_to_name[i], 0))} for i in sorted(id_to_name.keys())],
        'id_to_name': {str(i): id_to_name[i] for i in sorted(id_to_name.keys())},
    })
    SplitUtils.save_json(str(meta_dir / 'image_index.json'), {
        'images': [
            {'image_id': r.image_id, 'split': r.split, 'labels': r.labels, 'image_path': r.image_path}
            for r in image_records
        ]
    })

    root, children_map, parent_map, leaf_nodes = SplitUtils.load_tree_and_parent(
        str(resource_root / 'tree_validated.json'),
        str(resource_root / 'parent_map.json'))
    if root != 'entity':
        raise ValueError(f'fair1m tree root must be entity, got {root}')
    class_freq, _, _ = SplitUtils.load_class_freq(str(meta_dir / 'class_freq_train.json'))
    groups = SplitUtils.build_leaf_groups(
        dataset_name='FAIR1M',
        leaf_nodes=sorted(set(leaf_nodes)),
        parent_map=parent_map,
        children_map=children_map,
        class_freq=class_freq,
        min_base_train_per_parent=1,
        rare_leaf_threshold=20,
        seed=args.seed)
    base_train = set(groups['base_train'])
    base_test_only = set(groups['base_test_only'])
    novel = set(groups['novel'])
    base_train, base_test_only, novel = _rebalance_novel_ratio(
        leaf_nodes=sorted(set(leaf_nodes)),
        parent_map=parent_map,
        class_freq=class_freq,
        base_train=base_train,
        base_test_only=base_test_only,
        novel=novel,
        target_ratio=0.2)
    parent_to_leaves = {p: sorted([c for c in cs if c in leaf_nodes]) for p, cs in children_map.items() if cs}
    SplitUtils.validate_leaf_groups(parent_to_leaves, set(leaf_nodes), base_train, base_test_only, novel)

    splits = SplitUtils.split_images(image_records, base_train, base_test_only, novel)
    SplitUtils.save_json(str(resource_root / 'base_train_leaf_ids.json'), sorted([name_to_id[x] for x in base_train]))
    SplitUtils.save_json(str(resource_root / 'base_test_only_leaf_ids.json'), sorted([name_to_id[x] for x in base_test_only]))
    SplitUtils.save_json(str(resource_root / 'novel_leaf_ids.json'), sorted([name_to_id[x] for x in novel]))

    SplitUtils.save_txt(str(split_main / 'fair1m_base_train.txt'), SplitUtils.ids_from_records(splits['train_labeled']))
    SplitUtils.save_txt(str(split_main / 'fair1m_train_unlabeled.txt'), SplitUtils.ids_from_records(splits['train_unlabeled']))
    SplitUtils.save_txt(str(split_main / 'fair1m_test_all.txt'), SplitUtils.ids_from_records(splits['test_all']))
    SplitUtils.save_txt(str(split_main / 'fair1m_test_base_train.txt'), SplitUtils.ids_from_records(splits['test_base_train']))
    SplitUtils.save_txt(str(split_main / 'fair1m_test_base_test_only.txt'), SplitUtils.ids_from_records(splits['test_base_test_only']))
    SplitUtils.save_txt(str(split_main / 'fair1m_test_novel.txt'), SplitUtils.ids_from_records(splits['test_novel']))
    SplitUtils.save_txt(str(split_main / 'fair1m_initialize.txt'), SplitUtils.initialize_lines(splits['train_unlabeled'], use_image_path=False))

    report = {
        'seed': args.seed,
        'leaf_class_count': len(names),
        'base_train_count': len(base_train),
        'base_test_only_count': len(base_test_only),
        'novel_count': len(novel),
        'base_ratio': (len(base_train) + len(base_test_only)) / max(len(names), 1),
        'novel_ratio': len(novel) / max(len(names), 1),
        'image_split_counts': {k: len(v) for k, v in splits.items()},
    }
    SplitUtils.save_json(str(meta_dir / 'open_vocab_split_report.json'), report)

    # Build CastDet-compatible directory
    (compat_root / 'ImageSets' / 'Main').mkdir(parents=True, exist_ok=True)
    for src in split_main.glob('*.txt'):
        shutil.copy2(src, compat_root / 'ImageSets' / 'Main' / src.name)
    if not (compat_root / 'annfiles').exists():
        (compat_root / 'annfiles').symlink_to(ann_dir, target_is_directory=True)
    if args.extract_images and not (compat_root / 'images').exists():
        (compat_root / 'images').symlink_to(img_dir, target_is_directory=True)

    repo_link = Path(args.repo_data_link)
    repo_link.parent.mkdir(parents=True, exist_ok=True)
    if repo_link.exists() or repo_link.is_symlink():
        repo_link.unlink()
    repo_link.symlink_to(compat_root, target_is_directory=True)

    print(f'Prepared FAIR1M open-vocab data at: {target_root}')
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
