import argparse
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set

from projects.HyDet.hydet.hyper.utils.split_utils import ImageRecord, SplitUtils


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Build parent-aware split for HRSC2016/FAIR1M')
    parser.add_argument('--dataset-name', required=True, choices=['HRSC2016', 'FAIR1M'])
    parser.add_argument('--tree-validated', required=True, help='path to tree_validated.json')
    parser.add_argument('--parent-map', required=True, help='path to parent_map.json')
    parser.add_argument('--class-freq', required=True, help='path to class frequency json')
    parser.add_argument('--image-index', required=True, help='path to image/annotation index json or jsonl')
    parser.add_argument('--out-dir', required=True, help='output directory')
    parser.add_argument('--seed', type=int, default=3407, help='split seed')
    parser.add_argument('--min-base-train-per-parent', type=int, default=1)
    parser.add_argument('--rare-leaf-threshold', type=int, default=20)
    parser.add_argument('--initialize-use-image-path', action='store_true')
    return parser.parse_args()


def _leaf_sets_as_ids(
    base_train: Set[str],
    base_test_only: Set[str],
    novel: Set[str],
    name_to_id: Dict[str, int]
) -> Dict[str, List[int]]:
    return {
        'base_train': sorted([int(name_to_id[x]) for x in base_train]),
        'base_test_only': sorted([int(name_to_id[x]) for x in base_test_only]),
        'novel': sorted([int(name_to_id[x]) for x in novel]),
    }


def _build_parent_to_leaves(leaf_nodes: Sequence[str], parent_map: Dict[str, Optional[str]]) -> Dict[str, List[str]]:
    out: Dict[str, List[str]] = {}
    for leaf in leaf_nodes:
        parent = parent_map.get(leaf)
        if parent is None:
            raise ValueError(f'leaf has no parent: {leaf}')
        out.setdefault(parent, []).append(leaf)
    for p in out:
        out[p] = sorted(out[p])
    return out


def _ensure_train_has_no_unseen_labels(
    records: Sequence[ImageRecord],
    base_train: Set[str]
) -> None:
    for rec in records:
        labels = set(rec.labels)
        if labels and not labels.issubset(base_train):
            raise ValueError(f'train labeled contains non-base-train labels: {rec.image_id}')


def _build_split_report(
    dataset_name: str,
    seed: int,
    base_train: Set[str],
    base_test_only: Set[str],
    novel: Set[str],
    parent_stats: Dict[str, Dict[str, Any]],
    image_splits: Dict[str, List[ImageRecord]]
) -> Dict[str, Any]:
    return {
        'dataset_name': dataset_name,
        'seed': seed,
        'leaf_group_counts': {
            'base_train': len(base_train),
            'base_test_only': len(base_test_only),
            'novel': len(novel),
        },
        'leaf_groups': {
            'base_train': sorted(base_train),
            'base_test_only': sorted(base_test_only),
            'novel': sorted(novel),
        },
        'parent_stats': parent_stats,
        'image_split_counts': {
            'train_labeled': len(image_splits['train_labeled']),
            'train_unlabeled': len(image_splits['train_unlabeled']),
            'test_all': len(image_splits['test_all']),
            'test_base_train': len(image_splits['test_base_train']),
            'test_base_test_only': len(image_splits['test_base_test_only']),
            'test_novel': len(image_splits['test_novel']),
        }
    }


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    dataset_name = args.dataset_name
    lower_name = dataset_name.lower()
    if lower_name == 'hrsc2016':
        prefix = 'hrsc'
    elif lower_name == 'fair1m':
        prefix = 'fair1m'
    else:
        raise ValueError('dataset-name must be HRSC2016 or FAIR1M')

    root, children_map, parent_map, leaf_nodes = SplitUtils.load_tree_and_parent(
        args.tree_validated,
        args.parent_map
    )
    if root != 'entity':
        raise ValueError(f'root must be entity, got {root}')

    class_freq, name_to_id, id_to_name = SplitUtils.load_class_freq(args.class_freq)
    leaf_nodes = sorted(set(leaf_nodes))
    missing_freq = [x for x in leaf_nodes if x not in class_freq]
    if missing_freq:
        raise ValueError(f'missing class frequency for leaves: {missing_freq[:20]}')
    missing_id = [x for x in leaf_nodes if x not in name_to_id]
    if missing_id:
        raise ValueError(f'missing class id for leaves: {missing_id[:20]}')

    result = SplitUtils.build_leaf_groups(
        dataset_name=dataset_name,
        leaf_nodes=leaf_nodes,
        parent_map=parent_map,
        children_map=children_map,
        class_freq=class_freq,
        min_base_train_per_parent=args.min_base_train_per_parent,
        rare_leaf_threshold=args.rare_leaf_threshold,
        seed=args.seed
    )
    base_train = set(result['base_train'])
    base_test_only = set(result['base_test_only'])
    novel = set(result['novel'])

    parent_to_leaves = _build_parent_to_leaves(leaf_nodes, parent_map)
    SplitUtils.validate_leaf_groups(parent_to_leaves, set(leaf_nodes), base_train, base_test_only, novel)

    image_records = SplitUtils.load_image_index(args.image_index, id_to_name=id_to_name)
    image_splits = SplitUtils.split_images(image_records, base_train, base_test_only, novel)
    _ensure_train_has_no_unseen_labels(image_splits['train_labeled'], base_train)

    base_ids = _leaf_sets_as_ids(base_train, base_test_only, novel, name_to_id)
    SplitUtils.save_json(str(out_dir / 'base_train_leaf_ids.json'), base_ids['base_train'])
    SplitUtils.save_json(str(out_dir / 'base_test_only_leaf_ids.json'), base_ids['base_test_only'])
    SplitUtils.save_json(str(out_dir / 'novel_leaf_ids.json'), base_ids['novel'])

    SplitUtils.save_txt(str(out_dir / f'{prefix}_base_train.txt'), SplitUtils.ids_from_records(image_splits['train_labeled']))
    SplitUtils.save_txt(str(out_dir / f'{prefix}_train_unlabeled.txt'), SplitUtils.ids_from_records(image_splits['train_unlabeled']))
    SplitUtils.save_txt(str(out_dir / f'{prefix}_test_all.txt'), SplitUtils.ids_from_records(image_splits['test_all']))
    SplitUtils.save_txt(str(out_dir / f'{prefix}_test_base_train.txt'), SplitUtils.ids_from_records(image_splits['test_base_train']))
    SplitUtils.save_txt(str(out_dir / f'{prefix}_test_base_test_only.txt'), SplitUtils.ids_from_records(image_splits['test_base_test_only']))
    SplitUtils.save_txt(str(out_dir / f'{prefix}_test_novel.txt'), SplitUtils.ids_from_records(image_splits['test_novel']))

    init_lines = SplitUtils.initialize_lines(
        records=image_splits['train_unlabeled'],
        use_image_path=bool(args.initialize_use_image_path)
    )
    SplitUtils.save_txt(str(out_dir / f'{prefix}_initialize.txt'), init_lines)

    parent_stats = SplitUtils.parent_stats(parent_to_leaves, base_train, base_test_only, novel)
    report = _build_split_report(
        dataset_name=dataset_name,
        seed=args.seed,
        base_train=base_train,
        base_test_only=base_test_only,
        novel=novel,
        parent_stats=parent_stats,
        image_splits=image_splits
    )
    SplitUtils.save_json(str(out_dir / 'split_report.json'), report)
    print(f'parent-aware split saved to: {out_dir}')


if __name__ == '__main__':
    main()
