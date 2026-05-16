import argparse
import json
import shutil
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch

from hrsc_open_vocab_builder import (
    _build_tree_artifacts,
    _class_name,
    _parse_annotation_labels,
    _parse_sysdata,
    _read_ids,
    _save_json,
    _save_txt,
    ensure_hrsc_open_vocab_layout,
)


def _read_lines(path: Path) -> List[str]:
    return [x.strip() for x in path.read_text(encoding='utf-8').splitlines() if x.strip()]


def _load_tree_prompts(meta_dir: Path) -> Dict[str, str]:
    tree = json.loads((meta_dir / 'tree_validated.json').read_text(encoding='utf-8'))
    prompts = {}
    for node in tree.get('nodes', []):
        name = str(node['name'])
        prompt = str(node.get('prompt') or f"a remote sensing image of a {name.replace('_', ' ')}")
        prompts[name] = prompt
    return prompts


def _encode_texts(model_path: Path, texts: List[str], batch_size: int) -> np.ndarray:
    import open_clip

    model_name = 'RN50'
    model, _, _ = open_clip.create_model_and_transforms(model_name)
    tokenizer = open_clip.get_tokenizer(model_name)
    ckpt = torch.load(str(model_path), map_location='cpu')
    if isinstance(ckpt, dict) and 'state_dict' in ckpt:
        ckpt = ckpt['state_dict']
    msg = model.load_state_dict(ckpt, strict=False)
    print(f'loaded RemoteCLIP: {msg}')
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device).eval()

    features = []
    with torch.no_grad():
        for start in range(0, len(texts), batch_size):
            batch = texts[start:start + batch_size]
            tokens = tokenizer(batch).to(device)
            with torch.cuda.amp.autocast(enabled=device == 'cuda'):
                feat = model.encode_text(tokens)
                feat = feat / feat.norm(dim=-1, keepdim=True).clamp(min=1e-6)
            features.append(feat.detach().cpu().float())
    return torch.cat(features, dim=0).numpy()


def _copy_meta_files(meta_dir: Path, resource_dir: Path) -> None:
    resource_dir.mkdir(parents=True, exist_ok=True)
    for src in meta_dir.iterdir():
        if src.is_file() and src.suffix in {'.txt', '.json', '.npy'}:
            shutil.copy2(src, resource_dir / src.name)


def _preserve_fixed_class_split(source_root: Path, target_root: Path, fixed_meta_dir: Path) -> None:
    """Restore a user-provided class split while rebuilding tree/prompts.

    This keeps the experiment's base/novel protocol stable and only refreshes
    semantic resources. It is used for the existing 29-class HRSC split.
    """
    fixed_leaf = fixed_meta_dir / 'class_names_leaf.txt'
    fixed_base = fixed_meta_dir / 'base_train_leaf_ids.json'
    fixed_base_test = fixed_meta_dir / 'base_test_only_leaf_ids.json'
    fixed_novel = fixed_meta_dir / 'novel_leaf_ids.json'
    if not (fixed_leaf.is_file() and fixed_base.is_file() and fixed_novel.is_file()):
        return

    meta_dir = target_root / 'meta'
    split_main = target_root / 'ImageSets' / 'Main'
    leaf_names = _read_lines(fixed_leaf)
    leaf_ids = [name[len('class_'):] for name in leaf_names if name.startswith('class_')]
    base_ids_idx = json.loads(fixed_base.read_text(encoding='utf-8'))
    base_test_idx = json.loads(fixed_base_test.read_text(encoding='utf-8')) if fixed_base_test.is_file() else []
    novel_idx = json.loads(fixed_novel.read_text(encoding='utf-8'))
    base_names = {leaf_names[int(i)] for i in base_ids_idx}
    base_test_names = {leaf_names[int(i)] for i in base_test_idx}
    novel_names = {leaf_names[int(i)] for i in novel_idx}

    ann_dir = source_root / 'FullDataSet' / 'Annotations'
    train_ids = set(_read_ids(source_root / 'ImageSets' / 'trainval.txt'))
    test_ids = set(_read_ids(source_root / 'ImageSets' / 'test.txt'))
    train_labeled: List[str] = []
    test_all: List[str] = []
    test_base: List[str] = []
    test_base_only: List[str] = []
    test_novel: List[str] = []
    class_count_total: Dict[str, int] = {}
    class_count_train: Dict[str, int] = {}
    image_index = []
    for ann_file in sorted(ann_dir.glob('*.xml')):
        image_id = ann_file.stem
        labels = [_class_name(cid) for cid in _parse_annotation_labels(ann_file)]
        label_set = set(labels)
        for name in labels:
            raw = name[len('class_'):]
            class_count_total[raw] = class_count_total.get(raw, 0) + 1
            if image_id in train_ids:
                class_count_train[raw] = class_count_train.get(raw, 0) + 1
        split = 'test' if image_id in test_ids else 'train'
        image_index.append({'image_id': image_id, 'split': split, 'labels': labels})
        if split == 'train':
            if label_set and label_set.issubset(base_names):
                train_labeled.append(image_id)
        else:
            test_all.append(image_id)
            if label_set & base_names:
                test_base.append(image_id)
            if label_set & base_test_names:
                test_base_only.append(image_id)
            if label_set & novel_names:
                test_novel.append(image_id)

    class_defs = _parse_sysdata(source_root / 'FullDataSet' / 'sysdata.xml')
    tree_artifacts = _build_tree_artifacts(
        class_defs=class_defs,
        leaf_ids=leaf_ids,
        seen_ids={leaf_names[int(i)][len('class_'):] for i in base_ids_idx},
        unseen_ids={leaf_names[int(i)][len('class_'):] for i in novel_idx},
    )

    _save_txt(split_main / 'hrsc_base_train.txt', sorted(set(train_labeled)))
    _save_txt(split_main / 'hrsc_train_unlabeled.txt', [])
    _save_txt(split_main / 'hrsc_test_all.txt', sorted(set(test_all)))
    _save_txt(split_main / 'hrsc_test_base_train.txt', sorted(set(test_base)))
    _save_txt(split_main / 'hrsc_test_base_test_only.txt', sorted(set(test_base_only)))
    _save_txt(split_main / 'hrsc_test_novel.txt', sorted(set(test_novel)))
    _save_txt(meta_dir / 'class_names_leaf.txt', leaf_names)
    _save_txt(meta_dir / 'class_names_all_nodes.txt', tree_artifacts['class_names_all_nodes'])
    _save_json(meta_dir / 'base_train_leaf_ids.json', [int(x) for x in base_ids_idx])
    _save_json(meta_dir / 'base_test_only_leaf_ids.json', [int(x) for x in base_test_idx])
    _save_json(meta_dir / 'novel_leaf_ids.json', [int(x) for x in novel_idx])
    _save_json(meta_dir / 'class_freq_total.json', class_count_total)
    _save_json(meta_dir / 'class_freq_train.json', class_count_train)
    _save_json(meta_dir / 'image_index.json', {'images': image_index})
    _save_json(meta_dir / 'tree_validated.json', tree_artifacts['tree_validated'])
    _save_json(meta_dir / 'parent_map.json', tree_artifacts['parent_map'])
    _save_json(meta_dir / 'node_meta.json', tree_artifacts['node_meta'])
    _save_json(meta_dir / 'tree_with_split_roles.json', tree_artifacts['tree_with_split_roles'])


def main() -> None:
    parser = argparse.ArgumentParser(description='Prepare HRSC HyDet tree and RemoteCLIP text banks.')
    parser.add_argument('--source-root', default='data/HRSC2016_raw')
    parser.add_argument('--target-root', default='data/HRSC2016')
    parser.add_argument('--repo-data-link', default='data/HRSC2016')
    parser.add_argument('--resource-dir', default='projects/HyDet/resources/hrsc_hier')
    parser.add_argument('--model-path', default='checkpoints/RemoteCLIP-RN50.pt')
    parser.add_argument('--seed', type=int, default=3407)
    parser.add_argument('--force-rebuild', action='store_true')
    parser.add_argument('--skip-embeddings', action='store_true')
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--fixed-meta-dir', default='')
    args = parser.parse_args()

    report = ensure_hrsc_open_vocab_layout(
        source_root=args.source_root,
        target_root=args.target_root,
        repo_data_link=args.repo_data_link,
        seed=args.seed,
        force_rebuild=args.force_rebuild,
    )

    target_root = Path(args.target_root)
    meta_dir = target_root / 'meta'
    resource_dir = Path(args.resource_dir)
    if args.fixed_meta_dir:
        _preserve_fixed_class_split(Path(args.source_root), target_root, Path(args.fixed_meta_dir))
    _copy_meta_files(meta_dir, resource_dir)

    leaf_names = _read_lines(meta_dir / 'class_names_leaf.txt')
    all_node_names = _read_lines(meta_dir / 'class_names_all_nodes.txt')
    prompts = _load_tree_prompts(meta_dir)
    leaf_prompts = [prompts.get(name, f"a remote sensing image of a {name.replace('_', ' ')}") for name in leaf_names]
    all_prompts = [prompts.get(name, f"a remote sensing image of a {name.replace('_', ' ')}") for name in all_node_names]

    prompt_map = {
        'template': 'a remote sensing image of {article} {display_name}',
        'leaf_prompts': dict(zip(leaf_names, leaf_prompts)),
        'all_node_prompts': dict(zip(all_node_names, all_prompts)),
    }
    for out_dir in (meta_dir, resource_dir):
        (out_dir / 'prompt_map.json').write_text(json.dumps(prompt_map, ensure_ascii=False, indent=2), encoding='utf-8')

    if not args.skip_embeddings:
        model_path = Path(args.model_path)
        if not model_path.is_file():
            raise FileNotFoundError(f'RemoteCLIP checkpoint not found: {model_path}')
        leaf = _encode_texts(model_path, leaf_prompts, args.batch_size)
        all_nodes = _encode_texts(model_path, all_prompts, args.batch_size)
        bg = leaf.mean(axis=0, keepdims=True)
        bg = bg / np.linalg.norm(bg, axis=1, keepdims=True).clip(min=1e-6)
        leaf_bg = np.concatenate([leaf, bg.astype(leaf.dtype)], axis=0)
        for out_dir in (meta_dir, resource_dir):
            np.save(out_dir / 'leaf_text_embeddings_euc.npy', leaf)
            np.save(out_dir / 'leaf_text_embeddings_with_bg_euc.npy', leaf_bg)
            np.save(out_dir / 'all_nodes_text_embeddings_euc.npy', all_nodes)
            np.save(out_dir / 'hrsc_all_leaf_embeddings.npy', leaf)
            np.save(out_dir / 'hrsc_all_leaf_plus_bg_embeddings.npy', leaf_bg)

    print(json.dumps({
        'leaf_class_count': report.get('leaf_class_count'),
        'seen_class_count': report.get('seen_class_count'),
        'unseen_class_count': report.get('unseen_class_count'),
        'resource_dir': str(resource_dir),
        'target_root': str(target_root),
        'embeddings': not args.skip_embeddings,
    }, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
