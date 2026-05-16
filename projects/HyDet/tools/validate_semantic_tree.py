import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Set, Tuple

from projects.HyDet.hydet.hyper.structures.semantic_tree import SemanticTree
from projects.HyDet.hydet.hyper.utils.tree_utils import TreeUtils


@dataclass
class ValidationIssue:
    rule: str
    message: str

    def as_dict(self) -> Dict[str, str]:
        return {'rule': self.rule, 'message': self.message}


def _norm(name: str) -> str:
    return str(name).strip().lower()


def _token_set(name: str) -> Set[str]:
    text = _norm(name)
    for ch in ['-', '/', ',', '(', ')', '.']:
        text = text.replace(ch, ' ')
    return {x for x in text.split() if x}


def _reverse_pair_detected(parent: str, child: str) -> bool:
    parent_tokens = _token_set(parent)
    child_tokens = _token_set(child)

    super_tokens = {
        'ship', 'vessel', 'boat', 'aircraft', 'airplane', 'plane', 'vehicle',
        'facility', 'port', 'harbor', 'field', 'court', 'airport',
        'military', 'civilian', 'fixed', 'rotary', 'wing'
    }
    subtype_tokens = {
        'destroyer', 'carrier', 'frigate', 'submarine', 'tanker', 'container',
        'fishing', 'passenger', 'fighter', 'bomber', 'airliner', 'helicopter',
        'uav', 'drone', 'tank', 'truck', 'bus', 'car', 'tractor'
    }

    parent_is_subtype = len(parent_tokens & subtype_tokens) > 0
    child_is_supertype = len(child_tokens & super_tokens) > 0
    if parent_is_subtype and child_is_supertype:
        return True

    if _norm(parent) in {'military ship', 'warship'} and _norm(child) in {'ship', 'vessel'}:
        return True
    if _norm(parent) in {'fighter', 'bomber', 'airliner'} and _norm(child) in {'aircraft', 'airplane'}:
        return True
    if _norm(parent) in {'tank', 'truck', 'bus'} and _norm(child) in {'vehicle'}:
        return True
    return False


def _collect_reachable(children_map: Mapping[str, Sequence[str]], root: str) -> Set[str]:
    visited: Set[str] = set()
    stack: List[str] = [root]
    while stack:
        node = stack.pop()
        if node in visited:
            continue
        visited.add(node)
        stack.extend(children_map.get(node, []))
    return visited


def _validate_tree(
    root: str,
    children_map: Dict[str, List[str]],
    leaf_names: List[str]
) -> Tuple[List[ValidationIssue], Dict[str, Optional[str]], SemanticTree]:
    issues: List[ValidationIssue] = []
    leaf_set = set(leaf_names)

    try:
        parent_map = TreeUtils.children_map_to_parent_map(children_map)
    except ValueError as exc:
        issues.append(ValidationIssue(rule='R4', message=str(exc)))
        parent_map = {}

    roots = TreeUtils.infer_roots(children_map)
    if len(roots) != 1:
        issues.append(ValidationIssue(rule='R2', message=f'root count must be 1, got {len(roots)}: {roots}'))
    if root != 'entity':
        issues.append(ValidationIssue(rule='R2', message=f'root must be "entity", got "{root}"'))
    if len(roots) == 1 and roots[0] != 'entity':
        issues.append(ValidationIssue(rule='R2', message=f'unique inferred root is "{roots[0]}", expected "entity"'))

    missing_leaf = sorted(leaf_set - set(children_map.keys()))
    if missing_leaf:
        issues.append(ValidationIssue(rule='R1', message=f'missing leaf nodes: {missing_leaf}'))

    for leaf in sorted(leaf_set & set(children_map.keys())):
        if len(children_map.get(leaf, [])) > 0:
            issues.append(ValidationIssue(rule='R6', message=f'leaf node used as parent: {leaf}'))

    for parent, children in children_map.items():
        if _norm(parent) == '':
            issues.append(ValidationIssue(rule='R8', message='empty node name detected'))
        seen_sibling = set()
        for child in children:
            if _norm(parent) == _norm(child):
                issues.append(ValidationIssue(rule='R8', message=f'parent and child have same name: {parent}'))
            key = _norm(child)
            if key in seen_sibling:
                issues.append(ValidationIssue(rule='R8', message=f'duplicated sibling under parent {parent}: {child}'))
            seen_sibling.add(key)
            if _reverse_pair_detected(parent, child):
                issues.append(ValidationIssue(rule='R7', message=f'possible parent-child reversal: {parent} -> {child}'))

    for node, children in children_map.items():
        if node not in leaf_set and node != 'entity' and len(children) == 0:
            issues.append(ValidationIssue(rule='R9', message=f'non-leaf node has empty children: {node}'))

    try:
        topo_nodes = TreeUtils.topological_nodes(children_map, root)
    except ValueError as exc:
        issues.append(ValidationIssue(rule='R3', message=str(exc)))
        topo_nodes = []

    if topo_nodes:
        reachable = _collect_reachable(children_map, root)
        unreachable = sorted(set(topo_nodes) - reachable)
        if unreachable:
            issues.append(ValidationIssue(rule='R5', message=f'unreachable nodes from root: {unreachable}'))

    if issues:
        raise ValueError('\n'.join([f'[{it.rule}] {it.message}' for it in issues]))

    tree = SemanticTree(tree_data=children_map, root=root)
    return issues, parent_map, tree


def _build_node_meta(
    tree: SemanticTree,
    node_explanations: Mapping[str, str]
) -> Dict[str, Dict[str, Any]]:
    node_meta: Dict[str, Dict[str, Any]] = {}
    for node in tree.all_nodes():
        node_meta[node] = {
            'parent': tree.parent(node),
            'children': tree.children(node),
            'depth': tree.depth(node),
            'is_leaf': tree.is_leaf(node),
            'path': tree.path_from_root(node),
            'explanation': node_explanations.get(node, ''),
        }
    return node_meta


def _build_tree_validated(
    tree: SemanticTree,
    node_meta: Mapping[str, Mapping[str, Any]]
) -> Dict[str, Any]:
    nodes = []
    for node in tree.topological_nodes():
        meta = node_meta[node]
        nodes.append({
            'name': node,
            'parent': meta['parent'],
            'children': meta['children'],
            'is_leaf': meta['is_leaf'],
            'depth': meta['depth'],
            'explanation': meta['explanation'],
        })
    return {'root': tree.root, 'nodes': nodes}


def _build_review_report(
    tree: SemanticTree,
    node_meta: Mapping[str, Mapping[str, Any]]
) -> Dict[str, Any]:
    leaf_reports: List[Dict[str, Any]] = []
    for leaf in tree.leaf_nodes():
        parent = tree.parent(leaf)
        siblings = tree.siblings(leaf)
        leaf_reports.append({
            'leaf': leaf,
            'path': tree.path_from_root(leaf),
            'siblings': siblings,
            'parent': parent,
            'parent_explanation': '' if parent is None else node_meta[parent].get('explanation', ''),
        })
    return {
        'summary': {
            'root': tree.root,
            'node_count': tree.stats().node_count,
            'edge_count': tree.stats().edge_count,
            'leaf_count': tree.stats().leaf_count,
            'internal_count': tree.stats().internal_count,
        },
        'leaf_reports': leaf_reports
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Validate semantic tree generated by AI')
    parser.add_argument('--leaf-path', required=True, help='path to class_names_leaf.txt')
    parser.add_argument('--raw-tree-path', required=True, help='path to tree_raw_gpt.json')
    parser.add_argument('--out-dir', required=True, help='output directory')
    parser.add_argument('--dataset-name', default='unknown', help='dataset name for logging')
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    leaf_names = TreeUtils.read_leaf_names(args.leaf_path)
    raw = TreeUtils.load_json(args.raw_tree_path)
    parsed = TreeUtils.parse_raw_tree(raw)

    _, parent_map, tree = _validate_tree(parsed.root, parsed.children_map, leaf_names)
    node_meta = _build_node_meta(tree, parsed.node_explanations)
    tree_validated = _build_tree_validated(tree, node_meta)
    report = _build_review_report(tree, node_meta)
    report['dataset_name'] = args.dataset_name

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    TreeUtils.save_json(tree_validated, str(out_dir / 'tree_validated.json'))
    TreeUtils.save_json(parent_map, str(out_dir / 'parent_map.json'))
    TreeUtils.save_json(node_meta, str(out_dir / 'node_meta.json'))
    TreeUtils.save_json(report, str(out_dir / 'tree_review_report.json'))

    print(f'validated tree saved to: {out_dir}')


if __name__ == '__main__':
    main()
