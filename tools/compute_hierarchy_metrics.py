import argparse
import json
from collections import defaultdict, deque
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from mmengine.fileio import load


def _to_tensor(x: Any) -> Optional[torch.Tensor]:
    if x is None:
        return None
    if hasattr(x, 'tensor'):
        x = x.tensor
    if isinstance(x, torch.Tensor):
        return x.detach().cpu()
    try:
        return torch.as_tensor(x).detach().cpu()
    except Exception:
        return None


def _instances(sample: Any, name: str) -> Any:
    if hasattr(sample, name):
        return getattr(sample, name)
    if isinstance(sample, dict):
        return sample.get(name)
    return None


def _field(inst: Any, name: str) -> Any:
    if inst is None:
        return None
    if hasattr(inst, name):
        return getattr(inst, name)
    if isinstance(inst, dict):
        return inst.get(name)
    return None


def _load_tree(tree_path: Path, leaf_path: Path) -> Tuple[List[str], Dict[str, Optional[str]], Dict[Tuple[str, str], int]]:
    leaves = [x.strip() for x in leaf_path.read_text(encoding='utf-8').splitlines() if x.strip()]
    tree = json.loads(tree_path.read_text(encoding='utf-8'))
    parent = {str(n['name']): n.get('parent') for n in tree.get('nodes', [])}
    graph: Dict[str, List[str]] = defaultdict(list)
    for node, par in parent.items():
        if par is not None:
            graph[node].append(par)
            graph[par].append(node)
    dist: Dict[Tuple[str, str], int] = {}
    for src in leaves:
        q = deque([(src, 0)])
        seen = {src}
        while q:
            cur, d = q.popleft()
            dist[(src, cur)] = d
            for nxt in graph.get(cur, []):
                if nxt not in seen:
                    seen.add(nxt)
                    q.append((nxt, d + 1))
    return leaves, parent, dist


def _parent_name(label: int, leaves: List[str], parent: Dict[str, Optional[str]]) -> Optional[str]:
    if label < 0 or label >= len(leaves):
        return None
    return parent.get(leaves[label])


def _rotated_iou(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    try:
        from mmcv.ops import box_iou_rotated
        return box_iou_rotated(a.float(), b.float())
    except Exception:
        # Fallback for malformed inputs: no matches rather than misleading HCE.
        return torch.zeros((a.shape[0], b.shape[0]), dtype=torch.float32)


def compute(samples: List[Any], tree_path: Path, leaf_path: Path, iou_thr: float, score_thr: float) -> Dict[str, Any]:
    leaves, parent, tree_dist = _load_tree(tree_path, leaf_path)
    matched = 0
    correct = 0
    hce_sum = 0.0
    cross_parent_wrong = 0
    wrong = 0
    sibling_pairs = 0
    sibling_correct = 0
    pred_count = 0
    confusion: Dict[str, int] = defaultdict(int)

    for sample in samples:
        pred = _instances(sample, 'pred_instances')
        gt = _instances(sample, 'gt_instances')
        pb = _to_tensor(_field(pred, 'bboxes'))
        pl = _to_tensor(_field(pred, 'labels'))
        ps = _to_tensor(_field(pred, 'scores'))
        gb = _to_tensor(_field(gt, 'bboxes'))
        gl = _to_tensor(_field(gt, 'labels'))
        if pb is None or pl is None or ps is None or gb is None or gl is None:
            continue
        if pb.numel() == 0 or gb.numel() == 0:
            continue
        keep = (ps >= score_thr) & (pl >= 0) & (pl < len(leaves))
        pb, pl, ps = pb[keep], pl[keep].long(), ps[keep]
        if pb.numel() == 0:
            continue
        pred_count += int(pb.shape[0])
        ious = _rotated_iou(gb.float(), pb.float())
        used_pred = set()
        for gi in range(gb.shape[0]):
            vals, order = torch.sort(ious[gi], descending=True)
            chosen = -1
            for val, pi in zip(vals.tolist(), order.tolist()):
                if val < iou_thr:
                    break
                if pi not in used_pred:
                    chosen = pi
                    break
            if chosen < 0:
                continue
            used_pred.add(chosen)
            true_l = int(gl[gi].item())
            pred_l = int(pl[chosen].item())
            if true_l < 0 or true_l >= len(leaves):
                continue
            matched += 1
            confusion[f'{leaves[true_l]}->{leaves[pred_l]}'] += 1
            if pred_l == true_l:
                correct += 1
            else:
                wrong += 1
                hce_sum += float(tree_dist.get((leaves[true_l], leaves[pred_l]), 999))
                if _parent_name(true_l, leaves, parent) != _parent_name(pred_l, leaves, parent):
                    cross_parent_wrong += 1
            if _parent_name(true_l, leaves, parent) == _parent_name(pred_l, leaves, parent):
                sibling_pairs += 1
                if pred_l == true_l:
                    sibling_correct += 1

    return {
        'matched_gt': matched,
        'matched_acc': correct / matched if matched else 0.0,
        'HCE': hce_sum / wrong if wrong else 0.0,
        'CPR': cross_parent_wrong / wrong if wrong else 0.0,
        'Sibling-Acc': sibling_correct / sibling_pairs if sibling_pairs else 0.0,
        'wrong_matches': wrong,
        'pred_count': pred_count,
        'score_thr': score_thr,
        'iou_thr': iou_thr,
        'confusion_top': sorted(confusion.items(), key=lambda kv: kv[1], reverse=True)[:100],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description='Compute HyDet hierarchy metrics from MMEngine DumpResults pkl.')
    parser.add_argument('--preds', required=True)
    parser.add_argument('--tree', default='projects/HyDet/resources/hrsc_hier/tree_validated.json')
    parser.add_argument('--leaf-names', default='projects/HyDet/resources/hrsc_hier/class_names_leaf.txt')
    parser.add_argument('--out', required=True)
    parser.add_argument('--iou-thr', type=float, default=0.5)
    parser.add_argument('--score-thr', type=float, default=0.05)
    args = parser.parse_args()

    samples = load(args.preds)
    if not isinstance(samples, list):
        raise TypeError(f'expected list predictions, got {type(samples)}')
    metrics = compute(samples, Path(args.tree), Path(args.leaf_names), args.iou_thr, args.score_thr)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
