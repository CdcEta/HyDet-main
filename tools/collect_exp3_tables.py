#!/usr/bin/env python3
import argparse
import csv
import json
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional


TRAIN_KEYS = [
    'loss',
    'loss_cls',
    'loss_bbox',
    'mod_tree_cone',
    'mod_tree_radial',
    'mod_tree_text_image',
    'mod_hyperbolic_contrast',
    'mod_hac_radius',
    'mod_hac_cross_parent',
    'mod_hac_sibling',
]

EVAL_KEYS = [
    'bbox_mAP',
    'bbox_mAP_50',
    'dota/mAP',
    'mAP',
    'matched_acc',
    'HCE',
    'CPR',
    'Sibling-Acc',
    'pred_count',
]


def _read_json_lines(path: Path) -> List[Dict]:
    rows = []
    if not path.is_file():
        return rows
    with path.open('r', encoding='utf-8') as fp:
        for line in fp:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def _latest_scalars(work_dir: Path) -> Optional[Path]:
    matches = sorted(work_dir.glob('*/vis_data/scalars.json'))
    return matches[-1] if matches else None


def _avg(rows: List[Dict], key: str, tail: int) -> Optional[float]:
    values = [row.get(key) for row in rows[-tail:] if isinstance(row.get(key), (int, float))]
    if not values:
        return None
    return float(sum(values) / len(values))


def _last_numeric(rows: List[Dict], keys: Iterable[str]) -> Optional[float]:
    for row in reversed(rows):
        for key in keys:
            value = row.get(key)
            if isinstance(value, (int, float)):
                return float(value)
    return None


def _read_json(path: Path) -> Dict:
    if not path.is_file():
        return {}
    try:
        return json.loads(path.read_text(encoding='utf-8'))
    except json.JSONDecodeError:
        return {}


def _find_eval_log_metrics(eval_dir: Path) -> Dict[str, float]:
    out: Dict[str, float] = {}
    logs = sorted(eval_dir.glob('**/*.log'))
    pattern = re.compile(r'dota/mAP:\s*([0-9.]+)\s+dota/AP50:\s*([0-9.]+)')
    for log_path in reversed(logs):
        try:
            text = log_path.read_text(encoding='utf-8', errors='ignore')
        except OSError:
            continue
        matches = pattern.findall(text)
        if matches:
            m_ap, ap50 = matches[-1]
            out['dota/mAP'] = float(m_ap)
            out['bbox_mAP_50'] = float(ap50)
            out['bbox_mAP'] = float(m_ap)
            out['mAP'] = float(m_ap)
            break
    return out


def _find_eval_metrics(root: Path, variant: str) -> Dict[str, float]:
    out: Dict[str, float] = {}
    eval_dirs = sorted(root.glob(f'{variant}_eval*'))
    for eval_dir in reversed(eval_dirs):
        out.update(_find_eval_log_metrics(eval_dir))
        for candidate in [
            eval_dir / 'hierarchy_metrics.json',
            eval_dir / 'strict_openvocab' / 'openvocab_summary.json',
            eval_dir / 'strict_openvocab' / 'hierarchy_metrics.json',
        ]:
            metrics = _read_json(candidate)
            for key in EVAL_KEYS:
                value = metrics.get(key)
                if isinstance(value, (int, float)):
                    out[key] = float(value)
        if out:
            break
    return out


def _collect_variant(root: Path, variant: str, tail: int) -> Dict[str, object]:
    variant_dir = root / variant
    row: Dict[str, object] = {'variant': variant}
    scalars_path = _latest_scalars(variant_dir)
    if scalars_path is not None:
        rows = _read_json_lines(scalars_path)
        row['train_log'] = str(scalars_path)
        row['last_iter'] = int(_last_numeric(rows, ['iter', 'step']) or 0)
        for key in TRAIN_KEYS:
            avg = _avg(rows, key, tail)
            if avg is not None:
                row[key] = round(avg, 6)
    row.update({k: round(v, 6) for k, v in _find_eval_metrics(root, variant).items()})
    return row


def _to_markdown(rows: List[Dict[str, object]], headers: List[str]) -> str:
    lines = []
    lines.append('| ' + ' | '.join(headers) + ' |')
    lines.append('| ' + ' | '.join(['---'] * len(headers)) + ' |')
    for row in rows:
        values = []
        for header in headers:
            value = row.get(header, '')
            if isinstance(value, float):
                values.append(f'{value:.4f}')
            else:
                values.append(str(value))
        lines.append('| ' + ' | '.join(values) + ' |')
    return '\n'.join(lines) + '\n'


def main() -> None:
    parser = argparse.ArgumentParser(description='Collect experiment-3 training and evaluation statistics into a table.')
    parser.add_argument('--work-dir', required=True)
    parser.add_argument('--variants', nargs='+', required=True)
    parser.add_argument('--tail', type=int, default=50, help='Average the last N scalar rows for train statistics.')
    parser.add_argument('--out-md', required=True)
    parser.add_argument('--out-csv', required=True)
    args = parser.parse_args()

    root = Path(args.work_dir)
    rows = [_collect_variant(root, variant, args.tail) for variant in args.variants]
    headers = ['variant', 'last_iter', 'train_log'] + TRAIN_KEYS + EVAL_KEYS

    md_path = Path(args.out_md)
    md_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.write_text(_to_markdown(rows, headers), encoding='utf-8')

    csv_path = Path(args.out_csv)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open('w', encoding='utf-8', newline='') as fp:
        writer = csv.DictWriter(fp, fieldnames=headers)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    print(f'Wrote {md_path}')
    print(f'Wrote {csv_path}')


if __name__ == '__main__':
    main()
