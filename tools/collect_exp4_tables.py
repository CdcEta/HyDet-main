#!/usr/bin/env python3
import argparse
import csv
import json
import re
from pathlib import Path
from typing import Dict, List, Optional


VARIANT_FLAGS = {
    'base': dict(Align='-', Rad='-', Sep='-', Sib='-'),
    'align': dict(Align='Y', Rad='-', Sep='-', Sib='-'),
    'rad': dict(Align='Y', Rad='Y', Sep='-', Sib='-'),
    'sep': dict(Align='Y', Rad='Y', Sep='Y', Sib='-'),
    'sib': dict(Align='Y', Rad='Y', Sep='Y', Sib='Y'),
    'hra': dict(Align='Y', Rad='Y', Sep='Y', Sib='Y'),
}

LOG_PATTERN = re.compile(r'dota/mAP:\s*([0-9.]+)\s+dota/AP50:\s*([0-9.]+)')


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


def _read_json(path: Path) -> Dict:
    if not path.is_file():
        return {}
    try:
        return json.loads(path.read_text(encoding='utf-8'))
    except json.JSONDecodeError:
        return {}


def _find_eval_metrics(eval_dir: Path) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for log_path in sorted(eval_dir.glob('**/*.log'), reverse=True):
        try:
            text = log_path.read_text(encoding='utf-8', errors='ignore')
        except OSError:
            continue
        matches = LOG_PATTERN.findall(text)
        if matches:
            m_ap, ap50 = matches[-1]
            out['mAP'] = float(m_ap)
            out['AP50'] = float(ap50)
            break
    metrics = _read_json(eval_dir / 'hierarchy_metrics.json')
    for key in ['matched_acc', 'HCE', 'CPR', 'Sibling-Acc', 'pred_count']:
        value = metrics.get(key)
        if isinstance(value, (int, float)):
            out[key] = float(value)
    return out


def _collect_variant(root: Path, variant: str, tail: int) -> Dict[str, object]:
    source_variant = 'sib' if variant == 'hra' else variant
    train_dir = root / source_variant
    eval_dir = root / f'{source_variant}_eval'
    row: Dict[str, object] = {'variant': variant, 'Setting': variant}
    row.update(VARIANT_FLAGS.get(variant, {}))

    scalars = _latest_scalars(train_dir)
    if scalars is not None:
        rows = _read_json_lines(scalars)
        row['train_log'] = str(scalars)
        row['loss'] = round(_avg(rows, 'loss', tail) or 0.0, 6)
        row['loss_cls'] = round(_avg(rows, 'loss_cls', tail) or 0.0, 6)
        row['loss_bbox'] = round(_avg(rows, 'loss_bbox', tail) or 0.0, 6)
        for key in ['mod_hyperbolic_contrast', 'mod_hac_radius', 'mod_hac_cross_parent', 'mod_hac_sibling']:
            avg = _avg(rows, key, tail)
            if avg is not None:
                row[key] = round(avg, 6)

    row.update({k: round(v, 6) for k, v in _find_eval_metrics(eval_dir).items()})
    return row


def _to_markdown(rows: List[Dict[str, object]], headers: List[str]) -> str:
    lines = ['| ' + ' | '.join(headers) + ' |', '| ' + ' | '.join(['---'] * len(headers)) + ' |']
    for row in rows:
        vals = []
        for h in headers:
            v = row.get(h, '')
            if isinstance(v, float):
                vals.append(f'{v:.4f}')
            else:
                vals.append(str(v))
        lines.append('| ' + ' | '.join(vals) + ' |')
    return '\n'.join(lines) + '\n'


def main() -> None:
    parser = argparse.ArgumentParser(description='Collect experiment-4 HRA ablation statistics into a table.')
    parser.add_argument('--work-dir', required=True)
    parser.add_argument('--variants', nargs='+', required=True)
    parser.add_argument('--tail', type=int, default=50)
    parser.add_argument('--out-md', required=True)
    parser.add_argument('--out-csv', required=True)
    args = parser.parse_args()

    root = Path(args.work_dir)
    rows = [_collect_variant(root, variant, args.tail) for variant in args.variants]
    headers = [
        'variant', 'Setting', 'Align', 'Rad', 'Sep', 'Sib',
        'mAP', 'AP50', 'matched_acc', 'HCE', 'CPR', 'Sibling-Acc', 'pred_count',
        'loss', 'loss_cls', 'loss_bbox',
        'mod_hyperbolic_contrast', 'mod_hac_radius', 'mod_hac_cross_parent', 'mod_hac_sibling',
        'train_log'
    ]

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
