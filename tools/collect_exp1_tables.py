#!/usr/bin/env python3
import argparse
import csv
import json
import re
from pathlib import Path
from typing import Dict, List


LOG_PATTERN = re.compile(r'dota/mAP:\s*([0-9.]+)\s+dota/AP50:\s*([0-9.]+)')


def _read_json(path: Path) -> Dict:
    if not path.is_file():
        return {}
    try:
        return json.loads(path.read_text(encoding='utf-8'))
    except json.JSONDecodeError:
        return {}


def _find_log_metrics(eval_dir: Path) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for log_path in sorted(eval_dir.glob('**/*.log'), reverse=True):
        text = log_path.read_text(encoding='utf-8', errors='ignore')
        matches = LOG_PATTERN.findall(text)
        if matches:
            m_ap, ap50 = matches[-1]
            out['mAP'] = float(m_ap)
            out['AP50'] = float(ap50)
            break
    return out


def _collect_variant(root: Path, variant: str) -> Dict[str, object]:
    eval_dir = root / f'{variant}_eval'
    row: Dict[str, object] = {'Method': variant}
    row.update(_find_log_metrics(eval_dir))
    metrics = _read_json(eval_dir / 'hierarchy_metrics.json')
    for key in ['matched_acc', 'HCE', 'CPR', 'Sibling-Acc', 'pred_count']:
        value = metrics.get(key)
        if isinstance(value, (int, float)):
            row[key] = value
    return row


def _to_md(rows: List[Dict[str, object]], headers: List[str]) -> str:
    lines = ['| ' + ' | '.join(headers) + ' |', '| ' + ' | '.join(['---'] * len(headers)) + ' |']
    for row in rows:
        vals = []
        for h in headers:
            v = row.get(h, '')
            vals.append(f'{v:.4f}' if isinstance(v, float) else str(v))
        lines.append('| ' + ' | '.join(vals) + ' |')
    return '\n'.join(lines) + '\n'


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--work-dir', required=True)
    parser.add_argument('--variants', nargs='+', required=True)
    parser.add_argument('--out-md', required=True)
    parser.add_argument('--out-csv', required=True)
    args = parser.parse_args()

    root = Path(args.work_dir)
    rows = [_collect_variant(root, v) for v in args.variants]
    headers = ['Method', 'mAP', 'AP50', 'matched_acc', 'HCE', 'CPR', 'Sibling-Acc', 'pred_count']

    md_path = Path(args.out_md)
    md_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.write_text(_to_md(rows, headers), encoding='utf-8')

    csv_path = Path(args.out_csv)
    with csv_path.open('w', encoding='utf-8', newline='') as fp:
        writer = csv.DictWriter(fp, fieldnames=headers)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


if __name__ == '__main__':
    main()
