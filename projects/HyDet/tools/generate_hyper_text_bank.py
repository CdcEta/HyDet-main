import argparse
from pathlib import Path
from typing import Dict

import torch

from projects.HyDet.hydet.hyper.models.hyper_mapper import SharedHyperbolicMapper
from projects.HyDet.hydet.hyper.structures.text_bank import HyperTextBank


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Generate hyperbolic text bank cache from euclidean npy embeddings')
    parser.add_argument('--leaf-euc', required=True, help='path to leaf_text_embeddings_euc.npy')
    parser.add_argument('--all-euc', required=True, help='path to all_nodes_text_embeddings_euc.npy')
    parser.add_argument('--tree-validated', required=True, help='path to tree_validated.json')
    parser.add_argument('--parent-map', required=True, help='path to parent_map.json')
    parser.add_argument('--leaf-names', required=True, help='path to class_names_leaf.txt')
    parser.add_argument('--all-node-names', required=True, help='path to class_names_all_nodes.txt')
    parser.add_argument('--out-dir', required=True, help='output directory')
    parser.add_argument('--in-dim', type=int, default=1024)
    parser.add_argument('--out-dim', type=int, default=1024)
    parser.add_argument('--curvature', type=float, default=1.0)
    parser.add_argument('--dropout', type=float, default=0.0)
    parser.add_argument('--use-residual', action='store_true')
    parser.add_argument('--enabled', action='store_true')
    parser.add_argument('--use-hyper-branch', action='store_true')
    parser.add_argument('--cache-hyp-bank', action='store_true')
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--leaf-out-name', default='leaf_text_embeddings_hyp.pt')
    parser.add_argument('--all-out-name', default='all_nodes_text_embeddings_hyp.pt')
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    plugin_cfg: Dict[str, bool] = {
        'use_hyper_branch': bool(args.use_hyper_branch),
        'cache_hyp_bank': bool(args.cache_hyp_bank),
    }
    bank = HyperTextBank(
        leaf_text_embeddings_euc_path=args.leaf_euc,
        all_nodes_text_embeddings_euc_path=args.all_euc,
        tree_validated_path=args.tree_validated,
        parent_map_path=args.parent_map,
        class_names_leaf_path=args.leaf_names,
        class_names_all_nodes_path=args.all_node_names,
        plugin_cfg=plugin_cfg,
        device=args.device,
    )

    mapper = SharedHyperbolicMapper(
        in_dim=args.in_dim,
        out_dim=args.out_dim,
        curvature=args.curvature,
        dropout=args.dropout,
        use_residual=bool(args.use_residual),
        enabled=bool(args.enabled),
        plugin_cfg=plugin_cfg,
    ).to(torch.device(args.device))
    mapper.eval()

    bank.build_leaf_bank_hyp(mapper)
    bank.build_all_bank_hyp(mapper)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_paths = bank.export_cache(
        out_dir=str(out_dir),
        leaf_file_name=args.leaf_out_name,
        all_file_name=args.all_out_name,
    )
    print(f'saved leaf hyp bank: {out_paths["leaf"]}')
    print(f'saved all-node hyp bank: {out_paths["all_nodes"]}')


if __name__ == '__main__':
    main()
