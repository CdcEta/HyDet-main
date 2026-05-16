import argparse

from hrsc_open_vocab_builder import ensure_hrsc_open_vocab_layout


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Prepare HRSC open-vocabulary split (8:2 classes, ~8.5:1.5 seen/unseen instances) '
                    'and generate semantic tree artifacts.')
    parser.add_argument(
        '--source-root',
        default='data/HRSC2016_raw',
        help='HRSC2016 root that contains FullDataSet and ImageSets.')
    parser.add_argument(
        '--target-root',
        default='data/HRSC2016',
        help='Output root that stores split txt and tree/meta files.')
    parser.add_argument(
        '--repo-data-link',
        default='data/HRSC2016',
        help='Optional symlink path used by training config.')
    parser.add_argument('--seed', type=int, default=3407)
    parser.add_argument('--unseen-class-ratio', type=float, default=0.2)
    parser.add_argument('--unseen-instance-ratio', type=float, default=0.15)
    parser.add_argument('--force-rebuild', action='store_true')
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = ensure_hrsc_open_vocab_layout(
        source_root=args.source_root,
        target_root=args.target_root,
        repo_data_link=args.repo_data_link,
        seed=args.seed,
        unseen_class_ratio=args.unseen_class_ratio,
        unseen_instance_ratio=args.unseen_instance_ratio,
        force_rebuild=args.force_rebuild,
    )
    print('prepared hrsc open-vocabulary data:')
    for k, v in report.items():
        if k in {'seen_classes', 'unseen_classes'}:
            print(f'- {k}: {len(v)} classes')
        else:
            print(f'- {k}: {v}')


if __name__ == '__main__':
    main()
