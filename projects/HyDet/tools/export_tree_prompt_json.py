import argparse
import json
from pathlib import Path
from typing import Dict, List


def read_leaf_classes(path: str) -> List[str]:
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f'class_names_leaf.txt not found: {path}')
    classes = [line.strip() for line in file_path.read_text(encoding='utf-8').splitlines() if line.strip()]
    if not classes:
        raise ValueError('class_names_leaf.txt is empty')
    dedup = []
    seen = set()
    for c in classes:
        key = c.lower()
        if key in seen:
            continue
        seen.add(key)
        dedup.append(c)
    return dedup


def build_few_shot_examples() -> List[Dict]:
    ship_example = {
        'name': '舰船层级示例',
        'input_leaf_classes': [
            'aircraft carrier',
            'destroyer',
            'frigate',
            'container ship',
            'oil tanker',
            'fishing vessel'
        ],
        'output_tree': {
            'root': 'entity',
            'nodes': [
                {'name': 'entity', 'children': ['ship'], 'explanation': '所有遥感对象的总类'},
                {'name': 'ship', 'children': ['military ship', 'civilian ship'], 'explanation': '水面舰船总类'},
                {'name': 'military ship', 'children': ['aircraft carrier', 'destroyer', 'frigate'], 'explanation': '执行军事任务的舰艇'},
                {'name': 'civilian ship', 'children': ['container ship', 'oil tanker', 'fishing vessel'], 'explanation': '民用运输与渔业船舶'},
                {'name': 'aircraft carrier', 'children': [], 'explanation': '可搭载舰载机的大型军舰'},
                {'name': 'destroyer', 'children': [], 'explanation': '中大型水面作战舰艇'},
                {'name': 'frigate', 'children': [], 'explanation': '中小型多用途作战舰艇'},
                {'name': 'container ship', 'children': [], 'explanation': '运输集装箱的商船'},
                {'name': 'oil tanker', 'children': [], 'explanation': '运输液体燃料的油轮'},
                {'name': 'fishing vessel', 'children': [], 'explanation': '从事渔业活动的船舶'}
            ]
        }
    }
    air_vehicle_example = {
        'name': '航空器与车辆层级示例',
        'input_leaf_classes': [
            'fighter jet',
            'airliner',
            'helicopter',
            'tank',
            'bus',
            'cargo truck'
        ],
        'output_tree': {
            'root': 'entity',
            'nodes': [
                {'name': 'entity', 'children': ['aircraft', 'ground vehicle'], 'explanation': '遥感目标总类'},
                {'name': 'aircraft', 'children': ['fixed-wing aircraft', 'rotary-wing aircraft'], 'explanation': '在空中飞行的飞行器'},
                {'name': 'fixed-wing aircraft', 'children': ['fighter jet', 'airliner'], 'explanation': '依赖固定翼产生升力的航空器'},
                {'name': 'rotary-wing aircraft', 'children': ['helicopter'], 'explanation': '依赖旋翼产生升力的航空器'},
                {'name': 'ground vehicle', 'children': ['military vehicle', 'civilian vehicle'], 'explanation': '在地面道路或地表行驶的车辆'},
                {'name': 'military vehicle', 'children': ['tank'], 'explanation': '用于军事任务的地面车辆'},
                {'name': 'civilian vehicle', 'children': ['bus', 'cargo truck'], 'explanation': '用于公共运输或物流的民用车辆'},
                {'name': 'fighter jet', 'children': [], 'explanation': '用于空战任务的高速固定翼军机'},
                {'name': 'airliner', 'children': [], 'explanation': '用于客运航线的固定翼民航飞机'},
                {'name': 'helicopter', 'children': [], 'explanation': '使用旋翼起降与悬停的航空器'},
                {'name': 'tank', 'children': [], 'explanation': '履带式装甲战斗车辆'},
                {'name': 'bus', 'children': [], 'explanation': '用于公共客运的大型客车'},
                {'name': 'cargo truck', 'children': [], 'explanation': '用于货运的道路卡车'}
            ]
        }
    }
    return [ship_example, air_vehicle_example]


def build_prompt(dataset_name: str, leaf_classes: List[str], examples: List[Dict]) -> str:
    leaf_block = '\n'.join([f'- {c}' for c in leaf_classes])
    examples_text = json.dumps(examples, ensure_ascii=False, indent=2)
    prompt = f"""你是一个遥感语义层级建模专家。请为数据集 {dataset_name} 生成高质量层级树。

你必须严格满足以下要求：
1. 只依据现实世界合理父子关系建树。
2. 不允许为了凑树而创造不自然父类。
3. 每个叶子类只能有一个父类。
4. 允许虚拟父类，但必须是常识性类别。
5. 优先最短、最清晰、最稳定的路径。
6. 给出每个父类设置的简短解释。
7. 输出 JSON，便于后续程序解析。
8. 若类名存在歧义，优先选择遥感/交通/军事常见语义。

额外约束：
- 根节点唯一，名称固定为 entity。
- 不允许出现环。
- 不允许把叶子类再作为父类。
- 节点名称简洁、稳定、可复用。

输入叶子类别如下：
{leaf_block}

请参考以下 few-shot 示例风格：
{examples_text}

输出格式必须是严格 JSON，对象结构如下：
{{
  "root": "entity",
  "nodes": [
    {{
      "name": "entity",
      "children": ["..."],
      "explanation": "..."
    }}
  ]
}}

请仅输出 JSON 本体，不要输出额外说明文本。
"""
    return prompt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Export GPT-5.3 tree-construction prompt template')
    parser.add_argument('--dataset-name', required=True, help='HRSC2016 or FAIR1M')
    parser.add_argument('--leaf-path', required=True, help='path to class_names_leaf.txt')
    parser.add_argument('--out-dir', default='.', help='output directory')
    parser.add_argument('--txt-out', default='tree_prompt.txt', help='output txt file name')
    parser.add_argument('--json-out', default='tree_prompt.json', help='output json file name')
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_name = args.dataset_name.strip()
    if dataset_name not in {'HRSC2016', 'FAIR1M'}:
        raise ValueError('dataset_name must be HRSC2016 or FAIR1M')

    leaf_classes = read_leaf_classes(args.leaf_path)
    examples = build_few_shot_examples()
    prompt = build_prompt(dataset_name, leaf_classes, examples)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    txt_path = out_dir / args.txt_out
    json_path = out_dir / args.json_out

    txt_path.write_text(prompt, encoding='utf-8')

    prompt_json = {
        'dataset_name': dataset_name,
        'model_hint': 'GPT-5.3',
        'leaf_class_count': len(leaf_classes),
        'leaf_classes': leaf_classes,
        'constraints': [
            '只依据现实世界合理父子关系建树',
            '不允许为了凑树而创造不自然父类',
            '每个叶子类只能有一个父类',
            '允许虚拟父类，但必须是常识性类别',
            '优先最短、最清晰、最稳定的路径',
            '给出每个父类设置的简短解释',
            '输出 JSON，便于后续程序解析',
            '若类名存在歧义，优先选择遥感/交通/军事常见语义'
        ],
        'few_shot_examples': examples,
        'output_schema': {
            'root': 'entity',
            'nodes': [
                {'name': 'entity', 'children': ['...'], 'explanation': '...'}
            ]
        },
        'prompt': prompt
    }
    json_path.write_text(json.dumps(prompt_json, ensure_ascii=False, indent=2), encoding='utf-8')

    print(f'saved prompt txt: {txt_path}')
    print(f'saved prompt json: {json_path}')


if __name__ == '__main__':
    main()
