import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from projects.HyDet.hydet.hyper.utils.tree_utils import TreeUtils


def _format_leaf_list(leaf_names: List[str]) -> str:
    return '\n'.join([f'- {name}' for name in leaf_names])


def build_prompt(dataset_name: str, leaf_names: List[str]) -> str:
    hints = TreeUtils.dataset_ontology_hints(dataset_name)
    hint_lines = '\n'.join([f'- {k}: {", ".join(v)}' for k, v in hints.items()])
    leaf_block = _format_leaf_list(leaf_names)
    prompt = f"""你是一个遥感目标类别本体工程师。请基于现实世界常识与公开知识，为数据集 {dataset_name} 构建语义层级树初稿。

必须严格遵守以下规则：
1) 只使用现实世界合理父子关系，禁止凭空拼接、禁止语义漂移、禁止不自然挂靠。
2) 每个叶子类必须且只能有一个父类，不能多父、不能丢失。
3) 允许虚拟父类，但必须是常识性抽象类，且可解释。
4) 优先最短、最清晰、最稳定路径，不要过深树，不要过度细分。
5) 不允许为了凑树而制造不自然父类。
6) 根节点必须唯一且名字固定为 entity。
7) 父类必须比子类更抽象，不允许父子反转。
8) 不能把叶子类再作为父类使用。
9) 节点名要简洁稳定，建议英文小写短语，语义一致。
10) 输出必须是严格 JSON，不要输出额外解释文本。

优先参考的现实逻辑簇：
{hint_lines}

给定叶子类别如下：
{leaf_block}

输出 JSON 模式如下：
{{
  "root": "entity",
  "nodes": [
    {{
      "name": "entity",
      "children": ["child_a", "child_b"],
      "explanation": "实体总类"
    }},
    {{
      "name": "child_a",
      "children": ["leaf_x"],
      "explanation": "该父类的现实世界语义解释"
    }},
    {{
      "name": "leaf_x",
      "children": [],
      "explanation": "叶子类别解释"
    }}
  ]
}}

输出质量要求：
- 所有叶子类必须完整出现且仅出现一次。
- 关系应贴合常识，例如 舰船->军舰/民船，航空器->固定翼/旋翼，地面车辆->军用/民用，场地设施->体育/工业/港口。
- 如果某叶子类缺乏可靠上位类，使用最合理的常识父类，不要生造冷门节点。
"""
    return prompt


def _load_raw_response(raw_response_path: Optional[str], raw_response_json: Optional[str]) -> Optional[Dict[str, Any]]:
    if raw_response_path is None and raw_response_json is None:
        return None
    if raw_response_path is not None and raw_response_json is not None:
        raise ValueError('only one of --raw-response-path and --raw-response-json can be used')
    if raw_response_path is not None:
        text = Path(raw_response_path).read_text(encoding='utf-8')
    else:
        text = str(raw_response_json)
    data = json.loads(text)
    if not isinstance(data, dict):
        raise ValueError('raw response json must be an object')
    return data


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Build GPT prompt for semantic tree generation')
    parser.add_argument('--leaf-path', required=True, help='path to class_names_leaf.txt')
    parser.add_argument('--dataset-name', required=True, help='dataset name, e.g. HRSC2016 or FAIR1M')
    parser.add_argument('--prompt-out', default='semantic_tree_prompt.txt', help='path to save generated prompt')
    parser.add_argument('--raw-response-path', default=None, help='path to AI raw json response')
    parser.add_argument('--raw-response-json', default=None, help='raw json string from AI response')
    parser.add_argument('--raw-out', default='tree_raw_gpt.json', help='path to save tree_raw_gpt.json')
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    leaf_names = TreeUtils.read_leaf_names(args.leaf_path)
    prompt = build_prompt(args.dataset_name, leaf_names)
    Path(args.prompt_out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.prompt_out).write_text(prompt, encoding='utf-8')
    print(prompt)

    raw_data = _load_raw_response(args.raw_response_path, args.raw_response_json)
    if raw_data is not None:
        TreeUtils.save_json(raw_data, args.raw_out)
        print(f'saved raw tree json: {args.raw_out}')


if __name__ == '__main__':
    main()
