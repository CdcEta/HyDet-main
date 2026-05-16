from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
from mmrotate.registry import MODELS

from ...pseudo_label_queue import PseudoQueue
from ..utils import HyperPluginSwitch


@MODELS.register_module()
class HierPseudoQueue(PseudoQueue):
    def __init__(
        self,
        plugin_cfg: Optional[Mapping[str, Any]] = None,
        use_hier_queue_filter: bool = False,
        queue_hier_consistency_thr: float = 0.5,
        tau_leaf: float = 0.8,
        tau_anc: float = 0.7,
        tau_gap: float = 0.05,
        hier_score_weights: Tuple[float, float, float] = (0.5, 0.3, 0.2),
        low_weight_factor: float = 0.5,
        class_sampling: Optional[bool] = None,
        **kwargs
    ) -> None:
        init_kwargs = dict(kwargs)
        if class_sampling is not None:
            init_kwargs['class_sampling'] = class_sampling
        super().__init__(**init_kwargs)
        self.hyper_plugin_switch = HyperPluginSwitch.parse(plugin_cfg or kwargs)
        self.use_hier_queue_filter = bool(use_hier_queue_filter) and bool(self.hyper_plugin_switch.use_hier_queue_filter)
        self.queue_hier_consistency_thr = float(queue_hier_consistency_thr)
        self.tau_leaf = float(tau_leaf)
        self.tau_anc = float(tau_anc)
        self.tau_gap = float(tau_gap)
        self.hier_score_weights = tuple(float(x) for x in hier_score_weights)
        if len(self.hier_score_weights) != 3:
            raise ValueError('hier_score_weights must have length 3')
        self.low_weight_factor = float(low_weight_factor)
        self.class_sampling = self.class_sampling if class_sampling is None else bool(class_sampling)

        class_to_parent = kwargs.get('class_to_parent', {})
        class_to_ancestors = kwargs.get('class_to_ancestors', {})
        class_to_siblings = kwargs.get('class_to_siblings', {})
        self.class_to_parent = self._normalize_id_dict(class_to_parent)
        self.class_to_ancestors = self._normalize_id_list_dict(class_to_ancestors)
        self.class_to_siblings = self._normalize_id_list_dict(class_to_siblings)

    def _normalize_id_dict(self, mapping: Mapping[Any, Any]) -> Dict[int, int]:
        out: Dict[int, int] = {}
        for k, v in dict(mapping).items():
            try:
                out[int(k)] = int(v)
            except (TypeError, ValueError):
                continue
        return out

    def _normalize_id_list_dict(self, mapping: Mapping[Any, Sequence[Any]]) -> Dict[int, List[int]]:
        out: Dict[int, List[int]] = {}
        for k, values in dict(mapping).items():
            try:
                key = int(k)
            except (TypeError, ValueError):
                continue
            cur: List[int] = []
            for item in values:
                try:
                    cur.append(int(item))
                except (TypeError, ValueError):
                    continue
            out[key] = cur
        return out

    def compute_leaf_score(
        self,
        label: int,
        score: Optional[float] = None,
        cls_probs: Optional[np.ndarray] = None
    ) -> float:
        if score is not None:
            return float(score)
        if cls_probs is None or len(cls_probs) == 0:
            return 0.0
        idx = int(label)
        if 0 <= idx < len(cls_probs):
            return float(cls_probs[idx])
        return float(np.max(cls_probs))

    def compute_ancestor_score(
        self,
        label: int,
        cls_probs: Optional[np.ndarray] = None,
        ancestor_ids: Optional[Sequence[int]] = None
    ) -> float:
        ids = list(ancestor_ids) if ancestor_ids is not None else self.class_to_ancestors.get(int(label), [])
        if cls_probs is None or len(cls_probs) == 0:
            return 0.0
        valid = [int(i) for i in ids if 0 <= int(i) < len(cls_probs)]
        if not valid:
            idx = int(label)
            if 0 <= idx < len(cls_probs):
                return float(cls_probs[idx])
            return 0.0
        values = [float(cls_probs[i]) for i in valid]
        return float(np.max(values))

    def compute_sibling_gap(
        self,
        label: int,
        cls_probs: Optional[np.ndarray] = None
    ) -> float:
        if cls_probs is None or len(cls_probs) == 0:
            return 0.0
        idx = int(label)
        if idx < 0 or idx >= len(cls_probs):
            return 0.0
        sibs = self.class_to_siblings.get(idx, [])
        valid = [int(i) for i in sibs if 0 <= int(i) < len(cls_probs) and int(i) != idx]
        leaf_prob = float(cls_probs[idx])
        if not valid:
            top2 = np.partition(cls_probs, -2)[-2:] if len(cls_probs) > 1 else np.array([leaf_prob, 0.0])
            max_comp = float(top2[-2] if float(top2[-1]) == leaf_prob else top2[-1])
            return float(leaf_prob - max_comp)
        sib_max = max(float(cls_probs[i]) for i in valid)
        return float(leaf_prob - sib_max)

    def compute_hier_score(self, leaf_score: float, ancestor_score: float, sibling_gap: float) -> float:
        w1, w2, w3 = self.hier_score_weights
        return float(w1 * leaf_score + w2 * ancestor_score + w3 * max(sibling_gap, 0.0))

    def filter_and_pack(
        self,
        bboxes: np.ndarray,
        labels: np.ndarray,
        scores: Optional[np.ndarray] = None,
        cls_probs: Optional[np.ndarray] = None,
        ancestor_ids: Optional[Sequence[Sequence[int]]] = None,
        source: str = 'external_teacher'
    ) -> Dict[str, Any]:
        if bboxes is None or labels is None or len(labels) == 0:
            return {
                'bboxes': np.zeros((0, 5), dtype=np.float32),
                'labels': np.zeros((0,), dtype=np.int64),
                'scores': np.zeros((0,), dtype=np.float32),
                'leaf_score': np.zeros((0,), dtype=np.float32),
                'ancestor_score': np.zeros((0,), dtype=np.float32),
                'sibling_gap': np.zeros((0,), dtype=np.float32),
                'hier_score': np.zeros((0,), dtype=np.float32),
                'ancestor_ids': [],
                'source': [],
                'weights': np.zeros((0,), dtype=np.float32),
            }

        labels = np.asarray(labels).astype(np.int64)
        bboxes = np.asarray(bboxes)
        score_arr = np.asarray(scores).astype(np.float32) if scores is not None else None
        cls_probs_arr = np.asarray(cls_probs) if cls_probs is not None else None
        if cls_probs_arr is not None and cls_probs_arr.ndim == 1:
            cls_probs_arr = np.repeat(cls_probs_arr[None, :], len(labels), axis=0)

        keep_bboxes: List[np.ndarray] = []
        keep_labels: List[int] = []
        keep_scores: List[float] = []
        keep_leaf: List[float] = []
        keep_anc: List[float] = []
        keep_gap: List[float] = []
        keep_hier: List[float] = []
        keep_ancestor_ids: List[List[int]] = []
        keep_source: List[str] = []
        keep_weights: List[float] = []

        for i in range(len(labels)):
            label = int(labels[i])
            score = float(score_arr[i]) if score_arr is not None and i < len(score_arr) else None
            probs = cls_probs_arr[i] if cls_probs_arr is not None and i < len(cls_probs_arr) else None
            anc_ids = list(ancestor_ids[i]) if ancestor_ids is not None and i < len(ancestor_ids) else self.class_to_ancestors.get(label, [])

            leaf_score = self.compute_leaf_score(label=label, score=score, cls_probs=probs)
            anc_score = self.compute_ancestor_score(label=label, cls_probs=probs, ancestor_ids=anc_ids)
            gap = self.compute_sibling_gap(label=label, cls_probs=probs)
            hier_score = self.compute_hier_score(leaf_score, anc_score, gap)

            direct_accept = (
                leaf_score >= self.tau_leaf and
                anc_score >= self.tau_anc and
                gap >= self.tau_gap and
                hier_score >= self.queue_hier_consistency_thr
            )
            low_weight_accept = (
                anc_score >= self.tau_anc and
                leaf_score >= self.tau_leaf * 0.9 and
                gap >= 0.0 and
                hier_score >= self.queue_hier_consistency_thr * 0.9
            )
            path_inconsistent = anc_score < min(self.tau_anc * 0.8, max(leaf_score * 0.7, 1e-6))
            sibling_conflict = gap < self.tau_gap
            reject = (not direct_accept and not low_weight_accept) or path_inconsistent or sibling_conflict
            if reject:
                continue

            weight = 1.0 if direct_accept else self.low_weight_factor
            if score is None:
                score_to_store = float(leaf_score) * float(weight)
            else:
                score_to_store = float(score) * float(weight)

            keep_bboxes.append(bboxes[i])
            keep_labels.append(label)
            keep_scores.append(score_to_store)
            keep_leaf.append(float(leaf_score))
            keep_anc.append(float(anc_score))
            keep_gap.append(float(gap))
            keep_hier.append(float(hier_score))
            keep_ancestor_ids.append([int(x) for x in anc_ids])
            keep_source.append(str(source))
            keep_weights.append(float(weight))

        if keep_bboxes:
            out_bboxes = np.stack(keep_bboxes, axis=0)
        else:
            out_bboxes = np.zeros((0, bboxes.shape[1] if bboxes.ndim == 2 else 5), dtype=np.float32)
        return {
            'bboxes': out_bboxes,
            'labels': np.asarray(keep_labels, dtype=np.int64),
            'scores': np.asarray(keep_scores, dtype=np.float32),
            'leaf_score': np.asarray(keep_leaf, dtype=np.float32),
            'ancestor_score': np.asarray(keep_anc, dtype=np.float32),
            'sibling_gap': np.asarray(keep_gap, dtype=np.float32),
            'hier_score': np.asarray(keep_hier, dtype=np.float32),
            'ancestor_ids': keep_ancestor_ids,
            'source': keep_source,
            'weights': np.asarray(keep_weights, dtype=np.float32),
        }

    def push_batch(
        self,
        img_path: str,
        bboxes: np.ndarray,
        labels: np.ndarray,
        scores: Optional[np.ndarray] = None,
        cls_probs: Optional[np.ndarray] = None,
        ancestor_ids: Optional[Sequence[Sequence[int]]] = None,
        source: str = 'external_teacher'
    ) -> None:
        packed = self.filter_and_pack(
            bboxes=bboxes,
            labels=labels,
            scores=scores,
            cls_probs=cls_probs,
            ancestor_ids=ancestor_ids,
            source=source,
        )
        super().update_pseudo_queue(
            img_path=img_path,
            bboxes=packed['bboxes'],
            labels=packed['labels'],
            scores=packed['scores']
        )
        img_name = img_path.split('/')[-1]
        if img_name in self.name2detSample:
            self.name2detSample[img_name]['leaf_score'] = packed['leaf_score']
            self.name2detSample[img_name]['ancestor_score'] = packed['ancestor_score']
            self.name2detSample[img_name]['sibling_gap'] = packed['sibling_gap']
            self.name2detSample[img_name]['hier_score'] = packed['hier_score']
            self.name2detSample[img_name]['ancestor_ids'] = packed['ancestor_ids']
            self.name2detSample[img_name]['source'] = packed['source']
            self.name2detSample[img_name]['weights'] = packed['weights']

    def update_pseudo_queue(
        self,
        img_path,
        bboxes,
        labels,
        scores=None,
        cls_probs: Optional[np.ndarray] = None,
        ancestor_ids: Optional[Sequence[Sequence[int]]] = None,
        source: str = 'external_teacher'
    ):
        if not self.use_hier_queue_filter:
            return super().update_pseudo_queue(img_path=img_path, bboxes=bboxes, labels=labels, scores=scores)
        return self.push_batch(
            img_path=img_path,
            bboxes=bboxes,
            labels=labels,
            scores=scores,
            cls_probs=cls_probs,
            ancestor_ids=ancestor_ids,
            source=source
        )
