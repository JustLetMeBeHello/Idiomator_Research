from __future__ import annotations

import json
from pathlib import Path

ID_TO_LABEL = {
    "system_a_mbert_pipeline": "A",
    "system_b_gpt_pipeline": "B",
    "system_b4_gpt_pipeline_4shot": "B4",
    "system_c_gpt_single": "C",
    "system_c4_gpt_single_4shot": "C4",
    "system_d_mbert_s1_joint_span": "D",
    "system_e_joint_end_to_end": "E",
    "system_f_sequential_phase1_ph2": "F",
    "system_g_bio_tagger": "G",
}


def label_for(system_id: str) -> str:
    if system_id not in ID_TO_LABEL:
        raise KeyError(f"Unknown system id: {system_id}")
    return ID_TO_LABEL[system_id]


def load_results(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f"results json not found: {path}")
    return json.loads(path.read_text())


def _dig(node: dict, *path):
    """Walk nested dicts; return None if any key is missing/non-dict."""
    cur = node
    for k in path:
        if not isinstance(cur, dict) or k not in cur:
            return None
        cur = cur[k]
    return cur


def live_by_label(path: Path) -> dict:
    """Reduce the full nested JSON to {label: {joint_f1, stability, cls_f1}} scalars."""
    raw = load_results(path)
    out = {}
    for sid, node in raw.items():
        label = ID_TO_LABEL.get(sid)
        if label is None:
            continue
        out[label] = {
            # headline macro-avg Joint F1 lives at joint_f1.Overall.macro_avg_f1
            "joint_f1": _dig(node, "joint_f1", "Overall", "macro_avg_f1"),
            # scalar stability score lives at stability.stability
            "stability": _dig(node, "stability", "stability"),
            # overall classification F1 lives at cls_f1.Overall
            "cls_f1": _dig(node, "cls_f1", "Overall"),
        }
    return out
