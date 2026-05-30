#!/usr/bin/env python3
"""Print mBERT comparison baselines for the rigor experiments, read LIVE from
the ground-truth eval json — never hardcoded.

Why this exists: rigor launchers used to echo hardcoded baseline numbers
(e.g. "System G mBERT Telugu exact 0.0000"). When the eval was re-run after a
bugfix, those literals went stale and a stale copy was later reported as fact.
This script removes the rot vector: the ONLY source of a reported metric is the
committed results json (Tier 0). key_numbers.md (Tier 1) is the canonical prose
digest of the same file. Scripts must call this, not paste numbers.

Usage:
    python Additional_Rigor_Experiments/mbert_baselines.py            # full table
    python Additional_Rigor_Experiments/mbert_baselines.py --system E # one system
"""
import argparse
import json
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
JSON_PATH = os.path.join(REPO, "results", "pipeline_eval", "pipeline_eval_results.json")

# system label -> json top-level key
SYS = {
    "E": "system_e_joint_end_to_end",
    "G": "system_g_bio_tagger",
}
LANGS = ["English", "Spanish", "Hindi", "Telugu", "Indonesian"]
CODE = {"English": "EN", "Spanish": "ES", "Hindi": "HI", "Telugu": "TE", "Indonesian": "ID"}


def _scalar(v):
    """joint_f1[lang] is sometimes a scalar, sometimes {'macro_f1': ...}."""
    if isinstance(v, (int, float)):
        return v
    if isinstance(v, dict):
        return v.get("macro_f1", v.get("f1"))
    return None


def _get(d, *path):
    cur = d
    for p in path:
        if not isinstance(cur, dict) or p not in cur:
            return None
        cur = cur[p]
    return cur


def baselines(data):
    """Return {label: {metric: {lang: value}}} pulled live from json."""
    out = {}
    e = data.get(SYS["E"], {})
    out["E"] = {
        "Joint F1": {l: _scalar(_get(e, "joint_f1", l)) for l in LANGS},
    }
    g = data.get(SYS["G"], {})
    out["G"] = {
        "span exact": {l: _scalar(_get(g, "span_e2e", "exact", l)) for l in LANGS},
    }
    return out


def fmt(x):
    return f"{x:.2f}" if isinstance(x, (int, float)) else "  --"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--system", choices=["E", "G"], help="limit to one system")
    args = ap.parse_args()

    if not os.path.exists(JSON_PATH):
        sys.exit(f"ground-truth json not found: {JSON_PATH}\n"
                 f"Run Evaluation/Full_evaluation.py first.")
    data = json.load(open(JSON_PATH))
    bl = baselines(data)

    print(f"mBERT baselines (live from {os.path.relpath(JSON_PATH, REPO)}; 2dp):")
    for label in (["E", "G"] if not args.system else [args.system]):
        for metric, per_lang in bl.get(label, {}).items():
            cells = "  ".join(f"{CODE[l]}={fmt(v)}" for l, v in per_lang.items())
            print(f"  System {label} {metric:<10}: {cells}")


if __name__ == "__main__":
    main()
