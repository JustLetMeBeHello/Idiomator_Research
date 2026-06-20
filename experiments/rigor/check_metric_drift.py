#!/usr/bin/env python3
"""Tripwire for stale metric literals (Tier-2 drift).

Root cause of the recurring metric inconsistencies: a number is corrected in the
eval json + key_numbers.md, but hardcoded copies elsewhere (scripts, notebook,
CLAUDE.md, project_overview, paper drafts) are never swept and silently rot.

A plain value-set membership check is useless here: the eval json holds ~1000
numbers, so almost any 2-decimal literal coincidentally matches some unrelated
real metric and never flags. So this checker is SEMANTIC instead: each CLAIM
identifies lines asserting a specific fact (e.g. "System G Telugu exact match")
and the json path that fact must equal. A matching line whose number disagrees
with the json (at 2dp) is flagged as stale.

Add a CLAIM whenever a number starts getting copied around. Precise, low-noise.

Usage:
    python experiments/rigor/check_metric_drift.py
    python experiments/rigor/check_metric_drift.py --add <glob> ...
Exit 1 if any flags (gate-friendly).
"""
import argparse
import glob
import json
import os
import re
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
JSON_PATH = os.path.join(REPO, "results", "pipeline_eval", "pipeline_eval_results.json")
# run_08 diagnostic JSON — optional, skipped if not yet committed.
JSON_RUN08 = os.path.join(REPO, "Additional_Rigor_Experiments", "results",
                          "run_08_extended_gold_s42_123_7.json")
MEM = os.path.expanduser(
    "~/.claude/projects/-Users-shishirmaddineni-Desktop-Idiomator-Research-"
    "Research-And-Training/memory")

DECIMAL = re.compile(r"\b\d\.\d{2,4}\b")

# Each claim: a line matches if ALL `needs` regexes hit (case-insensitive); then
# the line's decimal literal(s) must include `expected` (2dp str from json).
# `expected_path` pulls ground truth live so the checker never hardcodes either.
CLAIMS = [
    {
        "name": "System G Telugu span exact",
        "needs": [r"telugu|\bTE\b", r"\bBIO\b|system\s*g", r"exact|\bEM\b"],
        "path": ["system_g_bio_tagger", "span_e2e", "exact", "Telugu"],
        "json": JSON_PATH,
    },
    {
        "name": "System E Telugu Joint F1",
        "needs": [r"telugu|\bTE\b", r"system\s*e", r"joint\s*f1"],
        "path": ["system_e_joint_end_to_end", "joint_f1", "Telugu"],
        "json": JSON_PATH,
    },
    {
        "name": "System E Indonesian zero-shot Joint F1",
        "needs": [r"indonesian", r"system\s*e", r"joint\s*f1|zero.?shot"],
        "path": ["system_e_joint_end_to_end", "joint_f1", "Indonesian"],
        "json": JSON_PATH,
    },
    # run_08 extended-gold claims (active once run_08_extended_gold_s42.json is committed).
    {
        "name": "XLM-R strip gap (SP QA-BIO artifact verdict)",
        "needs": [r"xlm.?r|xlmr", r"strip", r"gap|mean"],
        "path": ["encoders", "xlmr", "mean_gap_by_mode", "strip"],
        "json": JSON_RUN08,
    },
    {
        "name": "XLM-R original EM gap (SP original)",
        "needs": [r"xlm.?r|xlmr", r"original", r"gap|mean"],
        "path": ["encoders", "xlmr", "mean_gap_by_mode", "original"],
        "json": JSON_RUN08,
    },
    {
        "name": "SP family strip gap (Scenario A verdict number)",
        "needs": [r"sentencepiece|sp\s+family|sp=", r"strip"],
        "path": ["family_mean_gap", "SentencePiece", "strip"],
        "json": JSON_RUN08,
    },
]

DEFAULT_TARGETS = [
    "experiments/rigor/*.sh",
    "experiments/rigor/*.ipynb",
    "CLAUDE.md",
    "experiments/rigor/CLAUDE.md",
]


def scalar(v):
    if isinstance(v, (int, float)):
        return v
    if isinstance(v, dict):
        return v.get("macro_f1", v.get("f1"))
    return None


def get(d, path):
    cur = d
    for p in path:
        if not isinstance(cur, dict) or p not in cur:
            return None
        cur = cur[p]
    return scalar(cur)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--add", nargs="*", default=[], help="extra globs (relative to repo)")
    args = ap.parse_args()

    if not os.path.exists(JSON_PATH):
        sys.exit(f"ground-truth json not found: {JSON_PATH}")

    # load each unique JSON once
    _loaded = {}
    def load_json(path):
        if path not in _loaded:
            _loaded[path] = json.load(open(path)) if os.path.exists(path) else None
        return _loaded[path]

    # resolve each claim's expected 2dp value from its own json (skip if json absent)
    active_claims = []
    for c in CLAIMS:
        json_path = c.get("json", JSON_PATH)
        data = load_json(json_path)
        if data is None:
            continue  # optional JSON not yet committed — skip silently
        v = get(data, c["path"])
        c["expected"] = f"{v:.2f}" if isinstance(v, (int, float)) else None
        c["needs_re"] = [re.compile(n, re.I) for n in c["needs"]]
        active_claims.append(c)
    CLAIMS[:] = active_claims

    files = []
    for pat in DEFAULT_TARGETS + args.add:
        files += glob.glob(os.path.join(REPO, pat))
    # also scan memory markdown
    files += glob.glob(os.path.join(MEM, "*.md"))

    flags = []
    for f in sorted(set(files)):
        try:
            lines = open(f, encoding="utf-8").read().splitlines()
        except Exception:
            continue
        for i, line in enumerate(lines, 1):
            if "metric-drift:ok" in line:  # intentional debunk / historical quote
                continue
            for c in CLAIMS:
                if c["expected"] is None:
                    continue
                if not all(r.search(line) for r in c["needs_re"]):
                    continue
                lits = DECIMAL.findall(line)
                if not lits:
                    continue
                # pass if any literal equals expected at 2dp
                ok = any(f"{float(t):.2f}" == c["expected"] for t in lits)
                if not ok:
                    flags.append((os.path.relpath(f, REPO) if f.startswith(REPO) else f,
                                  i, c["name"], c["expected"], sorted(set(lits)),
                                  line.strip()[:80]))

    if not flags:
        print(f"✓ no metric drift across {len(CLAIMS)} active claims.")
        return 0
    print(f"⚠ {len(flags)} stale metric claim(s) — fix to match ground-truth json:\n")
    for path, ln, name, exp, lits, ctx in flags:
        print(f"  {path}:{ln}")
        print(f"     claim: {name} — expected {exp}, found {lits}")
        print(f"     line:  {ctx}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
