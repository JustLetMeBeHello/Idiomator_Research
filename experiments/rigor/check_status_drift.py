#!/usr/bin/env python3
"""Tripwire for stale STATUS claims (experiment done/pending drift).

Companion to check_metric_drift.py. That tool catches wrong *numbers*; this one
catches wrong *state* — a doc saying an experiment is "TODO / pending / not run"
after its result artifact already exists on disk, or claiming "DONE" when no
artifact is there. Both directions silently rot the planning docs (REVISION_PLAN,
CODEBASE_MAP, CLAUDE.md, project_overview) across sessions — exactly the drift
that had E2/N2 reading "pending" days after they landed, and the ablation note
reading "don't use" after the fix was confirmed.

Deterministic — no LLM. Ground truth = file existence on disk, never prose.
A tracked item is "done" iff ALL its `done_if` globs match at least one file.
Then every doc line matching the item's `needs` regexes is checked:
  - item DONE  + line has a PENDING word (and no DONE word) -> STALE  (fix the doc)
  - item NOT   + line has a DONE word    (and no PENDING)   -> OVERCLAIM (no evidence)

Lines containing `status-drift:ok` are skipped (intentional historical wording).
Add a TRACKED item whenever a new experiment's status starts being copied around.

Usage:
    .venv/bin/python3 experiments/rigor/check_status_drift.py
Exit 1 if any flags (gate-friendly; run before any handoff/commit/paper build).
"""
import glob
import os
import re
import sys

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# REPO = .../Idiomator_Research/IdiomBERT/experiments; umbrella is two levels up.
UMBRELLA = os.path.dirname(os.path.dirname(REPO))
PAPER = os.path.join(UMBRELLA, "Paper_Drafts")
ROOTS = {"repo": REPO, "paper": PAPER}

PENDING = re.compile(
    r"\b(TODO|pending|unrun|not[\s-]+(?:yet[\s-]+)?(?:run|built|done)|"
    r"GPU run pending|in progress|don'?t use)\b", re.I)
DONE = re.compile(r"(\bDONE\b|✅|\bcomplete(?:d)?\b|\bconfirmed\b)")

# Each tracked item: file-existence ground truth + the doc lines that track it.
# `done_if`: globs relative to repo root; item is DONE iff ALL match >=1 file.
# `docs`: (root_key, relpath) doc files to scan.
# `needs`: regexes that ALL must hit on a line for it to count as a status line.
TRACKED = [
    {
        "name": "E2 multi-seed main A-G tables (seeds 123/7)",
        "done_if": ["results/pipeline_eval_s123/pipeline_eval_results.json",
                    "results/pipeline_eval_s7/pipeline_eval_results.json"],
        "docs": [("paper", "REVISION_PLAN.md"), ("repo", "docs/CODEBASE_MAP.md")],
        "needs": [r"\bE2\b", r"multi.?seed|main.?table|A.?G table"],
    },
    {
        "name": "E1 RemBERT 3-seed @ batch32 (real run, not dry-run)",
        # non-dry-run output dir: name not starting with "_"
        "done_if": ["models/rembert32/[!_]*/metrics.json"],
        "docs": [("paper", "REVISION_PLAN.md"), ("repo", "docs/CODEBASE_MAP.md")],
        # anchor on E1's OWN row, not the run_08 row that mentions E1's layout
        "needs": [r"\bE1\b", r"rerun RemBERT|run_07b|3 seeds.{0,15}batch|RemBERT.{0,15}batch.?32"],
    },
    {
        "name": "N1 TOST equivalence (run_13)",
        "done_if": ["experiments/rigor/results/run_13*.json"],
        "docs": [("paper", "REVISION_PLAN.md"), ("repo", "docs/CODEBASE_MAP.md")],
        "needs": [r"\bN1\b", r"TOST|equivalence"],
    },
    {
        "name": "N2 Holm/BH correction (run_14, all seeds)",
        "done_if": ["experiments/rigor/results/run_14_s42.json",
                    "experiments/rigor/results/run_14_s123.json",
                    "experiments/rigor/results/run_14_s7.json"],
        "docs": [("paper", "REVISION_PLAN.md"), ("repo", "docs/CODEBASE_MAP.md")],
        "needs": [r"\bN2\b", r"holm|bonferroni|\bBH\b"],
    },
]
# NOTE: free-prose memory files (project_overview.md etc.) are deliberately NOT
# scanned — status words co-occur incidentally in narrative prose (false flags),
# and those files are hand-maintained as canonical state via the handoff step.
# This tool guards the *derived* planning docs (REVISION_PLAN, CODEBASE_MAP),
# which is where status drifts behind reality between sessions.


def is_done(item):
    return all(glob.glob(os.path.join(REPO, g)) for g in item["done_if"])


def scan():
    flags = []
    for item in TRACKED:
        done = is_done(item)
        needs = [re.compile(n, re.I) for n in item["needs"]]
        for root_key, rel in item["docs"]:
            path = os.path.join(ROOTS[root_key], rel)
            if not os.path.exists(path):
                continue
            for i, line in enumerate(open(path, encoding="utf-8").read().splitlines(), 1):
                if "status-drift:ok" in line:
                    continue
                if not all(r.search(line) for r in needs):
                    continue
                has_pending, has_done = PENDING.search(line), DONE.search(line)
                if done and has_pending and not has_done:
                    flags.append(("STALE", path, i, item["name"],
                                  "artifact exists but line still says pending", line))
                elif not done and has_done and not has_pending:
                    flags.append(("OVERCLAIM", path, i, item["name"],
                                  "line claims done but no artifact on disk", line))
    return flags


def main():
    flags = scan()
    if not flags:
        print(f"✓ no status drift across {len(TRACKED)} tracked items.")
        return 0
    print(f"⚠ {len(flags)} status-drift flag(s):\n")
    for kind, path, ln, name, why, line in flags:
        rel = os.path.relpath(path, UMBRELLA)
        print(f"  [{kind}] {rel}:{ln}")
        print(f"     item: {name}")
        print(f"     why:  {why}")
        print(f"     line: {line.strip()[:90]}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
