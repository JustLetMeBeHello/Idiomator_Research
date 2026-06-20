#!/usr/bin/env python
"""
One-shot fix for the buggy System G (BIO tagger) language-ablation matrix.

Background
----------
The committed 15-combo matrix in `results/language_ablation_matrix/<combo>/
pipeline_eval_results.json` was produced (2026-05-22/23) BEFORE the BIO
decoder fix (multi-subtoken endword truncation) landed in
`experiments/ablations/BiO_Task_mBERT_train.py`. So every combo-level System G number
(span, Joint F1, stability) is a buggy-decoder artifact — e.g. Telugu e2e
span overlap = 0.2673 instead of ~0.88.

Systems A/E/F are NOT affected by the BIO bug, and their per-combo prediction
files have largely been deleted to save Drive space. So we CANNOT re-run the
matrix eval wholesale (it would reload the missing A/E/F preds as empty and
overwrite their correct numbers with "—").

This script therefore fixes ONLY System G, in place:
  1. Delete stale BIO checkpoint sentinels so retrain is not skipped.
  2. Retrain BIO across all 15 combos (fixed decoder) via the matrix harness.
  3. Run a System-G-only eval per combo (all other systems pointed at a
     nonexistent file -> evaluate_system_x returns None, harmless).
     `evaluate_system_g` is fully self-contained in the bio preds jsonl.
  4. Patch the corrected `system_g_bio_tagger` block into each combo's EXISTING
     pipeline_eval_results.json (preserving the correct A/E/F blocks).
  5. Regenerate the paper CSVs (ablation_summary / stability_summary /
     transfer_matrix).

Run from the repo root on Colab:
    python -u notebooks/colab/fix_system_g_combos.py
"""
import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

# 15 training-language combos (must match COMBOS in run_language_ablation_matrix.py)
COMBOS = [
    "en", "es", "hi", "te",
    "en_es", "en_hi", "en_te", "es_hi", "es_te", "hi_te",
    "en_es_hi", "en_es_te", "en_hi_te", "es_hi_te", "en_es_hi_te",
]

# A path that is guaranteed not to exist -> load_jsonl() prints a warning and
# returns {} -> evaluate_system_{a,b,c,d,e,f} return None (shown as "—").
NONE = "__nonexistent_placeholder__.jsonl"


def sh(cmd, cwd):
    print(f"\n$ {' '.join(str(c) for c in cmd)}", flush=True)
    subprocess.run(cmd, cwd=cwd, check=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", default=".", help="Research_And_Training dir")
    p.add_argument("--python", default=sys.executable)
    p.add_argument("--ablation_dir", default="models/language_ablation_matrix")
    p.add_argument("--eval_dir", default="results/language_ablation_matrix")
    p.add_argument("--skip_retrain", action="store_true",
                   help="Skip steps 1-2 (BIO already retrained); only re-eval+patch+summarize.")
    args = p.parse_args()

    root = Path(args.root).resolve()
    abl = root / args.ablation_dir
    evd = root / args.eval_dir
    py = args.python

    print(f"Root         : {root}")
    print(f"Ablation dir : {abl}")
    print(f"Eval dir     : {evd}")

    # ── Step 1: delete stale BIO outputs so the harness actually retrains ──────
    # run_language_ablation_matrix.py skips a job when EITHER the sentinel says
    # "done" (checkpoint_done) OR the model outputs already exist (expected_done:
    # bio_tagger/metrics.json + test_predictions.jsonl). Deleting only the
    # sentinel is NOT enough — stale buggy-decoder preds still satisfy
    # expected_done and the retrain is silently skipped. Delete BOTH.
    if not args.skip_retrain:
        ckpt_dir = evd / "job_checkpoints"
        removed = 0
        if ckpt_dir.exists():
            for f in ckpt_dir.glob("*__bio.json"):
                f.unlink()
                removed += 1
        print(f"\n[1] Removed {removed} stale '*__bio.json' sentinels from {ckpt_dir}")

        wiped = 0
        for c in COMBOS:
            bio_dir = abl / c / "bio_tagger"
            if bio_dir.exists():
                shutil.rmtree(bio_dir)
                wiped += 1
        print(f"[1] Wiped {wiped} stale bio_tagger/ dirs (forces expected_done=False -> retrain)")

        # ── Step 2: retrain BIO across all 15 combos (fixed decoder) ──────────
        print("\n[2] Retraining BIO across all combos (fixed decoder)...")
        sh([py, "-u", "notebooks/colab/run_language_ablation_matrix.py",
            "--only_system", "bio",
            "--ablation_dir", args.ablation_dir,
            "--eval_dir", args.eval_dir], cwd=root)
    else:
        print("\n[1-2] --skip_retrain set: assuming BIO preds already regenerated.")

    # ── Step 3: System-G-only eval per combo ──────────────────────────────────
    print("\n[3] Running System-G-only eval per combo...")
    for c in COMBOS:
        bio_preds = abl / c / "bio_tagger" / "test_predictions.jsonl"
        if not bio_preds.exists():
            print(f"  ⚠ SKIP {c}: missing {bio_preds} (BIO retrain failed?)")
            continue
        out = evd / c / "g_only"
        out.mkdir(parents=True, exist_ok=True)
        sh([py, "-u", "Evaluation/Full_evaluation.py",
            "--bio_preds", str(bio_preds),
            "--stage1_mbert", NONE, "--stage2_mbert", NONE,
            "--joint_preds", NONE, "--span2_joint", NONE,
            "--seq_phase1", NONE, "--seq_phase2", NONE,
            "--stage1_gpt", NONE, "--stage2_gpt", NONE, "--single_gpt", NONE,
            "--output_dir", str(out)], cwd=root)

    # ── Step 4: patch corrected G block into existing combo results ───────────
    print("\n[4] Patching corrected system_g_bio_tagger into existing results...")
    patched, skipped = 0, 0
    for c in COMBOS:
        main_json = evd / c / "pipeline_eval_results.json"
        gonly_json = evd / c / "g_only" / "pipeline_eval_results.json"
        if not (main_json.exists() and gonly_json.exists()):
            print(f"  ⚠ SKIP {c}: missing {'main' if not main_json.exists() else 'g_only'} json")
            skipped += 1
            continue
        m = json.loads(main_json.read_text())
        g = json.loads(gonly_json.read_text())
        if "system_g_bio_tagger" not in g:
            print(f"  ⚠ SKIP {c}: g_only json has no system_g_bio_tagger block")
            skipped += 1
            continue
        m["system_g_bio_tagger"] = g["system_g_bio_tagger"]
        main_json.write_text(json.dumps(m, indent=2))
        patched += 1
        print(f"  ✓ patched {c}")
    print(f"  Patched {patched}, skipped {skipped}")

    # ── Step 5: regenerate paper CSVs ─────────────────────────────────────────
    print("\n[5] Regenerating summary CSVs...")
    sh([py, "-u", "notebooks/colab/summarize_language_ablation_matrix.py",
        "--eval_dir", args.eval_dir,
        "--output_dir", args.eval_dir], cwd=root)

    print("\nDone. Clean CSVs:")
    print(f"  {evd}/ablation_summary.csv")
    print(f"  {evd}/stability_summary.csv")
    print(f"  {evd}/transfer_matrix.csv")


if __name__ == "__main__":
    main()
