"""
run_eval.py

Runs Full_evaluation.py for all 15 language combos, then summarizes the matrix.

Usage (in Colab):
    !python notebooks/colab/Language_Ablations/run_eval.py \
        --ablation_dir /content/drive/MyDrive/IdiomBERT_experiments/ablations/merged \
        --eval_dir     /content/drive/MyDrive/IdiomBERT_experiments/ablations/results \
        --gpt_s1       models/gpt_baseline_stage1/test_predictions.jsonl \
        --gpt_s2       models/gpt_baseline_stage2/test_predictions.jsonl \
        --gpt_single   models/gpt_single_stage/test_predictions.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from tqdm import tqdm


COMBOS = [
    ("en",          ["English"]),
    ("es",          ["Spanish"]),
    ("hi",          ["Hindi"]),
    ("te",          ["Telugu"]),
    ("en_es",       ["English", "Spanish"]),
    ("en_hi",       ["English", "Hindi"]),
    ("en_te",       ["English", "Telugu"]),
    ("es_hi",       ["Spanish", "Hindi"]),
    ("es_te",       ["Spanish", "Telugu"]),
    ("hi_te",       ["Hindi", "Telugu"]),
    ("en_es_hi",    ["English", "Spanish", "Hindi"]),
    ("en_es_te",    ["English", "Spanish", "Telugu"]),
    ("en_hi_te",    ["English", "Hindi", "Telugu"]),
    ("es_hi_te",    ["Spanish", "Hindi", "Telugu"]),
    ("en_es_hi_te", ["English", "Spanish", "Hindi", "Telugu"]),
]

SYSTEM = "eval"


@dataclass(frozen=True)
class Job:
    combo: str

    @property
    def name(self) -> str:
        return f"{self.combo}__{SYSTEM}"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--root",         default=".")
    p.add_argument("--python",       default=sys.executable)
    p.add_argument("--ablation_dir", default="/content/drive/MyDrive/IdiomBERT_experiments/ablations/merged")
    p.add_argument("--eval_dir",     default="/content/drive/MyDrive/IdiomBERT_experiments/ablations/results")
    p.add_argument("--log_dir",      default="/content/drive/MyDrive/IdiomBERT_experiments/ablations/results/logs")
    p.add_argument("--checkpoint_dir", default="/content/drive/MyDrive/IdiomBERT_experiments/ablations/results/job_checkpoints")
    # GPT files are global (same for every combo) — lives in the repo
    p.add_argument("--gpt_s1",     default="models/gpt_baseline_stage1/test_predictions.jsonl")
    p.add_argument("--gpt_s2",     default="models/gpt_baseline_stage2/test_predictions.jsonl")
    p.add_argument("--gpt_single", default="models/gpt_single_stage/test_predictions.jsonl")
    p.add_argument("--only_combo", default=None)
    p.add_argument("--force",      action="store_true", help="Re-run even if already done")
    return p.parse_args()


def checkpoint_path(args: argparse.Namespace, job: Job) -> Path:
    return Path(args.checkpoint_dir) / f"{job.name}.json"


def is_done(args: argparse.Namespace, job: Job) -> bool:
    if args.force:
        return False
    # Only consider done if the output file actually exists and is non-empty
    out = Path(args.eval_dir) / job.combo / "pipeline_eval_results.json"
    return out.exists() and out.stat().st_size > 0


def mark_done(args: argparse.Namespace, job: Job) -> None:
    path = checkpoint_path(args, job)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump({"combo": job.combo, "system": SYSTEM, "status": "done"}, f, indent=2)


def run_live(args: argparse.Namespace, job: Job, cmd: list[str]) -> None:
    log_path = Path(args.log_dir) / f"{job.name}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"\n{'='*80}\n{job.name}\n{' '.join(cmd)}\n{'='*80}", flush=True)
    env = {**os.environ, "PYTHONUNBUFFERED": "1"}
    with log_path.open("w", encoding="utf-8") as log:
        proc = subprocess.Popen(
            cmd, cwd=args.root,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1, env=env,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            print(line, end="", flush=True)
            log.write(line)
            log.flush()
        rc = proc.wait()
    if rc != 0:
        raise subprocess.CalledProcessError(rc, cmd)


def main() -> None:
    args = parse_args()
    args.root = str(Path(args.root).resolve())

    for d in [args.eval_dir, args.log_dir, args.checkpoint_dir]:
        Path(d).mkdir(parents=True, exist_ok=True)

    print(f"System        : {SYSTEM}")
    print(f"Root          : {args.root}")
    print(f"Ablation dir  : {args.ablation_dir}")
    print(f"Eval dir      : {args.eval_dir}")
    print(f"GPT stage1    : {args.gpt_s1}")
    print(f"GPT stage2    : {args.gpt_s2}")
    print(f"GPT single    : {args.gpt_single}")

    jobs = [
        Job(combo=combo)
        for combo, _ in COMBOS
        if not args.only_combo or combo == args.only_combo
    ]

    completed = 0
    for job in tqdm(jobs, desc="eval jobs", unit="combo"):
        if is_done(args, job):
            mark_done(args, job)
            tqdm.write(f"✓ skip {job.name}")
            completed += 1
            continue

        base     = Path(args.ablation_dir) / job.combo
        eval_out = str(Path(args.eval_dir) / job.combo)
        Path(eval_out).mkdir(parents=True, exist_ok=True)

        # Resolve GPT paths — relative paths are relative to repo root
        def gpt_path(p):
            p = Path(p)
            return str(p) if p.is_absolute() else str(Path(args.root) / p)

        cmd = [
            args.python, "-u", "Evaluation/Full_evaluation.py",
            "--stage1_mbert", str(base / "stage1_mbert/test_predictions.jsonl"),
            "--stage2_mbert", str(base / "stage2_mbert/test_predictions.jsonl"),
            "--stage1_gpt",   gpt_path(args.gpt_s1),
            "--stage2_gpt",   gpt_path(args.gpt_s2),
            "--single_gpt",   gpt_path(args.gpt_single),
            "--joint_preds",  str(base / "joint_mbert/test_predictions.jsonl"),
            "--span2_joint",  str(base / "stage2_mbert/test_predictions.jsonl"),  # fallback to stage2
            "--seq_phase1",   str(base / "sequential_mbert/phase1/test_predictions.jsonl"),
            "--seq_phase2",   str(base / "sequential_mbert/phase2/test_predictions.jsonl"),
            "--bio_preds",    str(base / "bio_tagger/test_predictions.jsonl"),
            "--output_dir",   eval_out,
        ]
        run_live(args, job, cmd)
        mark_done(args, job)
        completed += 1

    # Summarize
    if not args.only_combo:
        print("\nSummarizing ablation matrix...", flush=True)
        subprocess.run(
            [args.python, "-u", "notebooks/colab/summarize_language_ablation_matrix.py",
             "--eval_dir", args.eval_dir, "--output_dir", args.eval_dir],
            cwd=args.root, check=True,
        )
        print(f"Summary: {args.eval_dir}/ablation_summary.csv")

    print(f"\nDone. Completed/skipped {completed}/{len(jobs)} combos.")


if __name__ == "__main__":
    main()