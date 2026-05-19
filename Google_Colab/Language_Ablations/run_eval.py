"""
run_eval.py

Runs Full_evaluation.py for all 15 language combos, then summarizes the matrix.
Run this after all 5 training scripts have completed.
Same setup/run pattern as run_stage1.py — just swap the script name.
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
    p.add_argument("--ablation_dir", default="models/language_ablation_matrix")
    p.add_argument("--eval_dir",     default="results/language_ablation_matrix")
    p.add_argument("--log_dir",      default="results/language_ablation_matrix/logs")
    p.add_argument("--checkpoint_dir", default="results/language_ablation_matrix/job_checkpoints")
    p.add_argument("--only_combo",   default=None)
    p.add_argument("--keep_checkpoints", action="store_true")
    return p.parse_args()


def checkpoint_path(args: argparse.Namespace, job: Job) -> Path:
    return Path(args.checkpoint_dir) / f"{job.name}.json"


def checkpoint_done(args: argparse.Namespace, job: Job) -> bool:
    path = checkpoint_path(args, job)
    if not path.exists():
        return False
    try:
        with path.open(encoding="utf-8") as f:
            return json.load(f).get("status") == "done"
    except (json.JSONDecodeError, OSError):
        return False


def expected_done(args: argparse.Namespace, job: Job) -> bool:
    return (Path(args.eval_dir) / job.combo / "pipeline_eval_results.json").exists()


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


def cleanup_checkpoints(args: argparse.Namespace, combo: str) -> None:
    if args.keep_checkpoints:
        return
    base = Path(args.ablation_dir) / combo
    for rel in [
        "stage1_mbert/best_model",
        "stage2_mbert/best_model",
        "joint_mbert/best_model",
        "sequential_mbert/phase1/best_model",
        "sequential_mbert/phase2/best_model",
        "bio_tagger/best_model",
    ]:
        import shutil
        shutil.rmtree(base / rel, ignore_errors=True)


def main() -> None:
    args = parse_args()
    args.root = str(Path(args.root).resolve())

    for d in [args.eval_dir, args.log_dir, args.checkpoint_dir]:
        Path(d).mkdir(parents=True, exist_ok=True)

    print(f"System        : {SYSTEM}")
    print(f"Root          : {args.root}")

    jobs = [
        Job(combo=combo)
        for combo, _ in COMBOS
        if not args.only_combo or combo == args.only_combo
    ]

    completed = 0
    for job in tqdm(jobs, desc="eval jobs", unit="combo"):
        if checkpoint_done(args, job) or expected_done(args, job):
            mark_done(args, job)
            tqdm.write(f"✓ skip {job.name}")
            completed += 1
            continue

        base = Path(args.ablation_dir) / job.combo
        eval_out = str(Path(args.eval_dir) / job.combo)
        Path(eval_out).mkdir(parents=True, exist_ok=True)

        cmd = [
            args.python, "-u", "Evaluation/Full_evaluation.py",
            "--stage1_mbert", str(base / "stage1_mbert/test_predictions.jsonl"),
            "--stage2_mbert", str(base / "stage2_mbert/test_predictions.jsonl"),
            "--stage1_gpt",   str(base / "missing_gpt_stage1.jsonl"),
            "--stage2_gpt",   str(base / "missing_gpt_stage2.jsonl"),
            "--single_gpt",   str(base / "missing_gpt_single.jsonl"),
            "--joint_preds",  str(base / "joint_mbert/test_predictions.jsonl"),
            "--span2_joint",  str(base / "missing_joint_span_only.jsonl"),
            "--seq_phase1",   str(base / "sequential_mbert/phase1/test_predictions.jsonl"),
            "--seq_phase2",   str(base / "sequential_mbert/phase2/test_predictions.jsonl"),
            "--bio_preds",    str(base / "bio_tagger/test_predictions.jsonl"),
            "--output_dir",   eval_out,
        ]
        run_live(args, job, cmd)
        mark_done(args, job)
        cleanup_checkpoints(args, job.combo)
        completed += 1

    # Summarize the full matrix once all combos are done
    if not args.only_combo:
        print("\nSummarizing ablation matrix...", flush=True)
        subprocess.run(
            [args.python, "-u", "Google_Colab/summarize_language_ablation_matrix.py",
             "--eval_dir", args.eval_dir, "--output_dir", args.eval_dir],
            cwd=args.root, check=True,
        )
        print(f"Summary: {args.eval_dir}/ablation_summary.csv")

    print(f"\nDone. Completed/skipped {completed}/{len(jobs)} combos.")


if __name__ == "__main__":
    main()
