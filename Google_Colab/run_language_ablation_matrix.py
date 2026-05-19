"""
run_language_ablation_matrix.py

Colab-friendly ablation runner with tqdm progress and resumable checkpoints.
This is the Python counterpart to run_language_ablation_matrix.sh, modeled after
the sequential hyperparameter tuner style:

  - visible tqdm progress across combo/system jobs
  - live subprocess output in Colab
  - one JSON checkpoint per completed job
  - safe to rerun after a Colab timeout
  - optional checkpoint cleanup to save Google Drive space

Usage:
    cd /content/drive/MyDrive/Idiomator_Research/Research_And_Training
    python -u Google_Colab/run_language_ablation_matrix.py

Useful options:
    python -u Google_Colab/run_language_ablation_matrix.py --only_combo en_es
    python -u Google_Colab/run_language_ablation_matrix.py --only_system joint
    python -u Google_Colab/run_language_ablation_matrix.py --keep_checkpoints
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from tqdm import tqdm


COMBOS = [
    ("en", ["English"]),
    ("es", ["Spanish"]),
    ("hi", ["Hindi"]),
    ("te", ["Telugu"]),
    ("en_es", ["English", "Spanish"]),
    ("en_hi", ["English", "Hindi"]),
    ("en_te", ["English", "Telugu"]),
    ("es_hi", ["Spanish", "Hindi"]),
    ("es_te", ["Spanish", "Telugu"]),
    ("hi_te", ["Hindi", "Telugu"]),
    ("en_es_hi", ["English", "Spanish", "Hindi"]),
    ("en_es_te", ["English", "Spanish", "Telugu"]),
    ("en_hi_te", ["English", "Hindi", "Telugu"]),
    ("es_hi_te", ["Spanish", "Hindi", "Telugu"]),
    ("en_es_hi_te", ["English", "Spanish", "Hindi", "Telugu"]),
]

ALL_TEST_LANGS = ["English", "Spanish", "Hindi", "Telugu"]


@dataclass(frozen=True)
class Job:
    combo: str
    langs: tuple[str, ...]
    system: str

    @property
    def name(self) -> str:
        return f"{self.combo}__{self.system}"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--root", default=".", help="Research_And_Training directory")
    p.add_argument("--python", default=sys.executable)
    p.add_argument("--data_dir", default="idioms_structured/Splits")
    p.add_argument("--model_name", default="bert-base-multilingual-cased")
    p.add_argument("--device", default="cuda")
    p.add_argument("--ablation_dir", default="models/language_ablation_matrix")
    p.add_argument("--eval_dir", default="results/language_ablation_matrix")
    p.add_argument("--log_dir", default="results/language_ablation_matrix/logs")
    p.add_argument("--checkpoint_dir", default="results/language_ablation_matrix/job_checkpoints")
    p.add_argument("--test_langs", nargs="+", default=ALL_TEST_LANGS)
    p.add_argument("--only_combo", default=None)
    p.add_argument(
        "--only_system",
        choices=["stage1", "stage2", "joint", "sequential", "bio", "eval"],
        default=None,
    )
    p.add_argument("--keep_checkpoints", action="store_true")

    # Locked paper/default hyperparameters.
    p.add_argument("--stage1_lr", default="3e-5")
    p.add_argument("--stage1_epochs", default="7")
    p.add_argument("--stage1_batch", default="32")
    p.add_argument("--stage2_lr", default="1e-5")
    p.add_argument("--stage2_epochs", default="7")
    p.add_argument("--stage2_batch", default="32")

    p.add_argument("--joint_lr", default="2e-5")
    p.add_argument("--joint_epochs", default="7")
    p.add_argument("--joint_batch", default="32")
    p.add_argument("--joint_cls_weight", default="0.3")
    p.add_argument("--joint_span_weight", default="1.9")

    p.add_argument("--seq_p1_lr", default="1e-5")
    p.add_argument("--seq_p1_epochs", default="7")
    p.add_argument("--seq_p1_batch", default="32")
    p.add_argument("--seq_p1_cls_weight", default="0.7")
    p.add_argument("--seq_p1_span_weight", default="0.3")
    p.add_argument("--seq_p2_lr", default="3e-5")
    p.add_argument("--seq_p2_epochs", default="5")
    p.add_argument("--seq_p2_batch", default="16")
    p.add_argument("--seq_p2_cls_weight", default="0.3")
    p.add_argument("--seq_p2_span_weight", default="0.7")
    p.add_argument("--seq_unfreeze_top", default="3")
    p.add_argument("--seq_dropout", default="0.1")

    p.add_argument("--bio_lr", default="3.27e-5")
    p.add_argument("--bio_epochs", default="6")
    p.add_argument("--bio_batch", default="32")
    p.add_argument("--bio_dropout", default="0.239431179668018881")
    p.add_argument("--bio_o_weight", default="0.104")
    return p.parse_args()


def checkpoint_path(args: argparse.Namespace, job: Job) -> Path:
    return Path(args.checkpoint_dir) / f"{job.name}.json"


def expected_done(args: argparse.Namespace, job: Job) -> bool:
    base = Path(args.ablation_dir) / job.combo
    eval_base = Path(args.eval_dir) / job.combo
    paths = {
        "stage1": [base / "stage1_mbert/metrics.json", base / "stage1_mbert/test_predictions.jsonl"],
        "stage2": [base / "stage2_mbert/metrics.json", base / "stage2_mbert/test_predictions.jsonl"],
        "joint": [base / "joint_mbert/metrics.json", base / "joint_mbert/test_predictions.jsonl"],
        "sequential": [
            base / "sequential_mbert/phase1/metrics.json",
            base / "sequential_mbert/phase1/test_predictions.jsonl",
            base / "sequential_mbert/phase2/metrics.json",
            base / "sequential_mbert/phase2/test_predictions.jsonl",
        ],
        "bio": [base / "bio_tagger/metrics.json", base / "bio_tagger/test_predictions.jsonl"],
        "eval": [eval_base / "pipeline_eval_results.json"],
    }
    return all(p.exists() for p in paths[job.system])


def checkpoint_done(args: argparse.Namespace, job: Job) -> bool:
    """Fast-path resume: trust the sentinel JSON written by a previous run."""
    path = checkpoint_path(args, job)
    if not path.exists():
        return False
    try:
        with path.open(encoding="utf-8") as f:
            data = json.load(f)
        return data.get("status") == "done"
    except (json.JSONDecodeError, OSError):
        return False


def mark_done(args: argparse.Namespace, job: Job) -> None:
    path = checkpoint_path(args, job)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump({"combo": job.combo, "system": job.system, "status": "done"}, f, indent=2)


def run_live(args: argparse.Namespace, job: Job, cmd: list[str]) -> None:
    log_path = Path(args.log_dir) / f"{job.name}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    print("\n" + "=" * 80, flush=True)
    print(job.name, flush=True)
    print(" ".join(cmd), flush=True)
    print("=" * 80, flush=True)

    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    with log_path.open("w", encoding="utf-8") as log:
        proc = subprocess.Popen(
            cmd,
            cwd=args.root,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=env,
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
        shutil.rmtree(base / rel, ignore_errors=True)


def cmd_for(args: argparse.Namespace, job: Job) -> list[str]:
    py = args.python
    base = Path(args.ablation_dir) / job.combo
    common = [
        "--model_name", args.model_name,
        "--data_dir", args.data_dir,
        "--langs", *job.langs,
        "--test_langs", *args.test_langs,
        "--device", args.device,
    ]

    if job.system == "stage1":
        return [
            py, "-u", "Base_Pipeline/Stage_1_training.py",
            *common,
            "--output_dir", str(base / "stage1_mbert"),
            "--epochs", args.stage1_epochs,
            "--batch_size", args.stage1_batch,
            "--lr", args.stage1_lr,
        ]
    if job.system == "stage2":
        return [
            py, "-u", "Base_Pipeline/Stage_2_training.py",
            *common,
            "--output_dir", str(base / "stage2_mbert"),
            "--epochs", args.stage2_epochs,
            "--batch_size", args.stage2_batch,
            "--lr", args.stage2_lr,
        ]
    if job.system == "joint":
        return [
            py, "-u", "Train_Join.py",
            *common,
            "--output_dir", str(base / "joint_mbert"),
            "--epochs", args.joint_epochs,
            "--batch_size", args.joint_batch,
            "--lr", args.joint_lr,
            "--cls_loss_weight", args.joint_cls_weight,
            "--span_loss_weight", args.joint_span_weight,
        ]
    if job.system == "sequential":
        return [
            py, "-u", "Train_Sequential.py",
            *common,
            "--output_dir", str(base / "sequential_mbert"),
            "--p1_epochs", args.seq_p1_epochs,
            "--p1_batch_size", args.seq_p1_batch,
            "--p1_lr", args.seq_p1_lr,
            "--p1_cls_weight", args.seq_p1_cls_weight,
            "--p1_span_weight", args.seq_p1_span_weight,
            "--p2_epochs", args.seq_p2_epochs,
            "--p2_batch_size", args.seq_p2_batch,
            "--p2_lr", args.seq_p2_lr,
            "--p2_cls_weight", args.seq_p2_cls_weight,
            "--p2_span_weight", args.seq_p2_span_weight,
            "--unfreeze_top_layers", args.seq_unfreeze_top,
            "--dropout", args.seq_dropout,
        ]
    if job.system == "bio":
        return [
            py, "-u", "Ablations/BiO_Task_mBERT_train.py",
            *common,
            "--output_dir", str(base / "bio_tagger"),
            "--epochs", args.bio_epochs,
            "--batch_size", args.bio_batch,
            "--lr", args.bio_lr,
            "--dropout", args.bio_dropout,
            "--o_weight", args.bio_o_weight,
        ]

    eval_out = Path(args.eval_dir) / job.combo
    return [
        py, "-u", "Evaluation/Full_evaluation.py",
        "--stage1_mbert", str(base / "stage1_mbert/test_predictions.jsonl"),
        "--stage2_mbert", str(base / "stage2_mbert/test_predictions.jsonl"),
        "--stage1_gpt", str(base / "missing_gpt_stage1.jsonl"),
        "--stage2_gpt", str(base / "missing_gpt_stage2.jsonl"),
        "--single_gpt", str(base / "missing_gpt_single.jsonl"),
        "--joint_preds", str(base / "joint_mbert/test_predictions.jsonl"),
        "--span2_joint", str(base / "missing_joint_span_only.jsonl"),
        "--seq_phase1", str(base / "sequential_mbert/phase1/test_predictions.jsonl"),
        "--seq_phase2", str(base / "sequential_mbert/phase2/test_predictions.jsonl"),
        "--bio_preds", str(base / "bio_tagger/test_predictions.jsonl"),
        "--output_dir", str(eval_out),
    ]


def build_jobs(args: argparse.Namespace) -> list[Job]:
    jobs = []
    systems = ["stage1", "stage2", "joint", "sequential", "bio", "eval"]
    for combo, langs in COMBOS:
        if args.only_combo and combo != args.only_combo:
            continue
        for system in systems:
            if args.only_system and system != args.only_system:
                continue
            jobs.append(Job(combo=combo, langs=tuple(langs), system=system))
    return jobs


def summarize(args: argparse.Namespace) -> None:
    cmd = [
        args.python, "-u", "Google_Colab/summarize_language_ablation_matrix.py",
        "--eval_dir", args.eval_dir,
        "--output_dir", args.eval_dir,
    ]
    print("\nSummarizing ablation matrix...", flush=True)
    subprocess.run(cmd, cwd=args.root, check=True)


def main() -> None:
    args = parse_args()
    args.root = str(Path(args.root).resolve())
    Path(args.ablation_dir).mkdir(parents=True, exist_ok=True)
    Path(args.eval_dir).mkdir(parents=True, exist_ok=True)
    Path(args.log_dir).mkdir(parents=True, exist_ok=True)
    Path(args.checkpoint_dir).mkdir(parents=True, exist_ok=True)

    print(f"Root          : {args.root}")
    print(f"Device        : {args.device}")
    print(f"Ablation dir  : {args.ablation_dir}")
    print(f"Eval dir      : {args.eval_dir}")
    print(f"Keep ckpts    : {args.keep_checkpoints}")

    jobs = build_jobs(args)
    completed = 0

    for job in tqdm(jobs, desc="Ablation jobs", unit="job"):
        if checkpoint_done(args, job) or expected_done(args, job):
            mark_done(args, job)
            tqdm.write(f"✓ skip {job.name}")
            completed += 1
            continue

        run_live(args, job, cmd_for(args, job))
        mark_done(args, job)
        completed += 1
        if job.system == "eval":
            cleanup_checkpoints(args, job.combo)

    if not args.only_combo and not args.only_system:
        summarize(args)

    print(f"\nCompleted/skipped {completed}/{len(jobs)} jobs.")
    print(f"Summaries: {args.eval_dir}/ablation_summary.csv")


if __name__ == "__main__":
    main()
