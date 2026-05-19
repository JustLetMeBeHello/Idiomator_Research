"""
run_sequential.py

Trains Sequential mBERT (phase 1 + phase 2) for all 15 language combos.
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

ALL_TEST_LANGS = ["English", "Spanish", "Hindi", "Telugu"]
SYSTEM = "sequential"


@dataclass(frozen=True)
class Job:
    combo: str
    langs: tuple[str, ...]

    @property
    def name(self) -> str:
        return f"{self.combo}__{SYSTEM}"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--root",         default=".")
    p.add_argument("--python",       default=sys.executable)
    p.add_argument("--data_dir",     default="idioms_structured/Splits")
    p.add_argument("--model_name",   default="bert-base-multilingual-cased")
    p.add_argument("--device",       default="cuda")
    p.add_argument("--ablation_dir", default="models/language_ablation_matrix")
    p.add_argument("--log_dir",      default="results/language_ablation_matrix/logs")
    p.add_argument("--checkpoint_dir", default="results/language_ablation_matrix/job_checkpoints")
    p.add_argument("--test_langs",   nargs="+", default=ALL_TEST_LANGS)
    p.add_argument("--only_combo",   default=None)
    p.add_argument("--keep_checkpoints", action="store_true")
    # Phase 1 hyperparameters
    p.add_argument("--p1_lr",         default="1e-5")
    p.add_argument("--p1_epochs",     default="7")
    p.add_argument("--p1_batch",      default="32")
    p.add_argument("--p1_cls_weight", default="0.7")
    p.add_argument("--p1_span_weight",default="0.3")
    # Phase 2 hyperparameters
    p.add_argument("--p2_lr",         default="3e-5")
    p.add_argument("--p2_epochs",     default="5")
    p.add_argument("--p2_batch",      default="16")
    p.add_argument("--p2_cls_weight", default="0.3")
    p.add_argument("--p2_span_weight",default="0.7")
    # Shared
    p.add_argument("--unfreeze_top",  default="3")
    p.add_argument("--dropout",       default="0.1")
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
    base = Path(args.ablation_dir) / job.combo / "sequential_mbert"
    return all((base / phase / fname).exists()
               for phase in ["phase1", "phase2"]
               for fname in ["metrics.json", "test_predictions.jsonl"])


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

    for d in [args.ablation_dir, args.log_dir, args.checkpoint_dir]:
        Path(d).mkdir(parents=True, exist_ok=True)

    print(f"System        : {SYSTEM}")
    print(f"Root          : {args.root}")
    print(f"Device        : {args.device}")

    jobs = [
        Job(combo=combo, langs=tuple(langs))
        for combo, langs in COMBOS
        if not args.only_combo or combo == args.only_combo
    ]

    completed = 0
    for job in tqdm(jobs, desc=f"{SYSTEM} jobs", unit="combo"):
        if checkpoint_done(args, job) or expected_done(args, job):
            mark_done(args, job)
            tqdm.write(f"✓ skip {job.name}")
            completed += 1
            continue

        out = str(Path(args.ablation_dir) / job.combo / "sequential_mbert")
        cmd = [
            args.python, "-u", "Train_Sequential.py",
            "--model_name",        args.model_name,
            "--data_dir",          args.data_dir,
            "--langs",             *job.langs,
            "--test_langs",        *args.test_langs,
            "--device",            args.device,
            "--output_dir",        out,
            "--p1_epochs",         args.p1_epochs,
            "--p1_batch_size",     args.p1_batch,
            "--p1_lr",             args.p1_lr,
            "--p1_cls_weight",     args.p1_cls_weight,
            "--p1_span_weight",    args.p1_span_weight,
            "--p2_epochs",         args.p2_epochs,
            "--p2_batch_size",     args.p2_batch,
            "--p2_lr",             args.p2_lr,
            "--p2_cls_weight",     args.p2_cls_weight,
            "--p2_span_weight",    args.p2_span_weight,
            "--unfreeze_top_layers", args.unfreeze_top,
            "--dropout",           args.dropout,
        ]
        run_live(args, job, cmd)
        mark_done(args, job)
        completed += 1

    print(f"\nDone. Completed/skipped {completed}/{len(jobs)} combos.")


if __name__ == "__main__":
    main()
