"""
sweep_joint.py

Bayesian hyperparameter optimisation for Train_Join.py using Optuna (TPE sampler).
Optimises for dev joint F1 = geometric mean of cls macro F1 and span overlap F1.

Why Optuna over grid/random search:
  - TPE builds a probabilistic model of which configs are promising
  - Each trial informs the next — it learns the landscape as it goes
  - Finds better configs in ~half the trials of random search
  - Supports pruning: cuts bad trials early to save time

Usage:
    # 20 trials, Bayesian TPE (recommended)
    python sweep_joint.py --n_trials 20

    # Resume a previous study (adds more trials to existing DB)
    python sweep_joint.py --n_trials 20 --study_name joint_sweep_v1

    # Prune bad trials early (saves ~30% time, slight risk of pruning good ones)
    python sweep_joint.py --n_trials 20 --pruning

See README_sweep.md for full Colab instructions.
"""

import json
import math
import subprocess
import argparse
import time
from pathlib import Path
from datetime import datetime

import optuna
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner


# ── Fixed config — edit to match your paths ───────────────────────────────────

FIXED = {
    'model_name': 'bert-base-multilingual-cased',
    'data_dir':   'idioms_structured/Splits',
    'langs':      ['English', 'Hindi', 'Telugu'],
    'max_len':    128,
    'seed':       42,
}

# Output dirs
SWEEP_DIR    = Path('sweep_results')
STUDY_DB     = SWEEP_DIR / 'optuna_study.db'
SUMMARY_PATH = SWEEP_DIR / 'summary.jsonl'


# ── Args ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--n_trials',     type=int,   default=20,
                   help='Number of Optuna trials to run')
    p.add_argument('--study_name',   default='joint_sweep_v1',
                   help='Study name — reuse to resume a previous sweep')
    p.add_argument('--train_script', default='Train_Join.py',
                   help='Path to Train_Join.py')
    p.add_argument('--pruning',      action='store_true',
                   help='Enable MedianPruner to cut bad trials early')
    p.add_argument('--timeout',      type=int,   default=None,
                   help='Stop after this many seconds regardless of n_trials')
    return p.parse_args()


# ── Search space ──────────────────────────────────────────────────────────────

def sample_config(trial: optuna.Trial) -> dict:
    """
    Define the hyperparameter search space.
    Optuna's TPE sampler will learn which regions are promising.
    """
    lr = trial.suggest_categorical('lr', [1e-5, 2e-5, 3e-5, 5e-5])

    # Loss weights: what matters is the RATIO, not the absolute values.
    # Parameterising as ratio + scale is cleaner than two independent floats.
    cls_weight  = trial.suggest_float('cls_loss_weight',  0.3, 2.0, step=0.1)
    span_weight = trial.suggest_float('span_loss_weight', 0.3, 2.0, step=0.1)

    warmup_ratio = trial.suggest_categorical('warmup_ratio', [0.05, 0.06, 0.1, 0.15])
    batch_size   = trial.suggest_categorical('batch_size',   [16, 32])
    epochs       = trial.suggest_categorical('epochs',       [5, 7, 10])
    dropout      = trial.suggest_float('dropout', 0.1, 0.3)

    return {
        'lr':               lr,
        'cls_loss_weight':  round(cls_weight, 2),
        'span_loss_weight': round(span_weight, 2),
        'warmup_ratio':     warmup_ratio,
        'batch_size':       batch_size,
        'epochs':           epochs,
        'dropout':          round(dropout, 3),
    }


# ── Trial runner ──────────────────────────────────────────────────────────────

def run_trial(config: dict, trial_dir: Path, train_script: str) -> dict | None:
    """Run one trial of Train_Join.py and return its metrics dict."""
    trial_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        'python', train_script,
        '--output_dir',       str(trial_dir),
        '--model_name',       FIXED['model_name'],
        '--data_dir',         FIXED['data_dir'],
        '--langs',            *FIXED['langs'],
        '--max_len',          str(FIXED['max_len']),
        '--seed',             str(FIXED['seed']),
        '--lr',               str(config['lr']),
        '--cls_loss_weight',  str(config['cls_loss_weight']),
        '--span_loss_weight', str(config['span_loss_weight']),
        '--warmup_ratio',     str(config['warmup_ratio']),
        '--batch_size',       str(config['batch_size']),
        '--epochs',           str(config['epochs']),
        '--dropout',          str(config['dropout']),
    ]

    print(f"\n{'─'*70}")
    print(f"Running: {' '.join(cmd)}")
    print(f"{'─'*70}")

    result = subprocess.run(cmd, capture_output=False)
    if result.returncode != 0:
        print(f"  ✗ Trial FAILED (exit {result.returncode})")
        return None

    metrics_path = trial_dir / 'metrics.json'
    if not metrics_path.exists():
        print(f"  ✗ metrics.json not found")
        return None

    return json.load(open(metrics_path))


# ── Optuna objective ──────────────────────────────────────────────────────────

def make_objective(args):
    def objective(trial: optuna.Trial) -> float:
        config    = sample_config(trial)
        trial_num = trial.number
        trial_dir = SWEEP_DIR / f"trial_{trial_num:03d}"

        print(f"\n{'='*70}")
        print(f"TRIAL {trial_num}  |  {datetime.now().strftime('%H:%M:%S')}")
        print(f"Config: {config}")
        print(f"{'='*70}")

        t0      = time.time()
        metrics = run_trial(config, trial_dir, args.train_script)
        elapsed = time.time() - t0

        if metrics is None:
            # Tell Optuna this trial failed — it will not sample from this region
            raise optuna.exceptions.TrialPruned()

        dev_cls_f1  = metrics.get('best_dev_cls_f1',   0.0)
        dev_joint   = metrics.get('best_dev_joint_f1', 0.0)
        test_cls_f1 = metrics.get('test_cls_macro_f1', 0.0)
        span_f1     = metrics.get('test_span_overlap',  0.0)

        # If train script doesn't have best_dev_joint_f1 yet, compute it
        if dev_joint == 0.0 and dev_cls_f1 > 0 and span_f1 > 0:
            dev_joint = math.sqrt(dev_cls_f1 * span_f1)

        print(f"\n  ✓ dev_joint_f1={dev_joint:.4f}  dev_cls_f1={dev_cls_f1:.4f}  "
              f"test_span_f1={span_f1:.4f}  ({elapsed/60:.1f} min)")

        # Log all metrics as Optuna user attributes for inspection later
        trial.set_user_attr('dev_cls_f1',   dev_cls_f1)
        trial.set_user_attr('dev_joint_f1', dev_joint)
        trial.set_user_attr('test_cls_f1',  test_cls_f1)
        trial.set_user_attr('test_span_f1', span_f1)
        trial.set_user_attr('elapsed_min',  round(elapsed / 60, 1))
        trial.set_user_attr('trial_dir',    str(trial_dir))

        # Save to summary log
        SWEEP_DIR.mkdir(parents=True, exist_ok=True)
        row = {
            'trial':        trial_num,
            'config':       config,
            'dev_joint_f1': round(dev_joint,   4),
            'dev_cls_f1':   round(dev_cls_f1,  4),
            'test_cls_f1':  round(test_cls_f1, 4),
            'test_span_f1': round(span_f1,     4),
            'elapsed_min':  round(elapsed / 60, 1),
            'timestamp':    datetime.now().isoformat(),
        }
        with open(SUMMARY_PATH, 'a') as f:
            f.write(json.dumps(row) + '\n')

        return dev_joint  # ← this is what Optuna maximises

    return objective


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    SWEEP_DIR.mkdir(parents=True, exist_ok=True)

    pruner  = MedianPruner(n_startup_trials=5, n_warmup_steps=2) if args.pruning else optuna.pruners.NopPruner()
    sampler = TPESampler(seed=42)

    # load_if_exists=True means you can resume a previous study by reusing the same study_name
    study = optuna.create_study(
        study_name    = args.study_name,
        direction     = 'maximize',
        sampler       = sampler,
        pruner        = pruner,
        storage       = f'sqlite:///{STUDY_DB}',
        load_if_exists= True,
    )

    print(f"Study: {args.study_name}")
    print(f"DB:    {STUDY_DB}")
    print(f"Optimising: dev joint F1 = geomean(cls_macro_f1, span_overlap_f1)")
    print(f"Running {args.n_trials} trials...\n")

    study.optimize(
        make_objective(args),
        n_trials  = args.n_trials,
        timeout   = args.timeout,
        show_progress_bar = True,
    )

    # ── Results ───────────────────────────────────────────────────────────────
    best = study.best_trial
    print(f"\n{'='*70}")
    print(f"SWEEP COMPLETE — {len(study.trials)} total trials")
    print(f"Best dev joint F1 : {best.value:.4f}")
    print(f"Best trial        : #{best.number}")
    print(f"Best config       :")
    for k, v in best.params.items():
        print(f"  {k:<22} {v}")
    print(f"{'='*70}")

    best_out = {
        'study_name':       args.study_name,
        'best_trial':       best.number,
        'best_dev_joint_f1': best.value,
        'best_dev_cls_f1':  best.user_attrs.get('dev_cls_f1'),
        'best_test_span_f1':best.user_attrs.get('test_span_f1'),
        'config':           best.params,
    }
    out_path = SWEEP_DIR / 'best_config.json'
    json.dump(best_out, open(out_path, 'w'), indent=2)
    print(f"\nBest config saved → {out_path}")

    # Leaderboard
    if SUMMARY_PATH.exists():
        rows = [json.loads(l) for l in open(SUMMARY_PATH)]
        rows.sort(key=lambda r: r['dev_joint_f1'], reverse=True)
        print(f"\n── Top 10 trials by dev joint F1 ──")
        print(f"{'#':<5} {'dev_joint':>10} {'dev_cls':>9} {'test_span':>10} {'time(min)':>10}  config")
        print('─' * 90)
        for r in rows[:10]:
            cfg_str = f"lr={r['config']['lr']}  cls={r['config']['cls_loss_weight']}  span={r['config']['span_loss_weight']}  bs={r['config']['batch_size']}  ep={r['config']['epochs']}  drop={r['config'].get('dropout', '—')}"
            print(f"{r['trial']:<5} {r['dev_joint_f1']:>10.4f} {r['dev_cls_f1']:>9.4f} {r['test_span_f1']:>10.4f} {r['elapsed_min']:>10.1f}  {cfg_str}")

    # Importance analysis — which hyperparameter mattered most
    try:
        importance = optuna.importance.get_param_importances(study)
        print(f"\n── Hyperparameter importance ──")
        for param, imp in sorted(importance.items(), key=lambda x: -x[1]):
            bar = '█' * int(imp * 40)
            print(f"  {param:<22} {imp:.3f}  {bar}")
    except Exception:
        pass  # needs >= 5 completed trials


if __name__ == '__main__':
    main()