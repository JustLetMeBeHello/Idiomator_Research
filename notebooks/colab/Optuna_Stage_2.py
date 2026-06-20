"""
Optuna_Stage_2.py

Optuna hyperparameter search for Stage 2 MWE span extraction.
Runs 30 trials, optimizing dev overlap F1 on Hindi+Telugu average.

Tunes:
  - lr           : 1e-5, 2e-5, 3e-5, 5e-5
  - epochs       : 3, 5, 7
  - batch_size   : 16, 32
  - warmup_ratio : 0.06, 0.1, 0.2

Usage:
    %%bash
    cd /content/drive/MyDrive/Idiomator_Research/Research_And_Training
    python notebooks/colab/Optuna_Stage_2.py \
        --langs English Hindi Telugu \
        --output_dir models/stage2_optuna \
        --n_trials 30 \
        --use_wandb
"""

import sys
sys.path.append('/content/drive/MyDrive/Idiomator_Research/Research_And_Training')

import json
import argparse
import numpy as np
from pathlib import Path
from collections import defaultdict

import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
from tqdm import tqdm

import optuna
from optuna.samplers import TPESampler

from Base_Pipeline.Stage_2_training import (
    SpanExtractor,
    SpanDataset,
    build_dataset_for_split,
    compute_overlap_f1,
    get_device,
)

try:
    import wandb
    WANDB = True
except ImportError:
    WANDB = False
    print("wandb not installed — skipping tracking.")


# ── Args ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--model_name', default='bert-base-multilingual-cased')
    p.add_argument('--data_dir',   default='data/idioms_structured/Splits')
    p.add_argument('--output_dir', default='models/stage2_optuna')
    p.add_argument('--langs',      nargs='+', default=['English', 'Hindi', 'Telugu'])
    p.add_argument('--n_trials',   type=int,   default=30)
    p.add_argument('--max_len',    type=int,   default=128)
    p.add_argument('--seed',       type=int,   default=42)
    p.add_argument('--device',     default=None)
    p.add_argument('--use_wandb',  action='store_true')
    return p.parse_args()


# ── Constants ─────────────────────────────────────────────────────────────────

LOW_RESOURCE = {'Hindi', 'Telugu'}


# ── Eval ──────────────────────────────────────────────────────────────────────

def evaluate_overlap(model, loader, examples, device):
    """Returns HI+TE average overlap F1 — our optimization target.
    Also returns per-language breakdown dict for wandb logging."""
    model.eval()
    lang_f1 = defaultdict(list)

    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            input_ids      = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            token_type_ids = batch['token_type_ids'].to(device)

            start_logits, end_logits = model(input_ids, attention_mask, token_type_ids)

            pred_starts = torch.argmax(start_logits, dim=-1).cpu().numpy()
            pred_ends   = torch.argmax(end_logits,   dim=-1).cpu().numpy()
            gold_starts = batch['start_positions'].numpy()
            gold_ends   = batch['end_positions'].numpy()

            batch_start = batch_idx * loader.batch_size
            for i in range(len(pred_starts)):
                ex_idx = batch_start + i
                if ex_idx >= len(examples):
                    break

                ex     = examples[ex_idx]
                lang   = ex['language']
                pred_s = int(pred_starts[i])
                pred_e = max(int(pred_ends[i]), pred_s)
                gold_s = int(gold_starts[i])
                gold_e = int(gold_ends[i])

                f1 = compute_overlap_f1(pred_s, pred_e, gold_s, gold_e)
                lang_f1[lang].append(f1)

    # Per-language scores
    lang_scores = {}
    lr_scores   = []
    for lang in sorted(lang_f1.keys()):
        score = np.mean(lang_f1[lang])
        lang_scores[lang] = round(float(score), 4)
        print(f"    {lang}: overlap F1 = {score:.4f} ({len(lang_f1[lang])} examples)")
        if lang in LOW_RESOURCE:
            lr_scores.append(score)

    # Optimization target
    target = np.mean(lr_scores) if lr_scores else np.mean([
        s for scores in lang_f1.values() for s in scores
    ])
    return float(target), lang_scores


# ── Incremental save callback ─────────────────────────────────────────────────

def make_save_callback(output_dir):
    """Returns an Optuna callback that flushes results to JSON after every trial.
    Safe to call on resume — rewrites the full sorted list each time."""
    def save_callback(study, trial):
        results = []
        for t in study.trials:
            if t.value is not None:
                results.append({
                    'trial': t.number,
                    'value': t.value,
                    **t.params,
                    'state': str(t.state),
                })
        results.sort(key=lambda r: r['value'], reverse=True)
        json.dump(results, open(output_dir / 'optuna_results.json', 'w'), indent=2)
        print(f"  [callback] Results flushed → optuna_results.json "
              f"({len(results)} completed trials)")
    return save_callback


# ── Single trial ──────────────────────────────────────────────────────────────

def run_trial(trial, args, train_ds, dev_ds, tokenizer, device, output_dir):
    """One Optuna trial — train and return dev HI+TE overlap F1."""

    # Sample hyperparameters
    lr           = trial.suggest_categorical('lr',           [1e-5, 2e-5, 3e-5, 5e-5])
    epochs       = trial.suggest_categorical('epochs',       [3, 5, 7])
    batch_size   = trial.suggest_categorical('batch_size',   [16, 32])
    warmup_ratio = trial.suggest_categorical('warmup_ratio', [0.06, 0.1, 0.2])

    print(f"\n── Trial {trial.number} ──")
    print(f"  lr={lr}  epochs={epochs}  bs={batch_size}  warmup={warmup_ratio}")

    # Start a fresh wandb run for this trial
    if WANDB and args.use_wandb:
        wandb.init(
            project='idiomator-app',
            name=f"stage2_trial_{trial.number}",
            config={
                'trial':         trial.number,
                'lr':            lr,
                'epochs':        epochs,
                'batch_size':    batch_size,
                'warmup_ratio':  warmup_ratio,
                'langs':         args.langs,
                'stage':         2,
            },
            reinit=True,
        )

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    dev_loader   = DataLoader(dev_ds,   batch_size=batch_size)

    model     = SpanExtractor(args.model_name).to(device)
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=0.01)

    total_steps  = len(train_loader) * epochs
    warmup_steps = int(total_steps * warmup_ratio)
    scheduler    = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    criterion    = torch.nn.CrossEntropyLoss()

    best_epoch_score = 0.0
    best_lang_scores = {}
    best_epoch       = 1

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0

        pbar = tqdm(train_loader,
                    desc=f"Trial {trial.number} | Epoch {epoch}/{epochs}",
                    unit='batch', leave=False)
        for batch in pbar:
            input_ids       = batch['input_ids'].to(device)
            attention_mask  = batch['attention_mask'].to(device)
            token_type_ids  = batch['token_type_ids'].to(device)
            start_positions = batch['start_positions'].to(device)
            end_positions   = batch['end_positions'].to(device)

            start_logits, end_logits = model(input_ids, attention_mask, token_type_ids)
            loss = (criterion(start_logits, start_positions) +
                    criterion(end_logits,   end_positions)) / 2

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            total_loss += loss.item()
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})

        avg_loss = total_loss / len(train_loader)
        print(f"  Epoch {epoch} avg loss: {avg_loss:.4f}")

        # Eval on dev — get target score + per-language breakdown
        score, lang_scores = evaluate_overlap(
            model, dev_loader, dev_ds.valid_examples, device
        )
        print(f"  → HI+TE overlap F1: {score:.4f}")

        # Log per-epoch metrics to wandb
        if WANDB and args.use_wandb:
            log_dict = {
                'epoch':          epoch,
                'train_loss':     avg_loss,
                'dev_hi_te_f1':   score,
            }
            for lang, lang_score in lang_scores.items():
                log_dict[f'dev_{lang.lower()}_overlap_f1'] = lang_score
            wandb.log(log_dict)

        # Optuna pruning
        trial.report(score, epoch)
        if trial.should_prune():
            print(f"  ✗ Trial pruned at epoch {epoch}")
            if WANDB and args.use_wandb:
                wandb.finish()
            raise optuna.exceptions.TrialPruned()

        # Track best epoch score — no model weights saved during search
        if score > best_epoch_score:
            best_epoch_score = score
            best_lang_scores = lang_scores
            best_epoch       = epoch

    # Save config only — no weights, keeps Drive usage minimal
    trial_dir = output_dir / f'trial_{trial.number}'
    trial_dir.mkdir(parents=True, exist_ok=True)
    json.dump({
        'lr': lr, 'epochs': epochs, 'batch_size': batch_size,
        'warmup_ratio': warmup_ratio, 'best_epoch': best_epoch,
        'dev_hi_te_overlap_f1': best_epoch_score,
        'lang_scores': best_lang_scores,
    }, open(trial_dir / 'config.json', 'w'), indent=2)

    if WANDB and args.use_wandb:
        wandb.log({'best_hi_te_overlap_f1': best_epoch_score})
        wandb.finish()

    return best_epoch_score


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args       = parse_args()
    device     = get_device(args.device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    print(f"Languages : {args.langs}")
    print(f"N trials  : {args.n_trials}")
    print(f"Output    : {output_dir}")

    # Load data once — shared across all trials
    print("\nLoading datasets...")
    train_examples = build_dataset_for_split('train', args.data_dir, args.langs, args.seed)
    dev_examples   = build_dataset_for_split('dev',   args.data_dir, args.langs, args.seed)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    print("Tokenizing train...")
    train_ds = SpanDataset(train_examples, tokenizer, args.max_len)
    print("Tokenizing dev...")
    dev_ds   = SpanDataset(dev_examples,   tokenizer, args.max_len)

    print(f"Train: {len(train_ds)} | Dev: {len(dev_ds)}")

    # ── Optuna study ──────────────────────────────────────────────────────────
    # SQLite storage on Drive — survives Colab timeouts and resumes automatically
    db_path = output_dir / 'optuna_study.db'
    storage = f"sqlite:///{db_path}"
    print(f"\nOptuna storage : {db_path}")
    if db_path.exists():
        print("  ↳ Existing study found — resuming from checkpoint")
    else:
        print("  ↳ No existing study — starting fresh")

    def objective(trial):
        return run_trial(
            trial, args, train_ds, dev_ds, tokenizer, device, output_dir
        )

    sampler = TPESampler(seed=args.seed)
    pruner  = optuna.pruners.MedianPruner(
        n_startup_trials=5,
        n_warmup_steps=2,
    )

    study = optuna.create_study(
        storage=storage,
        load_if_exists=True,      # ← resumes if session died mid-run
        direction='maximize',
        sampler=sampler,
        pruner=pruner,
        study_name='stage2_span_extraction',
    )

    # Clean up any trials stuck in RUNNING state from a crashed session
    from optuna.trial import TrialState
    for t in study.trials:
        if t.state == TrialState.RUNNING:
            study.tell(t.number, state=TrialState.FAIL)
            print(f"  ↳ Marked stuck trial {t.number} as FAILED")

    # How many trials are left to run
    completed = len([t for t in study.trials if t.value is not None])
    remaining = args.n_trials - completed
    print(f"Trials complete: {completed} / {args.n_trials}  →  running {remaining} more")

    if remaining <= 0:
        print("All trials already complete — nothing to run.")
    else:
        study.optimize(
            objective,
            n_trials=remaining,                    # only run what's left
            callbacks=[make_save_callback(output_dir)],  # flush JSON after each trial
        )

    # ── Results ───────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"Optuna search complete — {len(study.trials)} trials")
    print(f"{'='*60}")

    best = study.best_trial
    print(f"\nBest trial : {best.number}")
    print(f"Best score : {best.value:.4f} (HI+TE overlap F1)")
    print(f"Best params:")
    for k, v in best.params.items():
        print(f"  {k}: {v}")

    # Final save (also written incrementally by callback, this is the canonical copy)
    results = []
    for t in study.trials:
        if t.value is not None:
            results.append({
                'trial': t.number,
                'value': t.value,
                **t.params,
                'state': str(t.state),
            })
    results.sort(key=lambda r: r['value'], reverse=True)

    results_path = output_dir / 'optuna_results.json'
    json.dump(results, open(results_path, 'w'), indent=2)
    print(f"\nAll results saved → {results_path}")

    # Print ranked table
    print(f"\n{'Trial':<8} {'Score':<10} {'LR':<8} {'EP':<4} {'BS':<4} {'WR':<6}")
    print("─" * 44)
    for r in results[:10]:
        print(f"{r['trial']:<8} {r['value']:<10.4f} {str(r['lr']):<8} "
              f"{r['epochs']:<4} {r['batch_size']:<4} {r['warmup_ratio']:<6}")

    # Save best config
    best_config = {**best.params, 'model_name': args.model_name, 'langs': args.langs}
    json.dump(best_config, open(output_dir / 'best_config.json', 'w'), indent=2)
    print(f"Best config saved → {output_dir / 'best_config.json'}")

    print(f"\nTo train final model with best config:")
    print(f"  python Stage_2_training.py \\")
    print(f"      --langs {' '.join(args.langs)} \\")
    print(f"      --output_dir models/stage2_mbert_final \\")
    print(f"      --lr {best.params['lr']} \\")
    print(f"      --epochs {best.params['epochs']} \\")
    print(f"      --batch_size {best.params['batch_size']} \\")
    print(f"      --warmup_ratio {best.params['warmup_ratio']} \\")
    print(f"      --use_wandb")


if __name__ == '__main__':
    main()