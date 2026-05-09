"""
collect_results.py

Crawls all models/ subdirectories for metrics.json and test_predictions.jsonl,
computes per-language F1, and writes/updates results/stage1_results.csv.

Run from repo root:
    python collect_results.py

Or specify a custom models dir:
    python collect_results.py --models_dir models/sweep
"""

import json
import csv
import argparse
from pathlib import Path
from collections import defaultdict
from sklearn.metrics import f1_score

LABEL2ID = {'literal': 0, 'idiomatic': 1}

FIELDNAMES = [
    'run_id', 'model', 'langs', 'lr', 'epochs', 'batch_size',
    'dev_f1', 'test_f1', 'en_f1', 'hi_f1', 'te_f1',
    'train_size', 'test_size', 'notes'
]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--models_dir',  default='models')
    p.add_argument('--results_dir', default='results')
    p.add_argument('--output',      default='stage1_results.csv')
    return p.parse_args()


def get_lang_f1(preds_path):
    """Compute per-language macro F1 from test_predictions.jsonl."""
    lang_data = defaultdict(lambda: {'preds': [], 'labels': []})

    for line in open(preds_path, encoding='utf-8'):
        r = json.loads(line)
        lang = r.get('language', 'unknown')
        pred  = LABEL2ID.get(r.get('pred_idiomaticity', 'idiomatic'), 1)
        label = LABEL2ID.get(r.get('idiomaticity', 'idiomatic'), 1)
        lang_data[lang]['preds'].append(pred)
        lang_data[lang]['labels'].append(label)

    lang_f1 = {}
    for lang, data in lang_data.items():
        lang_f1[lang] = round(f1_score(data['labels'], data['preds'], average='macro'), 4)
    return lang_f1


def collect_run(run_dir):
    """Extract a result row from a single run directory."""
    metrics_path = run_dir / 'metrics.json'
    preds_path   = run_dir / 'test_predictions.jsonl'
    config_path  = run_dir / 'config.json'

    if not metrics_path.exists():
        return None

    m = json.load(open(metrics_path))
    c = json.load(open(config_path)) if config_path.exists() else {}

    # Per-language F1
    lang_f1 = {}
    if preds_path.exists():
        lang_f1 = get_lang_f1(preds_path)

    # Langs string
    langs = m.get('langs', c.get('langs', []))
    if isinstance(langs, list):
        langs = '+'.join(l[:2].upper() for l in langs)  # EN+HI+TE

    row = {
        'run_id':     run_dir.name,
        'model':      m.get('model', c.get('model_name', '')).replace('bert-base-', ''),
        'langs':      langs,
        'lr':         c.get('lr', ''),
        'epochs':     m.get('best_epoch', c.get('epochs', '')),
        'batch_size': c.get('batch_size', ''),
        'dev_f1':     round(m.get('best_dev_f1', 0), 4),
        'test_f1':    round(m.get('test_macro_f1', 0), 4),
        'en_f1':      lang_f1.get('English', ''),
        'hi_f1':      lang_f1.get('Hindi', ''),
        'te_f1':      lang_f1.get('Telugu', ''),
        'train_size': m.get('train_size', ''),
        'test_size':  m.get('test_size', ''),
        'notes':      '',
    }
    return row


def main():
    args   = parse_args()
    models_dir  = Path(args.models_dir)
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    out_path = results_dir / args.output

    # Load existing results to preserve manual notes
    existing = {}
    if out_path.exists():
        for row in csv.DictReader(open(out_path)):
            existing[row['run_id']] = row.get('notes', '')

    # Collect all runs
    rows = []
    for run_dir in sorted(models_dir.iterdir()):
        if not run_dir.is_dir():
            continue

        # Handle sweep subdirectories
        if (run_dir / 'metrics.json').exists():
            row = collect_run(run_dir)
            if row:
                row['notes'] = existing.get(row['run_id'], '')
                rows.append(row)
        else:
            # Look one level deeper (e.g. models/sweep/run1_lr3e-5_ep5/)
            for sub_dir in sorted(run_dir.iterdir()):
                if sub_dir.is_dir() and (sub_dir / 'metrics.json').exists():
                    row = collect_run(sub_dir)
                    if row:
                        row['notes'] = existing.get(row['run_id'], '')
                        rows.append(row)

    if not rows:
        print("No completed runs found in models/")
        return

    # Sort by test_f1 descending
    rows.sort(key=lambda r: float(r['test_f1']) if r['test_f1'] else 0, reverse=True)

    # Write CSV
    with open(out_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} runs → {out_path}\n")

    # Print summary table
    print(f"{'Run':<35} {'Model':<20} {'Langs':<12} {'Dev F1':<9} {'Test F1':<9} {'EN':<8} {'HI':<8} {'TE':<8}")
    print("─" * 109)
    for r in rows:
        print(f"{r['run_id']:<35} {r['model']:<20} {r['langs']:<12} "
              f"{str(r['dev_f1']):<9} {str(r['test_f1']):<9} "
              f"{str(r['en_f1']):<8} {str(r['hi_f1']):<8} {str(r['te_f1']):<8}")

    best = rows[0]
    print(f"\nBest so far: {best['run_id']} — test F1={best['test_f1']}  "
          f"EN={best['en_f1']}  HI={best['hi_f1']}  TE={best['te_f1']}")


if __name__ == '__main__':
    main()