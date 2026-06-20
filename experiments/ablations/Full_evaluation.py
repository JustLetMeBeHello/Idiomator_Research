"""
evaluate_pipeline.py

Full end-to-end pipeline evaluation chaining Stage 1 and Stage 2.
Compares four system configurations:

  A) mBERT Stage 1 → mBERT Stage 2   (full fine-tuned pipeline)
  B) GPT Stage 1   → GPT Stage 2     (full GPT two-stage pipeline)
  C) GPT single-stage                 (classify + extract in one prompt)
  D) mBERT Stage 2 standalone         (upper bound — assumes perfect Stage 1)

Evaluation:
  - Classification F1 (Stage 1)
  - Span extraction on idiomatic examples only (Stage 2)
  - End-to-end F1 (full pipeline, including Stage 1 errors)

Usage:
    python evaluate_pipeline.py

    # Custom paths
    python evaluate_pipeline.py \
        --stage1_mbert  models/stage1_mbert_en_hi_te/test_predictions.jsonl \
        --stage2_mbert  models/stage2_mbert_final/test_predictions.jsonl \
        --stage1_gpt    models/gpt_baseline_stage1/test_predictions.jsonl \
        --stage2_gpt    models/gpt_baseline_stage2/test_predictions.jsonl \
        --single_gpt    models/gpt_single_stage/test_predictions.jsonl \
        --output_dir    results/pipeline_eval
"""

import json
import argparse
import numpy as np
from pathlib import Path
from collections import defaultdict
from sklearn.metrics import f1_score


# ── Args ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--stage1_mbert', default='models/stage1_mbert_en_hi_te/test_predictions.jsonl')
    p.add_argument('--stage2_mbert', default='models/stage2_mbert_final/test_predictions.jsonl')
    p.add_argument('--stage1_gpt',   default='models/gpt_baseline_stage1/test_predictions.jsonl')
    p.add_argument('--stage2_gpt',   default='models/gpt_baseline_stage2/test_predictions.jsonl')
    p.add_argument('--single_gpt',   default='models/gpt_single_stage/test_predictions.jsonl')
    p.add_argument('--output_dir',   default='results/pipeline_eval')
    return p.parse_args()


# ── Helpers ───────────────────────────────────────────────────────────────────

LABEL2ID = {'literal': 0, 'idiomatic': 1}

def load_preds(path):
    """Load predictions JSONL keyed by sentence."""
    if not Path(path).exists():
        print(f"  ⚠ File not found: {path}")
        return {}
    preds = {}
    for line in open(path, encoding='utf-8'):
        r = json.loads(line)
        preds[r['sentence']] = r
    return preds


def compute_overlap_f1(pred_start, pred_end, gold_start, gold_end):
    """Character-level overlap F1."""
    if pred_start is None or pred_end is None:
        return 0.0
    pred_set = set(range(pred_start, pred_end))
    gold_set = set(range(gold_start, gold_end))
    if not pred_set or not gold_set:
        return 0.0
    overlap = len(pred_set & gold_set)
    if overlap == 0:
        return 0.0
    precision = overlap / len(pred_set)
    recall    = overlap / len(gold_set)
    return 2 * precision * recall / (precision + recall)


def cls_f1_per_lang(records, pred_key='pred_idiomaticity', gold_key='idiomaticity'):
    """Compute macro F1 per language for classification."""
    lang_preds  = defaultdict(list)
    lang_labels = defaultdict(list)
    for r in records:
        lang = r['language']
        pred  = LABEL2ID.get(r.get(pred_key, 'idiomatic'), 1)
        label = LABEL2ID.get(r.get(gold_key, 'idiomatic'), 1)
        lang_preds[lang].append(pred)
        lang_labels[lang].append(label)

    results = {}
    all_p, all_l = [], []
    for lang in sorted(lang_preds.keys()):
        f1 = f1_score(lang_labels[lang], lang_preds[lang], average='macro')
        results[lang] = round(f1, 4)
        all_p.extend(lang_preds[lang])
        all_l.extend(lang_labels[lang])
    results['Overall'] = round(f1_score(all_l, all_p, average='macro'), 4)
    return results


def span_f1_per_lang(records, pred_start_key='pred_span_start', pred_end_key='pred_span_end'):
    """Compute exact match and overlap F1 per language for span extraction."""
    lang_exact = defaultdict(list)
    lang_f1    = defaultdict(list)

    for r in records:
        lang      = r['language']
        pred_s    = r.get(pred_start_key)
        pred_e    = r.get(pred_end_key)
        gold_s    = r['span_start']
        gold_e    = r['span_end']

        exact   = int(pred_s == gold_s and pred_e == gold_e) if pred_s is not None else 0
        overlap = compute_overlap_f1(pred_s, pred_e, gold_s, gold_e)

        lang_exact[lang].append(exact)
        lang_f1[lang].append(overlap)

    exact_results   = {}
    overlap_results = {}
    all_exact, all_f1 = [], []

    for lang in sorted(lang_exact.keys()):
        exact_results[lang]   = round(np.mean(lang_exact[lang]), 4)
        overlap_results[lang] = round(np.mean(lang_f1[lang]), 4)
        all_exact.extend(lang_exact[lang])
        all_f1.extend(lang_f1[lang])

    exact_results['Overall']   = round(np.mean(all_exact), 4)
    overlap_results['Overall'] = round(np.mean(all_f1), 4)
    return exact_results, overlap_results


def print_cls_table(label, results):
    print(f"\n  {label}")
    print(f"  {'Language':<12} {'Macro F1':<10}")
    print(f"  {'-'*24}")
    for lang, f1 in results.items():
        print(f"  {lang:<12} {f1:<10.4f}")


def print_span_table(label, exact, overlap):
    print(f"\n  {label}")
    print(f"  {'Language':<12} {'Exact':<10} {'Overlap F1':<12}")
    print(f"  {'-'*36}")
    for lang in exact.keys():
        print(f"  {lang:<12} {exact[lang]:<10.4f} {overlap[lang]:<12.4f}")


# ── System A: mBERT Stage 1 → mBERT Stage 2 ──────────────────────────────────

def evaluate_system_a(s1_mbert, s2_mbert, all_records):
    """Full mBERT pipeline — Stage 1 filters, Stage 2 extracts span."""
    print("\n" + "="*60)
    print("System A: mBERT Stage 1 → mBERT Stage 2")
    print("="*60)

    if not s1_mbert or not s2_mbert:
        print("  ✗ Missing prediction files")
        return None

    # Stage 1 classification results
    s1_records = list(s1_mbert.values())
    cls_results = cls_f1_per_lang(s1_records, pred_key='pred_idiomaticity')
    print_cls_table("Stage 1 Classification:", cls_results)

    # Stage 2 standalone (upper bound — assumes correct Stage 1)
    s2_records = list(s2_mbert.values())
    exact_ub, overlap_ub = span_f1_per_lang(s2_records)
    print_span_table("Stage 2 Span (standalone — upper bound):", exact_ub, overlap_ub)

    # Full pipeline — only run Stage 2 on sentences Stage 1 called idiomatic
    pipeline_records = []
    for sentence, s1_pred in s1_mbert.items():
        if sentence not in s2_mbert:
            continue

        s2_pred = s2_mbert[sentence]
        gold_label = s1_pred['idiomaticity']

        if s1_pred['pred_idiomaticity'] == 'idiomatic':
            # Stage 1 said idiomatic → use Stage 2 span prediction
            pred_s = s2_pred.get('pred_span_start')
            pred_e = s2_pred.get('pred_span_end')
        else:
            # Stage 1 said literal → no span predicted → counts as miss
            pred_s = None
            pred_e = None

        # Only evaluate span on truly idiomatic sentences
        if gold_label == 'idiomatic':
            pipeline_records.append({
                **s1_pred,
                'pred_span_start': pred_s,
                'pred_span_end':   pred_e,
            })

    exact_e2e, overlap_e2e = span_f1_per_lang(pipeline_records)
    print_span_table("Full Pipeline Span (end-to-end — idiomatic only):", exact_e2e, overlap_e2e)

    return {
        'cls_f1':          cls_results,
        'span_standalone': {'exact': exact_ub, 'overlap': overlap_ub},
        'span_e2e':        {'exact': exact_e2e, 'overlap': overlap_e2e},
    }


# ── System B: GPT Stage 1 → GPT Stage 2 ──────────────────────────────────────

def evaluate_system_b(s1_gpt, s2_gpt):
    """Full GPT two-stage pipeline."""
    print("\n" + "="*60)
    print("System B: GPT-4o Stage 1 → GPT-4o Stage 2")
    print("="*60)

    if not s1_gpt or not s2_gpt:
        print("  ✗ Missing prediction files")
        return None

    # Stage 1 classification
    s1_records  = list(s1_gpt.values())
    cls_results = cls_f1_per_lang(s1_records, pred_key='pred_idiomaticity')
    print_cls_table("Stage 1 Classification:", cls_results)

    # Stage 2 standalone
    s2_records = list(s2_gpt.values())
    exact_ub, overlap_ub = span_f1_per_lang(s2_records)
    print_span_table("Stage 2 Span (standalone):", exact_ub, overlap_ub)

    # Full pipeline
    pipeline_records = []
    for sentence, s1_pred in s1_gpt.items():
        if sentence not in s2_gpt:
            continue
        s2_pred    = s2_gpt[sentence]
        gold_label = s1_pred['idiomaticity']

        if s1_pred['pred_idiomaticity'] == 'idiomatic':
            pred_s = s2_pred.get('pred_span_start')
            pred_e = s2_pred.get('pred_span_end')
        else:
            pred_s = None
            pred_e = None

        if gold_label == 'idiomatic':
            pipeline_records.append({
                **s1_pred,
                'pred_span_start': pred_s,
                'pred_span_end':   pred_e,
            })

    exact_e2e, overlap_e2e = span_f1_per_lang(pipeline_records)
    print_span_table("Full Pipeline Span (end-to-end):", exact_e2e, overlap_e2e)

    return {
        'cls_f1':          cls_results,
        'span_standalone': {'exact': exact_ub, 'overlap': overlap_ub},
        'span_e2e':        {'exact': exact_e2e, 'overlap': overlap_e2e},
    }


# ── System C: GPT single-stage ────────────────────────────────────────────────

def evaluate_system_c(single_gpt):
    """GPT single-stage — classify + extract in one prompt."""
    print("\n" + "="*60)
    print("System C: GPT-4o Single-Stage")
    print("="*60)

    if not single_gpt:
        print("  ✗ Missing prediction files")
        return None

    records = list(single_gpt.values())

    # Classification
    cls_results = cls_f1_per_lang(records, pred_key='pred_label')
    print_cls_table("Classification:", cls_results)

    # Span extraction (all examples)
    exact_all, overlap_all = span_f1_per_lang(records)
    print_span_table("Span Extraction (all examples):", exact_all, overlap_all)

    # Span extraction on idiomatic only
    idiomatic_records = [r for r in records if r['idiomaticity'] == 'idiomatic']
    exact_idio, overlap_idio = span_f1_per_lang(idiomatic_records)
    print_span_table("Span Extraction (idiomatic only):", exact_idio, overlap_idio)

    return {
        'cls_f1':       cls_results,
        'span_all':     {'exact': exact_all,  'overlap': overlap_all},
        'span_idio':    {'exact': exact_idio, 'overlap': overlap_idio},
    }


# ── Summary table ─────────────────────────────────────────────────────────────

def print_summary(results_a, results_b, results_c):
    print("\n" + "="*60)
    print("SUMMARY — Full Pipeline Comparison")
    print("="*60)

    print(f"\n{'System':<35} {'Cls F1':<10} {'Span Exact':<12} {'Span F1':<10}")
    print("-" * 70)

    if results_a:
        cls  = results_a['cls_f1'].get('Overall', '—')
        ex   = results_a['span_e2e']['exact'].get('Overall', '—')
        ov   = results_a['span_e2e']['overlap'].get('Overall', '—')
        print(f"{'A: mBERT S1 → mBERT S2 (pipeline)':<35} {cls:<10} {ex:<12} {ov:<10}")

        ex_ub = results_a['span_standalone']['exact'].get('Overall', '—')
        ov_ub = results_a['span_standalone']['overlap'].get('Overall', '—')
        print(f"{'   (Stage 2 standalone upper bound)':<35} {'—':<10} {ex_ub:<12} {ov_ub:<10}")

    if results_b:
        cls = results_b['cls_f1'].get('Overall', '—')
        ex  = results_b['span_e2e']['exact'].get('Overall', '—')
        ov  = results_b['span_e2e']['overlap'].get('Overall', '—')
        print(f"{'B: GPT S1 → GPT S2 (pipeline)':<35} {cls:<10} {ex:<12} {ov:<10}")

    if results_c:
        cls = results_c['cls_f1'].get('Overall', '—')
        ex  = results_c['span_idio']['exact'].get('Overall', '—')
        ov  = results_c['span_idio']['overlap'].get('Overall', '—')
        print(f"{'C: GPT single-stage':<35} {cls:<10} {ex:<12} {ov:<10}")

    print("\nNote: Span metrics for A/B are end-to-end (Stage 1 errors propagate).")
    print("      Span metrics for C are on idiomatic-only examples.")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args       = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading prediction files...")
    s1_mbert   = load_preds(args.stage1_mbert)
    s2_mbert   = load_preds(args.stage2_mbert)
    s1_gpt     = load_preds(args.stage1_gpt)
    s2_gpt     = load_preds(args.stage2_gpt)
    single_gpt = load_preds(args.single_gpt)

    print(f"  Stage 1 mBERT : {len(s1_mbert)} predictions")
    print(f"  Stage 2 mBERT : {len(s2_mbert)} predictions")
    print(f"  Stage 1 GPT   : {len(s1_gpt)} predictions")
    print(f"  Stage 2 GPT   : {len(s2_gpt)} predictions")
    print(f"  Single GPT    : {len(single_gpt)} predictions")

    # Filter all prediction dicts to the intersection of sentences
    # across all systems — ensures fair comparison on identical examples
    all_keys = [set(s1_mbert), set(s2_mbert), set(s1_gpt), set(s2_gpt), set(single_gpt)]
    # Only include non-empty dicts in intersection
    non_empty = [k for k in all_keys if len(k) > 0]
    common_sentences = set.intersection(*non_empty)

    print(f"\n  Common sentences across all systems: {len(common_sentences)}")
    print(f"  Filtering all predictions to common set for fair comparison...")

    s1_mbert   = {s: p for s, p in s1_mbert.items()   if s in common_sentences}
    s2_mbert   = {s: p for s, p in s2_mbert.items()   if s in common_sentences}
    s1_gpt     = {s: p for s, p in s1_gpt.items()     if s in common_sentences}
    s2_gpt     = {s: p for s, p in s2_gpt.items()     if s in common_sentences}
    single_gpt = {s: p for s, p in single_gpt.items() if s in common_sentences}

    print(f"  Stage 1 mBERT : {len(s1_mbert)} (filtered)")
    print(f"  Stage 2 mBERT : {len(s2_mbert)} (filtered)")
    print(f"  Stage 1 GPT   : {len(s1_gpt)} (filtered)")
    print(f"  Stage 2 GPT   : {len(s2_gpt)} (filtered)")
    print(f"  Single GPT    : {len(single_gpt)} (filtered)")

    # Run all evaluations
    results_a = evaluate_system_a(s1_mbert, s2_mbert, None)
    results_b = evaluate_system_b(s1_gpt, s2_gpt)
    results_c = evaluate_system_c(single_gpt)

    # Summary
    print_summary(results_a, results_b, results_c)

    # Save all results
    all_results = {
        'system_a_mbert_pipeline': results_a,
        'system_b_gpt_pipeline':   results_b,
        'system_c_gpt_single':     results_c,
    }
    out_path = output_dir / 'pipeline_eval_results.json'
    json.dump(all_results, open(out_path, 'w'), indent=2)
    print(f"\nFull results saved → {out_path}")


if __name__ == '__main__':
    main()