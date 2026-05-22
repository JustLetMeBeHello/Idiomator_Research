"""
Full_evaluation.py — fixed version
See docstring body for full change list.
"""

import json
import argparse
import numpy as np
from pathlib import Path
from collections import defaultdict
from sklearn.metrics import f1_score, precision_recall_fscore_support

HELD_OUT_LANG = "Indonesian"
SMALL_LANGS   = ["Hindi", "Telugu"]
IN_DIST_LANGS = ["English", "Spanish", "Hindi", "Telugu"]

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--stage1_mbert', default='models/stage1_mbert_en_hi_te/test_predictions.jsonl')
    p.add_argument('--stage2_mbert', default='models/stage2_mbert_en_hi_te/test_predictions.jsonl')
    p.add_argument('--stage1_gpt',   default='models/gpt_baseline_stage1/test_predictions.jsonl')
    p.add_argument('--stage2_gpt',   default='models/gpt_baseline_stage2/test_predictions.jsonl')
    p.add_argument('--single_gpt',   default='models/gpt_single_stage/test_predictions.jsonl')
    p.add_argument('--joint_preds',  default='models/joint_mbert_en_hi_te/test_predictions.jsonl')
    p.add_argument('--span2_joint',  default='models/stage2_mbert_shared/test_predictions.jsonl')
    p.add_argument('--seq_phase1',   default='models/sequential/phase1/test_predictions.jsonl')
    p.add_argument('--seq_phase2',   default='models/sequential/phase2/test_predictions.jsonl')
    p.add_argument('--bio_preds',    default='models/bio_tagger_en_hi_te/test_predictions.jsonl')
    p.add_argument('--output_dir',   default='results/pipeline_eval')
    p.add_argument('--n_bootstrap',  type=int, default=10000)
    return p.parse_args()

LABEL2ID = {'literal': 0, 'idiomatic': 1}

def load_preds(path):
    if not Path(path).exists():
        print(f"  ⚠ File not found: {path}")
        return {}
    preds = {}
    for line in open(path, encoding='utf-8'):
        r = json.loads(line)
        preds[r['sentence']] = r
    return preds

def compute_overlap_f1(pred_start, pred_end, gold_start, gold_end):
    if pred_start is None or pred_end is None:
        return 0.0
    pred_set = set(range(pred_start, pred_end))
    gold_set = set(range(gold_start, gold_end))
    if not pred_set or not gold_set:
        return 0.0
    overlap   = len(pred_set & gold_set)
    if overlap == 0:
        return 0.0
    precision = overlap / len(pred_set)
    recall    = overlap / len(gold_set)
    return 2 * precision * recall / (precision + recall)

def cls_f1_per_lang(records, pred_key='pred_idiomaticity', gold_key='idiomaticity'):
    lang_preds  = defaultdict(list)
    lang_labels = defaultdict(list)
    for r in records:
        lang  = r['language']
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
    lang_exact = defaultdict(list)
    lang_f1    = defaultdict(list)
    for r in records:
        lang   = r['language']
        pred_s = r.get(pred_start_key)
        pred_e = r.get(pred_end_key)
        gold_s = r['span_start']
        gold_e = r['span_end']
        exact   = int(pred_s == gold_s and pred_e == gold_e) if pred_s is not None else 0
        overlap = compute_overlap_f1(pred_s, pred_e, gold_s, gold_e)
        lang_exact[lang].append(exact)
        lang_f1[lang].append(overlap)
    exact_results, overlap_results = {}, {}
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

def compute_joint_acc(s1_preds, s2_preds, pred_label_key='pred_idiomaticity'):
    correct = 0
    total   = 0
    for sentence, s1 in s1_preds.items():
        gold_label  = s1['idiomaticity']
        pred_label  = s1.get(pred_label_key)
        cls_correct = (pred_label == gold_label)
        if gold_label == 'literal':
            correct += int(cls_correct)
        else:
            if cls_correct and sentence in s2_preds:
                s2 = s2_preds[sentence]
                span_exact = (s2.get('pred_span_start') == s1['span_start'] and
                              s2.get('pred_span_end')   == s1['span_end'])
                correct += int(span_exact)
        total += 1
    return correct / total if total > 0 else 0.0, correct, total

def compute_joint_f1(s1_preds, s2_preds, pred_label_key='pred_idiomaticity',
                     span_overlap_threshold=0.0):
    lang_gold = defaultdict(list)
    lang_pred = defaultdict(list)
    for sentence, s1 in s1_preds.items():
        gold_label = s1['idiomaticity']
        pred_label = s1.get(pred_label_key)
        lang       = s1['language']
        if gold_label == 'literal':
            gold_joint = 0
            pred_joint = 0 if pred_label == 'literal' else 1
        else:
            gold_joint = 1
            if pred_label != 'idiomatic':
                pred_joint = 0
            else:
                s2     = s2_preds.get(sentence, s1)
                pred_s = s2.get('pred_span_start')
                pred_e = s2.get('pred_span_end')
                gold_s = s1['span_start']
                gold_e = s1['span_end']
                overlap = compute_overlap_f1(pred_s, pred_e, gold_s, gold_e) \
                          if pred_s is not None else 0.0
                pred_joint = 1 if overlap > span_overlap_threshold else 0
        lang_gold[lang].append(gold_joint)
        lang_pred[lang].append(pred_joint)
    per_lang = {}
    all_gold, all_pred = [], []
    for lang in sorted(lang_gold.keys()):
        g, p = lang_gold[lang], lang_pred[lang]
        prec, rec, f1, _ = precision_recall_fscore_support(
            g, p, average=None, labels=[0, 1], zero_division=0)
        per_lang[lang] = {
            'macro_f1':  round(float(np.mean(f1)), 4),
            'literal':   {'P': round(prec[0],4), 'R': round(rec[0],4), 'F1': round(f1[0],4)},
            'idiomatic': {'P': round(prec[1],4), 'R': round(rec[1],4), 'F1': round(f1[1],4)},
        }
        all_gold.extend(g)
        all_pred.extend(p)
    prec, rec, f1, _ = precision_recall_fscore_support(
        all_gold, all_pred, average=None, labels=[0, 1], zero_division=0)
    in_dist_langs = [l for l in sorted(lang_gold.keys()) if l != HELD_OUT_LANG]
    macro_avg_f1  = round(float(np.mean(
        [per_lang[lang]['macro_f1'] for lang in in_dist_langs])), 4)
    per_lang['Overall'] = {
        'macro_f1':     round(float(np.mean(f1)), 4),
        'macro_avg_f1': macro_avg_f1,
        'literal':   {'P': round(prec[0],4), 'R': round(rec[0],4), 'F1': round(f1[0],4)},
        'idiomatic': {'P': round(prec[1],4), 'R': round(rec[1],4), 'F1': round(f1[1],4)},
    }
    return per_lang

def build_pipeline_records(s1_preds, s2_preds, only_correct_cls=False):
    pipeline_records = []
    for sentence, s1_pred in s1_preds.items():
        if sentence not in s2_preds:
            continue
        s2_pred    = s2_preds[sentence]
        gold_label = s1_pred['idiomaticity']
        pred_label = s1_pred.get('pred_idiomaticity', s1_pred.get('pred_label'))
        if only_correct_cls:
            if not (gold_label == 'idiomatic' and pred_label == 'idiomatic'):
                continue
        pred_s = s2_pred.get('pred_span_start') if pred_label == 'idiomatic' else None
        pred_e = s2_pred.get('pred_span_end')   if pred_label == 'idiomatic' else None
        if gold_label == 'idiomatic':
            pipeline_records.append({
                **s1_pred,
                'pred_span_start': pred_s,
                'pred_span_end':   pred_e,
            })
    return pipeline_records

def compute_stability(joint_f1_per_lang, langs=None):
    if langs is None:
        langs = [l for l in IN_DIST_LANGS if l in joint_f1_per_lang]
    scores = {}
    for lang in langs:
        entry = joint_f1_per_lang.get(lang)
        if entry is None:
            continue
        scores[lang] = entry.get('macro_f1', 0.0) if isinstance(entry, dict) else float(entry)
    if not scores:
        return {}
    vals       = list(scores.values())
    mean_j     = float(np.mean(vals))
    std_j      = float(np.std(vals))
    worst_lang = min(scores, key=scores.__getitem__)
    best_f1    = max(scores.values())
    return {
        'mean_joint': round(mean_j, 4),
        'std_joint':  round(std_j, 4),
        'worst_lang': worst_lang,
        'worst_f1':   round(scores[worst_lang], 4),
        'gap':        round(best_f1 - scores[worst_lang], 4),
        'stability':  round(mean_j - std_j, 4),
        'langs':      scores,
    }

def print_stability_table(label, stability):
    if not stability:
        print(f"\n  {label}  [no data]")
        return
    print(f"\n  {label}")
    print(f"  Mean Joint F1:  {stability['mean_joint']:.4f}")
    print(f"  Std (langs):    {stability['std_joint']:.4f}   (lower = more stable)")
    print(f"  Worst language: {stability['worst_lang']} = {stability['worst_f1']:.4f}")
    print(f"  Best-Worst gap: {stability['gap']:.4f}")
    print(f"  Stability score (mean - std): {stability['stability']:.4f}")
    print(f"  Per-language: " +
          "  ".join(f"{l}={v:.4f}" for l, v in stability['langs'].items()))

def bootstrap_ci(values, statistic_fn=np.mean, n_resamples=10000, ci=0.95, seed=42):
    rng     = np.random.default_rng(seed)
    vals    = np.array(values, dtype=float)
    point   = statistic_fn(vals)
    n       = len(vals)
    samples = [statistic_fn(rng.choice(vals, size=n, replace=True))
               for _ in range(n_resamples)]
    alpha   = (1 - ci) / 2
    lower   = float(np.quantile(samples, alpha))
    upper   = float(np.quantile(samples, 1 - alpha))
    return round(float(point), 4), round(lower, 4), round(upper, 4)

def _compute_lang_bootstrap(s1_preds, s2_preds, target_langs,
                             pred_label_key='pred_idiomaticity', n_resamples=10000):
    records = [r for r in s1_preds.values() if r['language'] in target_langs]
    if not records:
        return None
    cls_correct, span_exact_vals, span_overlap_vals, joint_correct = [], [], [], []
    for r in records:
        gold_label = r['idiomaticity']
        pred_label = r.get(pred_label_key, 'idiomatic')
        gold_s, gold_e = r['span_start'], r['span_end']
        cls_correct.append(int(pred_label == gold_label))
        s2     = s2_preds.get(r['sentence'], r)
        pred_s = s2.get('pred_span_start')
        pred_e = s2.get('pred_span_end')
        exact   = int(pred_s == gold_s and pred_e == gold_e) if pred_s is not None else 0
        overlap = compute_overlap_f1(pred_s, pred_e, gold_s, gold_e) if pred_s is not None else 0.0
        span_exact_vals.append(exact)
        span_overlap_vals.append(overlap)
        if gold_label == 'literal':
            joint_correct.append(int(pred_label == 'literal'))
        else:
            joint_correct.append(int(pred_label == 'idiomatic' and overlap > 0.0))
    return {
        'n_examples':   len(records),
        'cls_accuracy': bootstrap_ci(cls_correct,       n_resamples=n_resamples),
        'span_exact':   bootstrap_ci(span_exact_vals,   n_resamples=n_resamples),
        'span_overlap': bootstrap_ci(span_overlap_vals, n_resamples=n_resamples),
        'joint_f1':     bootstrap_ci(joint_correct,     n_resamples=n_resamples),
    }

def compute_indonesian_bootstrap(s1_preds, s2_preds,
                                  pred_label_key='pred_idiomaticity', n_resamples=10000):
    return _compute_lang_bootstrap(s1_preds, s2_preds, {HELD_OUT_LANG},
                                   pred_label_key=pred_label_key, n_resamples=n_resamples)

def compute_small_lang_bootstrap(s1_preds, s2_preds,
                                  pred_label_key='pred_idiomaticity', n_resamples=10000):
    return _compute_lang_bootstrap(s1_preds, s2_preds, set(SMALL_LANGS),
                                   pred_label_key=pred_label_key, n_resamples=n_resamples)

def print_bootstrap_ci(label, ci_result, lang_note=""):
    if ci_result is None:
        print(f"\n  {label}  [no examples found]")
        return
    print(f"\n  {label}  (n={ci_result['n_examples']}{lang_note}, 95% bootstrap CI)")
    print(f"  {'Metric':<18} {'Point':<10} {'95% CI':<20}")
    print(f"  {'-'*50}")
    for key, label_str in [('cls_accuracy','Cls Accuracy'),('span_exact','Span Exact'),
                            ('span_overlap','Span Overlap F1'),('joint_f1','Joint F1')]:
        entry = ci_result.get(key)
        if entry is None or (isinstance(entry, tuple) and entry[0] is None):
            print(f"  {label_str:<18} — (no classifier)")
            continue
        pt, lo, hi = entry
        print(f"  {label_str:<18} {pt:<10.4f} [{lo:.4f}, {hi:.4f}]")

def _build_return(cls_results, exact_ub, overlap_ub, exact_e2e, overlap_e2e,
                  exact_corr, overlap_corr, joint_acc, joint_f1,
                  stability, small_lang_ci, indonesian_ci):
    return {
        'cls_f1':          cls_results,
        'span_standalone': {'exact': exact_ub,   'overlap': overlap_ub},
        'span_e2e':        {'exact': exact_e2e,  'overlap': overlap_e2e},
        'span_correct_id': {'exact': exact_corr, 'overlap': overlap_corr},
        'joint_acc':       round(joint_acc, 4),
        'joint_f1':        joint_f1,
        'stability':       stability,
        'small_lang_ci':   small_lang_ci,
        'indonesian_ci':   indonesian_ci,
    }

def _print_joint_f1(joint_f1):
    print(f"\n  Joint F1 breakdown:")
    for lang, d in joint_f1.items():
        if isinstance(d, dict):
            suffix = (f"  macro_avg={d['macro_avg_f1']:.4f}"
                      if lang == 'Overall' and 'macro_avg_f1' in d else "")
            print(f"    {lang:<10} macro={d['macro_f1']:.4f}{suffix}  "
                  f"literal: P={d['literal']['P']:.4f} R={d['literal']['R']:.4f} F1={d['literal']['F1']:.4f}  "
                  f"idiomatic: P={d['idiomatic']['P']:.4f} R={d['idiomatic']['R']:.4f} F1={d['idiomatic']['F1']:.4f}")

def evaluate_system_a(s1_mbert, s2_mbert, n_bootstrap=10000):
    print("\n" + "="*60)
    print("System A: mBERT Stage 1 → mBERT Stage 2 (two-stage pipeline)")
    print("="*60)
    if not s1_mbert or not s2_mbert:
        print("  ✗ Missing prediction files"); return None
    cls_results = cls_f1_per_lang(list(s1_mbert.values()), pred_key='pred_idiomaticity')
    print_cls_table("Stage 1 Classification:", cls_results)
    exact_ub, overlap_ub = span_f1_per_lang(list(s2_mbert.values()))
    print_span_table("Stage 2 Span (standalone — upper bound):", exact_ub, overlap_ub)
    pipeline_records = build_pipeline_records(s1_mbert, s2_mbert, only_correct_cls=False)
    exact_e2e, overlap_e2e = span_f1_per_lang(pipeline_records)
    print_span_table("Full Pipeline Span (end-to-end):", exact_e2e, overlap_e2e)
    correct_records = build_pipeline_records(s1_mbert, s2_mbert, only_correct_cls=True)
    exact_corr, overlap_corr = span_f1_per_lang(correct_records)
    print_span_table("Span F1 (Correct Stage 1 Identifications Only):", exact_corr, overlap_corr)
    joint_acc, n_correct, n_total = compute_joint_acc(s1_mbert, s2_mbert)
    print(f"\n  Joint accuracy (cls + span both correct): {joint_acc:.4f} ({n_correct}/{n_total})")
    joint_f1 = compute_joint_f1(s1_mbert, s2_mbert)
    _print_joint_f1(joint_f1)
    stability     = compute_stability(joint_f1)
    small_lang_ci = compute_small_lang_bootstrap(s1_mbert, s2_mbert, n_resamples=n_bootstrap)
    indonesian_ci = compute_indonesian_bootstrap(s1_mbert, s2_mbert, n_resamples=n_bootstrap)
    print_stability_table("Stability metrics (in-distribution languages):", stability)
    print_bootstrap_ci("Hindi/Telugu bootstrap CIs:", small_lang_ci, " — Hindi+Telugu")
    print_bootstrap_ci("Indonesian held-out generalization:", indonesian_ci, " — zero-shot")
    return _build_return(cls_results, exact_ub, overlap_ub, exact_e2e, overlap_e2e,
                         exact_corr, overlap_corr, joint_acc, joint_f1,
                         stability, small_lang_ci, indonesian_ci)

def evaluate_system_b(s1_gpt, s2_gpt, n_bootstrap=10000):
    print("\n" + "="*60)
    print("System B: GPT-4o Stage 1 → GPT-4o Stage 2 (two-stage pipeline)")
    print("="*60)
    if not s1_gpt or not s2_gpt:
        print("  ✗ Missing prediction files"); return None
    cls_results = cls_f1_per_lang(list(s1_gpt.values()), pred_key='pred_idiomaticity')
    print_cls_table("Stage 1 Classification:", cls_results)
    exact_ub, overlap_ub = span_f1_per_lang(list(s2_gpt.values()))
    print_span_table("Stage 2 Span (standalone):", exact_ub, overlap_ub)
    pipeline_records = build_pipeline_records(s1_gpt, s2_gpt, only_correct_cls=False)
    exact_e2e, overlap_e2e = span_f1_per_lang(pipeline_records)
    print_span_table("Full Pipeline Span (end-to-end):", exact_e2e, overlap_e2e)
    correct_records = build_pipeline_records(s1_gpt, s2_gpt, only_correct_cls=True)
    exact_corr, overlap_corr = span_f1_per_lang(correct_records)
    print_span_table("Span F1 (Correct Stage 1 Identifications Only):", exact_corr, overlap_corr)
    joint_acc, n_correct, n_total = compute_joint_acc(s1_gpt, s2_gpt)
    print(f"\n  Joint accuracy (cls + span both correct): {joint_acc:.4f} ({n_correct}/{n_total})")
    joint_f1 = compute_joint_f1(s1_gpt, s2_gpt)
    _print_joint_f1(joint_f1)
    stability     = compute_stability(joint_f1)
    small_lang_ci = compute_small_lang_bootstrap(s1_gpt, s2_gpt, n_resamples=n_bootstrap)
    indonesian_ci = compute_indonesian_bootstrap(s1_gpt, s2_gpt, n_resamples=n_bootstrap)
    print_stability_table("Stability metrics (in-distribution languages):", stability)
    print_bootstrap_ci("Hindi/Telugu bootstrap CIs:", small_lang_ci, " — Hindi+Telugu")
    print_bootstrap_ci("Indonesian held-out generalization:", indonesian_ci, " — zero-shot")
    return _build_return(cls_results, exact_ub, overlap_ub, exact_e2e, overlap_e2e,
                         exact_corr, overlap_corr, joint_acc, joint_f1,
                         stability, small_lang_ci, indonesian_ci)

def evaluate_system_c(single_gpt, n_bootstrap=10000):
    print("\n" + "="*60)
    print("System C: GPT-4o Single-Stage (classify + extract in one prompt)")
    print("="*60)
    if not single_gpt:
        print("  ✗ Missing prediction files"); return None
    records = list(single_gpt.values())
    cls_results = cls_f1_per_lang(records, pred_key='pred_label')
    print_cls_table("Classification:", cls_results)
    idiomatic_records = [r for r in records if r['idiomaticity'] == 'idiomatic']
    exact_idio, overlap_idio = span_f1_per_lang(idiomatic_records)
    print_span_table("Span Extraction (Idiomatic Only):", exact_idio, overlap_idio)
    e2e_records = []
    for r in idiomatic_records:
        cls_correct = r.get('pred_label') == r.get('idiomaticity')
        e2e_records.append({**r,
            'pred_span_start': r.get('pred_span_start') if cls_correct else None,
            'pred_span_end':   r.get('pred_span_end')   if cls_correct else None})
    exact_e2e, overlap_e2e = span_f1_per_lang(e2e_records)
    print_span_table("Span F1 (E2E — cls errors zeroed):", exact_e2e, overlap_e2e)
    correct_records = [r for r in idiomatic_records if r.get('pred_label') == 'idiomatic']
    exact_corr, overlap_corr = span_f1_per_lang(correct_records)
    print_span_table("Span F1 (Correct Identifications Only):", exact_corr, overlap_corr)
    joint_acc, n_correct, n_total = compute_joint_acc(single_gpt, single_gpt, pred_label_key='pred_label')
    print(f"\n  Joint accuracy (cls + span both correct): {joint_acc:.4f} ({n_correct}/{n_total})")
    joint_f1  = compute_joint_f1(single_gpt, single_gpt, pred_label_key='pred_label')
    _print_joint_f1(joint_f1)
    stability     = compute_stability(joint_f1)
    # FIX: pass single_gpt twice; use pred_label_key='pred_label'
    small_lang_ci = compute_small_lang_bootstrap(single_gpt, single_gpt,
                                                  pred_label_key='pred_label', n_resamples=n_bootstrap)
    indonesian_ci = compute_indonesian_bootstrap(single_gpt, single_gpt,
                                                  pred_label_key='pred_label', n_resamples=n_bootstrap)
    print_stability_table("Stability metrics (in-distribution languages):", stability)
    print_bootstrap_ci("Hindi/Telugu bootstrap CIs:", small_lang_ci, " — Hindi+Telugu")
    print_bootstrap_ci("Indonesian held-out generalization:", indonesian_ci, " — zero-shot")
    # FIX: use exact_idio/overlap_idio (not undefined exact_ub/overlap_ub)
    return _build_return(cls_results, exact_idio, overlap_idio, exact_e2e, overlap_e2e,
                         exact_corr, overlap_corr, joint_acc, joint_f1,
                         stability, small_lang_ci, indonesian_ci)

def evaluate_system_d(s1_mbert, span2_joint, n_bootstrap=10000):
    print("\n" + "="*60)
    print("System D: mBERT Stage 1 → Joint Model Span Head")
    print("="*60)
    if not s1_mbert or not span2_joint:
        print("  ✗ Missing prediction files"); return None
    cls_results = cls_f1_per_lang(list(s1_mbert.values()), pred_key='pred_idiomaticity')
    print_cls_table("Stage 1 Classification:", cls_results)
    exact_ub, overlap_ub = span_f1_per_lang(list(span2_joint.values()))
    print_span_table("Joint Span Head (standalone — upper bound):", exact_ub, overlap_ub)
    pipeline_records = build_pipeline_records(s1_mbert, span2_joint, only_correct_cls=False)
    exact_e2e, overlap_e2e = span_f1_per_lang(pipeline_records)
    print_span_table("Full Pipeline Span (end-to-end):", exact_e2e, overlap_e2e)
    correct_records = build_pipeline_records(s1_mbert, span2_joint, only_correct_cls=True)
    exact_corr, overlap_corr = span_f1_per_lang(correct_records)
    print_span_table("Span F1 (Correct Stage 1 Identifications Only):", exact_corr, overlap_corr)
    joint_acc, n_correct, n_total = compute_joint_acc(s1_mbert, span2_joint)
    print(f"\n  Joint accuracy (cls + span both correct): {joint_acc:.4f} ({n_correct}/{n_total})")
    joint_f1 = compute_joint_f1(s1_mbert, span2_joint)
    _print_joint_f1(joint_f1)
    stability     = compute_stability(joint_f1)
    small_lang_ci = compute_small_lang_bootstrap(s1_mbert, span2_joint, n_resamples=n_bootstrap)
    indonesian_ci = compute_indonesian_bootstrap(s1_mbert, span2_joint, n_resamples=n_bootstrap)
    print_stability_table("Stability metrics (in-distribution languages):", stability)
    print_bootstrap_ci("Hindi/Telugu bootstrap CIs:", small_lang_ci, " — Hindi+Telugu")
    print_bootstrap_ci("Indonesian held-out generalization:", indonesian_ci, " — zero-shot")
    return _build_return(cls_results, exact_ub, overlap_ub, exact_e2e, overlap_e2e,
                         exact_corr, overlap_corr, joint_acc, joint_f1,
                         stability, small_lang_ci, indonesian_ci)

def evaluate_system_e(joint_preds, n_bootstrap=10000):
    print("\n" + "="*60)
    print("System E: Joint mBERT (single model — classify + span)")
    print("="*60)
    if not joint_preds:
        print("  ✗ Missing prediction files"); return None
    records = list(joint_preds.values())
    cls_results = cls_f1_per_lang(records, pred_key='pred_idiomaticity')
    print_cls_table("Classification:", cls_results)
    exact_all, overlap_all = span_f1_per_lang(records)
    print_span_table("Span Extraction (all examples):", exact_all, overlap_all)
    idiomatic_records = [r for r in records if r['idiomaticity'] == 'idiomatic']
    exact_idio, overlap_idio = span_f1_per_lang(idiomatic_records)
    print_span_table("Span Extraction (idiomatic only):", exact_idio, overlap_idio)
    e2e_records = []
    for r in idiomatic_records:
        cls_correct = r.get('pred_idiomaticity') == r.get('idiomaticity')
        e2e_records.append({**r,
            'pred_span_start': r.get('pred_span_start') if cls_correct else None,
            'pred_span_end':   r.get('pred_span_end')   if cls_correct else None})
    exact_e2e, overlap_e2e = span_f1_per_lang(e2e_records)
    print_span_table("Span F1 (E2E — cls errors zeroed):", exact_e2e, overlap_e2e)
    correct_records = [r for r in records
                       if r['idiomaticity'] == 'idiomatic'
                       and r.get('pred_idiomaticity') == 'idiomatic']
    exact_corr, overlap_corr = span_f1_per_lang(correct_records)
    print_span_table("Span F1 (Correct Identifications Only):", exact_corr, overlap_corr)
    joint_acc, n_correct, n_total = compute_joint_acc(joint_preds, joint_preds)
    print(f"\n  Joint accuracy (cls + span both correct): {joint_acc:.4f} ({n_correct}/{n_total})")
    joint_f1 = compute_joint_f1(joint_preds, joint_preds)
    _print_joint_f1(joint_f1)
    stability     = compute_stability(joint_f1)
    small_lang_ci = compute_small_lang_bootstrap(joint_preds, joint_preds, n_resamples=n_bootstrap)
    indonesian_ci = compute_indonesian_bootstrap(joint_preds, joint_preds, n_resamples=n_bootstrap)
    print_stability_table("Stability metrics (in-distribution languages):", stability)
    print_bootstrap_ci("Hindi/Telugu bootstrap CIs:", small_lang_ci, " — Hindi+Telugu")
    print_bootstrap_ci("Indonesian held-out generalization:", indonesian_ci, " — zero-shot")
    # FIX: use exact_idio/overlap_idio (not undefined exact_ub/overlap_ub)
    return _build_return(cls_results, exact_idio, overlap_idio, exact_e2e, overlap_e2e,
                         exact_corr, overlap_corr, joint_acc, joint_f1,
                         stability, small_lang_ci, indonesian_ci)

def evaluate_system_f(seq_phase1, seq_phase2, n_bootstrap=10000):
    print("\n" + "="*60)
    print("System F: Sequential mBERT (Phase 1 cls → Phase 2 span fine-tune)")
    print("="*60)
    if not seq_phase1 or not seq_phase2:
        print("  ✗ Missing prediction files — run Train_Sequential.py first"); return None
    cls_results = cls_f1_per_lang(list(seq_phase1.values()), pred_key='pred_idiomaticity')
    print_cls_table("Phase 1 Classification:", cls_results)
    exact_p2, overlap_p2 = span_f1_per_lang(list(seq_phase2.values()))
    print_span_table("Phase 2 Span (standalone):", exact_p2, overlap_p2)
    pipeline_records = build_pipeline_records(seq_phase1, seq_phase2, only_correct_cls=False)
    exact_e2e, overlap_e2e = span_f1_per_lang(pipeline_records)
    print_span_table("Full Sequential Pipeline (end-to-end):", exact_e2e, overlap_e2e)
    correct_records = build_pipeline_records(seq_phase1, seq_phase2, only_correct_cls=True)
    exact_corr, overlap_corr = span_f1_per_lang(correct_records)
    print_span_table("Span F1 (Correct Phase 1 Identifications Only):", exact_corr, overlap_corr)
    joint_acc, n_correct, n_total = compute_joint_acc(seq_phase1, seq_phase2)
    print(f"\n  Joint accuracy (cls + span both correct): {joint_acc:.4f} ({n_correct}/{n_total})")
    joint_f1 = compute_joint_f1(seq_phase1, seq_phase2)
    _print_joint_f1(joint_f1)
    stability = compute_stability(joint_f1)
    # FIX: was using s1_mbert/s2_mbert (wrong scope) — now correctly uses seq_phase1/seq_phase2
    small_lang_ci = compute_small_lang_bootstrap(seq_phase1, seq_phase2, n_resamples=n_bootstrap)
    indonesian_ci = compute_indonesian_bootstrap(seq_phase1, seq_phase2, n_resamples=n_bootstrap)
    print_stability_table("Stability metrics (in-distribution languages):", stability)
    print_bootstrap_ci("Hindi/Telugu bootstrap CIs:", small_lang_ci, " — Hindi+Telugu")
    print_bootstrap_ci("Indonesian held-out generalization:", indonesian_ci, " — zero-shot")
    return _build_return(cls_results, exact_p2, overlap_p2, exact_e2e, overlap_e2e,
                         exact_corr, overlap_corr, joint_acc, joint_f1,
                         stability, small_lang_ci, indonesian_ci)

def evaluate_system_g(bio_preds, n_bootstrap=10000):
    print("\n" + "="*60)
    print("System G: BIO Token-Level Tagger (span extraction only)")
    print("="*60)
    if not bio_preds:
        print("  ✗ Missing prediction files — run Train_BIO_Tagger.py first"); return None
    records = list(bio_preds.values())
    exact_all, overlap_all = span_f1_per_lang(records)
    print_span_table("Span Extraction (all examples):", exact_all, overlap_all)
    idiomatic_records = [r for r in records if r['idiomaticity'] == 'idiomatic']
    exact_idio, overlap_idio = span_f1_per_lang(idiomatic_records)
    print_span_table("Span Extraction (idiomatic only):", exact_idio, overlap_idio)
    literal_records = [r for r in records if r['idiomaticity'] == 'literal']
    if literal_records:
        exact_lit, overlap_lit = span_f1_per_lang(literal_records)
        print_span_table("Span Extraction (literal only — diagnostic):", exact_lit, overlap_lit)
    correct_exact = sum(
        int(r.get('pred_span_start') == r['span_start'] and r.get('pred_span_end') == r['span_end'])
        for r in records if r.get('pred_span_start') is not None)
    total = len(records)
    joint_acc_bio = correct_exact / total if total > 0 else 0.0
    print(f"\n  Joint accuracy (span exact match, no cls gate): {joint_acc_bio:.4f} ({correct_exact}/{total})")
    lang_gold = defaultdict(list)
    lang_pred = defaultdict(list)
    for r in records:
        lang   = r['language']
        pred_s = r.get('pred_span_start')
        pred_e = r.get('pred_span_end')
        overlap = compute_overlap_f1(pred_s, pred_e, r['span_start'], r['span_end']) \
                  if pred_s is not None else 0.0
        lang_gold[lang].append(1)
        lang_pred[lang].append(1 if overlap > 0.0 else 0)
    span_joint_f1 = {}
    all_g, all_p = [], []
    for lang in sorted(lang_gold.keys()):
        f1 = f1_score(lang_gold[lang], lang_pred[lang], average='macro')
        span_joint_f1[lang] = round(f1, 4)
        all_g.extend(lang_gold[lang])
        all_p.extend(lang_pred[lang])
    span_joint_f1['Overall'] = round(f1_score(all_g, all_p, average='macro'), 4)
    print(f"\n  Span-only Joint F1 (any overlap = correct): {span_joint_f1}")
    stability = compute_stability(span_joint_f1)
    print_stability_table("Stability metrics (in-distribution languages):", stability)
    id_records = [r for r in records if r['language'] == HELD_OUT_LANG]
    indonesian_ci = None
    if id_records:
        exact_vals   = [int(r.get('pred_span_start') == r['span_start'] and
                            r.get('pred_span_end') == r['span_end'])
                        if r.get('pred_span_start') is not None else 0 for r in id_records]
        overlap_vals = [compute_overlap_f1(r.get('pred_span_start'), r.get('pred_span_end'),
                                           r['span_start'], r['span_end'])
                        if r.get('pred_span_start') is not None else 0.0 for r in id_records]
        indonesian_ci = {
            'n_examples':   len(id_records),
            'cls_accuracy': None,
            'span_exact':   bootstrap_ci(exact_vals,   n_resamples=n_bootstrap),
            'span_overlap': bootstrap_ci(overlap_vals, n_resamples=n_bootstrap),
            'joint_f1':     bootstrap_ci(overlap_vals, n_resamples=n_bootstrap),
        }
        print_bootstrap_ci("Indonesian held-out generalization:", indonesian_ci,
                           " — zero-shot, span only")
    return {
        'cls_f1':          None,
        'span_all':        {'exact': exact_all,  'overlap': overlap_all},
        'span_e2e':        {'exact': exact_idio, 'overlap': overlap_idio},
        'span_correct_id': {'exact': exact_idio, 'overlap': overlap_idio},
        'joint_acc':       round(joint_acc_bio, 4),
        'joint_f1':        span_joint_f1,
        'stability':       stability,
        'small_lang_ci':   None,
        'indonesian_ci':   indonesian_ci,
    }

def print_summary(results_a, results_b, results_c, results_d,
                  results_e, results_f, results_g):
    LANGS   = ['English', 'Spanish', 'Hindi', 'Telugu']
    systems = [
        ("A: mBERT Pipeline (S1->S2)",    results_a),
        ("B: GPT-4o Pipeline (S1->S2)",    results_b),
        ("C: GPT-4o Single-Stage",         results_c),
        ("D: mBERT S1 -> Joint Span Head", results_d),
        ("E: Joint mBERT (one-pass)",      results_e),
        ("F: Sequential mBERT (Ph1->Ph2)", results_f),
        ("G: BIO Tagger (span only)",      results_g),
    ]
    def fmt(v):
        if isinstance(v, bool): return str(v)
        if isinstance(v, (float, int)): return f'{v:.4f}'
        return '—' if v is None else str(v)
    def get(res, *keys, default='—'):
        val = res
        for k in keys:
            if not isinstance(val, dict): return default
            val = val.get(k, default)
        return val
    print("\n" + "="*108)
    print("SUMMARY — Full Pipeline Comparison (Overall)")
    print("="*108)
    header = (f"{'System':<45} {'Cls F1':<10} {'E2E Span F1':<14} "
              f"{'Corr-ID Span F1':<16} {'Joint Acc':<12} {'Joint F1':<12} {'Joint F1 (macro-avg)':<20}")
    print(header)
    print("-" * len(header))
    for label, res in systems:
        if not res:
            print(f"{label:<45} {'—':<10} {'—':<14} {'—':<16} {'—':<12} {'—':<10}"); continue
        cls     = get(res, 'cls_f1',          'Overall')
        e2e     = get(res, 'span_e2e',        'overlap', 'Overall')
        corr    = get(res, 'span_correct_id', 'overlap', 'Overall')
        jacc    = get(res, 'joint_acc')
        jf1_d   = get(res, 'joint_f1', 'Overall')
        jf1     = jf1_d.get('macro_f1')     if isinstance(jf1_d, dict) else jf1_d
        jf1_avg = jf1_d.get('macro_avg_f1') if isinstance(jf1_d, dict) else None
        note    = ' *' if res.get('cls_f1') is None else ''
        print(f"{label+note:<45} {fmt(cls):<10} {fmt(e2e):<14} {fmt(corr):<16} "
              f"{fmt(jacc):<12} {fmt(jf1):<12} {fmt(jf1_avg):<20}")
    print("  * System G has no classifier — Cls F1 and Joint metrics are span-only.")
    lhdr = f"{'System':<45} " + "".join(f"{l:<14}" for l in LANGS) + f"{'Overall':<10}"
    for section_label, key_path, heading in [
        ("Classification Macro F1 — Per Language", ('cls_f1',), "Classification Macro F1 — Per Language"),
        ("E2E Span Overlap F1 — Per Language  (idiomatic, cls errors zeroed)",
         ('span_e2e','overlap'), "E2E Span Overlap F1 — Per Language"),
        ("Joint F1 — Per Language", ('joint_f1',), "Joint F1 — Per Language"),
        ("Corr-ID Span Overlap F1 — Per Language", ('span_correct_id','overlap'),
         "Corr-ID Span Overlap F1 — Per Language"),
    ]:
        print(f"\n{'─'*108}")
        print(heading)
        print(f"{'─'*108}")
        print(lhdr)
        print("-" * len(lhdr))
        for label, res in systems:
            if not res:
                print(f"{label:<45} " + "".join(f"{'—':<14}" for _ in LANGS) + f"{'—':<10}"); continue
            if key_path == ('cls_f1',):
                if res.get('cls_f1') is None:
                    print(f"{label:<45} " + "".join(f"{'—':<14}" for _ in LANGS) + f"{'—':<10}"); continue
                sf = res['cls_f1']
                row = f"{label:<45} " + "".join(f"{fmt(sf.get(l,'—')):<14}" for l in LANGS)
                row += f"{fmt(sf.get('Overall','—')):<10}"
            elif key_path == ('joint_f1',):
                jf = res.get('joint_f1', {})
                if not isinstance(jf, dict):
                    print(f"{label:<45} " + "".join(f"{'—':<14}" for _ in LANGS) + f"{'—':<10}"); continue
                def jf_macro(d):
                    return d.get('macro_avg_f1', d.get('macro_f1')) if isinstance(d, dict) else d
                row  = f"{label:<45} " + "".join(f"{fmt(jf_macro(jf.get(l,'—'))):<14}" for l in LANGS)
                row += f"{fmt(jf_macro(jf.get('Overall','—'))):<10}"
            else:
                sf = get(res, *key_path)
                if sf == '—' or not isinstance(sf, dict):
                    print(f"{label:<45} " + "".join(f"{'—':<14}" for _ in LANGS) + f"{'—':<10}"); continue
                row  = f"{label:<45} " + "".join(f"{fmt(sf.get(l,'—')):<14}" for l in LANGS)
                row += f"{fmt(sf.get('Overall','—')):<10}"
            print(row)
    print(f"\n{'─'*122}")
    print("Stability Metrics — Joint F1 across In-Distribution Languages")
    print(f"  (Stability = Mean - Std: higher = better average AND more consistent)")
    print(f"{'─'*122}")
    shdr = (f"{'System':<45} {'Mean':<10} {'Std':<10} {'Worst Lang':<16} "
            f"{'Worst F1':<12} {'Gap':<10} {'Stability':<12}")
    print(shdr)
    print("-" * len(shdr))
    for label, res in systems:
        if not res or not res.get('stability'):
            print(f"{label:<45} —"); continue
        s = res['stability']
        print(f"{label:<45} {fmt(s.get('mean_joint')):<10} {fmt(s.get('std_joint')):<10} "
              f"{s.get('worst_lang','—'):<16} {fmt(s.get('worst_f1')):<12} "
              f"{fmt(s.get('gap')):<10} {fmt(s.get('stability')):<12}")
    print(f"\n{'─'*108}")
    print("Indonesian Held-Out Generalization — 95% Bootstrap CIs  (n~33 test examples)")
    print(f"  Indonesian excluded from all training; zero-shot cross-lingual transfer.")
    print(f"{'─'*108}")
    ihdr = (f"{'System':<45} {'Cls Acc':<22} {'Span Exact':<22} "
            f"{'Span Overlap':<22} {'Joint F1':<22}")
    print(ihdr)
    print("-" * len(ihdr))
    for label, res in systems:
        if not res or not res.get('indonesian_ci'):
            print(f"{label:<45}   [no data]"); continue
        ci = res['indonesian_ci']
        def fmt_ci(tup):
            if tup is None or (isinstance(tup, tuple) and tup[0] is None): return "— (no cls)"
            pt, lo, hi = tup
            return f"{pt:.4f} [{lo:.4f},{hi:.4f}]"
        print(f"{label:<45} "
              f"{fmt_ci(ci.get('cls_accuracy')):<22} "
              f"{fmt_ci(ci.get('span_exact')):<22} "
              f"{fmt_ci(ci.get('span_overlap')):<22} "
              f"{fmt_ci(ci.get('joint_f1')):<22}")

def main():
    args       = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print("Loading prediction files...")
    s1_mbert    = load_preds(args.stage1_mbert)
    s2_mbert    = load_preds(args.stage2_mbert)
    s1_gpt      = load_preds(args.stage1_gpt)
    s2_gpt      = load_preds(args.stage2_gpt)
    single_gpt  = load_preds(args.single_gpt)
    joint_preds = load_preds(args.joint_preds)
    span2_joint = load_preds(args.span2_joint)
    seq_phase1  = load_preds(args.seq_phase1)
    seq_phase2  = load_preds(args.seq_phase2)
    bio_preds   = load_preds(args.bio_preds)
    print(f"  Stage 1 mBERT    : {len(s1_mbert)} predictions")
    print(f"  Stage 2 mBERT    : {len(s2_mbert)} predictions")
    print(f"  Stage 1 GPT      : {len(s1_gpt)} predictions")
    print(f"  Stage 2 GPT      : {len(s2_gpt)} predictions")
    print(f"  Single GPT       : {len(single_gpt)} predictions")
    print(f"  Joint mBERT      : {len(joint_preds)} predictions")
    print(f"  Joint span only  : {len(span2_joint)} predictions")
    print(f"  Sequential Ph1   : {len(seq_phase1)} predictions")
    print(f"  Sequential Ph2   : {len(seq_phase2)} predictions")
    print(f"  BIO Tagger       : {len(bio_preds)} predictions")
    print(f"  Bootstrap resamples: {args.n_bootstrap}")
    all_keys = [set(d) for d in [s1_mbert, s2_mbert, s1_gpt, s2_gpt, single_gpt,
                                  joint_preds, span2_joint, seq_phase1, seq_phase2, bio_preds] if d]
    common = set.intersection(*all_keys) if all_keys else set()
    print(f"\n  Common sentences across all systems: {len(common)}")
    print(f"  Filtering to common set for fair comparison...")
    def filt(d): return {s: p for s, p in d.items() if s in common}
    s1_mbert    = filt(s1_mbert);    s2_mbert    = filt(s2_mbert)
    s1_gpt      = filt(s1_gpt);      s2_gpt      = filt(s2_gpt)
    single_gpt  = filt(single_gpt);  joint_preds = filt(joint_preds)
    span2_joint = filt(span2_joint); seq_phase1  = filt(seq_phase1)
    seq_phase2  = filt(seq_phase2);  bio_preds   = filt(bio_preds)
    print(f"  After filtering: {len(s1_mbert)} examples per system\n")
    nb = args.n_bootstrap
    results_a = evaluate_system_a(s1_mbert, s2_mbert,       n_bootstrap=nb)
    results_b = evaluate_system_b(s1_gpt,   s2_gpt,         n_bootstrap=nb)
    results_c = evaluate_system_c(single_gpt,                n_bootstrap=nb)
    results_d = evaluate_system_d(s1_mbert, span2_joint,    n_bootstrap=nb)
    results_e = evaluate_system_e(joint_preds,               n_bootstrap=nb)
    results_f = evaluate_system_f(seq_phase1, seq_phase2,   n_bootstrap=nb)
    results_g = evaluate_system_g(bio_preds,                 n_bootstrap=nb)
    print_summary(results_a, results_b, results_c, results_d,
                  results_e, results_f, results_g)
    all_results = {
        'system_a_mbert_pipeline':        results_a,
        'system_b_gpt_pipeline':          results_b,
        'system_c_gpt_single':            results_c,
        'system_d_mbert_s1_joint_span':   results_d,
        'system_e_joint_end_to_end':      results_e,
        'system_f_sequential_phase1_ph2': results_f,
        'system_g_bio_tagger':            results_g,
    }
    out_path = output_dir / 'pipeline_eval_results.json'
    json.dump(all_results, open(out_path, 'w'), indent=2, default=str)
    print(f"\nFull results saved -> {out_path}")

if __name__ == '__main__':
    main()