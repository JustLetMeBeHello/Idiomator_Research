import json
import argparse
import numpy as np
from pathlib import Path
from collections import defaultdict, Counter

def parse_args():
    base = Path("/Users/shishirmaddineni/Desktop/Idiomator_Research/Research_And_Training")
    p = argparse.ArgumentParser()
    # mBERT Paths — all now point to *_shared.jsonl (379 examples, same test set as GPT)
    p.add_argument('--joint_mbert',  default=base / 'models/joint_mbert_en_hi_te/test_predictions_shared.jsonl')
    p.add_argument('--stage1_mbert', default=base / 'models/stage1_mbert_en_hi_te/test_predictions_shared.jsonl')
    p.add_argument('--stage2_mbert', default=base / 'models/stage2_mbert_shared/test_predictions_shared.jsonl')
    # GPT Paths (unchanged)
    p.add_argument('--stage1_gpt',   default=base / 'models/gpt_baseline_stage1/test_predictions.jsonl')
    p.add_argument('--stage2_gpt',   default=base / 'models/gpt_baseline_stage2/test_predictions.jsonl')
    p.add_argument('--single_gpt',   default=base / 'models/gpt_single_stage/test_predictions.jsonl')
    return p.parse_args()


def load_preds(path):
    if not Path(path).exists():
        print(f"  [WARNING] File not found: {path}")
        return {}
    return {
        json.loads(line)['sentence'].strip().lower(): json.loads(line)
        for line in open(path, encoding='utf-8')
    }


def compute_overlap_f1(pred_start, pred_end, gold_start, gold_end):
    if pred_start is None or pred_end is None:
        return 0.0
    pred_set = set(range(int(pred_start), int(pred_end)))
    gold_set = set(range(int(gold_start), int(gold_end)))
    if not pred_set or not gold_set:
        return 0.0
    intersect = len(pred_set & gold_set)
    if intersect == 0:
        return 0.0
    precision = intersect / len(pred_set)
    recall    = intersect / len(gold_set)
    return 2 * (precision * recall) / (precision + recall)


def evaluate_system(name, records):
    lang_stats = defaultdict(lambda: {'joint_acc': [], 'overlap_f1': []})

    for r in records:
        lang = r.get('language', 'Unknown')

        pred_label = r.get('pred_idiomaticity', r.get('pred_label'))
        gold_label = r.get('idiomaticity', r.get('label'))
        label_correct = (pred_label == gold_label)

        pred_s, pred_e = r.get('pred_span_start'), r.get('pred_span_end')
        gold_s, gold_e = r.get('span_start'), r.get('span_end')

        exact_match = (pred_s == gold_s and pred_e == gold_e)
        lang_stats[lang]['joint_acc'].append(int(label_correct and exact_match))

        f1 = compute_overlap_f1(pred_s, pred_e, gold_s, gold_e) if label_correct else 0.0
        lang_stats[lang]['overlap_f1'].append(f1)

    print(f"\n--- {name} ---")
    print(f"{'Language':<12} {'Joint Acc':<12} {'Overlap F1':<12} {'N':<6}")
    print("-" * 44)

    all_joint, all_f1 = [], []
    for lang in sorted(lang_stats.keys()):
        j_acc = np.mean(lang_stats[lang]['joint_acc'])
        o_f1  = np.mean(lang_stats[lang]['overlap_f1'])
        n     = len(lang_stats[lang]['joint_acc'])
        all_joint.extend(lang_stats[lang]['joint_acc'])
        all_f1.extend(lang_stats[lang]['overlap_f1'])
        print(f"{lang:<12} {j_acc:<12.4f} {o_f1:<12.4f} {n:<6}")

    print("-" * 44)
    print(f"{'OVERALL':<12} {np.mean(all_joint):<12.4f} {np.mean(all_f1):<12.4f} {len(all_joint):<6}")


def main():
    args = parse_args()

    j_mb   = load_preds(args.joint_mbert)
    s1_mb  = load_preds(args.stage1_mbert)
    s2_mb  = load_preds(args.stage2_mbert)
    s1_g   = load_preds(args.stage1_gpt)
    s2_g   = load_preds(args.stage2_gpt)
    sing_g = load_preds(args.single_gpt)

    # All systems now share the same 379-example test set — no intersection needed
    all_sentences = set(j_mb.keys()) | set(s1_mb.keys()) | set(s1_g.keys())
    print(f"\nTotal unique sentences across all files: {len(all_sentences)}")
    for name, preds in [("joint_mbert", j_mb), ("stage1_mbert", s1_mb),
                         ("stage2_mbert", s2_mb), ("stage1_gpt", s1_g),
                         ("stage2_gpt", s2_g)]:
        langs = Counter(v.get('language') for v in preds.values())
        print(f"  {name}: {len(preds)} examples {dict(langs)}")

    # 1. Joint mBERT (one-pass)
    evaluate_system("mBERT JOINT", list(j_mb.values()))

    # 2. mBERT Pipeline (Stage 1 cls + joint span head)
    mb_pipe = []
    for s, r in s1_mb.items():
        res = r.copy()
        res.update({
            'pred_span_start': s2_mb.get(s, {}).get('pred_span_start'),
            'pred_span_end':   s2_mb.get(s, {}).get('pred_span_end'),
        })
        mb_pipe.append(res)
    evaluate_system("mBERT PIPELINE (Stage 1 + Joint span head)", mb_pipe)

    # 3. GPT-4o Pipeline
    gpt_pipe = []
    for s, r in s1_g.items():
        res = r.copy()
        res.update({
            'pred_span_start': s2_g.get(s, {}).get('pred_span_start'),
            'pred_span_end':   s2_g.get(s, {}).get('pred_span_end'),
        })
        gpt_pipe.append(res)
    evaluate_system("GPT-4o PIPELINE (Stage 1 + 2)", gpt_pipe)

    # 4. GPT-4o Single-Stage
    evaluate_system("GPT-4o SINGLE-STAGE", list(sing_g.values()))


if __name__ == '__main__':
    main()