"""
gpt_baseline_single_stage.py

GPT-4o single-stage baseline — classifies idiomaticity AND extracts MWE span
in a single prompt. No pipeline, no two-stage design.

Used to compare against:
  - Two-stage GPT pipeline (Stage 1 → Stage 2 separately)
  - Two-stage mBERT pipeline
  - End-to-end mBERT

Evaluates:
  - Classification: macro F1 per language
  - Span extraction: exact match + overlap F1 per language
  - Both from a single GPT response

Usage:
    export OPENAI_API_KEY=sk-...
    python gpt_baseline_single_stage.py \
        --test_path  idioms_structured/Splits/test.jsonl \
        --output_dir models/gpt_single_stage

    # Dry run
    python gpt_baseline_single_stage.py --dry_run 10
"""

import os
import json
import time
import argparse
import numpy as np
from pathlib import Path
from collections import defaultdict

from openai import OpenAI
from sklearn.metrics import classification_report, f1_score
from tqdm import tqdm
from dotenv import load_dotenv
load_dotenv()

# ── Args ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--test_path',  default='idioms_structured/Splits/test.jsonl')
    p.add_argument('--output_dir', default='models/gpt_single_stage')
    p.add_argument('--model',      default='gpt-4o')
    p.add_argument('--dry_run',    type=int, default=None)
    p.add_argument('--sleep',      type=float, default=0.5)
    p.add_argument('--resume',     action='store_true')
    return p.parse_args()


# ── Prompt ────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are a linguist specializing in idiomatic language and multi-word expressions (MWEs).

Given a sentence, you must:
1. Classify whether it contains an idiomatic (figurative) or literal use of language
2. Extract the multi-word expression or key phrase from the sentence

Rules:
- Respond in EXACTLY this format, nothing else:
  Label: [idiomatic/literal]
  Span: [exact text of the expression as it appears in the sentence]

- Label must be exactly "idiomatic" or "literal"
- Span must be copied exactly from the sentence — do not paraphrase
- Always provide both Label and Span regardless of the label
- Do not add any explanation or extra text
"""

def make_prompt(sentence):
    return f'Analyze this sentence:\n\n"{sentence}"'


# ── API call ──────────────────────────────────────────────────────────────────

def classify_and_extract(client, sentence, model, retries=3):
    """Returns (label, span_text) or (None, None) on failure."""
    for attempt in range(retries):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user",   "content": make_prompt(sentence)},
                ],
                temperature=0,
                max_tokens=50,
            )
            raw = response.choices[0].message.content.strip()
            return parse_response(raw)

        except Exception as e:
            if attempt < retries - 1:
                print(f"\n  API error (attempt {attempt+1}): {e} — retrying in 5s...")
                time.sleep(5)
            else:
                print(f"\n  API error after {retries} attempts: {e}")
                return None, None


def parse_response(raw):
    """Parse GPT response into (label, span_text)."""
    label = None
    span  = None

    for line in raw.split('\n'):
        line = line.strip()
        if line.lower().startswith('label:'):
            val = line[6:].strip().lower()
            if 'idiomatic' in val:
                label = 'idiomatic'
            elif 'literal' in val:
                label = 'literal'
        elif line.lower().startswith('span:'):
            span = line[5:].strip().strip('"\'')

    return label, span


# ── Span matching ─────────────────────────────────────────────────────────────

def find_span_in_sentence(sentence, predicted_text):
    """Convert predicted span text to character offsets."""
    if not predicted_text:
        return None, None

    # Exact match
    idx = sentence.find(predicted_text)
    if idx != -1:
        return idx, idx + len(predicted_text)

    # Case-insensitive
    idx = sentence.lower().find(predicted_text.lower())
    if idx != -1:
        return idx, idx + len(predicted_text)

    # Partial match — progressively shorter substrings
    words = predicted_text.split()
    for length in range(len(words), 0, -1):
        for start in range(len(words) - length + 1):
            substr = ' '.join(words[start:start + length])
            idx = sentence.lower().find(substr.lower())
            if idx != -1:
                return idx, idx + len(substr)

    return None, None


# ── Metrics ───────────────────────────────────────────────────────────────────

LABEL2ID = {'literal': 0, 'idiomatic': 1}

def compute_overlap_f1(pred_start, pred_end, gold_start, gold_end):
    """Character-level overlap F1."""
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


def evaluate(predictions):
    print("\n── Classification Results ──")
    print(f"{'Language':<12} {'Macro F1':<12} {'N':<6}")
    print("-" * 32)

    all_preds  = []
    all_labels = []
    lang_cls   = defaultdict(lambda: {'preds': [], 'labels': []})

    for p in predictions:
        if p['pred_label'] is None:
            continue
        lang = p['language']
        pred  = LABEL2ID.get(p['pred_label'], 1)
        label = LABEL2ID.get(p['idiomaticity'], 1)
        all_preds.append(pred)
        all_labels.append(label)
        lang_cls[lang]['preds'].append(pred)
        lang_cls[lang]['labels'].append(label)

    cls_lang_f1 = {}
    for lang in sorted(lang_cls.keys()):
        f1 = f1_score(lang_cls[lang]['labels'], lang_cls[lang]['preds'], average='macro')
        cls_lang_f1[lang] = round(f1, 4)
        print(f"{lang:<12} {f1:<12.4f} {len(lang_cls[lang]['preds']):<6}")

    overall_cls_f1 = f1_score(all_labels, all_preds, average='macro')
    print(f"{'Overall':<12} {overall_cls_f1:<12.4f} {len(all_preds):<6}")

    print("\n── Span Extraction Results ──")
    print(f"{'Language':<12} {'Exact Match':<14} {'Overlap F1':<12} {'N':<6}")
    print("-" * 46)

    lang_exact = defaultdict(list)
    lang_f1    = defaultdict(list)

    for p in predictions:
        lang = p['language']
        lang_exact[lang].append(int(p['exact_match']))
        lang_f1[lang].append(float(p['overlap_f1']))

    span_lang_f1 = {}
    for lang in sorted(lang_exact.keys()):
        em = np.mean(lang_exact[lang])
        f1 = np.mean(lang_f1[lang])
        span_lang_f1[lang] = {'exact': round(em, 4), 'overlap': round(f1, 4)}
        print(f"{lang:<12} {em:<14.4f} {f1:<12.4f} {len(lang_exact[lang]):<6}")

    overall_exact  = np.mean([v for vals in lang_exact.values() for v in vals])
    overall_span_f1 = np.mean([v for vals in lang_f1.values() for v in vals])
    print(f"{'Overall':<12} {overall_exact:<14.4f} {overall_span_f1:<12.4f} {sum(len(v) for v in lang_exact.values()):<6}")

    return overall_cls_f1, overall_exact, overall_span_f1, cls_lang_f1, span_lang_f1


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY not set. Run: export OPENAI_API_KEY=sk-...")

    client     = OpenAI(api_key=api_key)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load test set
    test_records = [json.loads(l) for l in open(args.test_path, encoding='utf-8')]
    if args.dry_run:
        test_records = test_records[:args.dry_run]
        print(f"Dry run: {args.dry_run} examples")

    print(f"Test set : {len(test_records)} examples")
    print(f"Model    : {args.model}")

    # Resume support
    preds_path = output_dir / 'test_predictions.jsonl'
    completed  = {}
    if args.resume and preds_path.exists():
        for line in open(preds_path, encoding='utf-8'):
            r = json.loads(line)
            completed[r['sentence']] = r
        print(f"Resuming: {len(completed)} already completed")

    predictions   = []
    cost_estimate = 0.0
    parse_failures = 0
    span_not_found = 0

    with open(preds_path, 'a' if args.resume else 'w', encoding='utf-8') as f_out:
        for record in tqdm(test_records, desc="Single-stage GPT", unit="example"):
            sentence = record['sentence']

            if sentence in completed:
                predictions.append(completed[sentence])
                continue

            # Single GPT call — classify + extract
            pred_label, pred_span_text = classify_and_extract(client, sentence, args.model)

            if pred_label is None:
                parse_failures += 1
                pred_label = 'idiomatic'  # default

            # Find span in sentence
            if pred_span_text:
                pred_char_s, pred_char_e = find_span_in_sentence(sentence, pred_span_text)
            else:
                pred_char_s, pred_char_e = None, None

            if pred_char_s is None:
                span_not_found += 1
                pred_char_s = 0
                pred_char_e = 0

            gold_char_s = record['span_start']
            gold_char_e = record['span_end']

            # Guard against missing gold spans in the data
            if gold_char_s is None or gold_char_e is None:
                result = {
                    **record,
                    'pred_label':      pred_label,
                    'pred_span_text':  pred_span_text or '',
                    'pred_span_start': pred_char_s,
                    'pred_span_end':   pred_char_e,
                    'exact_match':     False,
                    'overlap_f1':      0.0,
                    'correct_label':   bool(pred_label == record['idiomaticity']),
                }
                predictions.append(result)
                f_out.write(json.dumps(result, ensure_ascii=False) + '\n')
                f_out.flush()
                continue

            exact   = bool(pred_char_s == gold_char_s and pred_char_e == gold_char_e)
            overlap = compute_overlap_f1(pred_char_s, pred_char_e, gold_char_s, gold_char_e)

            result = {
                **record,
                'pred_label':      pred_label,
                'pred_span_text':  pred_span_text or '',
                'pred_span_start': pred_char_s,
                'pred_span_end':   pred_char_e,
                'exact_match':     exact,
                'overlap_f1':      round(overlap, 4),
                'correct_label':   bool(pred_label == record['idiomaticity']),
            }

            predictions.append(result)
            f_out.write(json.dumps(result, ensure_ascii=False) + '\n')
            f_out.flush()

            # ~50 tokens/call
            cost_estimate += 50 * 2.50 / 1_000_000
            time.sleep(args.sleep)

    print(f"\nParse failures : {parse_failures}")
    print(f"Spans not found: {span_not_found}")
    print(f"Est. API cost  : ${cost_estimate:.3f}")

    # Evaluate
    cls_f1, exact, span_f1, cls_lang, span_lang = evaluate(predictions)

    # Save metrics
    metrics = {
        'model':                  args.model,
        'approach':               'single_stage',
        'test_size':              len(predictions),
        'classification_macro_f1': round(float(cls_f1), 4),
        'span_exact_match':       round(float(exact), 4),
        'span_overlap_f1':        round(float(span_f1), 4),
        'cls_per_lang':           cls_lang,
        'span_per_lang':          span_lang,
        'parse_failures':         parse_failures,
        'spans_not_found':        span_not_found,
        'cost_estimate':          round(cost_estimate, 4),
    }
    json.dump(metrics, open(output_dir / 'metrics.json', 'w'), indent=2)
    print(f"\nMetrics saved → {output_dir / 'metrics.json'}")

    # Compare against two-stage GPT
    print("\n── Single-stage vs Two-stage GPT comparison ──")
    print(f"  Single-stage GPT : cls_f1={cls_f1:.4f}  span_overlap={span_f1:.4f}")

    s1_path = Path('models/gpt_baseline_stage1/metrics.json')
    s2_path = Path('models/gpt_baseline_stage2/metrics.json')
    if s1_path.exists() and s2_path.exists():
        s1 = json.load(open(s1_path))
        s2 = json.load(open(s2_path))
        print(f"  Two-stage GPT    : cls_f1={s1['test_macro_f1']:.4f}  span_overlap={s2['test_overlap_f1']:.4f}")

    mbert_path = Path('models/stage2_mbert_final/metrics.json')
    if mbert_path.exists():
        mbert = json.load(open(mbert_path))
        print(f"  mBERT pipeline   : span_overlap={mbert['test_overlap_f1']:.4f}")


if __name__ == '__main__':
    main()