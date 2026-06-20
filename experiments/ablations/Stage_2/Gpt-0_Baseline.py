"""
gpt_baseline_stage2.py

Zero-shot GPT-4o baseline for MWE span extraction (Stage 2).
Input  : raw sentence only (no idiom hint)
Output : extracted span text → converted to character offsets

Evaluates on the test split across English, Hindi, and Telugu.
Reports exact match and overlap F1, matching Stage 2 mBERT format.

Usage:
    export OPENAI_API_KEY=sk-...
    python gpt_baseline_stage2.py \
        --test_path  data/idioms_structured/Splits/test.jsonl \
        --output_dir models/gpt_baseline_stage2

    # Dry run on first 10 examples
    python gpt_baseline_stage2.py \
        --test_path  data/idioms_structured/Splits/test.jsonl \
        --output_dir models/gpt_baseline_stage2 \
        --dry_run 10
"""

import os
import json
import time
import argparse
import numpy as np
from pathlib import Path
from collections import defaultdict

from openai import OpenAI
from tqdm import tqdm
from dotenv import load_dotenv
load_dotenv()


# ── Args ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--test_path',  default='data/idioms_structured/Splits/test.jsonl')
    p.add_argument('--output_dir', default='models/gpt_baseline_stage2')
    p.add_argument('--model',      default='gpt-4o')
    p.add_argument('--dry_run',    type=int, default=None)
    p.add_argument('--sleep',      type=float, default=0.5)
    p.add_argument('--resume',     action='store_true')
    return p.parse_args()


# ── Prompt ────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are a linguist specializing in multi-word expressions (MWEs).
Your task is to identify the multi-word expression or idiom in a sentence.

Rules:
- Return ONLY the exact text of the multi-word expression as it appears in the sentence
- The expression can be idiomatic (figurative) or literal — extract it either way
- Do not explain your answer
- Do not add quotes, punctuation, or any other text
- If you cannot find a multi-word expression, return the single most content-heavy word
"""

def make_prompt(sentence):
    return f'Extract the multi-word expression from this sentence:\n\n"{sentence}"'


# ── API call ──────────────────────────────────────────────────────────────────

def extract_span(client, sentence, model, retries=3):
    for attempt in range(retries):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user",   "content": make_prompt(sentence)},
                ],
                temperature=0,
                max_tokens=30,
            )
            raw = response.choices[0].message.content.strip()
            # Strip quotes if GPT added them
            raw = raw.strip('"\'')
            return raw

        except Exception as e:
            if attempt < retries - 1:
                print(f"\n  API error (attempt {attempt+1}): {e} — retrying in 5s...")
                time.sleep(5)
            else:
                print(f"\n  API error after {retries} attempts: {e}")
                return None


# ── Span matching ─────────────────────────────────────────────────────────────

def find_span_in_sentence(sentence, predicted_text):
    """
    Find character offsets of predicted_text in sentence.
    Tries exact match first, then case-insensitive.
    Returns (char_start, char_end) or (None, None) if not found.
    """
    if not predicted_text:
        return None, None

    # Exact match
    idx = sentence.find(predicted_text)
    if idx != -1:
        return idx, idx + len(predicted_text)

    # Case-insensitive match
    idx = sentence.lower().find(predicted_text.lower())
    if idx != -1:
        return idx, idx + len(predicted_text)

    # Partial match — find longest common substring
    # Try progressively shorter versions of the prediction
    words = predicted_text.split()
    for length in range(len(words), 0, -1):
        for start in range(len(words) - length + 1):
            substr = ' '.join(words[start:start + length])
            idx = sentence.lower().find(substr.lower())
            if idx != -1:
                return idx, idx + len(substr)

    return None, None


# ── Overlap F1 ────────────────────────────────────────────────────────────────

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


# ── Eval ──────────────────────────────────────────────────────────────────────

def evaluate(predictions):
    lang_exact = defaultdict(list)
    lang_f1    = defaultdict(list)

    for p in predictions:
        lang = p['language']
        lang_exact[lang].append(int(p['exact_match']))
        lang_f1[lang].append(float(p['overlap_f1']))

    print("\n── Test results ──")
    print(f"{'Language':<12} {'Exact Match':<14} {'Overlap F1':<12} {'N':<6}")
    print("-" * 46)

    overall_exact = []
    overall_f1    = []

    for lang in sorted(lang_exact.keys()):
        em = np.mean(lang_exact[lang])
        f1 = np.mean(lang_f1[lang])
        n  = len(lang_exact[lang])
        print(f"{lang:<12} {em:<14.4f} {f1:<12.4f} {n:<6}")
        overall_exact.extend(lang_exact[lang])
        overall_f1.extend(lang_f1[lang])

    print("-" * 46)
    print(f"{'Overall':<12} {np.mean(overall_exact):<14.4f} {np.mean(overall_f1):<12.4f} {len(overall_exact):<6}")

    return np.mean(overall_exact), np.mean(overall_f1)


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
        print(f"Dry run: using first {args.dry_run} examples")

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

    # Run extraction
    predictions  = []
    cost_estimate = 0.0
    not_found    = 0

    with open(preds_path, 'a' if args.resume else 'w', encoding='utf-8') as f_out:
        for record in tqdm(test_records, desc="Extracting spans", unit="example"):
            sentence = record['sentence']

            # Skip if already done
            if sentence in completed:
                predictions.append(completed[sentence])
                continue

            # Ask GPT for the span text
            pred_text = extract_span(client, sentence, args.model)

            # Find span in sentence
            if pred_text:
                pred_char_s, pred_char_e = find_span_in_sentence(sentence, pred_text)
            else:
                pred_char_s, pred_char_e = None, None

            if pred_char_s is None:
                not_found += 1
                pred_char_s = 0
                pred_char_e = 0

            # Gold span
            gold_char_s = record['span_start']
            gold_char_e = record['span_end']

            # Guard against missing gold spans in the data
            if gold_char_s is None or gold_char_e is None:
                result = {
                    **record,
                    'pred_span_text':  pred_text or '',
                    'pred_span_start': pred_char_s,
                    'pred_span_end':   pred_char_e,
                    'exact_match':     False,
                    'overlap_f1':      0.0,
                }
                predictions.append(result)
                f_out.write(json.dumps(result, ensure_ascii=False) + '\n')
                f_out.flush()
                continue

            # Exact match (character level)
            exact = bool(
                pred_char_s == gold_char_s and
                pred_char_e == gold_char_e
            )

            # Overlap F1 (character level)
            overlap = compute_overlap_f1(
                pred_char_s, pred_char_e,
                gold_char_s, gold_char_e
            )

            result = {
                **record,
                'pred_span_text':  pred_text or '',
                'pred_span_start': pred_char_s,
                'pred_span_end':   pred_char_e,
                'exact_match':     exact,
                'overlap_f1':      round(overlap, 4),
            }

            predictions.append(result)
            f_out.write(json.dumps(result, ensure_ascii=False) + '\n')
            f_out.flush()

            # Cost estimate (~30 tokens/call, gpt-4o $2.50/1M input)
            cost_estimate += 30 * 2.50 / 1_000_000
            time.sleep(args.sleep)

    print(f"\nSpans not found in sentence: {not_found} ({not_found/len(predictions)*100:.1f}%)")
    print(f"Estimated API cost: ${cost_estimate:.3f}")

    # Evaluate
    overall_exact, overall_f1 = evaluate(predictions)

    # Save metrics
    metrics = {
        'model':               args.model,
        'test_size':           len(predictions),
        'test_exact_match':    round(float(overall_exact), 4),
        'test_overlap_f1':     round(float(overall_f1), 4),
        'spans_not_found':     not_found,
        'cost_estimate':       round(cost_estimate, 4),
    }
    json.dump(metrics, open(output_dir / 'metrics.json', 'w'), indent=2)
    print(f"\nMetrics saved → {output_dir / 'metrics.json'}")

    # Compare against mBERT
    mbert_path = Path('models/stage2_mbert_final/metrics.json')
    if mbert_path.exists():
        mbert = json.load(open(mbert_path))
        print(f"\n── Comparison ──")
        print(f"  GPT-4o zero-shot : exact={overall_exact:.4f}  overlap={overall_f1:.4f}")
        print(f"  mBERT fine-tuned : exact={mbert['test_exact_match']:.4f}  overlap={mbert['test_overlap_f1']:.4f}")


if __name__ == '__main__':
    main()