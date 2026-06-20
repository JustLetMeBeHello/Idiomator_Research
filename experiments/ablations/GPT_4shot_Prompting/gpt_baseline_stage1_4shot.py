"""
gpt_baseline_stage1_4shot.py

GPT-4o Stage 1 — few-shot idiomaticity classification with 4 in-language exemplars.

Mirrors gpt_baseline_stage1.py (the two-stage GPT pipeline's Stage 1) but adds
4 in-language, balanced (2 idiomatic + 2 literal) few-shot examples drawn from
the training split for each query.  Exemplars are selected per-language and are
seeded for reproducibility.

Usage:
    export OPENAI_API_KEY=sk-...
    python gpt_baseline_stage1_4shot.py \
        --test_path   data/idioms_structured/Splits/test.jsonl \
        --train_path  data/idioms_structured/Splits/train.jsonl \
        --output_dir  models/gpt_stage1_4shot \
        --seed        42

    # Dry run
    python gpt_baseline_stage1_4shot.py --dry_run 10
"""

import os
import json
import time
import random
import argparse
import numpy as np
from pathlib import Path
from collections import defaultdict

from openai import OpenAI
from sklearn.metrics import f1_score
from tqdm import tqdm
from dotenv import load_dotenv
load_dotenv()

# ── Args ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--test_path',  default='data/idioms_structured/Splits/test.jsonl')
    p.add_argument('--train_path', default='data/idioms_structured/Splits/train.jsonl',
                   help='Training split used as exemplar pool.')
    p.add_argument('--output_dir', default='models/gpt_stage1_4shot')
    p.add_argument('--model',      default='gpt-4o')
    p.add_argument('--n_shot',     type=int, default=4,
                   help='Number of exemplars per query (must be even; half idiomatic, half literal).')
    p.add_argument('--seed',       type=int, default=42)
    p.add_argument('--dry_run',    type=int, default=None)
    p.add_argument('--sleep',      type=float, default=0.5)
    p.add_argument('--resume',     action='store_true')
    return p.parse_args()


# ── Prompts ───────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are a linguist specializing in idiomatic language and multi-word expressions (MWEs).

Given a sentence, classify whether it contains an idiomatic (figurative) or literal use of language.

Rules:
- Respond in EXACTLY this format, nothing else:
  Label: [idiomatic/literal]

- Label must be exactly "idiomatic" or "literal"
- Do not add any explanation or extra text
"""

def build_few_shot_messages(sentence: str, exemplars: list[dict]) -> list[dict]:
    """
    Build the messages list with alternating user/assistant turns for few-shot
    exemplars, followed by the actual query as the final user turn.

    exemplars: list of dicts with keys 'sentence' and 'idiomaticity'
    """
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for ex in exemplars:
        messages.append({
            "role": "user",
            "content": f'Classify this sentence:\n\n"{ex["sentence"]}"'
        })
        messages.append({
            "role": "assistant",
            "content": f'Label: {ex["idiomaticity"]}'
        })
    messages.append({
        "role": "user",
        "content": f'Classify this sentence:\n\n"{sentence}"'
    })
    return messages


# ── Exemplar pool ─────────────────────────────────────────────────────────────

def build_exemplar_pool(train_path: str, seed: int) -> dict[str, dict[str, list]]:
    """
    Returns {language: {'idiomatic': [...], 'literal': [...]}} from train split.
    Records are shuffled with the given seed so exemplar draws are reproducible.
    """
    rng = random.Random(seed)
    pool: dict[str, dict[str, list]] = defaultdict(lambda: {'idiomatic': [], 'literal': []})
    records = [json.loads(l) for l in open(train_path, encoding='utf-8')]
    rng.shuffle(records)
    for r in records:
        pool[r['language']][r['idiomaticity']].append(r)
    return pool


def sample_exemplars(pool: dict, language: str, n_shot: int, exclude_sentence: str) -> list[dict]:
    """
    Draw n_shot/2 idiomatic and n_shot/2 literal exemplars for the given language,
    excluding the query sentence itself (guard against train/test overlap).
    Returns a list of length n_shot in shuffled order.
    """
    half = n_shot // 2
    idiom_pool  = [r for r in pool[language]['idiomatic']  if r['sentence'] != exclude_sentence]
    literal_pool = [r for r in pool[language]['literal']   if r['sentence'] != exclude_sentence]
    selected = idiom_pool[:half] + literal_pool[:half]
    # Shuffle so the last example isn't always the same label
    random.shuffle(selected)
    return selected


# ── API call ──────────────────────────────────────────────────────────────────

LABEL2ID = {'literal': 0, 'idiomatic': 1}


def classify(client, messages: list[dict], model: str, retries: int = 3):
    """Call GPT and return the predicted label string or None on failure."""
    for attempt in range(retries):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=0,
                max_tokens=10,
            )
            raw = response.choices[0].message.content.strip()
            return parse_label(raw)
        except Exception as e:
            if attempt < retries - 1:
                print(f"\n  API error (attempt {attempt+1}): {e} — retrying in 5s...")
                time.sleep(5)
            else:
                print(f"\n  API error after {retries} attempts: {e}")
                return None


def parse_label(raw: str) -> str | None:
    for line in raw.split('\n'):
        line = line.strip()
        if line.lower().startswith('label:'):
            val = line[6:].strip().lower()
            if 'idiomatic' in val:
                return 'idiomatic'
            elif 'literal' in val:
                return 'literal'
    return None


# ── Metrics ───────────────────────────────────────────────────────────────────

def evaluate(predictions: list[dict]):
    print("\n── Classification Results (4-shot Stage 1) ──")
    print(f"{'Language':<12} {'Macro F1':<12} {'N':<6}")
    print("-" * 32)

    all_preds, all_labels = [], []
    lang_cls: dict[str, dict] = defaultdict(lambda: {'preds': [], 'labels': []})

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

    overall_f1 = f1_score(all_labels, all_preds, average='macro')
    print(f"{'Overall':<12} {overall_f1:<12.4f} {len(all_preds):<6}")
    return overall_f1, cls_lang_f1


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)

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

    # Build exemplar pool from training split
    exemplar_pool = build_exemplar_pool(args.train_path, args.seed)
    lang_counts = {lang: len(v['idiomatic']) + len(v['literal'])
                   for lang, v in exemplar_pool.items()}
    print(f"Exemplar pool sizes: {lang_counts}")

    print(f"Test set : {len(test_records)} examples")
    print(f"Model    : {args.model}")
    print(f"N-shot   : {args.n_shot} (per query, balanced idiomatic/literal)")

    # Resume support
    preds_path = output_dir / 'test_predictions.jsonl'
    completed  = {}
    if args.resume and preds_path.exists():
        for line in open(preds_path, encoding='utf-8'):
            r = json.loads(line)
            completed[r['sentence']] = r
        print(f"Resuming: {len(completed)} already completed")

    predictions    = []
    parse_failures = 0
    cost_estimate  = 0.0

    with open(preds_path, 'a' if args.resume else 'w', encoding='utf-8') as f_out:
        for record in tqdm(test_records, desc="4-shot Stage 1 GPT", unit="example"):
            sentence = record['sentence']
            language = record['language']

            if sentence in completed:
                predictions.append(completed[sentence])
                continue

            # Sample in-language exemplars
            exemplars = sample_exemplars(
                exemplar_pool, language, args.n_shot, exclude_sentence=sentence
            )

            messages   = build_few_shot_messages(sentence, exemplars)
            pred_label = classify(client, messages, args.model)

            if pred_label is None:
                parse_failures += 1
                pred_label = 'idiomatic'  # safe default

            result = {
                **record,
                'pred_label':    pred_label,
                'correct_label': bool(pred_label == record['idiomaticity']),
                'n_shot':        args.n_shot,
                'exemplar_ids':  [e.get('meaning_id', '') for e in exemplars],
            }

            predictions.append(result)
            f_out.write(json.dumps(result, ensure_ascii=False) + '\n')
            f_out.flush()

            # ~200 tokens/call with 4 exemplars
            cost_estimate += 200 * 2.50 / 1_000_000
            time.sleep(args.sleep)

    print(f"\nParse failures : {parse_failures}")
    print(f"Est. API cost  : ${cost_estimate:.3f}")

    # Evaluate
    cls_f1, cls_lang = evaluate(predictions)

    # Save metrics
    metrics = {
        'model':                   args.model,
        'approach':                'stage1_4shot',
        'n_shot':                  args.n_shot,
        'seed':                    args.seed,
        'test_size':               len(predictions),
        'test_macro_f1':           round(float(cls_f1), 4),
        'cls_per_lang':            cls_lang,
        'parse_failures':          parse_failures,
        'cost_estimate':           round(cost_estimate, 4),
    }
    json.dump(metrics, open(output_dir / 'metrics.json', 'w'), indent=2)
    print(f"\nMetrics saved → {output_dir / 'metrics.json'}")


if __name__ == '__main__':
    main()
