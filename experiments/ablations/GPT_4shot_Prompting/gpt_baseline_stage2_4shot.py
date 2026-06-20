"""
gpt_baseline_stage2_4shot.py

GPT-4o Stage 2 — few-shot MWE span extraction with 4 in-language exemplars.

Mirrors gpt_baseline_stage2.py (the two-stage GPT pipeline's Stage 2) but adds
4 in-language idiomatic exemplars (Stage 2 is idiom-span-only; all exemplars
are therefore idiomatic).  Designed to be run after Stage 1 predictions are
available; if stage1_preds_path is provided, only examples predicted as
idiomatic by Stage 1 are sent to Stage 2 (pipeline mode).  If omitted, Stage 2
runs on all examples (standalone upper-bound mode).

Usage:
    export OPENAI_API_KEY=sk-...

    # Pipeline mode (uses Stage 1 output as gate)
    python gpt_baseline_stage2_4shot.py \
        --test_path          data/idioms_structured/Splits/test.jsonl \
        --train_path         data/idioms_structured/Splits/train.jsonl \
        --stage1_preds_path  models/gpt_stage1_4shot/test_predictions.jsonl \
        --output_dir         models/gpt_stage2_4shot \
        --seed               42

    # Standalone (no Stage 1 gate — span quality upper bound)
    python gpt_baseline_stage2_4shot.py \
        --test_path   data/idioms_structured/Splits/test.jsonl \
        --train_path  data/idioms_structured/Splits/train.jsonl \
        --output_dir  models/gpt_stage2_4shot_standalone
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
    p.add_argument('--test_path',         default='data/idioms_structured/Splits/test.jsonl')
    p.add_argument('--train_path',        default='data/idioms_structured/Splits/train.jsonl',
                   help='Training split used as exemplar pool (idiomatic only).')
    p.add_argument('--stage1_preds_path', default=None,
                   help='Path to Stage 1 predictions JSONL. If set, only examples '
                        'predicted idiomatic by Stage 1 are sent to GPT (pipeline mode). '
                        'If omitted, all examples run through Stage 2 (standalone mode).')
    p.add_argument('--output_dir',        default='models/gpt_stage2_4shot')
    p.add_argument('--model',             default='gpt-4o')
    p.add_argument('--n_shot',            type=int, default=4,
                   help='Number of idiomatic exemplars to include per query.')
    p.add_argument('--seed',              type=int, default=42)
    p.add_argument('--dry_run',           type=int, default=None)
    p.add_argument('--sleep',             type=float, default=0.5)
    p.add_argument('--resume',            action='store_true')
    return p.parse_args()


# ── Prompts ───────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are a linguist specializing in idiomatic language and multi-word expressions (MWEs).

Given a sentence that contains an idiomatic expression, extract the exact multi-word expression or key phrase.

Rules:
- Respond in EXACTLY this format, nothing else:
  Span: [exact text of the expression as it appears in the sentence]

- The span must be copied exactly from the sentence — do not paraphrase or normalise
- Do not add any explanation or extra text
"""

def build_few_shot_messages(sentence: str, exemplars: list[dict]) -> list[dict]:
    """
    Build messages with few-shot turns (all idiomatic) then the query.
    Uses matched_span from training records as gold span text.
    """
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for ex in exemplars:
        span_text = ex.get('matched_span') or ex['sentence'][ex['span_start']:ex['span_end']]
        messages.append({
            "role": "user",
            "content": f'Extract the idiomatic expression from this sentence:\n\n"{ex["sentence"]}"'
        })
        messages.append({
            "role": "assistant",
            "content": f'Span: {span_text}'
        })
    messages.append({
        "role": "user",
        "content": f'Extract the idiomatic expression from this sentence:\n\n"{sentence}"'
    })
    return messages


# ── Exemplar pool ─────────────────────────────────────────────────────────────

def build_exemplar_pool(train_path: str, seed: int) -> dict[str, list]:
    """
    Returns {language: [idiomatic records]} shuffled with seed.
    Stage 2 exemplars are all idiomatic.
    """
    rng = random.Random(seed)
    pool: dict[str, list] = defaultdict(list)
    records = [json.loads(l) for l in open(train_path, encoding='utf-8')]
    rng.shuffle(records)
    for r in records:
        if r['idiomaticity'] == 'idiomatic':
            pool[r['language']].append(r)
    return pool


def sample_exemplars(pool: dict, language: str, n_shot: int, exclude_sentence: str) -> list[dict]:
    candidates = [r for r in pool[language] if r['sentence'] != exclude_sentence]
    return candidates[:n_shot]


# ── Span utilities ────────────────────────────────────────────────────────────

def parse_span(raw: str) -> str | None:
    for line in raw.split('\n'):
        line = line.strip()
        if line.lower().startswith('span:'):
            return line[5:].strip().strip('"\'')
    return None


def find_span_in_sentence(sentence: str, predicted_text: str):
    """Return (char_start, char_end) or (None, None)."""
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

    # Progressive substring fallback
    words = predicted_text.split()
    for length in range(len(words), 0, -1):
        for start in range(len(words) - length + 1):
            substr = ' '.join(words[start:start + length])
            idx = sentence.lower().find(substr.lower())
            if idx != -1:
                return idx, idx + len(substr)

    return None, None


def compute_overlap_f1(ps, pe, gs, ge) -> float:
    pred_set = set(range(ps, pe))
    gold_set  = set(range(gs, ge))
    if not pred_set or not gold_set:
        return 0.0
    overlap = len(pred_set & gold_set)
    if overlap == 0:
        return 0.0
    prec = overlap / len(pred_set)
    rec  = overlap / len(gold_set)
    return 2 * prec * rec / (prec + rec)


# ── API call ──────────────────────────────────────────────────────────────────

def extract_span(client, messages: list[dict], model: str, retries: int = 3):
    for attempt in range(retries):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=0,
                max_tokens=50,
            )
            raw = response.choices[0].message.content.strip()
            return parse_span(raw)
        except Exception as e:
            if attempt < retries - 1:
                print(f"\n  API error (attempt {attempt+1}): {e} — retrying in 5s...")
                time.sleep(5)
            else:
                print(f"\n  API error after {retries} attempts: {e}")
                return None


# ── Metrics ───────────────────────────────────────────────────────────────────

def evaluate(predictions: list[dict]):
    print("\n── Span Extraction Results (4-shot Stage 2) ──")
    print(f"{'Language':<12} {'Exact Match':<14} {'Overlap F1':<12} {'N':<6}")
    print("-" * 46)

    lang_exact: dict[str, list] = defaultdict(list)
    lang_f1:    dict[str, list] = defaultdict(list)

    for p in predictions:
        lang = p['language']
        lang_exact[lang].append(int(p['exact_match']))
        lang_f1[lang].append(float(p['overlap_f1']))

    span_lang = {}
    for lang in sorted(lang_exact.keys()):
        em = np.mean(lang_exact[lang])
        f1 = np.mean(lang_f1[lang])
        span_lang[lang] = {'exact': round(em, 4), 'overlap': round(f1, 4)}
        print(f"{lang:<12} {em:<14.4f} {f1:<12.4f} {len(lang_exact[lang]):<6}")

    overall_exact   = np.mean([v for vals in lang_exact.values() for v in vals])
    overall_overlap = np.mean([v for vals in lang_f1.values() for v in vals])
    n_total = sum(len(v) for v in lang_exact.values())
    print(f"{'Overall':<12} {overall_exact:<14.4f} {overall_overlap:<12.4f} {n_total:<6}")

    return overall_exact, overall_overlap, span_lang


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

    # Optional Stage 1 gate
    stage1_idiom_sentences: set[str] | None = None
    stage1_label_map: dict[str, str] = {}
    if args.stage1_preds_path:
        stage1_preds = [json.loads(l) for l in open(args.stage1_preds_path, encoding='utf-8')]
        stage1_idiom_sentences = {
            r['sentence'] for r in stage1_preds if r.get('pred_label') == 'idiomatic'
        }
        stage1_label_map = {r['sentence']: r['pred_label'] for r in stage1_preds}
        print(f"Stage 1 gate: {len(stage1_idiom_sentences)} / {len(stage1_preds)} "
              f"examples predicted idiomatic → sent to Stage 2")

    # Build exemplar pool
    exemplar_pool = build_exemplar_pool(args.train_path, args.seed)
    print(f"Test set : {len(test_records)} examples")
    print(f"Model    : {args.model}")
    print(f"N-shot   : {args.n_shot} (idiomatic exemplars per query)")

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
    span_not_found = 0
    cost_estimate  = 0.0

    with open(preds_path, 'a' if args.resume else 'w', encoding='utf-8') as f_out:
        for record in tqdm(test_records, desc="4-shot Stage 2 GPT", unit="example"):
            sentence = record['sentence']
            language = record['language']

            if sentence in completed:
                predictions.append(completed[sentence])
                continue

            gold_s = record['span_start']
            gold_e = record['span_end']

            # Determine label: from Stage 1 if available, else treat as idiomatic
            if stage1_label_map:
                pred_label = stage1_label_map.get(sentence, 'idiomatic')
            else:
                pred_label = 'idiomatic'

            # Stage 2 only runs on idiomatic predictions
            if stage1_idiom_sentences is not None and sentence not in stage1_idiom_sentences:
                # Literal prediction — no span extraction; assign null span
                pred_span_text = ''
                pred_s, pred_e = 0, 0
                exact   = False
                overlap = 0.0
            else:
                exemplars = sample_exemplars(exemplar_pool, language, args.n_shot, sentence)
                messages  = build_few_shot_messages(sentence, exemplars)
                span_text = extract_span(client, messages, args.model)

                if span_text is None:
                    parse_failures += 1
                    span_text = ''

                pred_span_text = span_text
                pred_s, pred_e = find_span_in_sentence(sentence, span_text)

                if pred_s is None:
                    span_not_found += 1
                    pred_s, pred_e = 0, 0

                if gold_s is None or gold_e is None:
                    exact, overlap = False, 0.0
                else:
                    exact   = bool(pred_s == gold_s and pred_e == gold_e)
                    overlap = compute_overlap_f1(pred_s, pred_e, gold_s, gold_e)

                cost_estimate += 250 * 2.50 / 1_000_000
                time.sleep(args.sleep)

            result = {
                **record,
                'pred_label':      pred_label,
                'pred_span_text':  pred_span_text,
                'pred_span_start': pred_s,
                'pred_span_end':   pred_e,
                'exact_match':     exact,
                'overlap_f1':      round(overlap, 4),
                'correct_label':   bool(pred_label == record['idiomaticity']),
                'n_shot':          args.n_shot,
            }

            predictions.append(result)
            f_out.write(json.dumps(result, ensure_ascii=False) + '\n')
            f_out.flush()

    print(f"\nParse failures : {parse_failures}")
    print(f"Spans not found: {span_not_found}")
    print(f"Est. API cost  : ${cost_estimate:.3f}")

    # Evaluate
    exact, overlap, span_lang = evaluate(predictions)

    metrics = {
        'model':             args.model,
        'approach':          'stage2_4shot',
        'n_shot':            args.n_shot,
        'seed':              args.seed,
        'pipeline_mode':     args.stage1_preds_path is not None,
        'test_size':         len(predictions),
        'test_exact_match':  round(float(exact), 4),
        'test_overlap_f1':   round(float(overlap), 4),
        'span_per_lang':     span_lang,
        'parse_failures':    parse_failures,
        'spans_not_found':   span_not_found,
        'cost_estimate':     round(cost_estimate, 4),
    }
    json.dump(metrics, open(output_dir / 'metrics.json', 'w'), indent=2)
    print(f"\nMetrics saved → {output_dir / 'metrics.json'}")


if __name__ == '__main__':
    main()
