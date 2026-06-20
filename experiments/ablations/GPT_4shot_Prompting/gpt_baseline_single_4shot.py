"""
gpt_baseline_single_4shot.py

GPT-4o single-stage few-shot baseline — classifies idiomaticity AND extracts
the MWE span in a single prompt, with 4 in-language exemplars (2 idiomatic +
2 literal) per query.

Mirrors GPT-Baseline.py (System C, zero-shot single-stage) but adds balanced
in-language few-shot context.  Results can be directly compared to System C
to measure the few-shot gain on single-stage span+classification.

Usage:
    export OPENAI_API_KEY=sk-...
    python gpt_baseline_single_4shot.py \
        --test_path  data/idioms_structured/Splits/test.jsonl \
        --train_path data/idioms_structured/Splits/train.jsonl \
        --output_dir models/gpt_single_4shot \
        --seed       42

    # Dry run
    python gpt_baseline_single_4shot.py --dry_run 10
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
    p.add_argument('--output_dir', default='models/gpt_single_4shot')
    p.add_argument('--model',      default='gpt-4o')
    p.add_argument('--n_shot',     type=int, default=4,
                   help='Exemplars per query (must be even; half idiomatic, half literal).')
    p.add_argument('--seed',       type=int, default=42)
    p.add_argument('--dry_run',    type=int, default=None)
    p.add_argument('--sleep',      type=float, default=0.5)
    p.add_argument('--resume',     action='store_true')
    return p.parse_args()


# ── Prompts ───────────────────────────────────────────────────────────────────

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


def make_user_message(sentence: str) -> str:
    return f'Analyze this sentence:\n\n"{sentence}"'


def make_assistant_message(record: dict) -> str:
    """Build the expected assistant response for a training exemplar."""
    span_text = (
        record.get('matched_span')
        or record['sentence'][record['span_start']:record['span_end']]
    )
    return f'Label: {record["idiomaticity"]}\nSpan: {span_text}'


def build_few_shot_messages(sentence: str, exemplars: list[dict]) -> list[dict]:
    """
    Interleave user/assistant turns for all exemplars, then append the query.
    Exemplars are a balanced mix of idiomatic and literal examples.
    """
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for ex in exemplars:
        messages.append({"role": "user",      "content": make_user_message(ex['sentence'])})
        messages.append({"role": "assistant", "content": make_assistant_message(ex)})
    messages.append({"role": "user", "content": make_user_message(sentence)})
    return messages


# ── Exemplar pool ─────────────────────────────────────────────────────────────

def build_exemplar_pool(train_path: str, seed: int) -> dict[str, dict[str, list]]:
    """
    Returns {language: {'idiomatic': [...], 'literal': [...]}} shuffled with seed.
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
    Draw n_shot/2 idiomatic + n_shot/2 literal exemplars for the given language.
    Excludes the query sentence itself.  Returned order is shuffled.
    """
    half     = n_shot // 2
    idiom    = [r for r in pool[language]['idiomatic'] if r['sentence'] != exclude_sentence]
    literal  = [r for r in pool[language]['literal']  if r['sentence'] != exclude_sentence]
    selected = idiom[:half] + literal[:half]
    random.shuffle(selected)
    return selected


# ── API call + parsing ────────────────────────────────────────────────────────

LABEL2ID = {'literal': 0, 'idiomatic': 1}


def classify_and_extract(client, messages: list[dict], model: str, retries: int = 3):
    """Returns (label, span_text) or (None, None) on failure."""
    for attempt in range(retries):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=0,
                max_tokens=60,
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


def parse_response(raw: str) -> tuple[str | None, str | None]:
    label, span = None, None
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


# ── Span utilities ────────────────────────────────────────────────────────────

def find_span_in_sentence(sentence: str, predicted_text: str):
    if not predicted_text:
        return None, None

    idx = sentence.find(predicted_text)
    if idx != -1:
        return idx, idx + len(predicted_text)

    idx = sentence.lower().find(predicted_text.lower())
    if idx != -1:
        return idx, idx + len(predicted_text)

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


# ── Metrics ───────────────────────────────────────────────────────────────────

def evaluate(predictions: list[dict]):
    print("\n── Classification Results (4-shot Single-Stage) ──")
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

    overall_cls_f1 = f1_score(all_labels, all_preds, average='macro')
    print(f"{'Overall':<12} {overall_cls_f1:<12.4f} {len(all_preds):<6}")

    print("\n── Span Extraction Results (4-shot Single-Stage) ──")
    print(f"{'Language':<12} {'Exact Match':<14} {'Overlap F1':<12} {'N':<6}")
    print("-" * 46)

    lang_exact: dict[str, list] = defaultdict(list)
    lang_f1:    dict[str, list] = defaultdict(list)

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

    overall_exact   = np.mean([v for vals in lang_exact.values() for v in vals])
    overall_span_f1 = np.mean([v for vals in lang_f1.values() for v in vals])
    n_total = sum(len(v) for v in lang_exact.values())
    print(f"{'Overall':<12} {overall_exact:<14.4f} {overall_span_f1:<12.4f} {n_total:<6}")

    return overall_cls_f1, overall_exact, overall_span_f1, cls_lang_f1, span_lang_f1


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

    # Build exemplar pool
    exemplar_pool = build_exemplar_pool(args.train_path, args.seed)
    lang_counts = {
        lang: len(v['idiomatic']) + len(v['literal'])
        for lang, v in exemplar_pool.items()
    }
    print(f"Exemplar pool sizes: {lang_counts}")
    print(f"Test set : {len(test_records)} examples")
    print(f"Model    : {args.model}")
    print(f"N-shot   : {args.n_shot} (2 idiomatic + 2 literal per query)")

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
        for record in tqdm(test_records, desc="4-shot Single-Stage GPT", unit="example"):
            sentence = record['sentence']
            language = record['language']

            if sentence in completed:
                predictions.append(completed[sentence])
                continue

            exemplars   = sample_exemplars(exemplar_pool, language, args.n_shot, sentence)
            messages    = build_few_shot_messages(sentence, exemplars)
            pred_label, pred_span_text = classify_and_extract(client, messages, args.model)

            if pred_label is None:
                parse_failures += 1
                pred_label = 'idiomatic'

            if pred_span_text:
                pred_s, pred_e = find_span_in_sentence(sentence, pred_span_text)
            else:
                pred_s, pred_e = None, None

            if pred_s is None:
                span_not_found += 1
                pred_s, pred_e = 0, 0

            gold_s = record['span_start']
            gold_e = record['span_end']

            if gold_s is None or gold_e is None:
                exact, overlap = False, 0.0
            else:
                exact   = bool(pred_s == gold_s and pred_e == gold_e)
                overlap = compute_overlap_f1(pred_s, pred_e, gold_s, gold_e)

            result = {
                **record,
                'pred_label':      pred_label,
                'pred_span_text':  pred_span_text or '',
                'pred_span_start': pred_s,
                'pred_span_end':   pred_e,
                'exact_match':     exact,
                'overlap_f1':      round(overlap, 4),
                'correct_label':   bool(pred_label == record['idiomaticity']),
                'n_shot':          args.n_shot,
                'exemplar_ids':    [e.get('meaning_id', '') for e in exemplars],
            }

            predictions.append(result)
            f_out.write(json.dumps(result, ensure_ascii=False) + '\n')
            f_out.flush()

            # ~300 tokens/call with 4 exemplars (label+span each)
            cost_estimate += 300 * 2.50 / 1_000_000
            time.sleep(args.sleep)

    print(f"\nParse failures : {parse_failures}")
    print(f"Spans not found: {span_not_found}")
    print(f"Est. API cost  : ${cost_estimate:.3f}")

    # Evaluate
    cls_f1, exact, span_f1, cls_lang, span_lang = evaluate(predictions)

    # Compare against zero-shot single-stage
    print("\n── 4-shot vs Zero-shot Single-Stage Comparison ──")
    print(f"  4-shot single-stage: cls_f1={cls_f1:.4f}  span_overlap={span_f1:.4f}")
    zs_path = Path('models/gpt_single_stage/metrics.json')
    if zs_path.exists():
        zs = json.load(open(zs_path))
        print(f"  Zero-shot (System C): cls_f1={zs['classification_macro_f1']:.4f}  "
              f"span_overlap={zs['span_overlap_f1']:.4f}")

    metrics = {
        'model':                    args.model,
        'approach':                 'single_stage_4shot',
        'n_shot':                   args.n_shot,
        'seed':                     args.seed,
        'test_size':                len(predictions),
        'classification_macro_f1':  round(float(cls_f1), 4),
        'span_exact_match':         round(float(exact), 4),
        'span_overlap_f1':          round(float(span_f1), 4),
        'cls_per_lang':             cls_lang,
        'span_per_lang':            span_lang,
        'parse_failures':           parse_failures,
        'spans_not_found':          span_not_found,
        'cost_estimate':            round(cost_estimate, 4),
    }
    json.dump(metrics, open(output_dir / 'metrics.json', 'w'), indent=2)
    print(f"\nMetrics saved → {output_dir / 'metrics.json'}")


if __name__ == '__main__':
    main()
