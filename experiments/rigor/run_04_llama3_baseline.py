"""
run_04_llama3_baseline.py

Llama-3.3-70B-Instruct baseline — second open-weights LLM evaluation to
complement the GPT-4o baselines (Systems B/C in the main paper).

Why this experiment exists
--------------------------
Council deliberation (2026-05-25) identified single-LLM comparison
(GPT-4o only) as an auto-reject signal at Main venues in 2026. Adding
Llama-3.3-70B closes that gap and lets the paper make defensible
"in-domain fine-tuning beats frontier LLMs" claims rather than
"in-domain fine-tuning beats one OpenAI model."

What it does
------------
Single-stage prompt — same SYSTEM_PROMPT and parse logic as
experiments/ablations/GPT-Baseline.py, only the API client is swapped. Output format
matches GPT-Baseline.py exactly so this slots into
Evaluation/Full_evaluation.py without any post-processing changes.

Supported providers (OpenAI-compatible APIs)
--------------------------------------------
  - Groq              (DEFAULT — free tier 14.4k RPD, LPU-quantized; preprint-grade)
  - DeepInfra         (BF16 full-precision; paper-grade — switch to this before ARR)
  - Together AI       (FP8 Turbo only at serverless tier; see "Precision note")
  - OpenRouter        (NOT recommended for paper baselines — routes to varying backends)
  - Local vLLM server (set --base_url http://localhost:8000/v1)

Dual-track precision strategy
-----------------------------
The Llama-3.3-70B baseline is on a two-stage cutover:

  • **Now → arXiv preprint**: `--provider groq` (free tier, LPU-quantized).
    Paper carries the footnote *"Llama-3.3-70B served via Groq's LPU
    inference (provider-quantized; AWQ-equivalent per Groq's published
    evals)."* arXiv reviewers don't ding precision — this is fine.

  • **Before ARR Aug 2026 submission**: switch to `--provider deepinfra`
    (BF16 full-precision; $5 minimum credit purchase, ~$0.25 per full
    test-set run). Paper updates to: *"Llama-3.3-70B-Instruct (BF16),
    served via DeepInfra's serverless endpoint, temperature=0,
    max_tokens=80."* Re-run experiment 04, replace the metrics, swap
    the footnote. ~10 minutes of work.

Together AI's serverless 70B inventory is FP8-quantized "Turbo" only —
keep it as a fallback but never as the paper-reported artifact (the FP8
confound is reviewer-flaggable). OpenRouter routes to whichever backend
has capacity; reproducibility is provider-dependent between runs.

Usage
-----
Preprint run (free, ~60 min on Groq's 30-RPM free tier):

    export GROQ_API_KEY=...
    python experiments/rigor/run_04_llama3_baseline.py \\
        --test_path data/idioms_structured/Splits/test.jsonl \\
        --output_dir models/rigor_llama3_single

Resume (auto — just re-run the same command; skips completed meaning_ids):

    python experiments/rigor/run_04_llama3_baseline.py  # same args

Force restart from scratch (discards existing predictions):

    python experiments/rigor/run_04_llama3_baseline.py --force

Pre-ARR re-run (~$0.25 on DeepInfra BF16, ~5-10 min):

    export DEEPINFRA_API_KEY=...
    python experiments/rigor/run_04_llama3_baseline.py \\
        --provider deepinfra \\
        --output_dir models/rigor_llama3_single_bf16

Dry run smoke test (10 examples, ~10 sec, free):

    python experiments/rigor/run_04_llama3_baseline.py --dry_run 10

Fallback (NOT paper-grade — for sanity checks only):

    export TOGETHER_API_KEY=...
    python ... --provider together   # FP8 Turbo
"""

import os
import json
import time
import argparse
import numpy as np
from pathlib import Path
from collections import defaultdict

from openai import OpenAI
from sklearn.metrics import f1_score
from tqdm import tqdm
from dotenv import load_dotenv
load_dotenv()


# ── Provider registry ────────────────────────────────────────────────────────

PROVIDERS = {
    # ── Preprint-grade default: Groq free tier ───────────────────────────────
    'groq': {
        'base_url': 'https://api.groq.com/openai/v1',
        'env_key':  'GROQ_API_KEY',
        # LPU custom quantization (Groq publishes AWQ-equivalent eval
        # parity). Free tier: 30 RPM, 6000 TPM, 14.4k requests/day —
        # comfortably enough for the 960-example test set in ~60 min.
        # Reportable for the preprint with a precision footnote.
        # Before ARR Aug 2026: re-run on --provider deepinfra and
        # update the metric + footnote (see docstring).
        'default_model': 'llama-3.3-70b-versatile',
    },
    # ── Paper-grade (ARR cutover): BF16 full-precision ───────────────────────
    'deepinfra': {
        'base_url': 'https://api.deepinfra.com/v1/openai',
        'env_key':  'DEEPINFRA_API_KEY',
        # BF16 full-precision — official Meta numerics. Switch the
        # default to this before the ARR submission and re-run exp 04
        # against this provider so the reported metric is BF16-grade.
        'default_model': 'meta-llama/Llama-3.3-70B-Instruct',
    },
    # ── Fallbacks (NOT paper-grade — see "Precision note" in docstring) ──────
    'together': {
        'base_url': 'https://api.together.xyz/v1',
        'env_key':  'TOGETHER_API_KEY',
        # WARNING: -Turbo is FP8-quantized. Together's serverless 70B
        # inventory is Turbo-only as of 2026-05. Use only for smoke tests
        # or non-reported sanity checks. Reviewers can (correctly) flag
        # FP8 as a confound for the "frontier open-weights LLM" framing.
        'default_model': 'meta-llama/Llama-3.3-70B-Instruct-Turbo',
    },
    'openrouter': {
        'base_url': 'https://openrouter.ai/api/v1',
        'env_key':  'OPENROUTER_API_KEY',
        # Routes to whichever backend has capacity. Reproducibility is
        # provider-dependent between runs. NEVER use for the paper artifact.
        'default_model': 'meta-llama/llama-3.3-70b-instruct',
    },
    'vllm_local': {
        'base_url': 'http://localhost:8000/v1',
        'env_key':  None,
        'default_model': 'meta-llama/Llama-3.3-70B-Instruct',
    },
}


# ── Args ─────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--test_path',  default='data/idioms_structured/Splits/test.jsonl')
    p.add_argument('--output_dir', default='models/rigor_llama3_single')
    p.add_argument('--provider',   default='groq', choices=list(PROVIDERS.keys()))
    p.add_argument('--model',      default=None,
                   help='Override provider default model.')
    p.add_argument('--base_url',   default=None,
                   help='Override provider default base_url (use for self-hosted vLLM, etc.).')
    p.add_argument('--api_key',    default=None,
                   help='Override env var lookup. Avoid in shared envs.')
    p.add_argument('--dry_run',    type=int, default=None,
                   help='Process only N examples for smoke-testing.')
    p.add_argument('--sleep',      type=float, default=2.0,
                   help='Inter-call delay (s). Groq free tier is 30 RPM → use 2.0 (default). '
                        'DeepInfra/Together tolerate 0.0; lower this when switching providers.')
    p.add_argument('--force',      action='store_true',
                   help='Overwrite existing predictions instead of resuming.')
    p.add_argument('--cost_per_million_tokens', type=float, default=0.0,
                   help='$/M tokens for cost-estimate line at the end of the run. Default 0.0 '
                        '(Groq free tier). Override for other providers: DeepInfra BF16 ~$0.65, '
                        'Groq paid ~$0.69, Together-Turbo ~$0.88, OpenRouter varies.')
    return p.parse_args()


# ── Prompt — identical to experiments/ablations/GPT-Baseline.py for apples-to-apples ─────

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


# ── API call ─────────────────────────────────────────────────────────────────

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
                max_tokens=80,
            )
            raw = response.choices[0].message.content.strip()
            return parse_response(raw)
        except Exception as e:
            if attempt < retries - 1:
                wait = 5 * (attempt + 1)
                print(f"\n  API error (attempt {attempt+1}): {e} — retrying in {wait}s...")
                time.sleep(wait)
            else:
                print(f"\n  API error after {retries} attempts: {e}")
                return None, None


def parse_response(raw):
    """Parse LLM response into (label, span_text). Matches GPT-Baseline.py."""
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


# ── Span matching — identical to GPT-Baseline.py ─────────────────────────────

def find_span_in_sentence(sentence, predicted_text):
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


# ── Metrics ──────────────────────────────────────────────────────────────────

LABEL2ID = {'literal': 0, 'idiomatic': 1}


def compute_overlap_f1(pred_start, pred_end, gold_start, gold_end):
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

    all_preds, all_labels = [], []
    lang_cls = defaultdict(lambda: {'preds': [], 'labels': []})

    for p in predictions:
        if p['pred_label'] is None:
            continue
        lang  = p['language']
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

    overall_exact   = np.mean([v for vals in lang_exact.values() for v in vals])
    overall_span_f1 = np.mean([v for vals in lang_f1.values() for v in vals])
    print(f"{'Overall':<12} {overall_exact:<14.4f} {overall_span_f1:<12.4f} "
          f"{sum(len(v) for v in lang_exact.values()):<6}")
    return overall_cls_f1, overall_exact, overall_span_f1, cls_lang_f1, span_lang_f1


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    provider = PROVIDERS[args.provider]
    base_url = args.base_url or provider['base_url']
    model    = args.model    or provider['default_model']

    if args.api_key:
        api_key = args.api_key
    elif provider['env_key']:
        api_key = os.getenv(provider['env_key'])
        if not api_key:
            raise ValueError(
                f"{provider['env_key']} not set. Run: export {provider['env_key']}=...")
    else:
        # vllm_local — no auth needed
        api_key = 'EMPTY'

    client     = OpenAI(api_key=api_key, base_url=base_url)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    test_records = [json.loads(l) for l in open(args.test_path, encoding='utf-8')]
    if args.dry_run:
        test_records = test_records[:args.dry_run]
        print(f"Dry run: {args.dry_run} examples")

    print(f"Provider : {args.provider}")
    print(f"Base URL : {base_url}")
    print(f"Model    : {model}")
    print(f"Test set : {len(test_records)} examples")

    preds_path = output_dir / 'test_predictions.jsonl'
    completed  = {}
    if not args.force and preds_path.exists():
        for line in open(preds_path, encoding='utf-8'):
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            completed[r['meaning_id']] = r
        if completed:
            print(f"Resuming: {len(completed)} already completed (use --force to restart)")

    predictions   = []
    cost_estimate = 0.0
    parse_failures = 0
    span_not_found = 0

    with open(preds_path, 'a' if (completed and not args.force) else 'w', encoding='utf-8') as f_out:
        for record in tqdm(test_records, desc=f"{args.provider}/{model.split('/')[-1]}", unit="ex"):
            sentence = record['sentence']
            if record['meaning_id'] in completed:
                predictions.append(completed[record['meaning_id']])
                continue

            pred_label, pred_span_text = classify_and_extract(client, sentence, model)

            if pred_label is None:
                parse_failures += 1
                pred_label = 'idiomatic'

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

            # ~80 tokens/call worst-case
            cost_estimate += 80 * args.cost_per_million_tokens / 1_000_000
            time.sleep(args.sleep)

    print(f"\nParse failures : {parse_failures}")
    print(f"Spans not found: {span_not_found}")
    print(f"Est. API cost  : ${cost_estimate:.3f}")

    cls_f1, exact, span_f1, cls_lang, span_lang = evaluate(predictions)

    metrics = {
        'provider':                args.provider,
        'model':                   model,
        'base_url':                base_url,
        'approach':                'single_stage',
        'test_size':               len(predictions),
        'classification_macro_f1': round(float(cls_f1), 4),
        'span_exact_match':        round(float(exact), 4),
        'span_overlap_f1':         round(float(span_f1), 4),
        'cls_per_lang':            cls_lang,
        'span_per_lang':           span_lang,
        'parse_failures':          parse_failures,
        'spans_not_found':         span_not_found,
        'cost_estimate':           round(cost_estimate, 4),
    }
    json.dump(metrics, open(output_dir / 'metrics.json', 'w'), indent=2)
    print(f"\nMetrics saved → {output_dir / 'metrics.json'}")

    print("\n── Llama-3.3-70B vs GPT-4o (single-stage) comparison ──")
    print(f"  Llama-3.3-70B    : cls_f1={cls_f1:.4f}  span_overlap={span_f1:.4f}")
    gpt_path = Path('models/gpt_single_stage/metrics.json')
    if gpt_path.exists():
        gpt = json.load(open(gpt_path))
        print(f"  GPT-4o single    : cls_f1={gpt['classification_macro_f1']:.4f}  "
              f"span_overlap={gpt['span_overlap_f1']:.4f}")


if __name__ == '__main__':
    main()
