"""
gpt_baseline_stage1.py

Zero-shot GPT-4o baseline for idiomaticity classification (Stage 1).
Input  : raw sentence only (no idiom expression given)
Output : idiomatic or literal

Evaluates on the test split across English, Hindi, and Telugu.
Results are saved to a predictions JSONL and a metrics JSON,
matching the format of the mBERT Stage 1 output for direct comparison.

Usage:
    export OPENAI_API_KEY=sk-...
    python gpt_baseline_stage1.py \
        --test_path  Research_And_Training/idioms_structured/Splits/test.jsonl \
        --output_dir models/gpt_baseline_stage1

    # Dry run on first 10 examples to check prompt/output
    python gpt_baseline_stage1.py \
        --test_path  Research_And_Training/idioms_structured/Splits/test.jsonl \
        --output_dir models/gpt_baseline_stage1 \
        --dry_run 10
"""

import os
import json
import time
import argparse
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
    p.add_argument('--test_path',   default='data/idioms_structured/Splits/test.jsonl')
    p.add_argument('--output_dir',  default='models/gpt_baseline_stage1')
    p.add_argument('--model',       default='gpt-4o')
    p.add_argument('--dry_run',     type=int, default=None,
                   help='Only run on first N examples (for testing)')
    p.add_argument('--sleep',       type=float, default=0.5,
                   help='Seconds to sleep between API calls (avoid rate limits)')
    p.add_argument('--resume',      action='store_true',
                   help='Resume from existing predictions file if interrupted')
    return p.parse_args()


# ── Prompt ────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are a linguist specializing in figurative language.
Your task is to classify whether a sentence contains an idiomatic (figurative) expression or uses language literally.

Rules:
- Answer with exactly one word: either "idiomatic" or "literal"
- "idiomatic" means the sentence contains a phrase whose meaning cannot be derived from the individual words (e.g. "kick the bucket" meaning to die)
- "literal" means all words are used with their standard, dictionary meanings
- Do not explain your answer
- Do not output anything other than "idiomatic" or "literal"
"""

def make_prompt(sentence):
    return f'Classify this sentence:\n\n"{sentence}"'



# ── API call ──────────────────────────────────────────────────────────────────

def classify(client, sentence, model, retries=3):
    for attempt in range(retries):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user",   "content": make_prompt(sentence)},
                ],
                temperature=0,
                max_tokens=5,
            )
            raw = response.choices[0].message.content.strip().lower()

            # Normalize output
            if 'idiomatic' in raw:
                return 'idiomatic', raw
            elif 'literal' in raw:
                return 'literal', raw
            else:
                # Unexpected output — default to idiomatic, flag it
                return 'idiomatic', f'UNEXPECTED: {raw}'

        except Exception as e:
            if attempt < retries - 1:
                print(f"\n  API error (attempt {attempt+1}): {e} — retrying in 5s...")
                time.sleep(5)
            else:
                print(f"\n  API error after {retries} attempts: {e}")
                return 'idiomatic', f'ERROR: {e}'


# ── Eval ──────────────────────────────────────────────────────────────────────

LABEL2ID = {'literal': 0, 'idiomatic': 1}

def evaluate(predictions):
    all_preds  = [LABEL2ID[p['pred_idiomaticity']] for p in predictions]
    all_labels = [LABEL2ID[p['idiomaticity']]      for p in predictions]

    report   = classification_report(all_labels, all_preds,
                                      target_names=['literal', 'idiomatic'], digits=4)
    macro_f1 = f1_score(all_labels, all_preds, average='macro')

    print("\n── Test results ──")
    print(report)

    # Per-language breakdown
    lang_f1 = {}
    for lang in sorted(set(p['language'] for p in predictions)):
        idxs     = [i for i, p in enumerate(predictions) if p['language'] == lang]
        l_preds  = [all_preds[i]  for i in idxs]
        l_labels = [all_labels[i] for i in idxs]
        lf1      = f1_score(l_labels, l_preds, average='macro')
        lang_f1[lang] = round(lf1, 4)
        print(f"  {lang}: macro F1 = {lf1:.4f} ({len(idxs)} examples)")

    return macro_f1, lang_f1


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

    print(f"Test set: {len(test_records)} examples")
    print(f"Model: {args.model}")

    # Resume support — load already-completed predictions
    preds_path  = output_dir / 'test_predictions.jsonl'
    completed   = {}
    if args.resume and preds_path.exists():
        for line in open(preds_path, encoding='utf-8'):
            r = json.loads(line)
            completed[r['sentence']] = r
        print(f"Resuming: {len(completed)} already completed")

    # Run classification
    predictions = []
    cost_estimate = 0.0

    with open(preds_path, 'a' if args.resume else 'w', encoding='utf-8') as f_out:
        for record in tqdm(test_records, desc="Classifying", unit="example"):
            sentence = record['sentence']

            # Skip if already done
            if sentence in completed:
                predictions.append(completed[sentence])
                continue

            pred_label, raw_output = classify(client, sentence, args.model)

            result = {
                **record,
                'pred_idiomaticity': pred_label,
                'raw_gpt_output':    raw_output,
                'correct':           bool(pred_label == record['idiomaticity']),
            }

            predictions.append(result)
            f_out.write(json.dumps(result, ensure_ascii=False) + '\n')
            f_out.flush()

            # Rough cost estimate (gpt-4o: ~$2.50/1M input tokens, ~50 tokens/call)
            cost_estimate += 50 * 2.50 / 1_000_000

            time.sleep(args.sleep)

    print(f"\nEstimated API cost: ${cost_estimate:.3f}")

    # Evaluate
    macro_f1, lang_f1 = evaluate(predictions)

    # Check for unexpected outputs
    unexpected = [p for p in predictions if p['raw_gpt_output'].startswith('UNEXPECTED')]
    if unexpected:
        print(f"\n⚠ {len(unexpected)} unexpected GPT outputs:")
        for p in unexpected[:5]:
            print(f"  '{p['sentence'][:60]}' → {p['raw_gpt_output']}")

    # Save metrics
    metrics = {
        'model':         args.model,
        'test_size':     len(predictions),
        'test_macro_f1': round(macro_f1, 4),
        'lang_f1':       lang_f1,
        'cost_estimate': round(cost_estimate, 4),
        'unexpected_outputs': len(unexpected),
    }
    json.dump(metrics, open(output_dir / 'metrics.json', 'w'), indent=2)
    print(f"\nMetrics saved → {output_dir / 'metrics.json'}")
    print(f"Predictions saved → {preds_path}")

    # Print comparison reminder
    print("\n── Compare against mBERT ──")
    print(f"  GPT-4o zero-shot : {macro_f1:.4f} overall")
    mbert_metrics_path = Path('models/stage1_mbert_en_hi_te/metrics.json')
    if mbert_metrics_path.exists():
        mbert = json.load(open(mbert_metrics_path))
        print(f"  mBERT fine-tuned : {mbert['test_macro_f1']:.4f} overall")


if __name__ == '__main__':
    main()