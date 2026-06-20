"""
prepare_dataset.py

Samples and splits data for mBERT idiomaticity fine-tuning.

Sampling strategy:
  - Hindi / Telugu : 318 per class (constrained by Telugu idiomatic minority)
  - English / Spanish : 1272 per class (2x Hindi+Telugu combined to prevent gradient dominance)

Split strategy:
  - 80/10/10 unseen idiom split (test idioms never seen in train)
  - Split is done per language at the idiom_id level BEFORE sampling to prevent leakage
  - Weighted loss weights written to a separate JSON file

Output files:
  - train.jsonl
  - dev.jsonl
  - test.jsonl
  - split_stats.json
  - loss_weights.json
"""

import json
import random
from collections import defaultdict
from pathlib import Path

# ── Config ────────────────────────────────────────────────────────────────────

random.seed(42)

DATA_DIR = Path('idioms_structured/Span_tagged_data')
OUT_DIR  = Path('data/idioms_structured/Splits')
OUT_DIR.mkdir(parents=True, exist_ok=True)

FILES = {
    'English': DATA_DIR / 'English/Final_English_MERGED_normalized.jsonl',
    'Spanish': DATA_DIR / 'Spanish/Final_Spanish_MERGED.jsonl',
    'Hindi':   DATA_DIR / 'Hindi/Final_Hindi_MERGED.jsonl',
    'Telugu':  DATA_DIR / 'Telugu/Final_Telugu_MERGED.jsonl',
}

N_PER_CLASS = {
    'Hindi':   318,
    'Telugu':  318,
    'English': 1272,   
    'Spanish': 1272,   
}

SPLIT_RATIOS = (0.80, 0.10, 0.10)  # train / dev / test

# ── Helpers ───────────────────────────────────────────────────────────────────

def load_and_flatten(path, lang):
    """Load a JSONL file and flatten to one dict per example."""
    records = [json.loads(l) for l in open(path, encoding='utf-8')]
    examples = []
    for r in records:
        for ex in r['examples']:
            if ex.get('span_flagged'):
                continue
            examples.append({
                'language':     lang,
                'idiom_id':     r['idiom_id'],
                'idiom':        r['idiom'],
                'meaning_id':   r['meaning_id'],
                'sense_number': r['sense_number'],
                'idiomaticity': r['Idiomaticity'],
                'register':     r['Register'],
                'region':       r['Region'],
                'sentence':     ex['sentence'],
                'span_start':   ex['span_start'],
                'span_end':     ex['span_end'],
                'matched_span': ex['matched_span'],
            })
    return examples


def sample_split_balanced(examples, target_per_class):
    """
    Safely samples up to target_per_class from a segregated split pool.
    If a split doesn't have enough examples due to strict idiom segregation,
    it gracefully falls back to using all available tokens in that split class.
    """
    by_class = defaultdict(list)
    for ex in examples:
        by_class[ex['idiomaticity']].append(ex)

    sampled = []
    for cls in ['idiomatic', 'literal']:
        available = by_class[cls]
        pull_size = min(target_per_class, len(available))
        if pull_size < target_per_class:
            print(f"    * Warning: Requested {target_per_class} {cls} examples, but split pool only had {len(available)}. Using all available.")
        sampled.extend(random.sample(available, pull_size))
    return sampled


def write_jsonl(path, records):
    """Writes a list of dictionaries to a JSONL file."""
    with open(path, 'w', encoding='utf-8') as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')


# ── Main ──────────────────────────────────────────────────────────────────────

all_splits = {'train': [], 'dev': [], 'test': []}
stats = {}

for lang, path in FILES.items():
    print(f"\n{'─'*50}")
    print(f"Processing {lang}...")

    # 1. Load data
    raw_examples = load_and_flatten(path, lang)
    
    # 2. Group raw data by unique idiom_id BEFORE any sampling happens
    by_idiom = defaultdict(list)
    for ex in raw_examples:
        by_idiom[ex['idiom_id']].append(ex)

    idiom_ids = list(by_idiom.keys())
    random.shuffle(idiom_ids)

    # 3. Cleanly slice the idiom types according to ratios
    n = len(idiom_ids)
    train_end = int(n * SPLIT_RATIOS[0])
    dev_end   = int(n * (SPLIT_RATIOS[0] + SPLIT_RATIOS[1]))

    train_ids = set(idiom_ids[:train_end])
    dev_ids   = set(idiom_ids[train_end:dev_end])
    test_ids  = set(idiom_ids[dev_end:])

    # 4. Construct un-leaked split pools of examples
    raw_train = [ex for iid in train_ids for ex in by_idiom[iid]]
    raw_dev   = [ex for iid in dev_ids   for ex in by_idiom[iid]]
    raw_test  = [ex for iid in test_ids  for ex in by_idiom[iid]]

    # 5. Extract balanced samples independently inside each isolated pool
    train = sample_split_balanced(raw_train, int(N_PER_CLASS[lang] * SPLIT_RATIOS[0]))
    dev   = sample_split_balanced(raw_dev,   int(N_PER_CLASS[lang] * SPLIT_RATIOS[1]))
    test  = sample_split_balanced(raw_test,  int(N_PER_CLASS[lang] * SPLIT_RATIOS[2]))

    for split_name, split_data in [('train', train), ('dev', dev), ('test', test)]:
        all_splits[split_name].extend(split_data)

    idiomatic = lambda exs: sum(1 for e in exs if e['idiomaticity'] == 'idiomatic')
    literal   = lambda exs: sum(1 for e in exs if e['idiomaticity'] == 'literal')

    n_sampled_total = len(train) + len(dev) + len(test)
    stats[lang] = {
        'sampled_total':  n_sampled_total,
        'unique_idioms':  {'train': len(train_ids), 'dev': len(dev_ids), 'test': len(test_ids)},
        'train':          {'total': len(train), 'idiomatic': idiomatic(train), 'literal': literal(train)},
        'dev':            {'total': len(dev),   'idiomatic': idiomatic(dev),   'literal': literal(dev)},
        'test':           {'total': len(test),  'idiomatic': idiomatic(test),  'literal': literal(test)},
    }

    print(f"  Total Cleanly Sampled : {n_sampled_total} across splits")
    print(f"  Train   : {len(train)} examples, {len(train_ids)} unique idioms")
    print(f"  Dev     : {len(dev)} examples, {len(dev_ids)} unique idioms")
    print(f"  Test    : {len(test)} examples, {len(test_ids)} unique idioms")

# Shuffle and write splits
for split_name, data in all_splits.items():
    random.shuffle(data)
    out_path = OUT_DIR / f'{split_name}.jsonl'
    write_jsonl(out_path, data)
    print(f"\nWrote {len(data)} total multilingual examples → {out_path}")

# Write stats
stats_path = OUT_DIR / 'split_stats.json'
with open(stats_path, 'w') as f:
    json.dump(stats, f, indent=2)
print(f"Wrote execution stats → {stats_path}")

# ── Loss weights ──────────────────────────────────────────────────────────────
train_lang_counts = defaultdict(int)
for ex in all_splits['train']:
    train_lang_counts[ex['language']] += 1

total_train = sum(train_lang_counts.values())
loss_weights = {
    lang: round(total_train / (len(train_lang_counts) * count), 4)
    for lang, count in train_lang_counts.items()
}

weights_path = OUT_DIR / 'loss_weights.json'
with open(weights_path, 'w') as f:
    json.dump(loss_weights, f, indent=2)

print(f"\nLoss weights (inverse frequency matrix): {loss_weights}")
print(f"Wrote adjusted loss weights → {weights_path}")