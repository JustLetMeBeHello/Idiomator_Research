"""
prepare_dataset.py

Samples and splits data for mBERT idiomaticity fine-tuning.

Sampling strategy:
  - Hindi / Telugu : 318 per class (constrained by Telugu idiomatic minority)
  - English        : 1272 per class (2x Hindi+Telugu combined)

Split strategy:
  - 80/10/10 unseen idiom split (test idioms never seen in train)
  - Split is done per language, stratified at idiom_id level
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

DATA_DIR = Path('Research_And_Training/idioms_structured/Span_tagged_data')
OUT_DIR  = Path('Research_And_Training/idioms_structured/Splits')
OUT_DIR.mkdir(parents=True, exist_ok=True)

FILES = {
    'English': DATA_DIR / 'English/Final_English_MERGED_normalized.jsonl',
    'Hindi':   DATA_DIR / 'Hindi/Final_Hindi_MERGED.jsonl',
    'Telugu':  DATA_DIR / 'Telugu/Final_Telugu_MERGED.jsonl',
}

N_PER_CLASS = {
    'Hindi':   318,
    'Telugu':  318,
    'English': 1272,   # 2x (Hindi + Telugu) per class
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


def sample_balanced(examples, n_per_class):
    """Sample n_per_class examples from each idiomaticity class."""
    by_class = defaultdict(list)
    for ex in examples:
        by_class[ex['idiomaticity']].append(ex)

    sampled = []
    for cls in ['idiomatic', 'literal']:
        available = by_class[cls]
        assert len(available) >= n_per_class, (
            f"Not enough {cls} examples: need {n_per_class}, have {len(available)}"
        )
        sampled.extend(random.sample(available, n_per_class))
    return sampled


def unseen_split(examples, ratios):
    """
    Split by unique idiom_id so test idioms are never seen in train.
    Returns (train, dev, test) lists.
    """
    # Group examples by idiom_id
    by_idiom = defaultdict(list)
    for ex in examples:
        by_idiom[ex['idiom_id']].append(ex)

    idiom_ids = list(by_idiom.keys())
    random.shuffle(idiom_ids)

    n = len(idiom_ids)
    train_end = int(n * ratios[0])
    dev_end   = int(n * (ratios[0] + ratios[1]))

    train_ids = set(idiom_ids[:train_end])
    dev_ids   = set(idiom_ids[train_end:dev_end])
    test_ids  = set(idiom_ids[dev_end:])

    train = [ex for iid in train_ids for ex in by_idiom[iid]]
    dev   = [ex for iid in dev_ids   for ex in by_idiom[iid]]
    test  = [ex for iid in test_ids  for ex in by_idiom[iid]]

    return train, dev, test, len(train_ids), len(dev_ids), len(test_ids)


def write_jsonl(path, records):
    with open(path, 'w', encoding='utf-8') as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')


# ── Main ──────────────────────────────────────────────────────────────────────

all_splits = {'train': [], 'dev': [], 'test': []}
stats = {}

for lang, path in FILES.items():
    print(f"\n{'─'*50}")
    print(f"Processing {lang}...")

    examples  = load_and_flatten(path, lang)
    sampled   = sample_balanced(examples, N_PER_CLASS[lang])
    train, dev, test, n_tr, n_dv, n_te = unseen_split(sampled, SPLIT_RATIOS)

    for split_name, split_data in [('train', train), ('dev', dev), ('test', test)]:
        all_splits[split_name].extend(split_data)

    idiomatic = lambda exs: sum(1 for e in exs if e['idiomaticity'] == 'idiomatic')
    literal   = lambda exs: sum(1 for e in exs if e['idiomaticity'] == 'literal')

    stats[lang] = {
        'sampled':        len(sampled),
        'unique_idioms':  {'train': n_tr, 'dev': n_dv, 'test': n_te},
        'train':          {'total': len(train), 'idiomatic': idiomatic(train), 'literal': literal(train)},
        'dev':            {'total': len(dev),   'idiomatic': idiomatic(dev),   'literal': literal(dev)},
        'test':           {'total': len(test),  'idiomatic': idiomatic(test),  'literal': literal(test)},
    }

    print(f"  Sampled : {len(sampled)} ({N_PER_CLASS[lang]} per class)")
    print(f"  Train   : {len(train)} examples, {n_tr} unique idioms")
    print(f"  Dev     : {len(dev)} examples, {n_dv} unique idioms")
    print(f"  Test    : {len(test)} examples, {n_te} unique idioms")

# Shuffle and write splits
for split_name, data in all_splits.items():
    random.shuffle(data)
    out_path = OUT_DIR / f'{split_name}.jsonl'
    write_jsonl(out_path, data)
    print(f"\nWrote {len(data)} examples → {out_path}")

# Write stats
stats_path = OUT_DIR / 'split_stats.json'
with open(stats_path, 'w') as f:
    json.dump(stats, f, indent=2)
print(f"Wrote stats → {stats_path}")

# ── Loss weights ──────────────────────────────────────────────────────────────
# Inverse of proportion of each language in training set
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

print(f"\nLoss weights (inverse frequency): {loss_weights}")
print(f"Wrote weights → {weights_path}")