"""
id10m_to_jsonl.py

Converts ID10M BIO TSV files → JSONL format compatible with
Ablations/BiO_Task_mBERT_train.py.

Output format per line:
    {
        "sentence":     "<reconstructed text>",
        "language":     "English",          # full name, matching BiO_Task langs
        "idiomaticity": "idiomatic|literal",
        "span_start":   <int|null>,         # char offset in reconstructed sentence
        "span_end":     <int|null>,
        "idiom":        "<span text>",
        "id10m_id":     "EN_42"
    }

Usage:
    python Additional_Rigor_Experiments/id10m_to_jsonl.py \\
        --data_dir /path/to/ID10M \\
        --output_dir /path/to/id10m_jsonl \\
        --langs EN ES \\
        --splits train dev test

Output files:
    id10m_jsonl/train.jsonl
    id10m_jsonl/dev.jsonl    (if exists)
    id10m_jsonl/test.jsonl
"""

import json
import argparse
from pathlib import Path

PUNCT = set('.,;:!?)]\'\"…—–')

LANG_MAP = {
    'EN': ('English',     'english'),
    'ES': ('Spanish',     'spanish'),
    'DE': ('German',      'german'),
    'FR': ('French',      'french'),
    'IT': ('Italian',     'italian'),
    'PT': ('Portuguese',  'portuguese'),
    'ZH': ('Chinese',     'chinese'),
    'JA': ('Japanese',    'japanese'),
    'NL': ('Dutch',       'dutch'),
    'PL': ('Polish',      'polish'),
}


def reconstruct_sentence(tokens):
    """Join tokens with space, no space before punctuation."""
    s = ''
    tok_starts = []
    for tok in tokens:
        if s and tok not in PUNCT and not tok.startswith("'"):
            s += ' '
        tok_starts.append(len(s))
        s += tok
    return s, tok_starts


def parse_bio_tsv(tsv_path, lang_label, split_name):
    """Parse ID10M BIO TSV → list of dicts in BiO_Task_mBERT_train format."""
    examples = []
    tokens, tags = [], []
    sent_id = 0

    def flush(tokens, tags, sent_id):
        if not tokens:
            return None

        sentence, tok_starts = reconstruct_sentence(tokens)
        tok_ends = [tok_starts[i] + len(tokens[i]) for i in range(len(tokens))]

        # Find first idiom span (B-IDIOM..I-IDIOM run)
        span_s_tok = None
        span_e_tok = None
        in_span    = False
        for i, tag in enumerate(tags):
            if tag == 'B-IDIOM':
                if not in_span:
                    span_s_tok = i
                    span_e_tok = i
                    in_span = True
                else:
                    # New B after previous B — take first span only
                    break
            elif tag == 'I-IDIOM' and in_span:
                span_e_tok = i
            elif in_span:
                break  # span ended

        has_idiom = span_s_tok is not None
        label     = 'idiomatic' if has_idiom else 'literal'

        span_start = tok_starts[span_s_tok]            if has_idiom else None
        span_end   = tok_ends[span_e_tok]              if has_idiom else None
        idiom      = sentence[span_start:span_end]     if has_idiom else ''

        return {
            'sentence':     sentence,
            'language':     lang_label,
            'idiomaticity': label,
            'span_start':   span_start,
            'span_end':     span_end,
            'idiom':        idiom,
            'id10m_id':     f'{lang_label[:2].upper()}_{sent_id}',
        }

    with open(tsv_path, encoding='utf-8') as f:
        for line in f:
            line = line.rstrip('\n')
            if not line.strip():
                ex = flush(tokens, tags, sent_id)
                if ex is not None:
                    examples.append(ex)
                    sent_id += 1
                tokens, tags = [], []
            else:
                parts = line.split('\t')
                if len(parts) >= 2:
                    tokens.append(parts[0])
                    tags.append(parts[1].strip())

    ex = flush(tokens, tags, sent_id)
    if ex is not None:
        examples.append(ex)

    dist = {k: sum(1 for e in examples if e['idiomaticity'] == k)
            for k in ('idiomatic', 'literal')}
    print(f'  [{lang_label} / {split_name}] {len(examples)} sentences | {dist}')
    return examples


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--data_dir',   required=True,
                   help='Root of cloned Babelscape/ID10M repo')
    p.add_argument('--output_dir', required=True,
                   help='Where to write train/dev/test.jsonl')
    p.add_argument('--langs',      nargs='+', default=['EN', 'ES'],
                   help='Language codes to include')
    p.add_argument('--splits',     nargs='+', default=['train', 'dev', 'test'])
    return p.parse_args()


def main():
    args    = parse_args()
    data_dir = Path(args.data_dir)
    out_dir  = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Collect per split across all langs
    split_examples: dict[str, list[dict]] = {s: [] for s in args.splits}

    for code in args.langs:
        if code not in LANG_MAP:
            print(f'  [{code}] Unknown lang code — skip')
            continue
        lang_label, dirname = LANG_MAP[code]

        for split in args.splits:
            tsv = data_dir / 'resources' / 'bio_format' / dirname / f'{split}_{dirname}.tsv'
            if not tsv.exists():
                print(f'  [{code}/{split}] Not found: {tsv} — skip')
                continue
            examples = parse_bio_tsv(tsv, lang_label, split)
            split_examples[split].extend(examples)

    for split, examples in split_examples.items():
        if not examples:
            print(f'  [skip] {split} — no examples')
            continue
        out_path = out_dir / f'{split}.jsonl'
        with out_path.open('w', encoding='utf-8') as f:
            for ex in examples:
                f.write(json.dumps(ex, ensure_ascii=False) + '\n')
        # Verify
        actual = sum(1 for _ in out_path.open())
        assert actual == len(examples), f'Persistence FAILED: {actual} vs {len(examples)}'
        print(f'  ✓ {split}.jsonl — {len(examples)} rows → {out_path}')

    print('\n✓ Conversion complete. Point BiO_Task_mBERT_train.py --data_dir at output_dir.')


if __name__ == '__main__':
    main()
