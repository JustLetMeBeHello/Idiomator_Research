"""
sample_error_analysis.py

Samples misclassified/mispredicted examples from System E (joint_mbert)
and System G (bio_tagger) for IdiomBERT Table 9 (§9 error analysis).

Heuristic categorization into the 7 error types defined in Table 8:
  FN-I  idiomatic labelled literal (false negative)
  FP-L  literal labelled idiomatic (false positive)
  SB    span boundary off (cls correct, overlap>0, not exact)
  SM    span missed entirely (overlap==0 on idiomatic prediction)
  SC    sense confusion candidate (sense_number>1, cls correct, span off)
  CL    cross-lingual transfer error (HI/TE/ID, cls wrong)
  AM    ambiguous gold candidate (overlap>0.5, not exact, flagged for re-annot)

Output:
  error_analysis_candidates.md  — human-readable, copy rows into Table 9
  error_analysis_candidates.jsonl  — machine-readable backup

Usage:
  .venv/bin/python3 experiments/rigor/sample_error_analysis.py
  .venv/bin/python3 experiments/rigor/sample_error_analysis.py \
      --n_per_cat 8 --seed 99
"""

import json
import random
import argparse
import textwrap
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

SYSTEMS = {
    'E': ROOT / 'models/en_es_hi_te/joint_mbert/test_predictions.jsonl',
    'G': ROOT / 'models/en_es_hi_te/bio_tagger/test_predictions.jsonl',
    'C': ROOT / 'models/gpt_single_stage/test_predictions.jsonl',
}

CATEGORIES = ['FN-I', 'FP-L', 'SB', 'SM', 'SC', 'CL', 'AM']
LANGS = ['English', 'Spanish', 'Hindi', 'Telugu']
LOW_RESOURCE = {'Hindi', 'Telugu', 'Indonesian'}

CAT_DEFS = {
    'FN-I': 'Idiomatic → literal (FN). Figurative use labelled literal.',
    'FP-L': 'Literal → idiomatic (FP). Literal use labelled idiomatic.',
    'SB':   'Span boundary. Idiom detected, span off by ≥1 token.',
    'SM':   'Span missed entirely. No span produced for detected idiom.',
    'SC':   'Sense confusion. Multi-sense idiom; wrong sense (candidate).',
    'CL':   'Cross-lingual transfer. Error in low-resource lang (HI/TE/ID).',
    'AM':   'Ambiguous gold. Borderline annotation; flagged for re-annotation.',
}


def load(path):
    if not Path(path).exists():
        return []
    return [json.loads(l) for l in open(path, encoding='utf-8')]


def bio_pred_cls(row):
    tags = row.get('pred_bio_tags', [])
    return 'idiomatic' if any(t == 'B-IDIOM' for t in tags) else 'literal'


def categorize(row, sys_label):
    gold   = row.get('idiomaticity', '')
    span_s = row.get('span_start')
    span_e = row.get('span_end')
    p_s    = row.get('pred_span_start')
    p_e    = row.get('pred_span_end')
    exact  = row.get('span_exact_match', False)
    ovlp   = float(row.get('span_overlap_f1') or 0)
    lang   = row.get('language', '')
    sense  = row.get('sense_number', 1)

    if sys_label == 'G':
        pred = bio_pred_cls(row)
        cls_ok = (pred == gold)
    else:
        pred   = row.get('pred_idiomaticity', '')
        cls_ok = row.get('cls_correct', pred == gold)

    cats = []

    # FN-I: gold idiomatic, predicted literal
    if gold == 'idiomatic' and pred == 'literal':
        if lang in LOW_RESOURCE:
            cats.append('CL')
        else:
            cats.append('FN-I')

    # FP-L: gold literal, predicted idiomatic
    if gold == 'literal' and pred == 'idiomatic':
        cats.append('FP-L')

    # Span errors — only meaningful when gold is idiomatic and span exists
    if gold == 'idiomatic' and span_s is not None and p_s is not None:
        if cls_ok:
            if not exact and ovlp > 0:
                # SC: multi-sense idiom with near-correct span
                if sense and sense > 1:
                    cats.append('SC')
                # AM: overlap high but not exact — borderline annotation
                elif ovlp >= 0.6:
                    cats.append('AM')
                else:
                    cats.append('SB')
            elif ovlp == 0.0 and pred == 'idiomatic':
                cats.append('SM')

    # CL catch-all: any error in low-resource lang not yet categorized
    if not cats and not cls_ok and lang in LOW_RESOURCE:
        cats.append('CL')

    # Deduplicate, keeping first
    seen = set()
    out = []
    for c in cats:
        if c not in seen:
            seen.add(c)
            out.append(c)
    return out


def fmt_span(sentence, start, end):
    if start is None or end is None:
        return '—'
    try:
        return sentence[int(start):int(end)]
    except Exception:
        return '?'


def truncate(s, n=80):
    return (s[:n] + '…') if len(s) > n else s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n_per_cat', type=int, default=6,
                    help='Candidates to sample per category (default 6; Table 9 needs ~5)')
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--output_dir', default=str(ROOT / 'experiments/rigor/results'))
    args = ap.parse_args()

    rng = random.Random(args.seed)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Load and categorize ────────────────────────────────────────────────────
    pool = defaultdict(list)   # cat → list of (sys, row)

    for sys_label, path in SYSTEMS.items():
        rows = load(path)
        if not rows:
            print(f'  [{sys_label}] not found: {path}')
            continue
        print(f'  [{sys_label}] {len(rows)} rows from {Path(path).parent.name}')
        for row in rows:
            # Only non-Indonesian for main table (Indonesian in held-out table)
            if row.get('language') == 'Indonesian':
                continue
            for cat in categorize(row, sys_label):
                pool[cat].append((sys_label, row))

    print()
    for cat in CATEGORIES:
        print(f'  {cat}: {len(pool[cat])} candidates')

    # ── Sample ────────────────────────────────────────────────────────────────
    sampled = {}
    for cat in CATEGORIES:
        candidates = pool[cat]
        if not candidates:
            sampled[cat] = []
            continue
        # Prefer diversity: at most 2 per language
        by_lang = defaultdict(list)
        for sys, row in candidates:
            by_lang[row.get('language', '?')].append((sys, row))
        diverse = []
        for lang in LANGS:
            items = by_lang.get(lang, [])
            rng.shuffle(items)
            diverse.extend(items[:2])
        # Fill up to n_per_cat from remainder
        remainder = [(s, r) for s, r in candidates if (s, r) not in diverse]
        rng.shuffle(remainder)
        combined = diverse + remainder
        sampled[cat] = combined[:args.n_per_cat]

    # ── Write markdown ────────────────────────────────────────────────────────
    md_path = out_dir / 'error_analysis_candidates.md'
    jsonl_path = out_dir / 'error_analysis_candidates.jsonl'

    md_lines = [
        '# IdiomBERT Table 9 — Error Analysis Candidates',
        '',
        'Auto-sampled heuristically. **Review each row: keep, edit, or drop.**',
        'Target: 20–40 rows total, ~5 per category, ≥3 per language.',
        'Copy accepted rows into Table 9 in the docx.',
        '',
        '> Category codes: ' + '  |  '.join(f'**{k}** = {v.split(".")[0]}' for k, v in CAT_DEFS.items()),
        '',
    ]

    all_records = []
    row_num = 1

    for cat in CATEGORIES:
        items = sampled[cat]
        md_lines.append(f'## {cat} — {CAT_DEFS[cat]}')
        md_lines.append('')
        if not items:
            md_lines.append('_No candidates found._')
            md_lines.append('')
            continue

        md_lines.append('| # | Sys | Lang | Sentence (idiom bold) | Gold | Pred | Notes |')
        md_lines.append('|---|-----|------|-----------------------|------|------|-------|')

        for sys_label, row in items:
            sentence = row.get('sentence', '')
            gold     = row.get('idiomaticity', '?')
            lang     = row.get('language', '?')
            idiom    = row.get('idiom', '')
            span_s   = row.get('span_start')
            span_e   = row.get('span_end')
            p_s      = row.get('pred_span_start')
            p_e      = row.get('pred_span_end')
            ovlp     = float(row.get('span_overlap_f1') or 0)
            sense    = row.get('sense_number', 1)

            if sys_label == 'G':
                pred = bio_pred_cls(row)
            else:
                pred = row.get('pred_idiomaticity', '?')

            gold_span_text = fmt_span(sentence, span_s, span_e)
            pred_span_text = fmt_span(sentence, p_s, p_e)

            # Bold the gold span in the sentence for readability
            display_sent = truncate(sentence, 90)
            if span_s is not None and span_e is not None:
                try:
                    gs, ge = int(span_s), int(span_e)
                    if ge <= len(sentence):
                        display_sent = (
                            sentence[:gs] + '**' + sentence[gs:ge] + '**' + sentence[ge:]
                        )
                        display_sent = truncate(display_sent, 100)
                except Exception:
                    pass

            notes_parts = []
            if ovlp > 0 and ovlp < 1:
                notes_parts.append(f'overlap={ovlp:.2f}')
            if pred_span_text and pred_span_text != gold_span_text and pred_span_text != '—':
                notes_parts.append(f'pred_span="{truncate(pred_span_text, 25)}"')
            if sense and sense > 1:
                notes_parts.append(f'sense={sense}')
            notes = '; '.join(notes_parts) if notes_parts else '[TODO: add notes]'

            md_lines.append(
                f'| {row_num} | {sys_label} | {lang[:2]} | {display_sent} '
                f'| {gold[:4]} | {pred[:4]} | {notes} |'
            )

            all_records.append({
                'row_num': row_num, 'cat': cat, 'sys': sys_label,
                'lang': lang, 'sentence': sentence, 'idiom': idiom,
                'gold': gold, 'pred': pred,
                'gold_span': gold_span_text, 'pred_span': pred_span_text,
                'span_overlap_f1': ovlp, 'sense_number': sense,
                'idiom_id': row.get('idiom_id', ''),
            })
            row_num += 1

        md_lines.append('')

    md_lines += [
        '---',
        '',
        f'Total candidates: {row_num - 1}',
        'Final Table 9 target: 20–40 rows, ~5 per category, ≥3 per language.',
        '',
        'After selecting rows: paste into Table 9 in IdiomBERT_Submission_Ready_v8.docx §9.',
    ]

    md_path.write_text('\n'.join(md_lines), encoding='utf-8')
    print(f'\n✓ Candidates → {md_path}')

    with open(jsonl_path, 'w', encoding='utf-8') as f:
        for r in all_records:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')
    actual = sum(1 for _ in open(jsonl_path))
    assert actual == len(all_records)
    print(f'✓ JSONL backup → {jsonl_path}')
    print(f'\nReview {md_path.name} and copy accepted rows into Table 9 in the docx.')


if __name__ == '__main__':
    main()
