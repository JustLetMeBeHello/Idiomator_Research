"""
compute_iaa.py

Inter-Annotator Agreement computation from two annotation export files.

Each export file is a JSONL produced by the IdiomBank Validator (either the
browser "Export results.jsonl" button or the /export/{annotator} API endpoint).

Usage:
    python compute_iaa.py annotator1.jsonl annotator2.jsonl

    # Restrict to a specific language
    python compute_iaa.py annotator1.jsonl annotator2.jsonl --lang Telugu

    # Save a machine-readable JSON summary
    python compute_iaa.py annotator1.jsonl annotator2.jsonl --out iaa_results.json

Outputs:
    - Per-language IAA table: % Agreement and Cohen's Kappa for two dimensions:
        1. Idiomaticity label (idiomaticity_verdict: idiomatic / literal)
        2. Span correctness (span_correct: True / False)
    - Counts of aligned vs. missing examples per annotator
    - Paper-ready table (LaTeX format with --latex flag)
"""

import json
import argparse
import sys
from pathlib import Path
from collections import defaultdict


# ── Cohen's Kappa (from scratch, no sklearn) ──────────────────────────────────

def cohen_kappa(labels_a, labels_b):
    """
    Compute Cohen's Kappa for two lists of categorical labels.
    Returns (kappa, po, pe) — observed agreement, expected agreement, kappa.
    Returns (None, None, None) if n < 2 or only one category present.
    """
    assert len(labels_a) == len(labels_b), "Lists must be same length"
    n = len(labels_a)
    if n < 2:
        return None, None, None

    categories = sorted(set(labels_a) | set(labels_b))
    if len(categories) < 2:
        return 1.0, 1.0, 1.0  # perfect agreement, only one category

    # Observed agreement
    po = sum(a == b for a, b in zip(labels_a, labels_b)) / n

    # Expected agreement
    pe = 0.0
    for cat in categories:
        pa = labels_a.count(cat) / n
        pb = labels_b.count(cat) / n
        pe += pa * pb

    if abs(1 - pe) < 1e-12:
        return 1.0, po, pe  # degenerate case

    kappa = (po - pe) / (1 - pe)
    return round(kappa, 4), round(po, 4), round(pe, 4)


def percent_agreement(labels_a, labels_b):
    n = len(labels_a)
    if n == 0:
        return None
    return round(sum(a == b for a, b in zip(labels_a, labels_b)) / n * 100, 1)


# ── Load & align ──────────────────────────────────────────────────────────────

def load_annotations(path):
    """Load JSONL annotation export. Returns dict keyed by meaning_id."""
    records = {}
    with open(path, encoding='utf-8') as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"  Warning: skipping malformed line {i} in {path}: {e}")
                continue
            mid = r.get('meaning_id')
            if not mid:
                print(f"  Warning: line {i} missing meaning_id, skipping")
                continue
            records[mid] = r
    return records


def align(ann_a, ann_b, lang_filter=None):
    """
    Return aligned (a_records, b_records) for the intersection of meaning_ids.
    Optionally filter to a specific language.
    """
    common_ids = set(ann_a) & set(ann_b)

    a_aligned, b_aligned = [], []
    for mid in sorted(common_ids):
        ra = ann_a[mid]
        rb = ann_b[mid]
        if lang_filter and ra.get('language') != lang_filter:
            continue
        a_aligned.append(ra)
        b_aligned.append(rb)

    return a_aligned, b_aligned


# ── IAA per language ──────────────────────────────────────────────────────────

def compute_iaa_for_pairs(a_records, b_records, language):
    """Given aligned record lists for one language, compute both IAA dimensions."""
    n = len(a_records)
    if n == 0:
        return None

    # Dimension 1: idiomaticity label
    idio_a = [r['idiomaticity_verdict'] for r in a_records]
    idio_b = [r['idiomaticity_verdict'] for r in b_records]
    idio_kappa, idio_po, idio_pe = cohen_kappa(idio_a, idio_b)
    idio_pct = percent_agreement(idio_a, idio_b)

    # Dimension 2: span correctness (bool → str for kappa)
    span_a = [str(r['span_correct']) for r in a_records]
    span_b = [str(r['span_correct']) for r in b_records]
    span_kappa, span_po, span_pe = cohen_kappa(span_a, span_b)
    span_pct = percent_agreement(span_a, span_b)

    return {
        'language': language,
        'n': n,
        'idiomaticity': {
            'pct_agreement': idio_pct,
            'kappa': idio_kappa,
            'po': idio_po,
            'pe': idio_pe,
        },
        'span_boundary': {
            'pct_agreement': span_pct,
            'kappa': span_kappa,
            'po': span_po,
            'pe': span_pe,
        },
    }


# ── Printing ──────────────────────────────────────────────────────────────────

def print_iaa_table(results_by_lang, name_a, name_b):
    """Print human-readable IAA table."""
    print(f"\n{'='*72}")
    print("Inter-Annotator Agreement")
    print(f"  Annotator A: {name_a}")
    print(f"  Annotator B: {name_b}")
    print(f"{'='*72}")
    print(f"\n{'Language':<14} {'Dimension':<22} {'N':>5} {'% Agree':>10} {'Kappa':>8}")
    print(f"{'-'*62}")
    for lang, res in sorted(results_by_lang.items()):
        if res is None:
            print(f"{lang:<14} {'—':22} {'—':>5} {'—':>10} {'—':>8}")
            continue
        n = res['n']
        for dim_key, dim_label in [('idiomaticity', 'Idiomaticity label'),
                                    ('span_boundary', 'Span boundary')]:
            d = res[dim_key]
            pct   = f"{d['pct_agreement']:.1f}%" if d['pct_agreement'] is not None else "—"
            kappa = f"{d['kappa']:.4f}"           if d['kappa']         is not None else "—"
            print(f"{lang:<14} {dim_label:<22} {n:>5} {pct:>10} {kappa:>8}")
        print()


def print_latex_table(results_by_lang):
    """Print LaTeX tabular for the paper (Section 5.1)."""
    print("\n% ── LaTeX IAA table for MultiIdiom Section 5.1 ──────────────────────")
    print(r"\begin{table}[t]")
    print(r"\centering")
    print(r"\small")
    print(r"\begin{tabular}{llrrr}")
    print(r"\toprule")
    print(r"\textbf{Language} & \textbf{Dimension} & \textbf{N} & \textbf{\% Agree} & \textbf{$\kappa$} \\")
    print(r"\midrule")
    for lang, res in sorted(results_by_lang.items()):
        if res is None:
            continue
        n = res['n']
        for dim_key, dim_label in [('idiomaticity', r'Idiomaticity label'),
                                    ('span_boundary', r'Span boundary')]:
            d = res[dim_key]
            pct   = f"{d['pct_agreement']:.1f}" if d['pct_agreement'] is not None else "—"
            kappa = f"{d['kappa']:.4f}"          if d['kappa']         is not None else "—"
            lang_cell = lang if dim_key == 'idiomaticity' else ""
            n_cell    = str(n) if dim_key == 'idiomaticity' else ""
            print(f"{lang_cell} & {dim_label} & {n_cell} & {pct} & {kappa} \\\\")
        print(r"\midrule")
    print(r"\bottomrule")
    print(r"\end{tabular}")
    print(r"\caption{Inter-annotator agreement (Cohen's $\kappa$) for doubly-annotated Telugu test examples and English crosscheck subset, computed on two dimensions separately.}")
    print(r"\label{tab:iaa}")
    print(r"\end{table}")


# ── Main ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Compute IAA from two IdiomBank Validator annotation exports."
    )
    p.add_argument('annotator_a', help='JSONL export from annotator A')
    p.add_argument('annotator_b', help='JSONL export from annotator B')
    p.add_argument('--lang', default=None,
                   help='Filter to a specific language (e.g. Telugu). Default: all languages.')
    p.add_argument('--out', default=None,
                   help='Optional path to save JSON results')
    p.add_argument('--latex', action='store_true',
                   help='Also print LaTeX table')
    return p.parse_args()


def main():
    args = parse_args()

    print(f"Loading {args.annotator_a}...")
    ann_a = load_annotations(args.annotator_a)
    print(f"  {len(ann_a)} annotations loaded")

    print(f"Loading {args.annotator_b}...")
    ann_b = load_annotations(args.annotator_b)
    print(f"  {len(ann_b)} annotations loaded")

    # Overlap stats
    only_a = set(ann_a) - set(ann_b)
    only_b = set(ann_b) - set(ann_a)
    common = set(ann_a) & set(ann_b)
    print(f"\n  Only in A: {len(only_a)}  |  Only in B: {len(only_b)}  |  Common: {len(common)}")
    if only_a or only_b:
        print("  Warning: annotators did not cover exactly the same set of examples.")

    if not common:
        print("\nNo common examples to compare — cannot compute IAA.")
        sys.exit(1)

    # Determine languages to report
    all_records_a = list(ann_a.values())
    all_records_b = list(ann_b.values())
    if args.lang:
        languages = [args.lang]
    else:
        languages = sorted(set(r['language'] for r in all_records_a))

    results_by_lang = {}
    for lang in languages:
        a_recs, b_recs = align(ann_a, ann_b, lang_filter=lang)
        if not a_recs:
            print(f"\n  Warning: no common examples for language '{lang}'")
            results_by_lang[lang] = None
            continue
        results_by_lang[lang] = compute_iaa_for_pairs(a_recs, b_recs, lang)

    name_a = Path(args.annotator_a).stem
    name_b = Path(args.annotator_b).stem
    print_iaa_table(results_by_lang, name_a, name_b)

    if args.latex:
        print_latex_table(results_by_lang)

    if args.out:
        with open(args.out, 'w', encoding='utf-8') as f:
            json.dump(results_by_lang, f, indent=2)
        print(f"\nJSON results saved → {args.out}")


if __name__ == '__main__':
    main()
