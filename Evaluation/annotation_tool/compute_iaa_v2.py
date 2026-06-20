"""
compute_iaa_v2.py

Corrected inter-annotator agreement for MultiIdiom / IdiomBERT.

Why v2 exists
-------------
The original compute_iaa.py reports two Cohen's kappa values:
  1. idiomaticity_verdict  (idiomatic / literal)   -- fine, keep it
  2. span_correct          (True / False)           -- a BINARY "is the pipeline
                                                       span correct?" judgment

Problem (council finding): span_correct kappa does NOT measure whether two
annotators agree on the *span boundary*. It is a verification judgment against
the pipeline's own span, so:
  - It is insensitive to the +/-1-token trailing-punctuation differences that
    IdiomBERT's C1 ("annotation artifact") claim lives or dies on.
  - It is anchored: when both annotators ACCEPT the pipeline span, their
    effective gold spans are identical BY CONSTRUCTION, inflating agreement.

A span-extraction benchmark whose headline effects are 0.01-0.03 in exact match
cannot be certified by a metric that is both insensitive to boundaries and
upward-biased. C1's exact-match metric is therefore currently UNVALIDATED.

What v2 adds
------------
For every doubly-annotated example we reconstruct each annotator's *effective
gold span* (character offsets into the sentence):
    span_correct == True   -> they accepted the pipeline span  (span_start, span_end)
    span_correct == False  -> locate their span_correction text in the sentence
Then we report, with bootstrap 95% CIs (seed=42, 10k resamples):
    A. Idiomaticity-label Cohen's kappa            (+ CI)
    B. Span-ACCEPTANCE kappa (old span_correct)    (+ CI)  -- relabeled honestly
    C. Boundary agreement  -- the metric C1 actually needs:
         - mean character-level IoU between the two effective spans
         - exact-boundary match rate (identical start AND end)
       reported TWICE:
         (C-all)       over all aligned items                 [anchored, UPPER BOUND]
         (C-unanchored) over items where >=1 annotator         [the real signal]
                        rejected the pipeline span

Interpreting C for the paper
----------------------------
- If C-unanchored mean-IoU is high and exact-boundary rate is high, exact-match
  span eval is defensible and C1 can be headlined on human gold.
- If boundary disagreement (1 - exact-rate, or 1 - mean-IoU) is COMPARABLE TO OR
  LARGER THAN the 0.01-0.03 artifact deltas, then exact-match is measuring
  annotation noise. Report span scores as IoU-thresholded (overlap), not exact,
  and state that exact-match is unreliable at the measured IAA. (This is the
  council's salvage path.)

English caveat
--------------
Cohen's kappa requires TWO annotators. EN currently has one. Either recruit a
second EN annotator on an overlap subset, or drop the EN kappa cell entirely.
This script will simply report no EN row if only one EN file is supplied.

Usage
-----
    python compute_iaa_v2.py annotatorA.jsonl annotatorB.jsonl
    python compute_iaa_v2.py a.jsonl b.jsonl --lang Telugu --pool-dir ./data
    python compute_iaa_v2.py a.jsonl b.jsonl --out iaa_results.json --latex
    python compute_iaa_v2.py a.jsonl b.jsonl --boot 10000 --seed 42
"""

import json
import argparse
import sys
import random
from pathlib import Path


# ── Cohen's Kappa ─────────────────────────────────────────────────────────────

def cohen_kappa(labels_a, labels_b):
    """Cohen's kappa for two equal-length lists of categorical labels."""
    n = len(labels_a)
    if n < 2:
        return None
    categories = sorted(set(labels_a) | set(labels_b))
    if len(categories) < 2:
        return 1.0  # only one category present -> perfect agreement
    po = sum(a == b for a, b in zip(labels_a, labels_b)) / n
    pe = sum((labels_a.count(c) / n) * (labels_b.count(c) / n) for c in categories)
    if abs(1 - pe) < 1e-12:
        return 1.0
    return (po - pe) / (1 - pe)


def percent_agreement(labels_a, labels_b):
    n = len(labels_a)
    return None if n == 0 else sum(a == b for a, b in zip(labels_a, labels_b)) / n


# ── Boundary geometry ─────────────────────────────────────────────────────────

def char_iou(span_a, span_b):
    """Character-set IoU between two (start, end) half-open intervals."""
    a0, a1 = span_a
    b0, b1 = span_b
    if a1 <= a0 or b1 <= b0:
        return 0.0
    inter = max(0, min(a1, b1) - max(a0, b0))
    union = (a1 - a0) + (b1 - b0) - inter
    return inter / union if union > 0 else 0.0


def exact_boundary(span_a, span_b):
    return 1.0 if span_a == span_b else 0.0


def effective_span(rec, sentence):
    """
    Reconstruct an annotator's effective gold span as (start, end) char offsets.
    Returns None if it cannot be determined (rejected with unusable correction).
    """
    if rec.get('span_correct') is True:
        s, e = rec.get('span_start'), rec.get('span_end')
        if s is None or e is None:
            return None
        return (int(s), int(e))
    # span rejected -> use the annotator's typed correction, located in sentence
    corr = (rec.get('span_correction') or '').strip()
    if not corr or sentence is None:
        return None
    idx = sentence.find(corr)
    if idx < 0:
        # try a whitespace-normalized fallback
        norm = ' '.join(corr.split())
        idx = sentence.find(norm)
        if idx < 0:
            return None
        corr = norm
    return (idx, idx + len(corr))


# ── Bootstrap CI ──────────────────────────────────────────────────────────────

def bootstrap_ci(items, metric_fn, n_boot, rng, lo=2.5, hi=97.5):
    """
    items: list of per-example tuples consumed by metric_fn(list)->float|None.
    Returns (point, ci_lo, ci_hi) or (point, None, None) if not estimable.
    """
    point = metric_fn(items)
    if point is None or len(items) < 2:
        return point, None, None
    n = len(items)
    samples = []
    for _ in range(n_boot):
        resample = [items[rng.randrange(n)] for _ in range(n)]
        v = metric_fn(resample)
        if v is not None:
            samples.append(v)
    if not samples:
        return point, None, None
    samples.sort()
    return (point,
            samples[int(lo / 100 * len(samples))],
            samples[min(len(samples) - 1, int(hi / 100 * len(samples)))])


# ── Metric closures over per-item records ─────────────────────────────────────

def _kappa_metric(key):
    def fn(items):
        a = [it[key + '_a'] for it in items]
        b = [it[key + '_b'] for it in items]
        return cohen_kappa(a, b)
    return fn


def _mean_iou(items):
    vals = [it['iou'] for it in items if it['iou'] is not None]
    return sum(vals) / len(vals) if vals else None


def _exact_rate(items):
    vals = [it['exact'] for it in items if it['exact'] is not None]
    return sum(vals) / len(vals) if vals else None


# ── Load / align ──────────────────────────────────────────────────────────────

def load_annotations(path):
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
            if mid:
                records[mid] = r
    return records


def load_pool_sentences(pool_dir):
    """meaning_id -> sentence, from data/<Lang>/annotation_pool.jsonl files."""
    sentences = {}
    if not pool_dir:
        return sentences
    for p in Path(pool_dir).glob('*/annotation_pool.jsonl'):
        with open(p, encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                mid = r.get('meaning_id')
                if mid and r.get('sentence'):
                    sentences[mid] = r['sentence']
    return sentences


# ── Per-language computation ──────────────────────────────────────────────────

def build_items(ann_a, ann_b, sentences, language):
    """Build per-example item dicts for one language over the common id set."""
    common = sorted(set(ann_a) & set(ann_b))
    items = []
    n_boundary_undeterminable = 0
    for mid in common:
        ra, rb = ann_a[mid], ann_b[mid]
        if ra.get('language') != language:
            continue
        sent = sentences.get(mid)
        sa = effective_span(ra, sent)
        sb = effective_span(rb, sent)
        anchored = (ra.get('span_correct') is True) and (rb.get('span_correct') is True)
        if sa is None or sb is None:
            iou = None
            exact = None
            n_boundary_undeterminable += 1
        else:
            iou = char_iou(sa, sb)
            exact = exact_boundary(sa, sb)
        items.append({
            'meaning_id': mid,
            'idio_a': ra.get('idiomaticity_verdict'),
            'idio_b': rb.get('idiomaticity_verdict'),
            'accept_a': str(ra.get('span_correct')),
            'accept_b': str(rb.get('span_correct')),
            'iou': iou,
            'exact': exact,
            'anchored': anchored,
        })
    return items, n_boundary_undeterminable


def compute_language(items, n_boot, seed):
    if not items:
        return None
    rng = random.Random(seed)
    boundary_items = [it for it in items if it['iou'] is not None]
    unanchored = [it for it in boundary_items if not it['anchored']]

    def ci(its, fn):
        return bootstrap_ci(its, fn, n_boot, random.Random(seed))

    return {
        'n': len(items),
        'n_boundary': len(boundary_items),
        'n_unanchored': len(unanchored),
        # A. idiomaticity label
        'idiomaticity_kappa': ci(items, _kappa_metric('idio')),
        'idiomaticity_pct': percent_agreement([i['idio_a'] for i in items],
                                              [i['idio_b'] for i in items]),
        # B. span acceptance (old span_correct), honestly relabeled
        'acceptance_kappa': ci(items, _kappa_metric('accept')),
        # C-all: boundary agreement, anchored upper bound
        'iou_all': ci(boundary_items, _mean_iou),
        'exact_all': ci(boundary_items, _exact_rate),
        # C-unanchored: boundary agreement, the real signal
        'iou_unanchored': ci(unanchored, _mean_iou),
        'exact_unanchored': ci(unanchored, _exact_rate),
    }


# ── Printing ──────────────────────────────────────────────────────────────────

def _fmt(triple):
    if triple is None:
        return "—"
    pt, lo, hi = triple
    if pt is None:
        return "—"
    if lo is None:
        return f"{pt:.3f}"
    return f"{pt:.3f} [{lo:.3f}, {hi:.3f}]"


def print_report(results, name_a, name_b):
    print(f"\n{'='*78}")
    print(f"Inter-Annotator Agreement v2   (A={name_a}  B={name_b})")
    print(f"{'='*78}")
    for lang, r in sorted(results.items()):
        if r is None:
            print(f"\n{lang}: no common examples")
            continue
        print(f"\n## {lang}   (n={r['n']}, boundary-determinable={r['n_boundary']}, "
              f"unanchored={r['n_unanchored']})")
        print(f"  A  Idiomaticity kappa        : {_fmt(r['idiomaticity_kappa'])}")
        print(f"     Idiomaticity % agreement  : "
              f"{r['idiomaticity_pct']*100:.1f}%" if r['idiomaticity_pct'] is not None else "—")
        print(f"  B  Span-acceptance kappa     : {_fmt(r['acceptance_kappa'])}  "
              f"(old span_correct; anchored, not a boundary metric)")
        print(f"  C  Boundary — ALL items (anchored upper bound):")
        print(f"       mean char-IoU           : {_fmt(r['iou_all'])}")
        print(f"       exact-boundary rate     : {_fmt(r['exact_all'])}")
        print(f"  C  Boundary — UNANCHORED (>=1 correction; the real signal):")
        print(f"       mean char-IoU           : {_fmt(r['iou_unanchored'])}")
        print(f"       exact-boundary rate     : {_fmt(r['exact_unanchored'])}")
        # decision hint
        ex = r['exact_unanchored']
        if ex and ex[0] is not None:
            disagree = 1 - ex[0]
            print(f"     -> boundary disagreement (unanchored) ~ {disagree:.3f}. "
                  f"Compare to your artifact deltas (0.01-0.03): "
                  f"{'EXCEEDS — exact-match unreliable, use IoU-thresholded' if disagree > 0.03 else 'within tolerance'}.")


def print_latex(results):
    print("\n% ── LaTeX IAA table (MultiIdiom Sec 5.1 / IdiomBERT Sec 5) ──")
    print(r"\begin{table}[t]\centering\small")
    print(r"\begin{tabular}{lrccc}")
    print(r"\toprule")
    print(r"\textbf{Lang} & \textbf{N} & \textbf{Idiom.\ $\kappa$} "
          r"& \textbf{Bnd.\ IoU (unanch.)} & \textbf{Exact bnd.\ (unanch.)} \\")
    print(r"\midrule")
    for lang, r in sorted(results.items()):
        if r is None:
            continue
        print(f"{lang} & {r['n']} & {_fmt(r['idiomaticity_kappa'])} "
              f"& {_fmt(r['iou_unanchored'])} & {_fmt(r['exact_unanchored'])} \\\\")
    print(r"\bottomrule\end{tabular}")
    print(r"\caption{Inter-annotator agreement on the doubly-annotated subset. "
          r"Idiomaticity $\kappa$ (Cohen); boundary agreement reported on the "
          r"\emph{unanchored} subset (at least one annotator rejected the "
          r"pipeline span) to remove acceptance-anchoring bias. 95\% bootstrap "
          r"CIs (10k resamples, seed 42).}")
    print(r"\label{tab:iaa}\end{table}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="Corrected IAA (boundary + bootstrap CIs).")
    ap.add_argument('annotator_a')
    ap.add_argument('annotator_b')
    ap.add_argument('--lang', default=None, help='Restrict to one language.')
    ap.add_argument('--pool-dir', default='./data',
                    help='Dir with <Lang>/annotation_pool.jsonl for sentence lookup.')
    ap.add_argument('--boot', type=int, default=10000)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--out', default=None)
    ap.add_argument('--latex', action='store_true')
    args = ap.parse_args()

    ann_a = load_annotations(args.annotator_a)
    ann_b = load_annotations(args.annotator_b)
    sentences = load_pool_sentences(args.pool_dir)
    print(f"Loaded A={len(ann_a)}  B={len(ann_b)}  pool_sentences={len(sentences)}")

    common = set(ann_a) & set(ann_b)
    if not common:
        print("No common meaning_ids — cannot compute IAA.")
        sys.exit(1)
    if not sentences:
        print("  WARNING: no pool sentences loaded; boundary metrics from corrections "
              "will be unavailable. Pass --pool-dir pointing at data/.")

    langs = [args.lang] if args.lang else sorted(
        {ann_a[m].get('language') for m in common if ann_a[m].get('language')})

    results = {}
    for lang in langs:
        items, undet = build_items(ann_a, ann_b, sentences, lang)
        if not items:
            results[lang] = None
            continue
        if undet:
            print(f"  {lang}: {undet} items had undeterminable boundaries "
                  f"(rejected w/o locatable correction).")
        results[lang] = compute_language(items, args.boot, args.seed)

    print_report(results, Path(args.annotator_a).stem, Path(args.annotator_b).stem)
    if args.latex:
        print_latex(results)
    if args.out:
        with open(args.out, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2)
        print(f"\nSaved -> {args.out}")


if __name__ == '__main__':
    main()
