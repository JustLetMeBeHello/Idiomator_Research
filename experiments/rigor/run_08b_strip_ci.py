"""
run_08b_strip_ci.py

Companion to run_08_extended_gold.py. Produces the single bootstrap CI that
the §7 [FILL] needs: the 95% CI on the SentencePiece FAMILY strip gap
(the headline +0.025), computed on EXACTLY the statistic the paper reports —
mean over SP encoders of (mean over languages of the per-example QA−BIO
strip-normalised exact-match gap). Macro, not micro: resamples examples
within each (encoder, language) cell and recomputes the macro family mean on
every bootstrap draw, so the interval is centred on the same +0.025 point
estimate that appears in Table 6 / §7, not on an example-count-weighted micro
average.

Reuses evaluate_encoder() + SOURCES from run_08 so the punctuation set,
offset fix, and strip logic are identical to the main analysis.

Usage (same preds-root layout as run_08; the strip rows were re-scored from
the saved test_predictions.jsonl files):

    python experiments/rigor/run_08b_strip_ci.py \
        --preds-root /content/drive/MyDrive/IdiomatorRigor \
        --seeds 42 123 7 \
        --mode strip

Prints:  SentencePiece family <mode> gap = +0.0xx  (95% CI +0.0xx, +0.0xx)  n_cells=K
Paste the lo/hi into the §7 sentence:
    "...places the SentencePiece family strip gap at +0.025 [95% CI lo, hi];..."

NOTE: confirm --seeds against what the strip/extend rows were actually scored
on. run_08's header re-scores seed-42 predictions and its --seeds default is
[42,123]; the paper claims 3 seeds (42/123/7). If only seed 42 (or 42+123)
exists on disk, the §7 "3 seeds each" claim for the normalised rows must be
corrected to match what this script actually finds (it reports n_seeds_found).
"""

import argparse
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_08_extended_gold import evaluate_encoder, SOURCES  # noqa: E402


def macro_family_gap(cells, rng=None):
    """cells: list of (encoder_id, lang, np.array_of_per_example_gaps).
    Returns the macro family gap = mean_enc( mean_lang( mean_example ) ).
    If rng is given, resample examples within each cell (bootstrap draw)."""
    by_enc = {}
    for enc, lang, gaps in cells:
        if len(gaps) == 0:
            continue
        if rng is not None:
            gaps = gaps[rng.integers(0, len(gaps), len(gaps))]
        by_enc.setdefault(enc, []).append(float(np.mean(gaps)))
    enc_means = [float(np.mean(lang_means)) for lang_means in by_enc.values() if lang_means]
    return float(np.mean(enc_means)) if enc_means else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds-root", required=True,
                    help="Same predictions root as run_08_extended_gold.py")
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 7])
    ap.add_argument("--mode", default="strip", choices=["original", "extend", "strip"])
    ap.add_argument("--family", default="SentencePiece", choices=["SentencePiece", "WordPiece"])
    ap.add_argument("--n-bootstrap", type=int, default=10000)
    ap.add_argument("--ci", type=float, default=0.95)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    # Gather per-(encoder, language) per-example gap arrays for the chosen family+mode.
    cells = []
    seeds_found = {}
    for enc, (_subdir, _tmpl, fam) in SOURCES.items():
        if fam != args.family:
            continue
        per_lang_mode_em, gap_values, n_seen = evaluate_encoder(args.preds_root, enc, args.seeds)
        seeds_found[enc] = n_seen
        if n_seen == 0:
            print(f"[warn] {enc}: no seed runs found under {args.preds_root}")
            continue
        for lang, modes in gap_values.items():
            arr = np.asarray(modes.get(args.mode, []), dtype=float)
            if arr.size:
                cells.append((enc, lang, arr))

    if not cells:
        raise SystemExit("No cells found — check --preds-root layout and --seeds.")

    point = macro_family_gap(cells)
    rng = np.random.default_rng(args.seed)
    draws = np.array([macro_family_gap(cells, rng) for _ in range(args.n_bootstrap)])
    lo = float(np.quantile(draws, (1 - args.ci) / 2))
    hi = float(np.quantile(draws, 1 - (1 - args.ci) / 2))

    print(f"\n{args.family} family {args.mode} gap = {point:+.3f} "
          f"({int(args.ci*100)}% CI {lo:+.3f}, {hi:+.3f})  "
          f"n_cells={len(cells)}  seeds_found={seeds_found}")
    print(f"\nPaste into §7:  ...at {point:+.3f} [{int(args.ci*100)}% CI {lo:+.3f}, {hi:+.3f}];...")
    if hi >= 0.032 >= lo:
        print("CI contains the +0.032 WordPiece reference -> 'not separable at this resolution' holds.")
    else:
        print("WARNING: CI does NOT contain +0.032 -> rephrase the §7 sentence to match the interval.")


if __name__ == "__main__":
    main()
