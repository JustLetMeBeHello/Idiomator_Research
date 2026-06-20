"""
run_13_tost_equivalence.py  —  REVISION_PLAN fix N1 (CPU, no GPU)

Replaces the soft C1 argument "the bootstrap CI contains the +0.032 WordPiece
reference value" with a PROPER equivalence test (TOST — Two One-Sided Tests).

Why this exists
---------------
"A 95% CI that happens to contain the reference value" is NOT evidence of
equivalence — failing to reject a difference is not the same as demonstrating
absence of one. C1 claims the SentencePiece QA-BIO exact-match advantage
*collapses* under strip-normalisation to a level indistinguishable from the
WordPiece tokenizer control. The correct statistical statement of that claim is
an equivalence test: the SP family strip gap is equivalent to the reference
(WordPiece strip gap, or zero) WITHIN a pre-specified margin of indifference Δ.

TOST gives that. We test the two one-sided nulls
    H0_low : D <= -Δ     and     H0_high : D >= +Δ
where D is the difference statistic (see --reference). Rejecting BOTH at level α
lets us conclude statistical equivalence: |D| < Δ. Equivalently (and what we
print), the (1 - 2α) bootstrap CI of D lies entirely inside (-Δ, +Δ).

Statistic (macro, identical resolution to run_08 / run_08b)
-----------------------------------------------------------
Per (encoder, language) cell we already have the paired per-example
(QA_em - BIO_em) strip gaps from run_08.evaluate_encoder. A family's macro gap
is  mean_enc( mean_lang( mean_example ) ). Then:

  --reference wp   (default):  D = SP_family_macro - WP_family_macro
                   "the SP strip advantage over the tokenizer control is
                    negligible" — the exact claim C1 needs.
  --reference zero          :  D = SP_family_macro
                   "the SP strip gap is equivalent to no effect."

Bootstrap: one resample draw resamples examples within EVERY cell (both
families) and recomputes D, so the SP and WP family macros move together draw to
draw. Two-sided 90%/(1-2α) percentile CI + explicit one-sided TOST p-values.

Equivalence margin Δ
--------------------
Default Δ = 0.02. This is the same smallest-meaningful-gap threshold already
baked into run_08's verdict logic (`qa_still_wins = sp[m] > wp_m + 0.02`): a
signed gap within ±0.02 of the control is treated as "no genuine SP advantage"
elsewhere in the analysis, so reusing it here keeps the paper internally
consistent. Override with --margin and STATE the chosen Δ in the paper — TOST is
only as meaningful as its pre-registered margin.

No reported metric is hardcoded (CLAUDE.md metric-integrity rule): every number
is recomputed live from the prediction files via run_08 machinery.

Usage (same preds-root layout as run_08; runs on CPU in seconds)
----------------------------------------------------------------
    .venv/bin/python3 experiments/rigor/run_13_tost_equivalence.py \
        --preds-root /content/drive/MyDrive/IdiomatorRigor \
        --seeds 42 123 7 \
        --mode strip --reference wp --margin 0.02

Prints the TOST verdict and the sentence to paste into §7 in place of the
"CI contains the reference" wording.
"""

import argparse
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_08_extended_gold import evaluate_encoder, SOURCES  # noqa: E402


def family_macro(cells_by_fam, fam, rng=None):
    """Macro gap for one family = mean_enc( mean_lang( mean_example ) ).
    cells_by_fam[fam]: list of (encoder_id, lang, np.array per-example gaps).
    If rng given, resample examples within each cell (paired bootstrap draw)."""
    by_enc = {}
    for enc, _lang, gaps in cells_by_fam.get(fam, []):
        if len(gaps) == 0:
            continue
        if rng is not None:
            gaps = gaps[rng.integers(0, len(gaps), len(gaps))]
        by_enc.setdefault(enc, []).append(float(np.mean(gaps)))
    enc_means = [float(np.mean(v)) for v in by_enc.values() if v]
    return float(np.mean(enc_means)) if enc_means else float("nan")


def difference(cells_by_fam, reference, rng=None):
    sp = family_macro(cells_by_fam, "SentencePiece", rng)
    if reference == "zero":
        return sp, sp, 0.0
    wp = family_macro(cells_by_fam, "WordPiece", rng)
    return sp - wp, sp, wp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds-root", required=True,
                    help="Same predictions root as run_08_extended_gold.py")
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 7])
    ap.add_argument("--mode", default="strip", choices=["original", "extend", "strip"])
    ap.add_argument("--reference", default="wp", choices=["wp", "zero"],
                    help="wp: D = SP-WP family gap (equivalence to tokenizer control). "
                         "zero: D = SP family gap (equivalence to no effect).")
    ap.add_argument("--margin", type=float, default=0.02,
                    help="Equivalence margin Δ (smallest meaningful gap). Default 0.02 "
                         "matches run_08's qa_still_wins threshold.")
    ap.add_argument("--alpha", type=float, default=0.05,
                    help="One-sided level for each TOST test (CI shown is 1-2*alpha).")
    ap.add_argument("--n-bootstrap", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default=None)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    out_path = Path(args.out) if args.out else (
        Path(__file__).resolve().parent / "results"
        / f"run_13_tost_{args.reference}_{args.mode}_d{args.margin}.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists() and not args.force:
        raise SystemExit(f"{out_path} exists — pass --force to overwrite (idempotency rule).")

    # Gather per-(encoder, language) per-example gap arrays, grouped by family.
    cells_by_fam = {"SentencePiece": [], "WordPiece": []}
    seeds_found = {}
    for enc, (_subdir, _tmpl, fam) in SOURCES.items():
        _per_lang, gap_values, n_seen = evaluate_encoder(args.preds_root, enc, args.seeds)
        seeds_found[enc] = n_seen
        if n_seen == 0:
            print(f"[warn] {enc}: no seed runs found under {args.preds_root}")
            continue
        for lang, modes in gap_values.items():
            arr = np.asarray(modes.get(args.mode, []), dtype=float)
            if arr.size:
                cells_by_fam[fam].append((enc, lang, arr))

    if not cells_by_fam["SentencePiece"]:
        raise SystemExit("No SentencePiece cells found — check --preds-root layout / --seeds.")
    if args.reference == "wp" and not cells_by_fam["WordPiece"]:
        raise SystemExit("--reference wp needs WordPiece cells but none found.")

    point, sp_pt, wp_pt = difference(cells_by_fam, args.reference)

    rng = np.random.default_rng(args.seed)
    draws = np.array([difference(cells_by_fam, args.reference, rng)[0]
                      for _ in range(args.n_bootstrap)])

    d = args.margin
    a = args.alpha
    lo = float(np.quantile(draws, a))            # (1 - 2*alpha) CI lower
    hi = float(np.quantile(draws, 1 - a))        # (1 - 2*alpha) CI upper

    # One-sided bootstrap TOST p-values.
    #   H0_low : D <= -Δ   -> p = P(D* <= -Δ)
    #   H0_high: D >= +Δ   -> p = P(D* >= +Δ)
    p_low = float(np.mean(draws <= -d))
    p_high = float(np.mean(draws >= d))
    p_tost = max(p_low, p_high)
    equivalent = (lo > -d) and (hi < d)          # == (p_tost < alpha) up to MC noise

    ci_pct = int(round((1 - 2 * a) * 100))
    ref_lbl = "WordPiece reference" if args.reference == "wp" else "zero (no effect)"

    print(f"\nTOST equivalence test  (N1)  mode={args.mode}  reference={ref_lbl}")
    print(f"  margin Δ = ±{d:.3f}   one-sided α = {a:.3f}   n_bootstrap = {args.n_bootstrap}")
    print(f"  seeds_found = {seeds_found}")
    print(f"  SP family {args.mode} gap = {sp_pt:+.3f}", end="")
    if args.reference == "wp":
        print(f"   WP family {args.mode} gap = {wp_pt:+.3f}")
    else:
        print()
    print(f"  difference D = {point:+.3f}   {ci_pct}% CI [{lo:+.3f}, {hi:+.3f}]")
    print(f"  one-sided p: lower(D≤-Δ)={p_low:.4f}  upper(D≥+Δ)={p_high:.4f}  TOST p=max={p_tost:.4f}")
    if equivalent:
        print(f"  ✓ EQUIVALENT: {ci_pct}% CI lies inside (-Δ,+Δ); reject both one-sided nulls "
              f"at α={a} → SP strip gap is statistically equivalent to the {ref_lbl} within ±{d:.3f}.")
    else:
        print(f"  ✗ NOT equivalent at Δ=±{d:.3f}: CI escapes the margin. "
              f"Report the CI honestly; do NOT claim equivalence (or widen Δ only with justification).")

    if equivalent:
        sentence = (f"A two one-sided tests (TOST) procedure with an equivalence margin of "
                    f"±{d:.3f} confirms the SentencePiece strip-normalised QA–BIO gap ({sp_pt:+.3f}) "
                    f"is statistically equivalent to the {ref_lbl} (difference {point:+.3f}, "
                    f"{ci_pct}% CI [{lo:+.3f}, {hi:+.3f}], TOST p = {p_tost:.3f}).")
    else:
        sentence = (f"A TOST procedure (margin ±{d:.3f}) does not establish equivalence "
                    f"(difference {point:+.3f}, {ci_pct}% CI [{lo:+.3f}, {hi:+.3f}], TOST p = {p_tost:.3f}); "
                    f"we report the interval rather than claiming the gap is null.")
    print(f"\n§7 sentence:\n  {sentence}")

    summary = {
        "fix": "N1_tost_equivalence", "mode": args.mode, "reference": args.reference,
        "margin": d, "alpha": a, "n_bootstrap": args.n_bootstrap,
        "seeds": args.seeds, "seeds_found": seeds_found,
        "sp_gap": round(sp_pt, 4), "wp_gap": round(wp_pt, 4) if args.reference == "wp" else None,
        "difference": round(point, 4), "ci_pct": ci_pct, "ci": [round(lo, 4), round(hi, 4)],
        "p_lower": round(p_low, 5), "p_upper": round(p_high, 5), "p_tost": round(p_tost, 5),
        "equivalent": bool(equivalent), "paper_sentence": sentence,
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"\nSummary JSON -> {out_path}")


if __name__ == "__main__":
    main()
