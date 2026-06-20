"""
run_14_holm_bh_correction.py  —  REVISION_PLAN fix N2 (CPU, no GPU)

Applies multiple-comparison correction (Holm-Bonferroni AND Benjamini-Hochberg)
to the family of per-language "fine-tuned mBERT beats GPT-4o" Joint-F1
comparisons, before the paper says the gap is "confirmed statistically."

Why this exists
---------------
The paper claims mBERT > GPT-4o per language across {EN, ES, HI, TE}. That is a
FAMILY of 4 simultaneous tests. Reading each language's CI / p-value at the
uncorrected α=0.05 inflates the family-wise error rate: with 4 tests the chance
of ≥1 false "significant" is ~1-0.95^4 ≈ 19%. A reviewer will flag any
"statistically confirms" that ignores this. N2 reports each comparison's raw
p-value alongside its Holm-adjusted (controls FWER) and BH-adjusted (controls
FDR) p-value, so the claim survives correction explicitly.

Test per language
-----------------
Statistic = mBERT_jointF1(lang) − GPT_jointF1(lang), where Joint-F1 is the macro
F1 over {literal, idiomatic-with-span-overlap} EXACTLY as
Evaluation/Full_evaluation.compute_joint_f1 defines it (recomputed live here, no
hardcoded metric — CLAUDE.md rule). Examples are paired by sentence (the two
systems are scored on the same test sentences); we intersect the two prediction
sets per language.

Significance via paired bootstrap (n=10,000): resample example indices with
replacement, recompute BOTH systems' Joint-F1 on the SAME resample, take the
difference. Two-sided percentile-bootstrap p-value
    p = 2 * min( P(diff* ≤ 0), P(diff* ≥ 0) ),  clipped to ≤ 1
plus the 95% CI of the difference. Bootstrapping a recomputed-per-draw F1 is the
right tool here (Joint-F1 is a set-level statistic, not a per-example mean).

Corrections (implemented inline, no scipy dependency)
-----------------------------------------------------
  Holm-Bonferroni (step-down, FWER): sort p ascending; adjusted p_(i) =
      max_{j≤i} min(1, (m-j+1) * p_(j)).
  Benjamini-Hochberg (step-up, FDR):  adjusted p_(i) =
      min_{k≥i} min(1, (m/k) * p_(k)).

Defaults: mBERT = System E (joint end-to-end), GPT = System C (single-stage
GPT-4o). Override with --mbert-preds / --gpt-preds (+ their label keys) to run
the correction over any system pair (e.g. System D vs System B).

Usage (runs on CPU in seconds against the committed local preds)
----------------------------------------------------------------
    .venv/bin/python3 experiments/rigor/run_14_holm_bh_correction.py
    .venv/bin/python3 experiments/rigor/run_14_holm_bh_correction.py \
        --langs English Spanish Hindi Telugu --alpha 0.05
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.metrics import precision_recall_fscore_support

LABELS = [0, 1]
DEFAULT_LANGS = ["English", "Spanish", "Hindi", "Telugu"]  # Indonesian = held-out


def load_preds(path):
    """sentence -> record. Mirrors Full_evaluation.load_preds."""
    preds = {}
    for line in open(path, encoding="utf-8"):
        r = json.loads(line)
        preds[r["sentence"]] = r
    return preds


def overlap_f1(ps, pe, gs, ge):
    if ps is None or pe is None or gs is None or ge is None:
        return 0.0
    pred, gold = set(range(ps, pe)), set(range(gs, ge))
    if not pred or not gold:
        return 0.0
    ov = len(pred & gold)
    if ov == 0:
        return 0.0
    p, r = ov / len(pred), ov / len(gold)
    return 2 * p * r / (p + r)


def joint_labels(rec, pred_label_key, overlap_threshold=0.0):
    """Return (gold_joint, pred_joint) in {0,1} for one record, EXACTLY matching
    Full_evaluation.compute_joint_f1 (single-file systems: s1 == s2)."""
    gold_label = rec["idiomaticity"]
    pred_label = rec.get(pred_label_key)
    if gold_label == "literal":
        return 0, (0 if pred_label == "literal" else 1)
    # idiomatic gold
    if pred_label != "idiomatic":
        return 1, 0
    ov = overlap_f1(rec.get("pred_span_start"), rec.get("pred_span_end"),
                    rec["span_start"], rec["span_end"])
    return 1, (1 if ov > overlap_threshold else 0)


def macro_f1(gold, pred):
    _p, _r, f1, _s = precision_recall_fscore_support(
        gold, pred, average=None, labels=LABELS, zero_division=0)
    return float(np.mean(f1))


def per_language_arrays(mbert, gpt, mbert_key, gpt_key, lang):
    """Aligned-by-sentence arrays for one language.
    Returns gold[], pred_mbert[], pred_gpt[] (all 0/1, same order)."""
    shared = [s for s in mbert if s in gpt
              and mbert[s]["language"] == lang]
    gold, pm, pg = [], [], []
    for s in shared:
        gj_m, pj_m = joint_labels(mbert[s], mbert_key)
        gj_g, pj_g = joint_labels(gpt[s], gpt_key)
        # gold is system-independent; both must agree (same gold span/label).
        gold.append(gj_m)
        pm.append(pj_m)
        pg.append(pj_g)
    return np.array(gold), np.array(pm), np.array(pg)


def paired_bootstrap_p(gold, pm, pg, n_boot=10000, seed=42):
    """Two-sided percentile-bootstrap p for (mBERT - GPT) Joint-F1 + 95% CI."""
    point = macro_f1(gold, pm) - macro_f1(gold, pg)
    n = len(gold)
    rng = np.random.default_rng(seed)
    diffs = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        diffs[b] = macro_f1(gold[idx], pm[idx]) - macro_f1(gold[idx], pg[idx])
    lo = float(np.quantile(diffs, 0.025))
    hi = float(np.quantile(diffs, 0.975))
    p = 2.0 * min(float(np.mean(diffs <= 0)), float(np.mean(diffs >= 0)))
    return point, lo, hi, min(p, 1.0)


def holm(pvals):
    """Holm-Bonferroni adjusted p-values, original order."""
    m = len(pvals)
    order = np.argsort(pvals)
    adj = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        val = min(1.0, (m - rank) * pvals[i])
        running = max(running, val)
        adj[i] = running
    return adj


def benjamini_hochberg(pvals):
    """BH (FDR) adjusted p-values, original order."""
    m = len(pvals)
    order = np.argsort(pvals)
    adj = np.empty(m)
    running = 1.0
    for rank in range(m - 1, -1, -1):
        i = order[rank]
        val = min(1.0, pvals[i] * m / (rank + 1))
        running = min(running, val)
        adj[i] = running
    return adj


def main():
    ap = argparse.ArgumentParser()
    repo = Path(__file__).resolve().parents[2]  # experiments/ repo root
    ap.add_argument("--mbert-preds",
                    default=str(repo / "models/en_es_hi_te/joint_mbert/test_predictions.jsonl"),
                    help="Best fine-tuned mBERT system (default: System E joint end-to-end).")
    ap.add_argument("--mbert-key", default="pred_idiomaticity")
    ap.add_argument("--gpt-preds",
                    default=str(repo / "models/gpt_single_stage/test_predictions.jsonl"),
                    help="GPT-4o system (default: System C single-stage).")
    ap.add_argument("--gpt-key", default="pred_label")
    ap.add_argument("--langs", nargs="+", default=DEFAULT_LANGS)
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--n-bootstrap", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default=None)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    out_path = Path(args.out) if args.out else (
        Path(__file__).resolve().parent / "results" / "run_14_holm_bh_correction.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists() and not args.force:
        raise SystemExit(f"{out_path} exists — pass --force to overwrite (idempotency rule).")

    mbert = load_preds(args.mbert_preds)
    gpt = load_preds(args.gpt_preds)
    if not mbert or not gpt:
        raise SystemExit("Empty prediction set — check --mbert-preds / --gpt-preds paths.")

    rows = []
    for lang in args.langs:
        gold, pm, pg = per_language_arrays(mbert, gpt, args.mbert_key, args.gpt_key, lang)
        if len(gold) == 0:
            print(f"[warn] {lang}: no shared examples — skipped.")
            continue
        point, lo, hi, p = paired_bootstrap_p(
            gold, pm, pg, n_boot=args.n_bootstrap, seed=args.seed)
        rows.append({
            "lang": lang, "n": int(len(gold)),
            "mbert_f1": round(macro_f1(gold, pm), 4),
            "gpt_f1": round(macro_f1(gold, pg), 4),
            "diff": round(point, 4), "ci": [round(lo, 4), round(hi, 4)],
            "p_raw": round(p, 5),
        })

    if not rows:
        raise SystemExit("No comparable languages found.")

    pvals = np.array([r["p_raw"] for r in rows])
    p_holm = holm(pvals)
    p_bh = benjamini_hochberg(pvals)
    for r, ph, pb in zip(rows, p_holm, p_bh):
        r["p_holm"] = round(float(ph), 5)
        r["p_bh"] = round(float(pb), 5)
        r["sig_raw"] = bool(r["p_raw"] < args.alpha)
        r["sig_holm"] = bool(ph < args.alpha)
        r["sig_bh"] = bool(pb < args.alpha)

    m = len(rows)
    print(f"\nN2 multiple-comparison correction  (family of m={m} per-language tests)")
    print(f"  mBERT preds: {args.mbert_preds}")
    print(f"  GPT   preds: {args.gpt_preds}")
    print(f"  α = {args.alpha}   n_bootstrap = {args.n_bootstrap}\n")
    hdr = (f"  {'lang':10}{'n':>5}{'mBERT':>8}{'GPT':>8}{'diff':>8}"
           f"{'95% CI':>20}{'p_raw':>9}{'p_holm':>9}{'p_BH':>9}  sig(raw/Holm/BH)")
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    for r in rows:
        ci = f"[{r['ci'][0]:+.3f},{r['ci'][1]:+.3f}]"
        flags = f"{'Y' if r['sig_raw'] else 'n'}/{'Y' if r['sig_holm'] else 'n'}/{'Y' if r['sig_bh'] else 'n'}"
        print(f"  {r['lang']:10}{r['n']:>5}{r['mbert_f1']:>8.3f}{r['gpt_f1']:>8.3f}"
              f"{r['diff']:>+8.3f}{ci:>20}{r['p_raw']:>9.4f}{r['p_holm']:>9.4f}"
              f"{r['p_bh']:>9.4f}      {flags}")

    n_holm = sum(r["sig_holm"] for r in rows)
    n_bh = sum(r["sig_bh"] for r in rows)
    survivors_holm = [r["lang"] for r in rows if r["sig_holm"]]
    print(f"\n  Significant after Holm (FWER<{args.alpha}): {n_holm}/{m}  {survivors_holm}")
    print(f"  Significant after BH   (FDR <{args.alpha}): {n_bh}/{m}  "
          f"{[r['lang'] for r in rows if r['sig_bh']]}")

    if n_holm == m:
        verdict = (f"mBERT outperforms GPT-4o in all {m} languages, and every comparison "
                   f"remains significant after Holm-Bonferroni correction (all adjusted "
                   f"p < {args.alpha}).")
    elif n_holm > 0:
        verdict = (f"After Holm-Bonferroni correction, {n_holm}/{m} per-language mBERT>GPT-4o "
                   f"comparisons remain significant ({', '.join(survivors_holm)}); "
                   f"the remaining languages are reported as not significant under correction.")
    else:
        verdict = (f"No per-language mBERT>GPT-4o comparison survives Holm-Bonferroni correction "
                   f"at α={args.alpha}; drop 'confirms statistically' and report descriptively.")
    print(f"\n  Paper wording:\n  {verdict}")

    summary = {
        "fix": "N2_holm_bh_correction", "alpha": args.alpha,
        "n_bootstrap": args.n_bootstrap, "family_size": m,
        "mbert_preds": args.mbert_preds, "gpt_preds": args.gpt_preds,
        "rows": rows, "n_sig_holm": n_holm, "n_sig_bh": n_bh,
        "verdict": verdict,
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"\nSummary JSON -> {out_path}")


if __name__ == "__main__":
    main()
