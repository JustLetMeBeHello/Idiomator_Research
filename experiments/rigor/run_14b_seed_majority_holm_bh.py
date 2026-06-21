"""
run_14b_seed_majority_holm_bh.py — N2 variant: 3-seed majority-vote ensemble

Why this exists
----------------
run_14_holm_bh_correction.py (N2) tests ONE seed's mBERT against GPT-4o per
language. With 3 seeds now available (42/123/7), the naive way to "use all
three" is to concatenate all three seeds' prediction rows into one bootstrap
pool. That is wrong: it's still only 62 (or 254) *sentences*, scored by 3
correlated model checkpoints, not 3 independent samples of test items —
treating the 3x-duplicated rows as independent draws is pseudo-replication
that artificially shrinks the bootstrap CI. Worse, GPT-4o is deterministic
(byte-identical across all 3 seeds, confirmed in key_numbers.md), so "pooling"
its side just triplicates the same 62 values — fake variance reduction on the
GPT side with zero new information.

What this does instead
-----------------------
Reduces the 3 seeds to ONE prediction per sentence via majority vote (mBERT
gets a sentence "right" if >=2 of 3 seeds got it right), then runs the exact
same paired-bootstrap + Holm/BH machinery as N2 on that single ensembled
prediction set vs GPT-4o. This keeps n at the true sentence count (no fake
inflation) while using the extra seeds to produce a less noisy single mBERT
verdict per sentence — a legitimate power gain, not a fake one.

Usage
-----
    .venv/bin/python3 experiments/rigor/run_14b_seed_majority_holm_bh.py
"""

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import precision_recall_fscore_support

LABELS = [0, 1]
DEFAULT_LANGS = ["English", "Spanish", "Hindi", "Telugu"]


def load_preds(path):
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
    gold_label = rec["idiomaticity"]
    pred_label = rec.get(pred_label_key)
    if gold_label == "literal":
        return 0, (0 if pred_label == "literal" else 1)
    if pred_label != "idiomatic":
        return 1, 0
    ov = overlap_f1(rec.get("pred_span_start"), rec.get("pred_span_end"),
                    rec["span_start"], rec["span_end"])
    return 1, (1 if ov > overlap_threshold else 0)


def macro_f1(gold, pred):
    _p, _r, f1, _s = precision_recall_fscore_support(
        gold, pred, average=None, labels=LABELS, zero_division=0)
    return float(np.mean(f1))


def per_language_arrays(mbert_seeds, gpt, mbert_key, gpt_key, lang):
    """Majority-vote mBERT prediction (>=2/3 seeds) vs GPT, per language.
    Returns gold[], pred_mbert_majority[], pred_gpt[], agreement_rate."""
    shared = [s for s in mbert_seeds[0] if all(s in m for m in mbert_seeds[1:])
              and s in gpt and mbert_seeds[0][s]["language"] == lang]
    gold, pm, pg, n_unanimous = [], [], [], 0
    for s in shared:
        seed_results = [joint_labels(m[s], mbert_key) for m in mbert_seeds]
        golds = {gj for gj, _ in seed_results}
        assert len(golds) == 1, f"gold mismatch across seeds for sentence: {s!r}"
        gj = seed_results[0][0]
        seed_preds = [pj for _, pj in seed_results]
        majority = 1 if sum(seed_preds) >= 2 else 0
        if len(set(seed_preds)) == 1:
            n_unanimous += 1
        gj_g, pj_g = joint_labels(gpt[s], gpt_key)
        gold.append(gj)
        pm.append(majority)
        pg.append(pj_g)
    agreement = n_unanimous / len(shared) if shared else 0.0
    return np.array(gold), np.array(pm), np.array(pg), agreement


def paired_bootstrap_p(gold, pm, pg, n_boot=10000, seed=42):
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
    repo = Path(__file__).resolve().parents[2]
    ap.add_argument("--mbert-preds", nargs=3, metavar=("S42", "S123", "S7"),
                    default=[
                        str(repo / "models/en_es_hi_te/joint_mbert/test_predictions.jsonl"),
                        str(repo / "models/main_s123/joint_mbert/test_predictions.jsonl"),
                        str(repo / "models/main_s7/joint_mbert/test_predictions.jsonl"),
                    ])
    ap.add_argument("--mbert-key", default="pred_idiomaticity")
    ap.add_argument("--gpt-preds",
                    default=str(repo / "models/gpt_single_stage/test_predictions.jsonl"))
    ap.add_argument("--gpt-key", default="pred_label")
    ap.add_argument("--langs", nargs="+", default=DEFAULT_LANGS)
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--n-bootstrap", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default=None)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    out_path = Path(args.out) if args.out else (
        Path(__file__).resolve().parent / "results" / "run_14b_seed_majority_holm_bh.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists() and not args.force:
        raise SystemExit(f"{out_path} exists — pass --force to overwrite (idempotency rule).")

    mbert_seeds = [load_preds(p) for p in args.mbert_preds]
    gpt = load_preds(args.gpt_preds)
    if not all(mbert_seeds) or not gpt:
        raise SystemExit("Empty prediction set — check --mbert-preds / --gpt-preds paths.")

    rows = []
    for lang in args.langs:
        gold, pm, pg, agreement = per_language_arrays(
            mbert_seeds, gpt, args.mbert_key, args.gpt_key, lang)
        if len(gold) == 0:
            print(f"[warn] {lang}: no shared examples — skipped.")
            continue
        point, lo, hi, p = paired_bootstrap_p(
            gold, pm, pg, n_boot=args.n_bootstrap, seed=args.seed)
        rows.append({
            "lang": lang, "n": int(len(gold)),
            "mbert_majority_f1": round(macro_f1(gold, pm), 4),
            "gpt_f1": round(macro_f1(gold, pg), 4),
            "seed_unanimous_rate": round(agreement, 4),
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
    print(f"\nN2-ensemble (3-seed majority vote vs GPT-4o, family of m={m} per-language tests)")
    print(f"  mBERT preds: {args.mbert_preds}")
    print(f"  GPT   preds: {args.gpt_preds}")
    print(f"  α = {args.alpha}   n_bootstrap = {args.n_bootstrap}\n")
    hdr = (f"  {'lang':10}{'n':>5}{'unanim':>8}{'mBERT':>8}{'GPT':>8}{'diff':>8}"
           f"{'95% CI':>20}{'p_raw':>9}{'p_holm':>9}{'p_BH':>9}  sig(raw/Holm/BH)")
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    for r in rows:
        ci = f"[{r['ci'][0]:+.3f},{r['ci'][1]:+.3f}]"
        flags = f"{'Y' if r['sig_raw'] else 'n'}/{'Y' if r['sig_holm'] else 'n'}/{'Y' if r['sig_bh'] else 'n'}"
        print(f"  {r['lang']:10}{r['n']:>5}{r['seed_unanimous_rate']:>8.3f}{r['mbert_majority_f1']:>8.3f}{r['gpt_f1']:>8.3f}"
              f"{r['diff']:>+8.3f}{ci:>20}{r['p_raw']:>9.4f}{r['p_holm']:>9.4f}"
              f"{r['p_bh']:>9.4f}      {flags}")

    n_holm = sum(r["sig_holm"] for r in rows)
    n_bh = sum(r["sig_bh"] for r in rows)
    survivors_holm = [r["lang"] for r in rows if r["sig_holm"]]
    print(f"\n  Significant after Holm (FWER<{args.alpha}): {n_holm}/{m}  {survivors_holm}")
    print(f"  Significant after BH   (FDR <{args.alpha}): {n_bh}/{m}  "
          f"{[r['lang'] for r in rows if r['sig_bh']]}")

    if n_holm == m:
        verdict = (f"3-seed-majority mBERT outperforms GPT-4o in all {m} languages, "
                   f"significant after Holm-Bonferroni (all adjusted p < {args.alpha}).")
    elif n_holm > 0:
        verdict = (f"After Holm-Bonferroni correction, {n_holm}/{m} per-language "
                   f"3-seed-majority-mBERT>GPT-4o comparisons remain significant "
                   f"({', '.join(survivors_holm)}).")
    else:
        verdict = (f"No per-language 3-seed-majority-mBERT>GPT-4o comparison survives "
                   f"Holm-Bonferroni at α={args.alpha}, even with the ensembled, "
                   f"lower-variance mBERT prediction.")
    print(f"\n  Paper wording:\n  {verdict}")

    summary = {
        "fix": "N2b_seed_majority_holm_bh", "alpha": args.alpha,
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
