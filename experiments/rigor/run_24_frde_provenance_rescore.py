"""
run_24_frde_provenance_rescore.py — N2 provenance / home-field-advantage stress test

Why this exists
----------------
The N2 result (fine-tuned mBERT > GPT-4o zero-shot, Holm/BH-significant across
Indonesian + French + German) rests on a test set whose French/German gold spans
are LLM-generated *silver* labels from the MultiIdiom pipeline — the SAME
generative provenance the mBERT training data descends from. A hostile methods
reviewer's attack (the "home-field advantage" confound): mBERT may win not on
idiom competence but because it learned the pipeline's span-boundary conventions
(trailing punctuation, whitespace offsets), while GPT-4o zero-shot is penalised
for producing correct-but-differently-conventioned spans. This is exactly the
C1 failure mode (a span-formatting artifact masquerading as a real effect) that
already killed one headline in this paper — so the surviving N2 effect has to be
shown immune to it before it can anchor anything.

What this does (three conditions, all on EXISTING prediction files — no model
runs, no API, no GPU):
  1. SYMMETRIC STRIP-NORMALIZATION — apply the canonical trailing-punct + leading-
     whitespace strip (from run_08's PUNCT_TRAIL) IDENTICALLY to gold, mBERT-pred,
     and GPT-pred spans, so neither system is advantaged by convention-matching.
  2. OVERLAP-THRESHOLD SWEEP — recompute Joint F1 at span-overlap thresholds
     {0.0 (current lenient), 0.5, exact-after-norm}. If the mBERT lead is a
     boundary-convention artifact, it shrinks as the threshold tightens ONLY for
     the raw (un-normalized) scoring and should be stable under symmetric norm.
  3. CLASSIFICATION-ONLY DECOMPOSITION — the decisive test. Score pure
     idiomatic/literal classification (span-agnostic) mBERT-majority vs GPT.
     Span convention is irrelevant here, so if mBERT still beats GPT on cls-only,
     the win CANNOT be a span-convention home-field artifact — at worst it's a
     label-provenance concern, which is what the human-gold spot-check addresses.

Method mirrors run_14b_seed_majority_holm_bh.py exactly (3-seed majority vote,
canonical per-instance macro-F1, paired bootstrap, Holm across the 3-lang family)
so the numbers are directly comparable to the headline N2 result.

Usage
-----
    .venv/bin/python3 experiments/rigor/run_24_frde_provenance_rescore.py \
        --mbert-preds <s42> <s123> <s7> --gpt-preds <gpt> \
        --langs Indonesian French German
"""

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import precision_recall_fscore_support

LABELS = [0, 1]

# Canonical trailing-punct set, copied verbatim from run_08_extended_gold.py so
# the normalization here is identical to the C1 strip analysis.
PUNCT_TRAIL = set(",.;:!?।॥。．！？"
                  "'\"’‘“”"
                  "—–…"
                  ")]}】）")


def load_preds(path):
    return {json.loads(l)["sentence"]: json.loads(l)
            for l in open(path, encoding="utf-8")}


def strip_span(text, start, end):
    """Symmetric normalization: drop trailing punct + leading whitespace.
    Applied identically to gold and BOTH systems' spans. Never empties the span."""
    if start is None or end is None:
        return start, end
    # trailing punct
    while end > start and end - 1 < len(text) and text[end - 1] in PUNCT_TRAIL:
        end -= 1
    # leading whitespace (SP ▁ offset artifact)
    while start < end and start < len(text) and text[start].isspace():
        start += 1
    return start, end


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


def joint_labels(rec, pred_label_key, sentence, normalize, span_mode, threshold):
    """Return (gold_joint, pred_joint) for one record.
    span_mode: 'joint' (cls AND span) or 'cls_only' (span-agnostic).
    normalize: apply symmetric strip to gold+pred spans before overlap.
    threshold: min overlap_f1 to count span correct ('exact' → require ==1.0)."""
    gold_label = rec["idiomaticity"]
    pred_label = rec.get(pred_label_key)

    if span_mode == "cls_only":
        gold_j = 1 if gold_label == "idiomatic" else 0
        pred_j = 1 if pred_label == "idiomatic" else 0
        return gold_j, pred_j

    # joint mode
    if gold_label == "literal":
        return 0, (0 if pred_label == "literal" else 1)
    if pred_label != "idiomatic":
        return 1, 0
    gs, ge = rec["span_start"], rec["span_end"]
    ps, pe = rec.get("pred_span_start"), rec.get("pred_span_end")
    if normalize:
        gs, ge = strip_span(sentence, gs, ge)
        ps, pe = strip_span(sentence, ps, pe)
    ov = overlap_f1(ps, pe, gs, ge)
    if threshold == "exact":
        correct = (ov >= 0.999)
    else:
        correct = (ov > threshold)
    return 1, (1 if correct else 0)


def macro_f1(gold, pred):
    _p, _r, f1, _s = precision_recall_fscore_support(
        gold, pred, average=None, labels=LABELS, zero_division=0)
    return float(np.mean(f1))


def per_language_arrays(mbert_seeds, gpt, mbert_key, gpt_key, lang,
                        normalize, span_mode, threshold):
    shared = [s for s in mbert_seeds[0]
              if all(s in m for m in mbert_seeds[1:]) and s in gpt
              and mbert_seeds[0][s]["language"] == lang]
    gold, pm, pg = [], [], []
    for s in shared:
        seed_results = [joint_labels(m[s], mbert_key, s, normalize, span_mode, threshold)
                        for m in mbert_seeds]
        gj = seed_results[0][0]
        seed_preds = [pj for _, pj in seed_results]
        majority = 1 if sum(seed_preds) >= 2 else 0
        _gj_g, pj_g = joint_labels(gpt[s], gpt_key, s, normalize, span_mode, threshold)
        gold.append(gj); pm.append(majority); pg.append(pj_g)
    return np.array(gold), np.array(pm), np.array(pg)


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


def run_condition(mbert_seeds, gpt, mbert_key, gpt_key, langs, label,
                  normalize, span_mode, threshold, alpha, n_boot):
    rows = []
    for lang in langs:
        gold, pm, pg = per_language_arrays(
            mbert_seeds, gpt, mbert_key, gpt_key, lang, normalize, span_mode, threshold)
        if len(gold) == 0:
            continue
        point, lo, hi, p = paired_bootstrap_p(gold, pm, pg, n_boot=n_boot)
        rows.append({"lang": lang, "n": int(len(gold)),
                     "mbert_f1": round(macro_f1(gold, pm), 4),
                     "gpt_f1": round(macro_f1(gold, pg), 4),
                     "diff": round(point, 4), "ci": [round(lo, 4), round(hi, 4)],
                     "p_raw": round(p, 5)})
    pvals = np.array([r["p_raw"] for r in rows])
    p_holm = holm(pvals)
    for r, ph in zip(rows, p_holm):
        r["p_holm"] = round(float(ph), 5)
        r["sig_holm"] = bool(ph < alpha)
    n_sig = sum(r["sig_holm"] for r in rows)
    print(f"\n=== {label} ===")
    print(f"  {'lang':11}{'n':>5}{'mBERT':>8}{'GPT':>8}{'diff':>8}{'95% CI':>20}{'p_holm':>9}  sig")
    for r in rows:
        ci = f"[{r['ci'][0]:+.3f},{r['ci'][1]:+.3f}]"
        print(f"  {r['lang']:11}{r['n']:>5}{r['mbert_f1']:>8.3f}{r['gpt_f1']:>8.3f}"
              f"{r['diff']:>+8.3f}{ci:>20}{r['p_holm']:>9.4f}  {'Y' if r['sig_holm'] else 'n'}")
    print(f"  → {n_sig}/{len(rows)} survive Holm at α={alpha}")
    return {"label": label, "normalize": normalize, "span_mode": span_mode,
            "threshold": str(threshold), "rows": rows, "n_sig_holm": n_sig,
            "family_size": len(rows)}


def main():
    ap = argparse.ArgumentParser()
    repo = Path(__file__).resolve().parents[2]
    ap.add_argument("--mbert-preds", nargs=3, required=True)
    ap.add_argument("--gpt-preds", required=True)
    ap.add_argument("--mbert-key", default="pred_idiomaticity")
    ap.add_argument("--gpt-key", default="pred_label")
    ap.add_argument("--langs", nargs="+", default=["Indonesian", "French", "German"])
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--n-bootstrap", type=int, default=10000)
    ap.add_argument("--out", default=str(repo / "experiments/rigor/results/run_24_frde_provenance_rescore.json"))
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists() and not args.force:
        raise SystemExit(f"{out_path} exists — pass --force (idempotency rule).")

    mbert_seeds = [load_preds(p) for p in args.mbert_preds]
    gpt = load_preds(args.gpt_preds)

    conditions = []
    # Baseline reproduction (raw, lenient overlap>0) — should match the headline N2.
    conditions.append(run_condition(mbert_seeds, gpt, args.mbert_key, args.gpt_key, args.langs,
        "BASELINE joint, raw spans, overlap>0 (reproduces headline N2)",
        normalize=False, span_mode="joint", threshold=0.0, alpha=args.alpha, n_boot=args.n_bootstrap))
    # Symmetric normalization, lenient — removes convention advantage, keeps lenient overlap.
    conditions.append(run_condition(mbert_seeds, gpt, args.mbert_key, args.gpt_key, args.langs,
        "SYMMETRIC-NORM joint, overlap>0 (both systems stripped identically)",
        normalize=True, span_mode="joint", threshold=0.0, alpha=args.alpha, n_boot=args.n_bootstrap))
    # Symmetric normalization, strict overlap 0.5 — boundary-convention-sensitive test.
    conditions.append(run_condition(mbert_seeds, gpt, args.mbert_key, args.gpt_key, args.langs,
        "SYMMETRIC-NORM joint, overlap>=0.5 (strict, boundary-sensitive)",
        normalize=True, span_mode="joint", threshold=0.5, alpha=args.alpha, n_boot=args.n_bootstrap))
    # Symmetric normalization, exact-after-norm — hardest span condition.
    conditions.append(run_condition(mbert_seeds, gpt, args.mbert_key, args.gpt_key, args.langs,
        "SYMMETRIC-NORM joint, EXACT-after-norm (hardest span)",
        normalize=True, span_mode="joint", threshold="exact", alpha=args.alpha, n_boot=args.n_bootstrap))
    # Classification-only — the decisive home-field test (span convention irrelevant).
    conditions.append(run_condition(mbert_seeds, gpt, args.mbert_key, args.gpt_key, args.langs,
        "CLS-ONLY, span-agnostic (decisive: if mBERT still wins, not a span artifact)",
        normalize=False, span_mode="cls_only", threshold=0.0, alpha=args.alpha, n_boot=args.n_bootstrap))

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"conditions": conditions}, f, ensure_ascii=False, indent=2)
    print(f"\nSummary JSON -> {out_path}")

    # Verdict
    baseline_sig = conditions[0]["n_sig_holm"]
    norm_sig = conditions[1]["n_sig_holm"]
    strict_sig = conditions[2]["n_sig_holm"]
    cls_sig = conditions[4]["n_sig_holm"]
    m = conditions[0]["family_size"]
    print("\n" + "=" * 70)
    print("VERDICT (home-field-advantage stress test):")
    print(f"  baseline joint:        {baseline_sig}/{m} survive Holm")
    print(f"  symmetric-norm joint:  {norm_sig}/{m}")
    print(f"  strict overlap>=0.5:   {strict_sig}/{m}")
    print(f"  CLS-ONLY (decisive):   {cls_sig}/{m}")
    if cls_sig == m and norm_sig == m:
        print("  → mBERT lead SURVIVES symmetric normalization AND holds on span-agnostic")
        print("    classification. The win is NOT a span-convention home-field artifact.")
        print("    Residual: label-provenance (needs the human-gold spot-check).")
    elif cls_sig < baseline_sig or norm_sig < baseline_sig:
        print("  → mBERT lead SHRINKS under normalization/cls-only — home-field advantage")
        print("    is partly real. Do NOT anchor §1 on the raw joint numbers; investigate.")


if __name__ == "__main__":
    main()
