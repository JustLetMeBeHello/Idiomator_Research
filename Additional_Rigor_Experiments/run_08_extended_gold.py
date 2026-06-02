"""
run_08_extended_gold.py

Decisive test of the SP QA-BIO gap mechanism (HANDOFF 2026-06-02):
is the SentencePiece QA>BIO exact-match gap an artifact of *trailing
punctuation attachment*, or a genuine SP effect?

Mechanism under test: SP pre-tokenization glues trailing punctuation
(`word,` `word.` `word।`) onto the preceding token. Annotation marks bare
`word`; the BIO head predicts the whole tokenizer-word incl. punct (exact
match fails); the QA span-pointer stops before the punct (exact match
passes). This would inflate the SP QA-BIO gap without any real modelling
advantage.

We re-score EXISTING seed-42 predictions (no GPU, no retrain, pure char
arithmetic) under three gold normalisations and read off whether the gap
closes:

  original : gold (span_start, span_end) as-is.
  extend   : advance span_end forward through any run of trailing
             punctuation  -> gold grows to match the SP tokenizer-word.
             Rewards BIO. If the gap closes here, BIO was only "wrong"
             about punctuation.
  strip    : remove trailing punctuation from BOTH gold and prediction
             ends before comparison -> symmetric normalisation that
             favours neither head. The fairest single number.

Verdict per SP encoder:
  gap collapses to ~WordPiece-control level under extend/strip
        -> Scenario A: "prior QA superiority is a decoder/annotation artifact".
  gap persists
        -> Scenario B: "SP genuinely amplifies QA via punct attachment".

WordPiece encoders (mBERT, MuRIL) are the null control: their gap should
be ~0 in every mode, and must not move much under normalisation.

Inputs are the per-run `test_predictions.jsonl` files laid out exactly as
the exp-07 notebook (DeBERTa_SentencePiece_Replication.ipynb, cell 16)
expects them under a predictions root:

    <root>/rembert/rembert_{joint,bio}_s{seed}/test_predictions.jsonl
    <root>/flip/{xlmr,mbert,muril}_{joint,bio}_s{seed}/test_predictions.jsonl

On Colab (where the files already live on the mounted Drive) run:

    !python Additional_Rigor_Experiments/run_08_extended_gold.py \
        --preds-root /content/drive/MyDrive/IdiomatorRigor

Locally, point --preds-root at a directory mirroring that layout.

Seed-42 Drive file manifest (for reference / re-pull; folder ids stable):
  rembert_joint_s42  1VN44cNrmLkN2eqeTzANUN3U1WoNvSfR3
  rembert_bio_s42    1Q0IaGtbjJO3GL6uFhzzvDI3od2H0f4Gr
  xlmr_joint_s42     16u9_UcR4D8U31n7773DWVidkIWEKBWWL
  xlmr_bio_s42       1De0ogcCM_lj_IXq_bb0HX1C4_fKmmT5G
  mbert_joint_s42    1Fp1kkJ6p6Ia9ry0TxkrKhW2yleTVXgiW
  mbert_bio_s42      1QYX7lwEtOviEa2lgr1v0mrzBwfK21r2f
  muril_joint_s42    1oELX1s0PmVPIt-1D3NvdsBTaDzh3Swig
  muril_bio_s42      12UZI85fCwSHFdF54p60iJwcDj_hP_qJC

No reported metric values are hardcoded here (CLAUDE.md metric-integrity
rule): every number is computed live from the prediction files.
"""
import os
import json
import argparse
from pathlib import Path
from collections import defaultdict

import numpy as np


# ── Trailing punctuation the SP tokenizer tends to glue onto a word ─────────────
# ASCII + Devanagari danda + CJK + curly quotes + dashes + ellipsis + closers.
PUNCT_TRAIL = set(",.;:!?।॥。．！？"   # , . ; : ! ?  । ॥ 。 ．！？
                  "'\"’‘“”"                  # ' "  ’ ‘ “ ”
                  "—–…"                            # — – …
                  ")]}】）")                              # ) ] } 】 ）


# ── Encoder layout — mirrors DeBERTa_SentencePiece_Replication.ipynb cell 16 ────
#   enc -> (subdir-under-root, name-template, family)
SOURCES = {
    "rembert": ("rembert", "rembert_{system}_s{seed}", "SentencePiece"),
    "xlmr":    ("flip",    "xlmr_{system}_s{seed}",    "SentencePiece"),
    "mbert":   ("flip",    "mbert_{system}_s{seed}",   "WordPiece"),
    "muril":   ("flip",    "muril_{system}_s{seed}",   "WordPiece"),
}
LANGS_ORDER = ["English", "Spanish", "Hindi", "Telugu", "Indonesian"]
GOLD_MODES  = ["original", "extend", "strip"]


# ── Metrics — exclusive-end convention, verbatim from run_05_decoder_rerun.py ───

def overlap_f1(ps, pe, gs, ge):
    if ps is None or pe is None:
        return 0.0
    pred_set = set(range(ps, pe))
    gold_set = set(range(gs, ge))
    if not pred_set or not gold_set:
        return 0.0
    ov = len(pred_set & gold_set)
    if ov == 0:
        return 0.0
    p = ov / len(pred_set)
    r = ov / len(gold_set)
    return 2 * p * r / (p + r)


def exact(ps, pe, gs, ge):
    return int(ps == gs and pe == ge) if ps is not None else 0


# ── Bootstrap CI — verbatim from Evaluation/Full_evaluation.py:385 ──────────────
# Copied (not imported) so the script is self-contained on a fresh Colab runtime.

def bootstrap_ci(values, statistic_fn=np.mean, n_resamples=10000, ci=0.95, seed=42):
    """Bootstrap CI for a scalar statistic over a list of values.
    Returns (point_estimate, lower, upper)."""
    if len(values) == 0:
        return (0.0, 0.0, 0.0)
    rng     = np.random.default_rng(seed)
    vals    = np.array(values, dtype=float)
    point   = statistic_fn(vals)
    n       = len(vals)
    samples = [statistic_fn(rng.choice(vals, size=n, replace=True))
               for _ in range(n_resamples)]
    alpha   = (1 - ci) / 2
    lower   = float(np.quantile(samples, alpha))
    upper   = float(np.quantile(samples, 1 - alpha))
    return round(float(point), 4), round(lower, 4), round(upper, 4)


# ── Gold / prediction normalisation ─────────────────────────────────────────────

def _strip_trailing_punct(text, start, end):
    """Walk `end` back while the char just inside the span is trailing punct.
    Never crosses `start` (keeps a non-empty span)."""
    if end is None or start is None:
        return start, end
    while end > start and end - 1 < len(text) and text[end - 1] in PUNCT_TRAIL:
        end -= 1
    return start, end


def _extend_gold(text, start, end):
    """Advance gold `end` forward through a run of trailing punctuation."""
    while end < len(text) and text[end] in PUNCT_TRAIL:
        end += 1
    return start, end


def score_row(row, mode):
    """Return (exact, overlap_f1) for one prediction row under a gold mode."""
    text = row["sentence"]
    gs, ge = row["span_start"], row["span_end"]
    ps, pe = row.get("pred_span_start"), row.get("pred_span_end")

    if mode == "extend":
        gs, ge = _extend_gold(text, gs, ge)
    elif mode == "strip":
        gs, ge = _strip_trailing_punct(text, gs, ge)
        ps, pe = _strip_trailing_punct(text, ps, pe)
    # "original": leave as-is.

    return exact(ps, pe, gs, ge), overlap_f1(ps, pe, gs, ge)


# ── Loading ─────────────────────────────────────────────────────────────────────

def load_rows(path):
    return [json.loads(l) for l in open(path, encoding="utf-8")]


def row_key(row):
    """Stable per-example key for pairing QA vs BIO predictions.

    Keyed on (sentence, gold span) — present in BOTH trainers' output and
    unique per test example. Deliberately NOT idiom_id: the QA (Train_Join)
    and BIO (BiO_Task) savers don't guarantee the same id field, and a
    mismatch there would silently zero out the paired intersection.
    """
    return (row["sentence"], row["span_start"], row["span_end"])


def pred_path(root, enc, system, seed):
    subdir, tmpl, _fam = SOURCES[enc]
    return Path(root) / subdir / tmpl.format(system=system, seed=seed) / "test_predictions.jsonl"


# ── Core ────────────────────────────────────────────────────────────────────────

def evaluate_encoder(root, enc, seeds):
    """Per-language paired QA/BIO scoring for one encoder, pooled over seeds.

    Returns:
      per_lang_mode_em : {lang: {mode: {"qa": float, "bio": float}}}
      gap_values       : {lang: {mode: [per-example (qa_em - bio_em)]}}  (paired)
      n_seen           : number of seed runs actually found (joint AND bio present)
    """
    per_lang_mode_em = defaultdict(lambda: defaultdict(lambda: {"qa": [], "bio": []}))
    gap_values       = defaultdict(lambda: defaultdict(list))
    n_seen = 0

    for seed in seeds:
        qa_p  = pred_path(root, enc, "joint", seed)
        bio_p = pred_path(root, enc, "bio",   seed)
        if not (qa_p.exists() and bio_p.exists()):
            continue
        n_seen += 1
        qa_rows  = {row_key(r): r for r in load_rows(qa_p)}
        bio_rows = {row_key(r): r for r in load_rows(bio_p)}
        shared   = [k for k in qa_rows if k in bio_rows]

        for mode in GOLD_MODES:
            for k in shared:
                qa_em,  _ = score_row(qa_rows[k],  mode)
                bio_em, _ = score_row(bio_rows[k], mode)
                lang = qa_rows[k]["language"]
                per_lang_mode_em[lang][mode]["qa"].append(qa_em)
                per_lang_mode_em[lang][mode]["bio"].append(bio_em)
                gap_values[lang][mode].append(qa_em - bio_em)

    return per_lang_mode_em, gap_values, n_seen


def fmt_ci(t):
    return f"{t[0]:+.3f} [{t[1]:+.3f},{t[2]:+.3f}]"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds-root", required=True,
                    help="Root containing rembert/ and flip/ subdirs (Drive mount on Colab).")
    ap.add_argument("--seeds", type=int, nargs="+", default=[42],
                    help="Seeds to pool (default: 42 — the only one fully on Drive).")
    ap.add_argument("--n-bootstrap", type=int, default=10000)
    ap.add_argument("--out", default=None,
                    help="Output JSON summary path (default: results/run_08_extended_gold_s<seeds>.json).")
    ap.add_argument("--force", action="store_true",
                    help="Overwrite an existing summary JSON.")
    args = ap.parse_args()

    root = args.preds_root
    seed_tag = "_".join(str(s) for s in args.seeds)
    out_path = Path(args.out) if args.out else (
        Path(__file__).resolve().parent / "results" / f"run_08_extended_gold_s{seed_tag}.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists() and not args.force:
        raise SystemExit(f"{out_path} exists — pass --force to overwrite (idempotency rule).")

    print(f"preds-root: {root}")
    print(f"seeds:      {args.seeds}")
    print(f"punct set:  {''.join(sorted(PUNCT_TRAIL))!r}\n")

    summary = {"preds_root": str(root), "seeds": args.seeds, "encoders": {}}
    fam_mode_meangap = defaultdict(lambda: defaultdict(list))  # family -> mode -> [per-enc mean gap]

    header = (f"{'enc':9}{'fam':14}{'lang':11}"
              f"{'QA':>6}{'BIO':>7}{'gap':>8}   "
              f"{'QA':>6}{'BIO':>7}{'gap':>8}   "
              f"{'QA':>6}{'BIO':>7}{'gap':>8}")
    sub = (f"{'':34}{'-- original --':>21}   {'-- extend --':>21}   {'-- strip --':>21}")

    for enc, (_subdir, _tmpl, fam) in SOURCES.items():
        per_lang_mode_em, gap_values, n_seen = evaluate_encoder(root, enc, args.seeds)
        if n_seen == 0:
            print(f"[skip] {enc}: no seed runs found under {root}")
            continue
        print("=" * 100)
        print(f"{enc}  ({fam})   seeds found: {n_seen}/{len(args.seeds)}")
        print(sub)
        print(header)
        print("-" * 100)

        enc_summary = {"family": fam, "n_seeds": n_seen, "langs": {}}
        enc_mode_meangap = defaultdict(list)

        for lang in LANGS_ORDER:
            if lang not in per_lang_mode_em:
                continue
            cells = []
            lang_summary = {}
            for mode in GOLD_MODES:
                qa  = per_lang_mode_em[lang][mode]["qa"]
                bio = per_lang_mode_em[lang][mode]["bio"]
                if not qa:
                    cells.append(f"{'-':>6}{'-':>7}{'-':>8}")
                    continue
                qa_em  = float(np.mean(qa))
                bio_em = float(np.mean(bio))
                gap_pt, gap_lo, gap_hi = bootstrap_ci(
                    gap_values[lang][mode], n_resamples=args.n_bootstrap)
                cells.append(f"{qa_em:>6.2f}{bio_em:>7.2f}{gap_pt:>+8.2f}")
                enc_mode_meangap[mode].append(gap_pt)
                lang_summary[mode] = {
                    "qa_em": round(qa_em, 4), "bio_em": round(bio_em, 4),
                    "gap": gap_pt, "gap_ci": [gap_lo, gap_hi], "n": len(qa),
                }
            print(f"{enc:9}{fam:14}{lang:11}" + "   ".join(cells))
            enc_summary["langs"][lang] = lang_summary

        # encoder-level mean gap across languages, per mode
        enc_summary["mean_gap_by_mode"] = {
            m: round(float(np.mean(v)), 4) for m, v in enc_mode_meangap.items() if v}
        for m, v in enc_mode_meangap.items():
            if v:
                fam_mode_meangap[fam][m].append(float(np.mean(v)))
        print("-" * 100)
        print("  mean gap across langs: " + "   ".join(
            f"{m}={enc_summary['mean_gap_by_mode'].get(m, float('nan')):+.3f}" for m in GOLD_MODES))
        summary["encoders"][enc] = enc_summary

    # ── Family-level verdict ────────────────────────────────────────────────────
    print("\n" + "=" * 100)
    print("FAMILY MEAN GAP (across encoders, across languages)")
    print(f"{'family':16}" + "".join(f"{m:>12}" for m in GOLD_MODES))
    fam_summary = {}
    for fam, modes in fam_mode_meangap.items():
        row = {m: round(float(np.mean(modes[m])), 4) for m in GOLD_MODES if m in modes}
        fam_summary[fam] = row
        print(f"{fam:16}" + "".join(f"{row.get(m, float('nan')):>12.3f}" for m in GOLD_MODES))
    summary["family_mean_gap"] = fam_summary

    # Directional read for the SP family.
    sp = fam_summary.get("SentencePiece", {})
    wp = fam_summary.get("WordPiece", {})
    if sp and wp:
        orig = sp.get("original")
        print("\nVERDICT (SentencePiece):")
        for m in ("extend", "strip"):
            if m in sp and orig is not None:
                wp_m = wp.get(m, 0.0)
                # QA "still meaningfully wins" only if the signed SP gap stays
                # above the WP-control band. A gap that collapses to ~0 OR flips
                # negative (BIO now ahead) is artifact evidence, not genuine SP
                # advantage — so use the SIGNED gap, never abs().
                qa_still_wins = sp[m] > wp_m + 0.02
                shrink = orig - sp[m]   # how much the (signed) QA lead shrank
                tag = ("Scenario B: gap persists -> SP genuinely amplifies QA"
                       if qa_still_wins else
                       "Scenario A: gap collapses to/below WP-control level -> "
                       "QA superiority is a punctuation/decoder artifact")
                print(f"  {m:8} SP gap {sp[m]:+.3f} (orig {orig:+.3f}, "
                      f"lead shrank {shrink:+.3f}) vs WP {wp_m:+.3f}  ->  {tag}")
        summary["verdict"] = {"sp": sp, "wp": wp}

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"\nSummary JSON -> {out_path}")


if __name__ == "__main__":
    main()
