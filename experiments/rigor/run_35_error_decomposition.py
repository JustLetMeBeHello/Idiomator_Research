"""
run_35_error_decomposition.py  —  Paper B experiment 6.12 + 6.16 (CPU, no GPU)

Classifies every test prediction against gold (original offsets) so QA and BIO can be
compared by HOW they fail, not just how often. Extends the 6.12 scratch script:
per-encoder rows, the `ow1` ABSA roots, and a leading-punctuation class.

Classes (first match wins):
  none        no predicted span
  exact       offsets identical
  punct_only  identical after stripping TRAILING punctuation from both gold and pred
  lead_punct  identical after also stripping LEADING opening punctuation (quotes, brackets)
  disjoint    no character overlap
  too_short   pred strictly inside gold (after trailing strip)
  too_long    pred contains gold
  shifted     overlap, neither contains the other

Output: per (root, encoder, language) the BIO-minus-QA difference in each class fraction
(positive = BIO makes that error more often), plus a CSV of raw fractions.

    .venv/bin/python3 experiments/rigor/run_35_error_decomposition.py \
        --drive "$HOME/Library/CloudStorage/GoogleDrive-.../My Drive" [--force]
"""
import argparse, collections, csv, os, sys
from pathlib import Path

os.environ.setdefault("BBPE", "1")
sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_08_extended_gold as r  # noqa: E402

ROOTS = ["NER_multi", "NER_lr3e5", "NER_absa_restaurants", "NER_absa_laptops",
         "NER_absa_restaurants_ow1", "NER_absa_laptops_ow1", "NER_en_bbpe", "NER_en_sj"]
CATS = ["exact", "punct_only", "lead_punct", "too_short", "too_long", "shifted", "disjoint", "none"]
LEAD_PUNCT = set("\"'([{“‘«¿¡")


def _strip_lead(text, s, e):
    while s < e and s < len(text) and text[s] in LEAD_PUNCT:
        s += 1
    return s, e


def classify(row):
    t, gs, ge = row["sentence"], row["span_start"], row["span_end"]
    ps, pe = row.get("pred_span_start"), row.get("pred_span_end")
    if ps is None or pe is None:
        return "none"
    ps, pe = r._strip_sp_leading_space(t, ps, pe)
    if (ps, pe) == (gs, ge):
        return "exact"
    g, p = r._strip_trailing_punct(t, gs, ge), r._strip_trailing_punct(t, ps, pe)
    if g == p:
        return "punct_only"
    if _strip_lead(t, *g) == _strip_lead(t, *p):
        return "lead_punct"
    (a, b), (c, d) = g, p
    if max(a, c) >= min(b, d):
        return "disjoint"
    if c >= a and d <= b:
        return "too_short"
    if c <= a and d >= b:
        return "too_long"
    return "shifted"


def selftest():
    mk = lambda gs, ge, ps, pe: dict(sentence='x "Marion", y', span_start=gs, span_end=ge, pred_span_start=ps, pred_span_end=pe)
    assert classify(mk(3, 9, 3, 9)) == "exact"
    assert classify(mk(3, 9, 3, 10)) == "punct_only"
    assert classify(mk(3, 9, 2, 9)) == "lead_punct"
    assert classify(mk(3, 9, 0, 1)) == "disjoint"
    assert classify(mk(3, 9, 3, 6)) == "too_short"
    assert classify(mk(3, 9, 0, 12)) == "too_long"
    print("selftest ok")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--drive")
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 7])
    ap.add_argument("--out", default=str(Path(__file__).resolve().parent / "results" / "run_35_error_decomposition.csv"))
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists() and not a.force:
        raise SystemExit(f"{out} exists — pass --force to overwrite (idempotency rule).")

    cnt = collections.defaultdict(collections.Counter)  # (root, enc, lang, system) -> class counts
    for root in ROOTS:
        base = Path(os.path.expanduser(a.drive)) / ("IdiomatorRigor_" + root)
        for enc in r.SOURCES:
            for sysm in ("joint", "bio"):
                for seed in a.seeds:
                    p = r.pred_path(base, enc, sysm, seed)
                    if p.exists():
                        for row in r.load_rows(p):
                            cnt[(root, enc, row["language"], sysm)][classify(row)] += 1

    frac = lambda c: {k: c[k] / sum(c.values()) for k in CATS}
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["root", "enc", "family", "lang", "system", "n"] + CATS)
        for (root, enc, lang, sysm), c in sorted(cnt.items()):
            fr = frac(c)
            w.writerow([root, enc, r.SOURCES[enc][2], lang, sysm, sum(c.values())] + [round(fr[k], 4) for k in CATS])

    show = [k for k in CATS if k != "exact"]
    print("BIO minus QA, fraction of test rows (positive = BIO makes that error more often)")
    print(f"{'task':25}{'enc':11}{'lang':11}{'nQA/nBIO':>10}" + "".join(f"{k:>11}" for k in show))
    for (root, enc, lang, sysm) in sorted(cnt):
        if sysm != "joint" or (root, enc, lang, "bio") not in cnt:
            continue
        q, b = cnt[(root, enc, lang, "joint")], cnt[(root, enc, lang, "bio")]
        fq, fb = frac(q), frac(b)
        print(f"{root:25}{enc:11}{lang:11}{sum(q.values()):>5}/{sum(b.values()):<4}" + "".join(f"{fb[k]-fq[k]:>+11.3f}" for k in show))
    print(f"\nCSV -> {out}")


if __name__ == "__main__":
    main()
