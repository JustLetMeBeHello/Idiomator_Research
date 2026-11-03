"""
run_34_metric_exposure.py  —  Paper B experiment 6.13 (CPU, no GPU)

Which standard span metrics expose the QA-vs-BIO trailing-punctuation artifact?
Rescores EXISTING test_predictions.jsonl (same layout as run_08, original gold,
no strip/extend) under five metrics and reports the paired QA - BIO gap per
metric, pooled over encoders of a family and over seeds.

Metrics (all binary per example except token_f1):
  strict_em   char-offset exact match (run_08 "original")
  squad_em    SQuAD normalize_answer(gold_text) == normalize_answer(pred_text)
              (lower, ASCII string.punctuation removed, articles removed)
  token_f1    SQuAD token-F1 on normalized tokens
  cuad_jacc   CUAD official hit: Jaccard of normalized word sets >= 0.5
  word_em     CoNLL/seqeval-style single-entity EM on whitespace words: a word is
              tagged if it overlaps the span; entity matches iff tagged word sets
              are identical (seqeval entity-level equals this for one span)

Usage (reads the Drive mount; BBPE=1 is set internally for the byte-BPE sources):
    .venv/bin/python3 experiments/rigor/run_34_metric_exposure.py \
        --drive "$HOME/Library/CloudStorage/GoogleDrive-.../My Drive" [--force]
"""
import argparse, collections, json, os, re, string, sys
from pathlib import Path

os.environ.setdefault("BBPE", "1")
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_08_extended_gold as r  # noqa: E402

ROOTS = ["NER_multi", "NER_lr3e5", "NER_absa_restaurants", "NER_absa_laptops", "NER_en_bbpe", "NER_en_sj", "NER_squad", "NER_squad_ow1"]
METRICS = ["strict_em", "squad_em", "token_f1", "cuad_jacc", "word_em"]
_PUNCT = set(string.punctuation)


def normalize_answer(s):
    s = s.lower()
    s = "".join(ch for ch in s if ch not in _PUNCT)
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def token_f1(g, p):
    gt, pt = g.split(), p.split()
    common = collections.Counter(gt) & collections.Counter(pt)
    ns = sum(common.values())
    if not gt or not pt:
        return float(gt == pt)
    if ns == 0:
        return 0.0
    pr, rc = ns / len(pt), ns / len(gt)
    return 2 * pr * rc / (pr + rc)


def jaccard_hit(g, p):
    a, b = set(g.split()), set(p.split())
    if not a and not b:
        return 1.0
    return float(len(a & b) / len(a | b) >= 0.5)


def word_set(text, s, e):
    if s is None or e is None:
        return frozenset()
    return frozenset(m.start() for m in re.finditer(r"\S+", text) if m.start() < e and m.end() > s)


def score(row):
    t = row["sentence"]
    gs, ge = row["span_start"], row["span_end"]
    ps, pe = r._strip_sp_leading_space(t, row.get("pred_span_start"), row.get("pred_span_end"))
    gt = t[gs:ge]
    pt = t[ps:pe] if ps is not None and pe is not None else ""
    ng, np_ = normalize_answer(gt), normalize_answer(pt)
    return dict(
        strict_em=r.exact(ps, pe, gs, ge),
        squad_em=float(ng == np_),
        token_f1=token_f1(ng, np_),
        cuad_jacc=jaccard_hit(ng, np_),
        word_em=float(word_set(t, gs, ge) == word_set(t, ps, pe)),
    )


def selftest():
    row = dict(sentence="Marion, the best", span_start=0, span_end=6, pred_span_start=0, pred_span_end=7)
    s = score(row)
    assert s["strict_em"] == 0 and s["squad_em"] == 1 and s["cuad_jacc"] == 1 and s["word_em"] == 1, s
    row2 = dict(sentence="a b c", span_start=0, span_end=1, pred_span_start=2, pred_span_end=3)
    assert score(row2)["token_f1"] == 0 and score(row2)["word_em"] == 0
    print("selftest ok")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--drive")
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 7])
    ap.add_argument("--n-bootstrap", type=int, default=10000)
    ap.add_argument("--out", default=str(Path(__file__).resolve().parent / "results" / "run_34_metric_exposure.json"))
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists() and not a.force:
        raise SystemExit(f"{out} exists — pass --force to overwrite (idempotency rule).")

    diffs = collections.defaultdict(lambda: {m: [] for m in METRICS})   # (root,lang,family) -> metric -> paired diffs
    qa_bio = collections.defaultdict(lambda: {m: [[], []] for m in METRICS})
    for root in ROOTS:
        base = Path(os.path.expanduser(a.drive)) / ("IdiomatorRigor_" + root)
        for enc, (_sub, _tmpl, fam) in r.SOURCES.items():
            for seed in a.seeds:
                qp, bp = r.pred_path(base, enc, "joint", seed), r.pred_path(base, enc, "bio", seed)
                if not (qp.exists() and bp.exists()):
                    continue
                qa = {r.row_key(x): x for x in r.load_rows(qp)}
                bio = {r.row_key(x): x for x in r.load_rows(bp)}
                for k in qa.keys() & bio.keys():
                    sq, sb = score(qa[k]), score(bio[k])
                    key = (root, qa[k]["language"], fam)
                    for m in METRICS:
                        diffs[key][m].append(sq[m] - sb[m])
                        qa_bio[key][m][0].append(sq[m]); qa_bio[key][m][1].append(sb[m])

    rng = np.random.default_rng(42)
    res = {}
    print(f"{'task':22}{'lang':11}{'family':14}{'n':>6}" + "".join(f"{m:>22}" for m in METRICS))
    for key in sorted(diffs):
        n = len(diffs[key][METRICS[0]])
        cells = {}
        for m in METRICS:
            d = np.asarray(diffs[key][m], float)
            bs = d[rng.integers(0, n, (a.n_bootstrap, n))].mean(1)
            cells[m] = dict(qa=round(float(np.mean(qa_bio[key][m][0])), 4), bio=round(float(np.mean(qa_bio[key][m][1])), 4),
                            gap=round(float(d.mean()), 4), ci=[round(float(np.quantile(bs, .025)), 4), round(float(np.quantile(bs, .975)), 4)])
        res["|".join(key)] = dict(n=n, metrics=cells)
        print(f"{key[0]:22}{key[1]:11}{key[2]:14}{n:>6}" + "".join(
            f"{cells[m]['gap']:>+9.3f}[{cells[m]['ci'][0]:+.2f},{cells[m]['ci'][1]:+.2f}]" for m in METRICS))
    json.dump(dict(seeds=a.seeds, metrics=METRICS, cells=res), open(out, "w"), indent=2)
    print(f"\nJSON -> {out}")


if __name__ == "__main__":
    main()
