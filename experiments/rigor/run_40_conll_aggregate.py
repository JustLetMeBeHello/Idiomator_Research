"""
run_40_conll_aggregate.py  —  Paper B experiment 6.22 (CPU): config selection + paired bootstrap / TOST

Reads the run dirs written by run_39_conll_qa_vs_bio.py under --root:
    <enc>_bio_ow<w>_s<seed>/   <enc>_mrc_pw<p>_s<seed>/      (metrics.json + test_predictions.jsonl)
and applies the PREREG_6.22 decision rules.

Arms per encoder:  A = BIO ow=1.0 argmax   B = BIO best-ow argmax   C = BIO best-ow BIO-valid Viterbi   D = MRC best-pw
Best config = highest seed-42 dev F1 (ties -> the weight closest to 1 / the smaller pos-weight).
Paired bootstrap over test sentences (same resample for every arm, seeds averaged), 95% CI for D-A, D-B, D-C;
TOST equivalence on D-C at +/-0.3 F1 points (90% CI).

    python run_40_conll_aggregate.py --root /content/drive/MyDrive/IdiomatorRigor_NER_conll --encs bert roberta xlmr
    python run_40_conll_aggregate.py --selftest
"""
import argparse
import json
from pathlib import Path

import numpy as np

OW = ["1.0", "0.5", "0.2", "0.1"]
PW = ["1.0", "5.0"]
SEEDS = [42, 123, 7]
DELTA = 0.3   # F1 points


def best_cfg(root, enc, mode):
    vals = OW if mode == "bio" else PW
    key = "ow" if mode == "bio" else "pw"
    sc = {}
    for v in vals:
        f = Path(root) / f"{enc}_{mode}_{key}{v}_s42" / "metrics.json"
        if f.exists():
            sc[v] = json.load(open(f))["dev_f1"]
    assert sc, f"no seed-42 sweep runs for {enc} {mode} under {root}"
    return max(sc, key=lambda v: (round(sc[v], 6), -abs(float(v) - 1.0)))


def select(root, enc):
    """(best O-weight for BIO, best positive-weight for MRC) from the seed-42 sweep dev F1."""
    return best_cfg(root, enc, "bio"), best_cfg(root, enc, "mrc")


def counts(path, field):
    out = []
    for l in open(path):
        r = json.loads(l)
        g, p = {tuple(x) for x in r["gold"]}, {tuple(x) for x in r[field]}
        out.append((len(g & p), len(p - g), len(g - p)))
    return np.array(out, dtype=np.int64)                       # [n_sent, (tp, fp, fn)]


def f1_from(c):                                                # c [..., 3] already summed over sentences
    return 2 * c[..., 0] / np.maximum(2 * c[..., 0] + c[..., 1] + c[..., 2], 1)


def selftest():
    c = np.array([[1, 0, 1], [1, 1, 0]])
    assert abs(float(f1_from(c.sum(0))) - 2 / 3) < 1e-9
    print("selftest ok")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root"); ap.add_argument("--encs", nargs="+")
    ap.add_argument("--n-bootstrap", type=int, default=5000)
    ap.add_argument("--out"); ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    rng = np.random.default_rng(42)
    res, verdicts = {}, {}
    for enc in a.encs:
        ow, pw = select(a.root, enc)
        spec = {"A": ("bio", "ow", "1.0", "pred"), "B": ("bio", "ow", ow, "pred"), "C": ("bio", "ow", ow, "pred_viterbi"), "D": ("mrc", "pw", pw, "pred")}
        arms = {}
        for k, (mode, key, v, field) in spec.items():
            cs = [counts(Path(a.root) / f"{enc}_{mode}_{key}{v}_s{s}" / "test_predictions.jsonl", field)
                  for s in SEEDS if (Path(a.root) / f"{enc}_{mode}_{key}{v}_s{s}" / "test_predictions.jsonl").exists()]
            arms[k] = np.stack(cs)                              # [seeds, n, 3]
        n_seeds = min(x.shape[0] for x in arms.values()); n = arms["A"].shape[1]
        assert all(x.shape[1] == n for x in arms.values()), f"{enc}: arms differ in test size {[x.shape[1] for x in arms.values()]}"
        arms = {k: v[:n_seeds] for k, v in arms.items()}
        point = {k: float(f1_from(v.sum(1)).mean()) * 100 for k, v in arms.items()}
        boot = {k: [] for k in arms}
        for _ in range(a.n_bootstrap // 500):
            idx = rng.integers(0, n, (500, n))
            for k, v in arms.items():
                boot[k].append(np.stack([f1_from(v[s][idx].sum(1)) for s in range(n_seeds)]).mean(0) * 100)
        boot = {k: np.concatenate(v) for k, v in boot.items()}
        row = {"best_ow": ow, "best_pw": pw, "seeds": n_seeds, "f1": {k: round(v, 2) for k, v in point.items()}}
        for x in "ABC":
            d = boot["D"] - boot[x]
            row[f"D-{x}"] = {"diff": round(point["D"] - point[x], 2), "ci95": [round(float(np.quantile(d, .025)), 2), round(float(np.quantile(d, .975)), 2)],
                             "ci90": [round(float(np.quantile(d, .05)), 2), round(float(np.quantile(d, .95)), 2)]}
        c_ = row["D-C"]
        row["tost_D-C"] = c_["ci90"][0] > -DELTA and c_["ci90"][1] < DELTA
        row["D_ahead_of_C"] = c_["ci95"][0] > 0 and c_["diff"] > 0.5
        res[enc] = row
        print(f"{enc}: best ow {ow}, pw {pw}, seeds {n_seeds}\n   F1 " + "  ".join(f"{k}={v:.2f}" for k, v in point.items()))
        for x in "ABC":
            print(f"   D-{x} {row[f'D-{x}']['diff']:+.2f} 95% CI {row[f'D-{x}']['ci95']}")
        print(f"   TOST D-C within ±{DELTA}: {row['tost_D-C']}   D>C by >0.5 with CI excl 0: {row['D_ahead_of_C']}")
    if all(r["tost_D-C"] for r in res.values()):
        v = "RULE 1: tuned BIO equivalent to MRC on every encoder -> reported advantage is baseline tuning (main-venue framing)"
    elif sum(r["D_ahead_of_C"] for r in res.values()) >= 2:
        v = "RULE 2: MRC still ahead on >=2 encoders -> Li's result stands on CoNLL; Paper B stays a measurement paper"
    else:
        v = "PARTIAL: neither rule met -> report as partial, no main-venue claim"
    print("\n" + v)
    if a.out:
        json.dump({"cells": res, "verdict": v}, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
