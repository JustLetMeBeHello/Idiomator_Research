"""
run_18_joint_justification.py — empirical justification for the Joint F1 metric.

Claim: reporting classification and span accuracy separately OVERSTATES joint
task capability, because the two error sets are (approximately) independent — the
examples a model misclassifies are largely different from the ones it
mislocalizes. If errors were independent, joint success = product of the
marginals; if disjoint, even lower. We show phi ~= 0 (independence) across the
three single-pass/sequential mBERT systems, 3 seeds each.

Scope: gold-idiomatic test examples only (literal examples have no span to
locate, so they reduce to classification). This is a NEW analysis — the numbers
here are not the canonical Joint F1 (that is macro-F1 over all examples, in
key_numbers.md). This measures the per-example error-overlap STRUCTURE that
motivates the metric, not the metric value itself.

Output: results/run_18_joint_justification.json + a printed table.
Reads existing test_predictions.jsonl only; CPU, seconds, no training.
"""
import json, math, os, statistics
from collections import defaultdict

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def P(rel):
    return os.path.join(REPO, "models", rel)

# Each (system, seed) -> (cls_path, span_path). Equal paths => single merged file
# (cls_correct + span_exact_match read from one file). Different paths => pipeline
# chained on 'sentence' (cls from stage_1, span from stage_2), matching the key
# convention in Evaluation/Full_evaluation.py.
SYSTEMS = {
    "E":  {  # single-pass QA (joint_mbert)
        42:  (P("en_es_hi_te/joint_mbert/test_predictions.jsonl"),) * 2,
        123: (P("main_s123/joint_mbert/test_predictions.jsonl"),) * 2,
        7:   (P("main_s7/joint_mbert/test_predictions.jsonl"),) * 2,
    },
    "E4": {  # single-pass BIO+CLS (bio_cls_joint)
        42:  (P("bio_cls_joint_mbert_s42/test_predictions.jsonl"),) * 2,
        123: (P("bio_cls_joint_mbert_s123/test_predictions.jsonl"),) * 2,
        7:   (P("bio_cls_joint_mbert_s7/test_predictions.jsonl"),) * 2,
    },
    "F":  {  # sequential QA
        42:  (P("en_es_hi_te/sequential_mbert/phase2/test_predictions.jsonl"),) * 2,
        123: (P("main_s123/sequential_mbert/stage_1_test_predictions.jsonl"),
              P("main_s123/sequential_mbert/stage_2_test_predictions.jsonl")),
        7:   (P("main_s7/sequential_mbert/stage_1_test_predictions.jsonl"),
              P("main_s7/sequential_mbert/stage_2_test_predictions.jsonl")),
    },
}

def load(path):
    return [json.loads(l) for l in open(path)]

def span_hit(rec, criterion):
    if criterion == "exact":
        return bool(rec["span_exact_match"])
    return float(rec.get("span_overlap_f1") or 0.0) > 0.0  # tau=0, matches Joint F1

def contingency(cls_path, span_path, criterion):
    """Return 2x2 counts (both, cls_only, span_only, neither) on gold-idiomatic."""
    if cls_path == span_path:
        recs = [r for r in load(cls_path) if r["idiomaticity"] == "idiomatic"]
        pairs = [(bool(r["cls_correct"]), span_hit(r, criterion)) for r in recs]
    else:
        cls = {r["sentence"]: r for r in load(cls_path)}
        spn = {r["sentence"]: r for r in load(span_path)}
        pairs = []
        for s, r in cls.items():
            if r["idiomaticity"] != "idiomatic":
                continue
            sr = spn.get(s)
            pairs.append((bool(r["cls_correct"]),
                          span_hit(sr, criterion) if sr else False))
    a = sum(1 for c, s in pairs if c and s)
    b = sum(1 for c, s in pairs if c and not s)
    c = sum(1 for c, s in pairs if not c and s)
    d = sum(1 for c, s in pairs if not c and not s)
    return a, b, c, d

def stats(a, b, c, d):
    n = a + b + c + d
    p_cls, p_span, p_both = (a + b) / n, (a + c) / n, a / n
    denom = math.sqrt((a + b) * (c + d) * (a + c) * (b + d))
    phi = (a * d - b * c) / denom if denom > 0 else float("nan")
    return dict(n=n, a=a, b=b, c=c, d=d, cls_acc=p_cls, span_acc=p_span,
                joint=p_both, indep=p_cls * p_span, disjoint=(b + c) / n, phi=phi)

def agg(vals):
    return (round(statistics.mean(vals), 4),
            round(statistics.stdev(vals), 4) if len(vals) > 1 else 0.0)

def run(criterion="exact"):
    out = {}
    for sysname, seeds in SYSTEMS.items():
        per_seed = {}
        for seed, (cp, sp) in seeds.items():
            per_seed[seed] = stats(*contingency(cp, sp, criterion))
        keys = ["cls_acc", "span_acc", "joint", "indep", "disjoint", "phi"]
        out[sysname] = {
            "per_seed": per_seed,
            "mean_std": {k: agg([per_seed[s][k] for s in per_seed]) for k in keys},
        }
    return out

def _selfcheck():
    # Seed-42 System E must reproduce the values reported in the figure.
    r = stats(*contingency(*SYSTEMS["E"][42], "exact"))
    assert abs(r["joint"] - 0.50) < 0.01 and abs(r["cls_acc"] - 0.76) < 0.01, r
    assert r["phi"] < 0.15, f"phi should be near zero, got {r['phi']}"

if __name__ == "__main__":
    _selfcheck()
    results = {crit: run(crit) for crit in ("exact", "overlap")}
    outdir = os.path.join(REPO, "results")
    os.makedirs(outdir, exist_ok=True)
    outpath = os.path.join(outdir, "run_18_joint_justification.json")
    with open(outpath, "w") as f:
        json.dump(results, f, indent=2)

    for crit in ("exact", "overlap"):
        print(f"\n=== span criterion: {crit} (3-seed mean +/- std, gold-idiomatic) ===")
        print(f"{'Sys':<4}{'clsAcc':>14}{'spanAcc':>14}{'JOINT':>14}"
              f"{'indep':>10}{'disjointErr':>14}{'phi':>12}")
        for sysname, d in results[crit].items():
            m = d["mean_std"]
            f2 = lambda k: f"{m[k][0]:.2f}+/-{m[k][1]:.2f}"
            print(f"{sysname:<4}{f2('cls_acc'):>14}{f2('span_acc'):>14}"
                  f"{f2('joint'):>14}{m['indep'][0]:>10.2f}"
                  f"{f2('disjoint'):>14}{f2('phi'):>12}")
    print(f"\nwrote {outpath}")
    print("clsAcc/spanAcc = marginal subtask accuracy; JOINT = P(both); "
          "indep = clsAcc*spanAcc; disjointErr = (cls-only + span-only)/n; "
          "phi ~ 0 => independent errors")
