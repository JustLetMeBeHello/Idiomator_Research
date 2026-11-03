"""
run_37_pad_density.py  —  Paper B experiment 6.20 (data prep, CPU)

Causal test of the sparsity hypothesis (6.19). WikiANN-EN rows are dense (span = ~73% of the words) and show
no residual QA-over-BIO gap; ABSA rows are sparse (~16%) and do. Dataset is confounded with density, so
here the SAME WikiANN rows (same spans, same labels, same test examples) get neutral distractor text added
around the span until the span is a target fraction of the words. If the residual gap opens as density falls,
density (BIO's O-dominated loss) is the cause; if it stays flat, it is something about ABSA itself.

Distractors come from the ABSA train sentences already built by run_33 (domain-neutral common-noun text).
Sentences with digits or any non-initial capitalised word (possible unlabeled named entities) are excluded,
so the only entity in a padded row is the gold span. Rows already at or below the target density are left
unchanged. Padding is seeded per example, identical across encoders/systems; test rows keep their order.

    python run_37_pad_density.py --density 0.30 --out-dir ../../data/wikiann_pad_d030/Splits
    python run_37_pad_density.py --selftest

Writes train/dev/test.jsonl + meta.json (realised density, punct-follow, suggested O_WEIGHT by the same
weighted-ratio rule as run_33).
"""
import argparse
import json
import random
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_33_absa_data_prep import TARGET_RATIO, stats  # noqa: E402

DATA = Path(__file__).resolve().parents[2] / "data"


def clean_pool(sentences):
    """Distractor sentences that cannot contain an unlabeled named entity."""
    ok = []
    for s in sentences:
        w = s.split()
        if len(w) >= 4 and not re.search(r"\d", s) and not any(x[0].isupper() and x != "I" for x in w[1:]):
            ok.append(s)
    return ok


def pad_row(row, density, pool, rng):
    t, s, e = row["sentence"], row["span_start"], row["span_end"]
    n_words, span_words = len(t.split()), len(t[s:e].split())
    extra = round(span_words / density) - n_words
    if extra <= 0:
        return dict(row)
    words = []
    while len(words) < extra:
        words += rng.choice(pool).split()
    words = words[:extra]
    k = round(extra * rng.random())
    left, right = " ".join(words[:k]), " ".join(words[k:])
    new = (left + " " if left else "") + t + (" " + right if right else "")
    shift = len(left) + 1 if left else 0
    out = dict(row, sentence=new, span_start=s + shift, span_end=e + shift)
    assert out["sentence"][out["span_start"]:out["span_end"]] == t[s:e]
    return out


def pad_split(rows, density, pool, seed, name):
    return [pad_row(r, density, pool, random.Random(f"{seed}-{name}-{i}")) for i, r in enumerate(rows)]


def selftest():
    pool = clean_pool(["the food was really good", "Great Pizza here", "we waited 20 minutes", "service was slow and rude"])
    assert pool == ["the food was really good", "service was slow and rude"], pool  # capital + digit rows dropped
    row = {"sentence": "Atlantic City , New Jersey", "span_start": 0, "span_end": 26, "language": "English"}
    out = pad_row(row, 0.25, pool, random.Random(0))
    seg = out["sentence"][out["span_start"]:out["span_end"]]
    assert seg == row["sentence"] and len(out["sentence"].split()) == round(5 / 0.25), out
    assert pad_row(row, 1.0, pool, random.Random(0)) == row                       # already dense enough -> unchanged
    assert pad_split([row] * 3, 0.25, pool, 1, "t") == pad_split([row] * 3, 0.25, pool, 1, "t")  # deterministic
    print("selftest ok")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--density", type=float, help="target span words / sentence words, e.g. 0.30")
    ap.add_argument("--src", default=str(DATA / "ner_wikiann_en" / "Splits"))
    ap.add_argument("--pool", nargs="+", default=[str(DATA / f"absa14_{d}" / "Splits" / "train.jsonl") for d in ("restaurants", "laptops")])
    ap.add_argument("--out-dir")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    if not (a.density and a.out_dir):
        ap.error("--density and --out-dir are required")
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for name in ("train", "dev", "test"):
        if (out / f"{name}.jsonl").exists() and not a.force:
            raise SystemExit(f"{out / (name + '.jsonl')} exists — pass --force to overwrite (idempotency rule).")
    pool = clean_pool(json.loads(l)["sentence"] for p in a.pool if Path(p).exists() for l in open(p, encoding="utf-8"))
    assert len(pool) >= 200, f"distractor pool too small ({len(pool)}): build the ABSA splits with run_33 first"

    meta = {"target_density": a.density, "seed": a.seed, "pool_size": len(pool), "splits": {}}
    for name in ("train", "dev", "test"):
        rows = [json.loads(l) for l in open(Path(a.src) / f"{name}.jsonl", encoding="utf-8")]
        padded = pad_split(rows, a.density, pool, a.seed, name)
        with open(out / f"{name}.jsonl", "w", encoding="utf-8") as f:
            for r in padded:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        meta["splits"][name] = {"before": stats(rows), "after": stats(padded)}
        print(f"{name}: {meta['splits'][name]}")
    frac = meta["splits"]["train"]["after"]["span_word_frac"]
    meta["o_weight"] = round(TARGET_RATIO * frac / (1 - frac), 3)
    json.dump(meta, open(out / "meta.json", "w"), indent=2)
    print(f"suggested O_WEIGHT={meta['o_weight']} (unpadded WikiANN ran at 0.615)")


if __name__ == "__main__":
    main()
