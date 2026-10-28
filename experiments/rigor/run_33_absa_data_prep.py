"""
run_33_absa_data_prep.py

Builds a SemEval-2014 Task 4 aspect-term dataset (restaurants or laptops) in the
JSONL schema Train_Join.py / BiO_Task_mBERT_train.py consume, so C1's
original/extend/strip diagnostic (run_08 / run_13) runs unchanged on a second
natural-text task. ABSA-14 is the high point of the dose-response: ~33-35% of
aspect terms are followed by attached punctuation (run_32_exposure_check.py).

Raw text with character offsets only: the pre-tokenized ABSA variants (Xu et al.
2018 style) are immune to the artifact and must not be used.

The pipeline is single-span, so we keep sentences with exactly one real aspect
term. Implicit aspects (term "NULL" / from == to) are ignored when counting.
Multi-aspect sentences are dropped, not expanded: expanding would label the other
aspects as O in BIO training, which is wrong supervision.

Source: HF datasets-server for `alexcadillon/SemEval2014Task4` (stdlib only, with
429 backoff; the dataset is script-based so `datasets>=4` cannot load it).
There is no validation split upstream: dev is carved from train (seeded).

Usage:
    python run_33_absa_data_prep.py --config restaurants --out-dir ../../data/absa14_restaurants/Splits
    python run_33_absa_data_prep.py --selftest

Writes train/dev/test.jsonl plus meta.json (counts, punct-follow rates, and the
O_WEIGHT that keeps the idiom runs' weighted O:span loss balance).
"""
import argparse
import json
import random
import time
import urllib.error
import urllib.request
from pathlib import Path

CLOSERS = set(",.;:!?'\"’”)]}…।॥")  # same set as run_31 / run_32
DATASET = "alexcadillon/SemEval2014Task4"
ROWS = "https://datasets-server.huggingface.co/rows?dataset={}&config={}&split={}&offset={}&length=100"
# Weighted O:span loss ratio of the idiom runs (o_weight 0.104, 21% span words) and
# the NER runs (0.615, 62%) both work out to w*(1-f)/f ~= 0.384.
TARGET_RATIO = 0.384


def get(url, tries=8):
    for i in range(tries):
        try:
            with urllib.request.urlopen(url, timeout=60) as r:
                return r.read().decode()
        except urllib.error.HTTPError as e:
            if e.code != 429 or i == tries - 1:
                raise
            time.sleep(2 ** i)  # run_32 hit 429s on this endpoint


def fetch_rows(config, split):
    out, off = [], 0
    while True:
        page = json.loads(get(ROWS.format(DATASET, config, split, off)))
        out += [x["row"] for x in page["rows"]]
        off += 100
        if off >= page["num_rows_total"]:
            return out


def single_aspect_examples(rows):
    """One example per sentence with exactly one real aspect term."""
    out = []
    for r in rows:
        terms = [a for a in r["aspectTerms"]
                 if a["term"] != "NULL" and int(a["to"]) > int(a["from"])]
        if len(terms) != 1:
            continue
        s, e = int(terms[0]["from"]), int(terms[0]["to"])
        assert r["text"][s:e] == terms[0]["term"], (r["text"], terms[0])  # offsets trustworthy
        out.append({"sentence": r["text"], "span_start": s, "span_end": e,
                    "language": "English", "idiomaticity": "idiomatic"})
    return out


def stats(rows):
    n = len(rows)
    pf = sum(r["span_end"] < len(r["sentence"]) and r["sentence"][r["span_end"]] in CLOSERS for r in rows)
    span_w = sum(len(r["sentence"][r["span_start"]:r["span_end"]].split()) for r in rows)
    all_w = sum(len(r["sentence"].split()) for r in rows)
    return {"n": n, "punct_follow_pct": round(100 * pf / n, 1) if n else None,
            "span_word_frac": round(span_w / all_w, 3) if all_w else None}


def build(train_rows, test_rows, dev_frac=0.1, seed=42):
    train = single_aspect_examples(train_rows)
    test = single_aspect_examples(test_rows)
    random.Random(seed).shuffle(train)
    n_dev = max(1, round(dev_frac * len(train)))
    return {"train": train[n_dev:], "dev": train[:n_dev], "test": test}


def selftest():
    mk = lambda text, *terms: {"text": text, "aspectTerms": [
        {"term": text[s:e] if t != "NULL" else "NULL", "from": s, "to": e} for t, s, e in terms]}
    t = "The pizza (cheese) was great."
    rows = [mk(t, ("x", 4, 9)),                              # single -> kept
            mk(t, ("x", 4, 9), ("x", 11, 17)),               # two aspects -> dropped
            mk(t, ("NULL", 0, 0)),                           # implicit only -> dropped
            mk(t, ("x", 4, 9), ("NULL", 0, 0)),              # one real + implicit -> kept
            mk("Great food.", ("x", 6, 10))]                 # span followed by '.'
    ex = single_aspect_examples(rows)
    assert [e["sentence"][e["span_start"]:e["span_end"]] for e in ex] == ["pizza", "pizza", "food"], ex
    st = stats(ex)
    assert st["n"] == 3 and st["punct_follow_pct"] == 33.3, st
    sp = build(rows, rows, dev_frac=0.34)
    assert len(sp["dev"]) + len(sp["train"]) == 3 and len(sp["test"]) == 3
    print("selftest ok")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", choices=["restaurants", "laptops"])
    ap.add_argument("--out-dir")
    ap.add_argument("--dev-frac", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        return selftest()
    if not (args.config and args.out_dir):
        ap.error("--config and --out-dir are required")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for name in ("train", "dev", "test"):
        if (out_dir / f"{name}.jsonl").exists() and not args.force:
            raise SystemExit(f"{out_dir / (name + '.jsonl')} exists — pass --force to overwrite (idempotency rule).")

    raw = {s: fetch_rows(args.config, s) for s in ("train", "test")}
    splits = build(raw["train"], raw["test"], args.dev_frac, args.seed)
    meta = {"config": args.config, "raw_sentences": {s: len(v) for s, v in raw.items()},
            "splits": {}, "source": DATASET}
    for name, rows in splits.items():
        with open(out_dir / f"{name}.jsonl", "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        meta["splits"][name] = stats(rows)
        print(f"{name}: {meta['splits'][name]}")
    frac = meta["splits"]["train"]["span_word_frac"]
    meta["o_weight"] = round(TARGET_RATIO * frac / (1 - frac), 3)
    json.dump(meta, open(out_dir / "meta.json", "w"), indent=2)
    print(f"raw sentences: {meta['raw_sentences']}  suggested O_WEIGHT={meta['o_weight']}")


if __name__ == "__main__":
    main()
