"""
run_38_squad_prep.py  —  Paper B experiment 6.21 (data prep)

Builds a SQuAD v1.1 extractive-QA task in the JSONL schema Train_Join.py / BiO_Task_mBERT_train.py consume,
so the QA-vs-BIO strip/extend diagnostic (run_08) runs unchanged on the standard QA benchmark.

Both trainers tokenize ONE string, so each example is  "<question> <answer sentence>"  with the answer's
character offsets shifted into the sentence part. The context is the single sentence that contains the answer
(not the paragraph: trainers cap at 128 tokens). QA and BIO see identical input; only the head differs. This is
a simplification of full SQuAD and is stated as such in the paper.

Filters (so exactly one unambiguous gold span exists): all gold answers of a dev question must be the same
string; the answer must sit inside one sentence and match the context text exactly; question <= 20 words,
sentence <= 40 words. Train/dev are split by article title (no paragraph shared across splits); test is
sampled from the SQuAD validation split. Pages are sampled at random offsets (SQuAD files are ordered by article).

Source: HF datasets-server for `rajpurkar/squad` (stdlib only, 429 backoff).

    python run_38_squad_prep.py --out-dir ../../data/squad_sent/Splits
    python run_38_squad_prep.py --selftest

Writes train/dev/test.jsonl + meta.json (counts, punct-follow, span-word fraction, suggested O_WEIGHT by the
same weighted-ratio rule as run_33).
"""
import argparse
import json
import random
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_33_absa_data_prep import TARGET_RATIO, get, stats  # noqa: E402

ROWS = "https://datasets-server.huggingface.co/rows?dataset=rajpurkar/squad&config=plain_text&split={}&offset={}&length=100"
TOTAL = {"train": 87599, "validation": 10570}
SENT_END = re.compile(r"(?<=[.!?])\s+(?=[A-Z\"'(\[])")


def fetch_pages(split, n_pages, rng):
    pages = rng.sample(range(0, TOTAL[split] // 100), n_pages)
    return [x["row"] for p in pages for x in json.loads(get(ROWS.format(split, p * 100)))["rows"]]


def sentence_bounds(context):
    out, start = [], 0
    for m in SENT_END.finditer(context):
        out.append((start, m.start()))
        start = m.end()
    out.append((start, len(context)))
    return out


def to_example(row, max_q=20, max_s=40):
    texts, starts = row["answers"]["text"], row["answers"]["answer_start"]
    if len(set(texts)) != 1:
        return None
    ctx, a_s = row["context"], starts[0]
    a_e = a_s + len(texts[0])
    if ctx[a_s:a_e] != texts[0]:
        return None
    for s, e in sentence_bounds(ctx):
        if s <= a_s and a_e <= e:
            sent = ctx[s:e]
            break
    else:
        return None
    q = row["question"].strip()
    if len(q.split()) > max_q or len(sent.split()) > max_s:
        return None
    shift = len(q) + 1 - s
    return {"sentence": q + " " + sent, "span_start": a_s + shift, "span_end": a_e + shift,
            "language": "English", "idiomaticity": "idiomatic", "qid": row["id"], "title": row["title"]}


def build(train_rows, val_rows, n_train=3000, n_dev=500, n_test=500, seed=42):
    rng = random.Random(seed)
    tr = [x for x in map(to_example, train_rows) if x]
    te = [x for x in map(to_example, val_rows) if x]
    titles = sorted({x["title"] for x in tr})
    rng.shuffle(titles)
    dev_titles, dev = set(), []
    for t in titles:
        if len(dev) >= n_dev:
            break
        dev_titles.add(t)
        dev = [x for x in tr if x["title"] in dev_titles]
    train = [x for x in tr if x["title"] not in dev_titles]
    for xs in (train, dev, te):
        rng.shuffle(xs)
    assert len(train) >= n_train and len(te) >= n_test, (len(train), len(te))
    return {"train": train[:n_train], "dev": dev[:n_dev], "test": te[:n_test]}


def selftest():
    ctx = "Denver won the game 24-10. The game was played in Santa Clara, California. Fans came."
    row = {"id": "q1", "title": "T", "context": ctx, "question": "Where was the game played?",
           "answers": {"text": ["Santa Clara, California"], "answer_start": [ctx.index("Santa")]}}
    ex = to_example(row)
    assert ex["sentence"][ex["span_start"]:ex["span_end"]] == "Santa Clara, California", ex
    assert ex["sentence"].startswith("Where was the game played? The game was played")
    bad = dict(row, answers={"text": ["Santa Clara, California", "Santa Clara"], "answer_start": [ctx.index("Santa")] * 2})
    assert to_example(bad) is None                                  # disagreeing gold answers dropped
    assert to_example(dict(row, question="word " * 25)) is None      # over-long question dropped
    print("selftest ok")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir")
    ap.add_argument("--train-pages", type=int, default=150)
    ap.add_argument("--val-pages", type=int, default=40)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    if not a.out_dir:
        ap.error("--out-dir is required")
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for name in ("train", "dev", "test"):
        if (out / f"{name}.jsonl").exists() and not a.force:
            raise SystemExit(f"{out / (name + '.jsonl')} exists — pass --force to overwrite (idempotency rule).")
    rng = random.Random(a.seed)
    splits = build(fetch_pages("train", a.train_pages, rng), fetch_pages("validation", a.val_pages, rng), seed=a.seed)
    meta = {"source": "rajpurkar/squad", "seed": a.seed, "splits": {}}
    for name, rows in splits.items():
        with open(out / f"{name}.jsonl", "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        meta["splits"][name] = stats(rows)
        print(f"{name}: {meta['splits'][name]}")
    frac = meta["splits"]["train"]["span_word_frac"]
    meta["o_weight"] = round(TARGET_RATIO * frac / (1 - frac), 3)
    json.dump(meta, open(out / "meta.json", "w"), indent=2)
    print(f"suggested O_WEIGHT={meta['o_weight']}")


if __name__ == "__main__":
    main()
