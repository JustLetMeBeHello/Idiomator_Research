"""
run_31_ner_data_prep.py

Builds a WikiANN single-entity dataset in the exact JSONL schema
Train_Join.py / BiO_Task_mBERT_train.py already consume, so C1's
three-mode (original/extend/strip) diagnostic (run_08_extended_gold.py /
run_13_tost_equivalence.py) can be re-run UNCHANGED on a second task.

WikiANN sentences are pre-tokenized (list of tokens + IOB2 tags over
PER/ORG/LOC). We:
  1. reconstruct sentence text by joining tokens with single spaces,
     tracking char offsets per token.
  2. collapse PER/ORG/LOC to one type — run_08/run_13 are span-offset-only
     and never look at label semantics, matching the idiom task's single
     B-IDIOM/I-IDIOM scheme.
  3. keep only sentences with exactly one entity chunk (single-span
     framing, matching the idiom task's one-gold-span-per-example design).
  4. emit {sentence, span_start, span_end, language, idiomaticity}.

Usage:
    pip install datasets
    python run_31_ner_data_prep.py --lang en --language-name English \
        --out-dir ../../data/ner_wikiann_en/Splits \
        --max-train 3000 --max-dev 500 --max-test 500

Note: the HF dataset id is 'unimelb-nlp/wikiann' — bare 'wikiann' no
longer resolves (moved org, verified 2026-08-13). Override with
--dataset-id if it moves again.
"""
import argparse
import json
import re
from pathlib import Path

from datasets import load_dataset


CLOSERS = set(",.;:!?'\"’”)]}…।॥")  # danda/double danda: Hindi sentence enders
OPENERS = set("([{‘“")


def reconstruct(tokens):
    """Detokenize: closing punctuation attaches to the previous token and
    opening brackets to the next, so `France .` becomes `France.` as in raw
    text. Space-joining (the original version) left 0% of gold spans followed
    by punctuation vs 13-23% in the idiom test sets, so the C1 artifact could
    not occur. Returns (sentence, char_spans) where char_spans[i] = (start,
    end) of tokens[i] in sentence."""
    sentence = ""
    spans = []
    for i, tok in enumerate(tokens):
        if i > 0 and not set(tok) <= CLOSERS and tokens[i - 1] not in OPENERS:
            sentence += " "
        start = len(sentence)
        sentence += tok
        spans.append((start, len(sentence)))
    return sentence, spans


def extract_single_entity(tokens, tags, tag_names):
    """Return (sentence, char_start, char_end, first_tok, last_tok) if the
    example has exactly one contiguous B-I* entity chunk (any type),
    else (sentence, None, None, None, None)."""
    # Drop wiki italic/bold markup tokens ('' ''' ``); left in, they made up
    # over half of the post-span "punctuation" and are not natural text.
    # WikiANN splits bold ''' into "'" + "''", so lone "'" tokens are markup
    # too (they were the top post-span "punctuation" in en/es/hi/id).
    kept = [(t, g) for t, g in zip(tokens, tags) if not re.fullmatch(r"['`]+", t)]
    tokens, tags = [t for t, _ in kept], [g for _, g in kept]
    sentence, char_spans = reconstruct(tokens)
    chunks = []
    cur = None
    for tok_idx, tag_id in enumerate(tags):
        name = tag_names[tag_id]
        if name.startswith("B-"):
            if cur is not None:
                chunks.append(cur)
            cur = [tok_idx, tok_idx]
        elif name.startswith("I-") and cur is not None:
            cur[1] = tok_idx
        else:
            if cur is not None:
                chunks.append(cur)
            cur = None
    if cur is not None:
        chunks.append(cur)
    if len(chunks) != 1:
        return sentence, None, None, None, None
    first, last = chunks[0]
    cs, ce = char_spans[first][0], char_spans[last][1]
    expected = reconstruct(tokens[first:last + 1])[0]
    assert sentence[cs:ce] == expected, (
        f"span mismatch: {sentence[cs:ce]!r} vs {expected!r}")
    return sentence, cs, ce, first, last


def build_split(hf_split, tag_names, language_name, limit):
    rows = []
    for ex in hf_split:
        sentence, cs, ce, ft, lt = extract_single_entity(
            ex["tokens"], ex["ner_tags"], tag_names)
        if cs is None:
            continue
        rows.append({
            "sentence": sentence,
            "span_start": cs,
            "span_end": ce,
            "language": language_name,
            "idiomaticity": "idiomatic",
        })
        if limit and len(rows) >= limit:
            break
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-id", default="unimelb-nlp/wikiann",
                     help="HF dataset id. Bare 'wikiann' no longer resolves (moved org).")
    ap.add_argument("--lang", default="en", help="WikiANN config name (ISO code).")
    ap.add_argument("--language-name", default="English",
                     help="Value written to the 'language' field (matches idiom task's naming, e.g. 'English').")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--max-train", type=int, default=3000)
    ap.add_argument("--max-dev", type=int, default=500)
    ap.add_argument("--max-test", type=int, default=500)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for split_name in ("train", "dev", "test"):
        p = out_dir / f"{split_name}.jsonl"
        if p.exists() and not args.force:
            raise SystemExit(f"{p} exists — pass --force to overwrite (idempotency rule).")

    ds = load_dataset(args.dataset_id, args.lang)
    tag_names = ds["train"].features["ner_tags"].feature.names

    splits = {
        "train": build_split(ds["train"], tag_names, args.language_name, args.max_train),
        "dev": build_split(ds["validation"], tag_names, args.language_name, args.max_dev),
        "test": build_split(ds["test"], tag_names, args.language_name, args.max_test),
    }

    for split_name, rows in splits.items():
        out_path = out_dir / f"{split_name}.jsonl"
        with open(out_path, "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"{split_name}: {len(rows)} single-entity examples -> {out_path}")


if __name__ == "__main__":
    main()
