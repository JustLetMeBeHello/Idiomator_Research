"""
Build data/{language}/annotation_pool.jsonl for languages whose test split
has < 250 examples (currently Hindi & Telugu).

The pool contains the full test set + a random sample from train+dev,
enriched with the `definition` field from idioms_structured/Span_tagged_data.
The full pool is shuffled (seed=42) so annotators don't see a "test then train"
regime change.

Run from anywhere:
    python Evaluation/annotation_tool/build_annotation_pool.py
"""

import json
import random
from collections import defaultdict
from pathlib import Path

REPO          = Path(__file__).resolve().parent.parent
SPLITS_DIR    = REPO / "idioms_structured" / "Splits"
MERGED_DIR    = REPO / "idioms_structured" / "Span_tagged_data"
ANNOT_DATA    = Path(__file__).resolve().parent / "data"

TARGET_TOTAL  = 300
SEED          = 42

LANGUAGES = ["Hindi", "Telugu","Spanish","English"]


def load_jsonl(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def build_definition_map(language: str) -> dict[str, str]:
    """meaning_id -> definition (definitions[0] from MERGED file)."""
    if language == "English":
        merged_path =  MERGED_DIR / language / f"Final_{language}_MERGED_normalized.jsonl"
    else:
        merged_path = MERGED_DIR / language / f"Final_{language}_MERGED.jsonl"
    out: dict[str, str] = {}
    for row in load_jsonl(merged_path):
        defs = row.get("definitions") or []
        if defs:
            out[row["meaning_id"]] = defs[0]
    return out


def sample_balanced(pool: list[dict], n_total: int, rng: random.Random) -> list[dict]:
    """
    Mirror of Sampler.sample_split_balanced: split pool by idiomaticity class
    and pull n_total/2 from each. Falls back to all-available if a class is short.
    """
    by_class: dict[str, list[dict]] = defaultdict(list)
    for r in pool:
        by_class[r["idiomaticity"]].append(r)

    per_class = n_total // 2
    sampled: list[dict] = []
    for cls in ("idiomatic", "literal"):
        available = by_class[cls]
        pull = min(per_class, len(available))
        if pull < per_class:
            print(f"    * Warning: requested {per_class} {cls} extras, only {len(available)} available.")
        sampled.extend(rng.sample(available, pull))
    return sampled


def build_pool(language: str) -> list[dict]:
    test_path = ANNOT_DATA / language / "test.jsonl"
    test_rows = load_jsonl(test_path)
    train_rows = [r for r in load_jsonl(SPLITS_DIR / "train.jsonl") if r["language"] == language]
    dev_rows   = [r for r in load_jsonl(SPLITS_DIR / "dev.jsonl")   if r["language"] == language]
    extra_pool = train_rows + dev_rows

    # Defensive: drop any (meaning_id, sentence) already in test
    test_keys = {(r["meaning_id"], r["sentence"]) for r in test_rows}
    extra_pool = [r for r in extra_pool if (r["meaning_id"], r["sentence"]) not in test_keys]

    n_needed = max(0, TARGET_TOTAL - len(test_rows))
    n_sample = min(n_needed, len(extra_pool))

    rng = random.Random(SEED)
    sampled = sample_balanced(extra_pool, n_sample, rng)

    def_map = build_definition_map(language)
    enriched = []
    for r in sampled:
        r = dict(r)
        r["definition"] = def_map.get(r["meaning_id"], "")
        enriched.append(r)

    combined = test_rows + enriched
    rng.shuffle(combined)
    return combined


def main() -> None:
    for language in LANGUAGES:
        pool = build_pool(language)
        out_path = ANNOT_DATA / language / "annotation_pool.jsonl"
        with open(out_path, "w", encoding="utf-8") as f:
            for row in pool:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

        test_n  = sum(1 for line in open(ANNOT_DATA / language / "test.jsonl", encoding="utf-8") if line.strip())
        extra_n = len(pool) - test_n
        missing_def = sum(1 for r in pool if not r.get("definition"))
        n_idio = sum(1 for r in pool if r["idiomaticity"] == "idiomatic")
        n_lit  = sum(1 for r in pool if r["idiomaticity"] == "literal")
        print(f"{language}: wrote {len(pool)} rows -> {out_path.relative_to(REPO)}")
        print(f"  test={test_n}  extra(class-balanced)={extra_n}  "
              f"idiomatic={n_idio}  literal={n_lit}  missing_definition={missing_def}")


if __name__ == "__main__":
    main()
