"""
Subset builder for the scratch curriculum.

Stdlib only — this must run on any machine, with or without torch, so that
Tier 2 and Tier 4 stay dependency-free.

Builds a small, class-balanced, language-stratified subset of the canonical
splits so every tier trains in minutes on a laptop instead of hours on Colab.
The subset is idiom-disjoint by construction because the source splits already
are (see Sampler.py, 80/10/10 at the idiom_id level).
"""

import json
import random
from pathlib import Path

REPO_ROOT   = Path(__file__).resolve().parents[2]
SPLITS_DIR  = REPO_ROOT / "data" / "idioms_structured" / "Splits"
SUBSET_DIR  = Path(__file__).resolve().parents[1] / "subset"

# Keep all four in-distribution languages so cross-lingual effects are visible
# at subset scale. HI/TE are small in the real splits too, so the imbalance the
# subset shows you is the imbalance the real pipeline has.
LANGS = ["English", "Spanish", "Hindi", "Telugu"]

SEED = 42


def load_split(name):
    """Load train/dev/test as a list of dicts."""
    path = SPLITS_DIR / f"{name}.jsonl"
    with path.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def build_subset(n_per_lang_per_class=25, seed=SEED):
    """
    Balanced subset: n examples per (language x idiomaticity) cell, per split.

    Default 25 gives 25 * 2 classes * 4 langs = 200 train examples, which is the
    size the curriculum assumes. Bump it once a tier passes and you want to see
    whether a result is real or small-sample noise.
    """
    rng = random.Random(seed)
    out = {}

    for split in ("train", "dev", "test"):
        try:
            rows = load_split(split)
        except FileNotFoundError:
            print(f"  ! {split}.jsonl not found under {SPLITS_DIR} — skipping")
            continue

        buckets = {}
        for row in rows:
            key = (row.get("language"), row.get("idiomaticity"))
            buckets.setdefault(key, []).append(row)

        picked = []
        for lang in LANGS:
            for label in ("idiomatic", "literal"):
                pool = buckets.get((lang, label), [])
                if not pool:
                    print(f"  ! no {lang}/{label} rows in {split}")
                    continue
                rng.shuffle(pool)
                take = pool[:n_per_lang_per_class]
                if len(take) < n_per_lang_per_class:
                    print(f"  ! {split} {lang}/{label}: only {len(take)} "
                          f"available (wanted {n_per_lang_per_class})")
                picked.extend(take)

        rng.shuffle(picked)
        out[split] = picked

    return out


def write_subset(subset, out_dir=SUBSET_DIR):
    out_dir.mkdir(parents=True, exist_ok=True)
    for split, rows in subset.items():
        path = out_dir / f"{split}.jsonl"
        with path.open("w", encoding="utf-8") as fh:
            for row in rows:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"  wrote {len(rows):4d} rows -> {path.relative_to(REPO_ROOT)}")


def load_subset(split, out_dir=SUBSET_DIR):
    """Used by every tier. Raises a useful error if the subset isn't built."""
    path = out_dir / f"{split}.jsonl"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} missing — run:  python3 scratch/common/data.py"
        )
    with path.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def sanity_check(subset):
    """
    Assert the invariants the curriculum depends on. A silently malformed
    subset would make every downstream tier lie to you.
    """
    problems = []
    for split, rows in subset.items():
        for i, row in enumerate(rows):
            s, e = row.get("span_start"), row.get("span_end")
            sent = row.get("sentence", "")
            if s is None or e is None:
                problems.append(f"{split}[{i}]: missing span offsets")
                continue
            if not (0 <= s < e <= len(sent)):
                problems.append(f"{split}[{i}]: span {s}:{e} out of range "
                                f"for len(sentence)={len(sent)}")
                continue
            # The offsets are CHARACTER offsets. Tier 2 lives or dies on this.
            if sent[s:e] != row.get("matched_span"):
                problems.append(
                    f"{split}[{i}]: sentence[{s}:{e}]={sent[s:e]!r} != "
                    f"matched_span={row.get('matched_span')!r}")
    return problems


if __name__ == "__main__":
    print(f"Reading canonical splits from {SPLITS_DIR}")
    subset = build_subset()

    print("\nComposition:")
    for split, rows in subset.items():
        counts = {}
        for row in rows:
            counts[row["language"]] = counts.get(row["language"], 0) + 1
        print(f"  {split:5s} n={len(rows):4d}  " +
              "  ".join(f"{k}={v}" for k, v in sorted(counts.items())))

    problems = sanity_check(subset)
    if problems:
        print(f"\n!! {len(problems)} span-offset problems (first 10):")
        for p in problems[:10]:
            print("   ", p)
        print("\nThese are character-offset mismatches in the SOURCE data.")
        print("Note them — Tier 2 will make you deal with them properly.")
    else:
        print("\nSpan offsets clean: sentence[span_start:span_end] == matched_span")

    print()
    write_subset(subset)
