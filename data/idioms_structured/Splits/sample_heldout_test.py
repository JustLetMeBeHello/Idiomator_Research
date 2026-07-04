"""
sample_heldout_test.py

Build a BALANCED held-out zero-shot test set for new languages (N2 extension:
French, German). Unlike the training languages, these are never in train — the
whole gold set is valid test — so there is NO 80/10/10 split (that would waste
90% of the data). We only balance classes and cap n, mirroring the Indonesian
anchor's ~50/50 composition (156 idiomatic / 172 literal, n=328).

Same output schema as Update_indonesian.py / Sampler.py's test rows, so
infer_shared_test.py and run_14/run_14b consume it unchanged.

Drops span_flagged rows and null-span rows (unusable for span scoring), then
samples N_PER_CLASS idiomatic + N_PER_CLASS literal per language, seed 42.

Usage:
  python sample_heldout_test.py                 # defaults: fr+de, 400/class
  python sample_heldout_test.py --n_per_class 400 --langs French German
"""

import argparse
import json
import random
from collections import Counter
from pathlib import Path

DATA_DIR = Path("idioms_structured/Span_tagged_data")
OUT_PATH = Path("idioms_structured/Splits/test_heldout_fr_de.jsonl")

FILES = {
    "French": DATA_DIR / "French/Final_French_MERGED.jsonl",
    "German": DATA_DIR / "German/Final_German_MERGED.jsonl",
}


def load_and_flatten(path, lang):
    """One dict per usable example, matching Update_indonesian.py's schema."""
    out = []
    for line in open(path, encoding="utf-8"):
        if not line.strip():
            continue
        r = json.loads(line)
        for ex in (r.get("examples") or []):
            if ex.get("span_flagged"):
                continue
            if ex.get("span_start") is None or ex.get("span_end") is None:
                continue
            out.append({
                "language":     lang,
                "idiom_id":     r.get("idiom_id"),
                "idiom":        r.get("idiom"),
                "meaning_id":   r.get("meaning_id"),
                "sense_number": r.get("sense_number"),
                "idiomaticity": (r.get("Idiomaticity") or "").lower(),
                "register":     r.get("Register", []),
                "region":       r.get("Region", []),
                "sentence":     ex.get("sentence"),
                "span_start":   ex.get("span_start"),
                "span_end":     ex.get("span_end"),
                "matched_span": ex.get("matched_span"),
            })
    return out


def balanced_sample(examples, n_per_class, rng):
    by_class = {"idiomatic": [], "literal": []}
    for ex in examples:
        if ex["idiomaticity"] in by_class:
            by_class[ex["idiomaticity"]].append(ex)
    sampled = []
    for cls in ("idiomatic", "literal"):
        pool = by_class[cls]
        take = min(n_per_class, len(pool))
        if take < n_per_class:
            print(f"    * WARNING: wanted {n_per_class} {cls}, pool only {len(pool)} — using all.")
        sampled.extend(rng.sample(pool, take))
    return sampled


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_per_class", type=int, default=400)
    ap.add_argument("--langs", nargs="+", default=list(FILES))
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default=str(OUT_PATH))
    args = ap.parse_args()

    rng = random.Random(args.seed)
    all_rows = []
    for lang in args.langs:
        path = FILES[lang]
        if not path.exists():
            raise SystemExit(f"Missing gold: {path}")
        flat = load_and_flatten(path, lang)
        sampled = balanced_sample(flat, args.n_per_class, rng)
        c = Counter(e["idiomaticity"] for e in sampled)
        n_idioms = len({e["idiom_id"] for e in sampled})
        print(f"{lang}: pool {len(flat)} -> sampled {len(sampled)} "
              f"(idiomatic {c['idiomatic']}, literal {c['literal']}, {n_idioms} unique idioms)")
        all_rows.extend(sampled)

    rng.shuffle(all_rows)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        for r in all_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"\nWrote {len(all_rows)} held-out test rows -> {out}")


if __name__ == "__main__":
    main()
