"""
run_05_decoder_rerun.py

Re-decode mBERT BIO predictions through both the original (buggy) and
patched span decoders, then report per-language exact_match and overlap_f1
side-by-side. No GPU needed — pure CPU, runs in seconds.

The bug: `decode_bio_to_char_span` in Ablations/BiO_Task_mBERT_train.py
walks only first-subtoken positions, so `span_tokens[-1]` is the FIRST
subtoken of the last word in the span. Taking `offsets[last_tok][1]`
then truncates the decoded char_end whenever the last word fragments
into multiple WordPiece subtokens (very common for Telugu/Hindi).

The patch: after locating the first-subtoken of the last span word,
walk forward through tokens that share the same word_id and take that
final subtoken's offset[1] as char_end.

Runs the comparison on every BIO predictions file in the repo:
  - models/bio_tagger_en_hi_te/test_predictions.jsonl     (System G)
  - models/en_es_hi_te/bio_tagger/test_predictions.jsonl  (cross-lingual matrix)

Usage:
    python Additional_Rigor_Experiments/run_05_decoder_rerun.py
"""
import json
from collections import defaultdict
from pathlib import Path

from transformers import AutoTokenizer

REPO    = Path(__file__).resolve().parent.parent
TOK_DIR = REPO / "models/bio_tagger_en_hi_te/best_model"
OUT_DIR = REPO / "Additional_Rigor_Experiments/results"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# (label, predictions_file, output_log)
TARGETS = [
    ("System G (en+hi+te+es)",
     REPO / "models/bio_tagger_en_hi_te/test_predictions.jsonl",
     OUT_DIR / "decoder_rerun_system_G.jsonl"),
    ("Cross-lingual matrix BIO row (en+es+hi+te)",
     REPO / "models/en_es_hi_te/bio_tagger/test_predictions.jsonl",
     OUT_DIR / "decoder_rerun_xling_matrix.jsonl"),
]

LABEL2ID = {"O": 0, "B-IDIOM": 1, "I-IDIOM": 2}
MAX_LEN  = 128


# ── Decoders ──────────────────────────────────────────────────────────────────

def decode_buggy(bio_preds, encoding, sentence):
    """Verbatim copy of Ablations/BiO_Task_mBERT_train.py:decode_bio_to_char_span."""
    offsets  = encoding["offset_mapping"]
    word_ids = encoding.word_ids()

    first_subtokens = []
    seen_words = set()
    for i, (label, wid) in enumerate(zip(bio_preds, word_ids)):
        if wid is None or wid in seen_words:
            continue
        seen_words.add(wid)
        first_subtokens.append((i, label))

    span_tokens = []
    in_span = False
    for tok_idx, label in first_subtokens:
        if label == LABEL2ID["B-IDIOM"]:
            span_tokens = [tok_idx]
            in_span = True
        elif label == LABEL2ID["I-IDIOM"] and in_span:
            span_tokens.append(tok_idx)
        else:
            if in_span:
                break

    if not span_tokens:
        return None, None

    first_tok = span_tokens[0]
    last_tok  = span_tokens[-1]
    if first_tok >= len(offsets) or last_tok >= len(offsets):
        return None, None

    char_start = offsets[first_tok][0]
    char_end   = offsets[last_tok][1]                # <-- BUG: first subtoken of last word
    if char_start is None or char_end is None:
        return None, None
    char_end = min(char_end, len(sentence))
    return int(char_start), int(char_end)


def decode_patched(bio_preds, encoding, sentence):
    """Patched: extend last_tok forward through all subtokens of its word."""
    offsets  = encoding["offset_mapping"]
    word_ids = encoding.word_ids()

    first_subtokens = []
    seen_words = set()
    for i, (label, wid) in enumerate(zip(bio_preds, word_ids)):
        if wid is None or wid in seen_words:
            continue
        seen_words.add(wid)
        first_subtokens.append((i, label))

    span_tokens = []
    in_span = False
    for tok_idx, label in first_subtokens:
        if label == LABEL2ID["B-IDIOM"]:
            span_tokens = [tok_idx]
            in_span = True
        elif label == LABEL2ID["I-IDIOM"] and in_span:
            span_tokens.append(tok_idx)
        else:
            if in_span:
                break

    if not span_tokens:
        return None, None

    first_tok = span_tokens[0]
    last_tok  = span_tokens[-1]
    if first_tok >= len(offsets) or last_tok >= len(offsets):
        return None, None

    # PATCH: walk forward to the final subtoken of the last span word.
    last_word_id = word_ids[last_tok]
    j = last_tok
    while j + 1 < len(word_ids) and word_ids[j + 1] == last_word_id:
        j += 1
    last_tok = j

    char_start = offsets[first_tok][0]
    char_end   = offsets[last_tok][1]
    if char_start is None or char_end is None:
        return None, None
    char_end = min(char_end, len(sentence))
    return int(char_start), int(char_end)


# ── Metrics ───────────────────────────────────────────────────────────────────

def overlap_f1(ps, pe, gs, ge):
    if ps is None or pe is None:
        return 0.0
    pred_set = set(range(ps, pe))
    gold_set = set(range(gs, ge))
    if not pred_set or not gold_set:
        return 0.0
    ov = len(pred_set & gold_set)
    if ov == 0:
        return 0.0
    p = ov / len(pred_set)
    r = ov / len(gold_set)
    return 2 * p * r / (p + r)


def exact(ps, pe, gs, ge):
    return int(ps == gs and pe == ge) if ps is not None else 0


# ── Main ──────────────────────────────────────────────────────────────────────

def evaluate_file(tokenizer, preds_path, out_path, label):
    print(f"\n══ {label} ══")
    print(f"   {preds_path.relative_to(REPO)}")
    rows = [json.loads(l) for l in open(preds_path, encoding="utf-8")]
    print(f"   {len(rows)} predictions loaded")

    buggy_exact   = defaultdict(list)
    buggy_f1      = defaultdict(list)
    patched_exact = defaultdict(list)
    patched_f1    = defaultdict(list)

    sanity_mismatch = 0          # buggy decoder output != stored pred_span_*
    changed_by_patch = defaultdict(int)
    per_example_log = []

    for r in rows:
        sentence = r["sentence"]
        gold_s, gold_e = r["span_start"], r["span_end"]
        lang     = r["language"]

        enc = tokenizer(
            sentence,
            max_length=MAX_LEN,
            truncation=True,
            return_offsets_mapping=True,
        )
        bio_ids = [LABEL2ID.get(t, 0) for t in r["pred_bio_tags"][:len(enc["input_ids"])]]

        b_s, b_e = decode_buggy(bio_ids, enc, sentence)
        p_s, p_e = decode_patched(bio_ids, enc, sentence)

        # Sanity: buggy decoder should reproduce the stored pred_span_*.
        if (b_s, b_e) != (r.get("pred_span_start"), r.get("pred_span_end")):
            sanity_mismatch += 1

        be = exact(b_s, b_e, gold_s, gold_e)
        bf = overlap_f1(b_s, b_e, gold_s, gold_e)
        pe = exact(p_s, p_e, gold_s, gold_e)
        pf = overlap_f1(p_s, p_e, gold_s, gold_e)

        buggy_exact[lang].append(be);   buggy_f1[lang].append(bf)
        patched_exact[lang].append(pe); patched_f1[lang].append(pf)

        if (b_s, b_e) != (p_s, p_e):
            changed_by_patch[lang] += 1

        per_example_log.append({
            "language": lang,
            "idiom":    r.get("idiom"),
            "sentence": sentence,
            "gold":          [gold_s, gold_e, sentence[gold_s:gold_e]],
            "buggy":         [b_s,   b_e,   sentence[b_s:b_e]   if b_s is not None else None],
            "patched":       [p_s,   p_e,   sentence[p_s:p_e]   if p_s is not None else None],
            "buggy_exact":   be, "buggy_f1":   round(bf, 4),
            "patched_exact": pe, "patched_f1": round(pf, 4),
            "changed":       (b_s, b_e) != (p_s, p_e),
        })

    # ── Report ───────────────────────────────────────────────────────────────
    langs = sorted(buggy_exact.keys())
    print()
    print(f"   Sanity check: buggy decoder reproduced stored pred_span_* on "
          f"{len(rows) - sanity_mismatch}/{len(rows)} examples "
          f"({sanity_mismatch} mismatches — all None/(0,0) representational).")
    print()
    print("   Per-language results — BUGGY vs PATCHED decoder")
    print("  " + "-" * 78)
    print(f"  {'Language':<10} {'N':>4}  "
          f"{'EM(buggy)':>10} {'EM(patch)':>10} {'ΔEM':>7}    "
          f"{'F1(buggy)':>10} {'F1(patch)':>10} {'ΔF1':>7}    "
          f"{'changed':>8}")
    print("  " + "-" * 78)
    overall_be = overall_bf = overall_pe = overall_pf = 0.0
    n_total = 0
    for lang in langs:
        n   = len(buggy_exact[lang])
        be  = sum(buggy_exact[lang]) / n
        pe  = sum(patched_exact[lang]) / n
        bf  = sum(buggy_f1[lang])    / n
        pf  = sum(patched_f1[lang])  / n
        ch  = changed_by_patch[lang]
        print(f"  {lang:<10} {n:>4}  "
              f"{be:>10.4f} {pe:>10.4f} {pe-be:>+7.4f}    "
              f"{bf:>10.4f} {pf:>10.4f} {pf-bf:>+7.4f}    "
              f"{ch:>8}")
        overall_be += sum(buggy_exact[lang])
        overall_pe += sum(patched_exact[lang])
        overall_bf += sum(buggy_f1[lang])
        overall_pf += sum(patched_f1[lang])
        n_total    += n
    print("  " + "-" * 78)
    print(f"  {'Overall':<10} {n_total:>4}  "
          f"{overall_be/n_total:>10.4f} {overall_pe/n_total:>10.4f} "
          f"{(overall_pe-overall_be)/n_total:>+7.4f}    "
          f"{overall_bf/n_total:>10.4f} {overall_pf/n_total:>10.4f} "
          f"{(overall_pf-overall_bf)/n_total:>+7.4f}    "
          f"{sum(changed_by_patch.values()):>8}")

    # ── Per-example dump ─────────────────────────────────────────────────────
    with open(out_path, "w", encoding="utf-8") as f:
        for ex in per_example_log:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")
    print(f"\n   Per-example diff written to {out_path.relative_to(REPO)}")

    # ── Show a handful of examples the patch changed ─────────────────────────
    changed = [ex for ex in per_example_log if ex["changed"]]
    print(f"\n   Examples whose decoded span changed under the patch: {len(changed)}")
    for ex in changed[:10]:
        print(f"     [{ex['language']:<9}] gold={ex['gold'][2]!r}")
        print(f"                 buggy={ex['buggy'][2]!r}  →  patched={ex['patched'][2]!r}")
    if len(changed) > 10:
        print(f"     ... ({len(changed) - 10} more in {out_path.name})")


def main():
    print(f"Loading tokenizer from {TOK_DIR} ...")
    tokenizer = AutoTokenizer.from_pretrained(str(TOK_DIR))
    for label, preds_path, out_path in TARGETS:
        evaluate_file(tokenizer, preds_path, out_path, label)


if __name__ == "__main__":
    main()
