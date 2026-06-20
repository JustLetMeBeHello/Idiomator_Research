#!/usr/bin/env python3
"""
Dataset quality audit — idiom structured data + splits.
Writes:
  Evaluation/annotation_tool/DATASET_AUDIT_REPORT.md
  Evaluation/annotation_tool/dataset_audit_metrics.json
"""

import json, collections, glob, os, sys

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(BASE, "Annotation_tool")

# ── Inventory file map ──────────────────────────────────────────────────────
INVENTORY_FILES = {
    "English": os.path.join(BASE, "idioms_structured/Span_tagged_data/English/Final_English_MERGED_normalized.jsonl"),
    "Spanish": os.path.join(BASE, "idioms_structured/Span_tagged_data/Spanish/Final_Spanish_MERGED.jsonl"),
    "Hindi":   os.path.join(BASE, "idioms_structured/Span_tagged_data/Hindi/Final_Hindi_MERGED.jsonl"),
    "Telugu":  os.path.join(BASE, "idioms_structured/Span_tagged_data/Telugu/Final_Telugu_MERGED.jsonl"),
}
# Indonesian — glob
id_glob = glob.glob(os.path.join(BASE, "idioms_structured/Span_tagged_data/Indonesian/Final_*MERGED*.jsonl"))
if id_glob:
    INVENTORY_FILES["Indonesian"] = id_glob[0]

# Split files
SPLIT_FILES = {
    "train": os.path.join(BASE, "data/idioms_structured/Splits/train.jsonl"),
    "dev":   os.path.join(BASE, "data/idioms_structured/Splits/dev.jsonl"),
    "test":  os.path.join(BASE, "data/idioms_structured/Splits/test.jsonl"),
}

# Language prefix map
LANG_PREFIX = {"en_": "English", "es_": "Spanish", "hi_": "Hindi", "te_": "Telugu", "id_": "Indonesian"}

# ── Helpers ─────────────────────────────────────────────────────────────────

def iter_jsonl(path):
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)

def get_sentence(e):
    if isinstance(e, dict):
        return e.get("sentence", "")
    return str(e)

# ── Metric 1 + 2: Inventory audit ───────────────────────────────────────────

def audit_inventory(lang, path):
    if not os.path.exists(path):
        return None

    by = collections.defaultdict(dict)  # idiom_id -> sense_number -> frozenset(sentences)
    total_records = 0

    for r in iter_jsonl(path):
        iid = r.get("idiom_id")
        sn  = r.get("sense_number")
        ex  = r.get("examples") or []
        if isinstance(ex, list):
            sents = frozenset(get_sentence(e) for e in ex if get_sentence(e))
        else:
            sents = frozenset()
        if iid is not None and sn is not None:
            by[iid][sn] = sents
            total_records += 1

    total_idioms = len(by)
    multi_sense_idioms = sum(1 for senses in by.values() if len(senses) > 1)

    # Metric 1: per-idiom duplicate sense detection
    dup_sense_set = set()   # (idiom_id, sense_number)
    for iid, senses in by.items():
        seen = set()
        for sn, sset in sorted(senses.items()):
            if not sset:
                continue
            if sset in seen:
                dup_sense_set.add((iid, sn))
            else:
                seen.add(sset)

    idioms_with_dup = len(set(iid for iid, _ in dup_sense_set))
    n_dup_records = len(dup_sense_set)

    # Metric 2: cross-sense bleed — same sentence under >=2 senses of same idiom
    bleed_sentences = 0
    bleed_idioms = set()
    for iid, senses in by.items():
        sent_to_senses = collections.defaultdict(set)
        for sn, sset in senses.items():
            for s in sset:
                sent_to_senses[s].add(sn)
        bleed = [s for s, sns in sent_to_senses.items() if len(sns) >= 2]
        if bleed:
            bleed_sentences += len(bleed)
            bleed_idioms.add(iid)

    return {
        "lang": lang,
        "total_records": total_records,
        "total_idioms": total_idioms,
        "multi_sense_idioms": multi_sense_idioms,
        "idioms_with_dup_sense": idioms_with_dup,
        "dup_sense_records": n_dup_records,
        "pct_multisense_with_dup": round(100 * idioms_with_dup / multi_sense_idioms, 2) if multi_sense_idioms else 0.0,
        "pct_records_duplicated": round(100 * n_dup_records / total_records, 2) if total_records else 0.0,
        "cross_sense_bleed_sentences": bleed_sentences,
        "cross_sense_bleed_idioms": len(bleed_idioms),
        "dup_sense_set": dup_sense_set,  # will be stripped before JSON serialise
    }

# ── Load all inventory results ───────────────────────────────────────────────

inv_results = {}
for lang, path in INVENTORY_FILES.items():
    res = audit_inventory(lang, path)
    if res is None:
        print(f"[WARN] Missing inventory file: {path}", file=sys.stderr)
    else:
        inv_results[lang] = res

# Build global dup_sense lookup: (idiom_id, sense_number) -> True
global_dup = set()
for res in inv_results.values():
    global_dup |= res["dup_sense_set"]

# ── Metric 3-6: Split audit ──────────────────────────────────────────────────

def lang_from_id(idiom_id):
    for prefix, lang in LANG_PREFIX.items():
        if str(idiom_id).startswith(prefix):
            return lang
    return "Unknown"

def audit_split(split_name, path):
    """Returns per-language breakdown within this split."""
    rows = list(iter_jsonl(path))

    # Group rows by language
    by_lang = collections.defaultdict(list)
    for r in rows:
        # prefer explicit 'language' field, else infer from idiom_id
        lang = r.get("language") or lang_from_id(r.get("idiom_id", ""))
        # normalise: "English" not "english"
        lang = lang.strip().title() if lang else "Unknown"
        by_lang[lang].append(r)

    lang_stats = {}
    for lang, lang_rows in by_lang.items():
        n = len(lang_rows)

        # Metric 3: mislabel risk
        dup_rows = [(r, r.get("idiomaticity","").lower()) for r in lang_rows
                    if (r.get("idiom_id"), r.get("sense_number")) in global_dup]
        dup_idiomatic = sum(1 for _, lab in dup_rows if lab == "idiomatic")
        dup_literal   = sum(1 for _, lab in dup_rows if lab == "literal")

        # Metric 4: span integrity
        span_flags = {
            "span_start_none": 0,
            "span_end_none": 0,
            "start_ge_end": 0,
            "end_gt_len_sentence": 0,
            "start_negative": 0,
            "matched_span_mismatch": 0,
        }
        total_bad_span = 0
        for r in lang_rows:
            ss = r.get("span_start")
            se = r.get("span_end")
            sent = r.get("sentence") or ""
            ms   = (r.get("matched_span") or "").strip()
            bad = False
            if ss is None:
                span_flags["span_start_none"] += 1; bad = True
            if se is None:
                span_flags["span_end_none"] += 1; bad = True
            if ss is not None and se is not None:
                if ss >= se:
                    span_flags["start_ge_end"] += 1; bad = True
                if se > len(sent):
                    span_flags["end_gt_len_sentence"] += 1; bad = True
                if ss < 0:
                    span_flags["start_negative"] += 1; bad = True
                if ss >= 0 and se <= len(sent) and ss < se:
                    extracted = sent[ss:se].strip()
                    if extracted != ms:
                        span_flags["matched_span_mismatch"] += 1; bad = True
            if bad:
                total_bad_span += 1

        # Metric 5: exact duplicate sentences (within this split × language)
        sent_counts = collections.Counter(r.get("sentence","") for r in lang_rows)
        dup_sent_strings = sum(1 for s, c in sent_counts.items() if c >= 2)
        redundant_rows   = sum(c - 1 for s, c in sent_counts.items() if c >= 2)

        # Metric 6: missing fields
        # 'definition' field is absent from split schema entirely — flag that
        definition_field_absent = not any("definition" in r for r in lang_rows)
        missing_def  = sum(1 for r in lang_rows if not r.get("definition"))
        missing_sn   = sum(1 for r in lang_rows if r.get("sense_number") is None)
        missing_sent = sum(1 for r in lang_rows if not r.get("sentence"))

        lang_stats[lang] = {
            "n_rows": n,
            "dup_sense_rows": len(dup_rows),
            "dup_sense_idiomatic": dup_idiomatic,
            "dup_sense_literal": dup_literal,
            "pct_on_dup_sense": round(100 * len(dup_rows) / n, 2) if n else 0.0,
            "span_flags": span_flags,
            "total_bad_span": total_bad_span,
            "dup_sent_strings": dup_sent_strings,
            "redundant_rows": redundant_rows,
            "definition_field_absent": definition_field_absent,
            "missing_definition": missing_def,
            "missing_sense_number": missing_sn,
            "missing_sentence": missing_sent,
        }
    return lang_stats

split_results = {}
for split_name, path in SPLIT_FILES.items():
    split_results[split_name] = audit_split(split_name, path)

# ── Serialisable metrics dict ────────────────────────────────────────────────

metrics = {
    "inventory": {
        lang: {k: v for k, v in res.items() if k != "dup_sense_set"}
        for lang, res in inv_results.items()
    },
    "splits": split_results,
}

os.makedirs(OUT_DIR, exist_ok=True)
with open(os.path.join(OUT_DIR, "dataset_audit_metrics.json"), "w", encoding="utf-8") as f:
    json.dump(metrics, f, indent=2, ensure_ascii=False)

# ── Collect all languages seen in splits ─────────────────────────────────────
all_split_langs = sorted(set(
    lang for sp in split_results.values() for lang in sp
))

# ── Build markdown report ────────────────────────────────────────────────────

def pct(num, den):
    if den == 0: return "N/A"
    return f"{100*num/den:.1f}%"

lines = []
A = lines.append

A("# Dataset Audit Report")
A(f"\n_Generated by run_audit.py | Data root: {BASE}_\n")

# ── Section 1: Inventory duplication ─────────────────────────────────────────
A("## 1. Sense-Example Duplication (Inventory)")
A("")
A("A 'duplicated sense' is one whose full example-set is identical to an earlier sense of the same idiom.")
A("")
A("| Language | Inventory idioms | Multi-sense idioms | % multi-sense w/ dup sense | Dup sense-records | % sense-records dup |")
A("|----------|-----------------|-------------------|---------------------------|-------------------|----------------------|")

totals = {"idioms":0,"multi":0,"idsup":0,"dup_rec":0,"total_rec":0}
for lang in ["English","Spanish","Hindi","Telugu","Indonesian"]:
    if lang not in inv_results:
        A(f"| {lang} | MISSING | — | — | — | — |")
        continue
    r = inv_results[lang]
    A(f"| {lang} | {r['total_idioms']:,} | {r['multi_sense_idioms']:,} | "
      f"{r['pct_multisense_with_dup']}% | {r['dup_sense_records']:,} | {r['pct_records_duplicated']}% |")
    totals["idioms"]   += r["total_idioms"]
    totals["multi"]    += r["multi_sense_idioms"]
    totals["idsup"]    += r["idioms_with_dup_sense"]
    totals["dup_rec"]  += r["dup_sense_records"]
    totals["total_rec"]+= r["total_records"]

tot_pct_multi = round(100*totals["idsup"]/totals["multi"],2) if totals["multi"] else 0
tot_pct_rec   = round(100*totals["dup_rec"]/totals["total_rec"],2) if totals["total_rec"] else 0
A(f"| **TOTAL** | **{totals['idioms']:,}** | **{totals['multi']:,}** | **{tot_pct_multi}%** | **{totals['dup_rec']:,}** | **{tot_pct_rec}%** |")

A("")
A("**Interpretation:** A duplicated sense means the annotator assigned the same example pool to two different senses of an idiom — this makes those two senses indistinguishable by the model and guarantees label noise wherever both senses are used. The percentage of sense-records that are duplicated tells you how much of the training signal is contaminated.")

# ── Section 2: Cross-sense bleed ─────────────────────────────────────────────
A("")
A("## 2. Cross-Sense Example Bleed (Inventory)")
A("")
A("A 'bleed sentence' appears under ≥2 different sense_numbers of the same idiom.")
A("")
A("| Language | Bleed sentences | Idioms affected |")
A("|----------|----------------|-----------------|")
tot_bleed_s = 0; tot_bleed_i = 0
for lang in ["English","Spanish","Hindi","Telugu","Indonesian"]:
    if lang not in inv_results:
        A(f"| {lang} | MISSING | — |")
        continue
    r = inv_results[lang]
    A(f"| {lang} | {r['cross_sense_bleed_sentences']:,} | {r['cross_sense_bleed_idioms']:,} |")
    tot_bleed_s += r["cross_sense_bleed_sentences"]
    tot_bleed_i += r["cross_sense_bleed_idioms"]
A(f"| **TOTAL** | **{tot_bleed_s:,}** | **{tot_bleed_i:,}** |")

A("")
A("**Interpretation:** Each bleed sentence is a training example that could legitimately carry either sense label, creating irreducible ambiguity. If these sentences end up in different splits with different sense labels, they become contradictory supervision signals.")

# ── Section 3: Mislabel risk ──────────────────────────────────────────────────
A("")
A("## 3. Mislabel-Risk: Split Examples on Duplicated Senses")
A("")
A("'Duplicated sense' = a sense whose example-set exactly matches another sense of the same idiom. An 'idiomatic-labeled on a duplicated sense' example is the headline mislabel-risk figure.")
A("")
A("| Language | Split | Rows | Rows on dup sense | % on dup sense | Idiomatic-on-dup (mislabel risk) | Literal-on-dup |")
A("|----------|-------|------|------------------|----------------|----------------------------------|----------------|")

for lang in sorted(all_split_langs):
    for split in ["train","dev","test"]:
        s = split_results.get(split, {}).get(lang)
        if s is None:
            A(f"| {lang} | {split} | 0 | 0 | — | 0 | 0 |")
            continue
        A(f"| {lang} | {split} | {s['n_rows']:,} | {s['dup_sense_rows']:,} | {s['pct_on_dup_sense']}% | {s['dup_sense_idiomatic']:,} | {s['dup_sense_literal']:,} |")

A("")
A("**Interpretation:** Rows sitting on a duplicated sense have an uncertain ground-truth label — the model may be trained/evaluated with one label but the annotation could equally support another. The idiomatic-on-dup count is the worst-case misclassification exposure.")

# ── Section 4: Span integrity ─────────────────────────────────────────────────
A("")
A("## 4. Span Integrity (Splits)")
A("")
A("Flags: span_start None, span_end None, start≥end, end>len(sentence), start<0, matched_span mismatch.")
A("")
A("| Language | Split | Rows | Bad spans | start_none | end_none | start≥end | end>len | start<0 | ms_mismatch |")
A("|----------|-------|------|-----------|-----------|---------|---------|---------|-------|-------------|")

for lang in sorted(all_split_langs):
    for split in ["train","dev","test"]:
        s = split_results.get(split, {}).get(lang)
        if s is None:
            continue
        sf = s["span_flags"]
        A(f"| {lang} | {split} | {s['n_rows']:,} | {s['total_bad_span']:,} | "
          f"{sf['span_start_none']} | {sf['span_end_none']} | {sf['start_ge_end']} | "
          f"{sf['end_gt_len_sentence']} | {sf['start_negative']} | {sf['matched_span_mismatch']} |")

A("")
A("**Interpretation:** A bad span means the model cannot reliably locate the idiom token(s) in the sentence during span-extraction or tagging tasks. matched_span mismatches are the most actionable — the stored span label disagrees with what slicing the sentence actually yields.")

# ── Section 5: Exact duplicate sentences ─────────────────────────────────────
A("")
A("## 5. Exact Duplicate Sentences (per split × language)")
A("")
A("| Language | Split | Dup sentence strings | Redundant rows |")
A("|----------|-------|--------------------|----------------|")

for lang in sorted(all_split_langs):
    for split in ["train","dev","test"]:
        s = split_results.get(split, {}).get(lang)
        if s is None:
            continue
        A(f"| {lang} | {split} | {s['dup_sent_strings']:,} | {s['redundant_rows']:,} |")

A("")
A("**Interpretation:** Redundant rows inflate evaluation counts and, if the same sentence appears with different labels, create direct contradictions. Within train, duplicates artificially upweight certain idioms.")

# ── Section 6: Missing fields ─────────────────────────────────────────────────
A("")
A("## 6. Missing / Empty Fields (Splits)")
A("")
A("**Note:** The `definition` field is entirely absent from all three split files (structural schema gap — not just empty values). This means every split row formally 'misses' a definition. The table below marks this as ABSENT rather than counting rows.")
A("")
A("| Language | Split | Rows | Definition field | Missing sense_number | Missing sentence |")
A("|----------|-------|------|-----------------|----------------------|-----------------|")

for lang in sorted(all_split_langs):
    for split in ["train","dev","test"]:
        s = split_results.get(split, {}).get(lang)
        if s is None:
            continue
        def_note = "ABSENT (all rows)" if s.get("definition_field_absent") else f"{s['missing_definition']:,}"
        A(f"| {lang} | {split} | {s['n_rows']:,} | {def_note} | {s['missing_sense_number']:,} | {s['missing_sentence']:,} |")

A("")
A("**Interpretation:** The `definition` field is absent from the split schema entirely — this is a structural gap, not just missing values. Any model or evaluation code that expects `definition` from the splits will silently get None for every row. Missing sense_numbers make metric-3 checks impossible for those examples.")

# ── Section 7: Exposure table ─────────────────────────────────────────────────
A("")
A("## 7. Exposure Summary: % Examples on Duplicated Sense per Language × Split")
A("")
splits_h = " | ".join(["train %","dev %","test %"])
A(f"| Language | {splits_h} |")
A("|----------|---------|---------|---------| ")

for lang in sorted(all_split_langs):
    row_vals = []
    for split in ["train","dev","test"]:
        s = split_results.get(split, {}).get(lang)
        if s:
            row_vals.append(f"{s['pct_on_dup_sense']}% ({s['dup_sense_rows']}/{s['n_rows']})")
        else:
            row_vals.append("—")
    A(f"| {lang} | " + " | ".join(row_vals) + " |")

A("")
A("**Interpretation:** High exposure percentages mean a non-trivial fraction of the model's learning/evaluation signal is drawn from ambiguous or duplicated senses. Corrective action: deduplicate the inventory before regenerating splits.")

# ── Output files note ─────────────────────────────────────────────────────────
A("")
A("---")
A(f"Machine-readable metrics: `{os.path.join(OUT_DIR, 'dataset_audit_metrics.json')}`")
A(f"This report: `{os.path.join(OUT_DIR, 'DATASET_AUDIT_REPORT.md')}`")

report_text = "\n".join(lines)
with open(os.path.join(OUT_DIR, "DATASET_AUDIT_REPORT.md"), "w", encoding="utf-8") as f:
    f.write(report_text)

print("DONE")
print(f"  Report : {os.path.join(OUT_DIR, 'DATASET_AUDIT_REPORT.md')}")
print(f"  Metrics: {os.path.join(OUT_DIR, 'dataset_audit_metrics.json')}")
