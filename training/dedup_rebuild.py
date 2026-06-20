#!/usr/bin/env python3
"""
dedup_rebuild.py
Fix English idiom inventory duplication bug:
  - duplicate senses (copied example sentences, wrong label) -> remap to canonical
  - 3 broken Indonesian spans in test
Produces .dedup.jsonl outputs + DEDUP_REBUILD_DIFF.md report.
NEVER overwrites originals.
"""

import json
import os
import sys
from collections import defaultdict

BASE = "/Users/shishirmaddineni/Desktop/Idiomator_Research/Research_And_Training"

INVENTORY_IN  = f"{BASE}/idioms_structured/Span_tagged_data/English/Final_English_MERGED_normalized.jsonl"
INVENTORY_OUT = f"{BASE}/idioms_structured/Span_tagged_data/English/Final_English_MERGED_normalized.dedup.jsonl"

TEST_IN   = f"{BASE}/idioms_structured/Splits/test.jsonl"
TEST_OUT  = f"{BASE}/idioms_structured/Splits/test.dedup.jsonl"
DEV_IN    = f"{BASE}/idioms_structured/Splits/dev.jsonl"
DEV_OUT   = f"{BASE}/idioms_structured/Splits/dev.dedup.jsonl"

POOL_IN   = f"{BASE}/Evaluation/annotation_tool/data/English/annotation_pool.jsonl"
POOL_OUT  = f"{BASE}/Evaluation/annotation_tool/data/English/annotation_pool.dedup.jsonl"

REPORT    = f"{BASE}/Evaluation/annotation_tool/DEDUP_REBUILD_DIFF.md"


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def load_jsonl(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def sentence_set(examples):
    """Return frozenset of normalised sentence strings from examples list."""
    sents = set()
    for ex in examples:
        if isinstance(ex, dict):
            s = ex.get("sentence", "")
        else:
            s = str(ex)
        sents.add(s.strip())
    return frozenset(sents)


# ---------------------------------------------------------------------------
# Step 1 — build canonical map from inventory (English only)
# ---------------------------------------------------------------------------

print("Step 1: building canonical map from inventory...", flush=True)

inventory = load_jsonl(INVENTORY_IN)
print(f"  Loaded {len(inventory)} sense-records from inventory.", flush=True)

# group senses per idiom_id
by_idiom = defaultdict(list)
for rec in inventory:
    if rec["idiom_id"].startswith("en_"):
        by_idiom[rec["idiom_id"]].append(rec)

# for each idiom, group senses by example sentence set
# canon_map[(idiom_id, dup_sense_number)] = canonical_sense_number
canon_map = {}          # (idiom_id, dup_sense) -> canonical_sense
canon_info = {}         # (idiom_id, canonical_sense) -> {idiomaticity, definition}
dup_groups_sample = []  # for report

for idiom_id, senses in by_idiom.items():
    # sort by sense_number ascending so lowest = canonical
    senses_sorted = sorted(senses, key=lambda r: r["sense_number"])

    # group by example sentence frozenset
    group_by_exset = defaultdict(list)
    for sense in senses_sorted:
        key = sentence_set(sense.get("examples", []))
        group_by_exset[key].append(sense)

    for exset, group in group_by_exset.items():
        if len(group) < 2:
            continue
        # canonical = lowest sense_number (first after sort)
        canonical = group[0]
        can_sense = canonical["sense_number"]
        can_idiom = canonical.get("Idiomaticity", "")
        can_defs  = canonical.get("definitions", [])
        can_def   = can_defs[0] if can_defs else ""

        canon_info[(idiom_id, can_sense)] = {
            "idiomaticity": can_idiom.lower() if can_idiom else can_idiom,
            "definition": can_def,
        }

        for dup_sense in group[1:]:
            dup_sn = dup_sense["sense_number"]
            canon_map[(idiom_id, dup_sn)] = can_sense
            dup_groups_sample.append((idiom_id, dup_sn, can_sense))

print(f"  Duplicate senses found: {len(canon_map)}", flush=True)
print(f"  Canonical senses involved: {len(canon_info)}", flush=True)

# Build a lookup: (idiom_id, canonical_sense) -> {idiomaticity, definition}
# Also need it for senses that are canonical but not yet in canon_info
# (they don't need remapping themselves but we need their label for remappers)
# -> canon_info already has those keyed by (idiom_id, canonical_sense)

# Also build a full lookup from inventory for any (idiom_id, sense_number)
inv_lookup = {}
for rec in inventory:
    iid = rec["idiom_id"]
    sn  = rec["sense_number"]
    inv_lookup[(iid, sn)] = rec

# Fill canon_info for any canonical senses not yet populated
for (idiom_id, dup_sn), can_sn in canon_map.items():
    key = (idiom_id, can_sn)
    if key not in canon_info:
        can_rec = inv_lookup.get(key, {})
        can_idiom = can_rec.get("Idiomaticity", "")
        can_defs  = can_rec.get("definitions", [])
        can_def   = can_defs[0] if can_defs else ""
        canon_info[key] = {
            "idiomaticity": can_idiom.lower() if can_idiom else can_idiom,
            "definition": can_def,
        }


# ---------------------------------------------------------------------------
# Step 2 — produce cleaned inventory
# ---------------------------------------------------------------------------

print("Step 2: writing cleaned inventory...", flush=True)

dup_set = set(canon_map.keys())  # (idiom_id, sense_number) pairs to drop

kept = []
dropped = []
for rec in inventory:
    key = (rec["idiom_id"], rec["sense_number"])
    if key in dup_set:
        dropped.append(rec)
    else:
        kept.append(rec)

write_jsonl(INVENTORY_OUT, kept)
print(f"  Kept: {len(kept)}  Dropped: {len(dropped)}", flush=True)


# ---------------------------------------------------------------------------
# Step 3 — rebuild test + dev splits
# ---------------------------------------------------------------------------

def remap_split(rows, canon_map, canon_info, split_name):
    """Remap English rows; return (new_rows, remap_log)."""
    new_rows = []
    remap_log = []

    for row in rows:
        iid = row.get("idiom_id", "")
        sn  = row.get("sense_number")

        if not iid.startswith("en_"):
            new_rows.append(row)
            continue

        key = (iid, sn)
        if key in canon_map:
            can_sn  = canon_map[key]
            can_key = (iid, can_sn)
            info    = canon_info.get(can_key, {})

            old_sn    = sn
            old_label = row.get("idiomaticity", "")
            old_def   = row.get("definition", "")

            new_row = dict(row)
            new_row["sense_number"]  = can_sn
            new_row["idiomaticity"]  = info.get("idiomaticity", old_label)
            new_row["definition"]    = info.get("definition", old_def)

            new_rows.append(new_row)
            remap_log.append({
                "idiom_id":   iid,
                "old_sense":  old_sn,
                "new_sense":  can_sn,
                "old_label":  old_label,
                "new_label":  info.get("idiomaticity", old_label),
                "sentence":   row.get("sentence", "")[:60],
            })
        else:
            new_rows.append(row)

    return new_rows, remap_log


print("Step 3: rebuilding test + dev splits...", flush=True)

test_rows = load_jsonl(TEST_IN)
dev_rows  = load_jsonl(DEV_IN)

test_new, test_remap = remap_split(test_rows, canon_map, canon_info, "test")
dev_new,  dev_remap  = remap_split(dev_rows,  canon_map, canon_info, "dev")

assert len(test_new) == len(test_rows), "test row count mismatch!"
assert len(dev_new)  == len(dev_rows),  "dev row count mismatch!"


# ---------------------------------------------------------------------------
# Step 4 — rebuild annotation pool
# ---------------------------------------------------------------------------

print("Step 4: rebuilding annotation pool...", flush=True)

pool_rows = load_jsonl(POOL_IN)
pool_new, pool_remap = remap_split(pool_rows, canon_map, canon_info, "pool")

assert len(pool_new) == len(pool_rows), "pool row count mismatch!"


# ---------------------------------------------------------------------------
# Step 5 — fix 3 broken Indonesian spans in test
# ---------------------------------------------------------------------------

print("Step 5: fixing Indonesian spans in test...", flush=True)

TARGET_IDS = {"id_0004", "id_0022", "id_0026"}
span_fixes = []

for row in test_new:
    if row.get("idiom_id") not in TARGET_IDS:
        continue

    iid          = row["idiom_id"]
    span_start   = row.get("span_start")
    span_end     = row.get("span_end")
    matched_span = row.get("matched_span", "")
    sentence     = row.get("sentence", "")

    before = {"span_start": span_start, "span_end": span_end}

    if span_start is not None:
        # already has a value — record and skip
        span_fixes.append({
            "idiom_id": iid, "status": "already_set",
            "before": before, "after": before,
        })
        continue

    if matched_span and matched_span in sentence:
        # exact match
        new_start = sentence.find(matched_span)
        new_end   = new_start + len(matched_span)
        row["span_start"] = new_start
        row["span_end"]   = new_end
        span_fixes.append({
            "idiom_id": iid, "status": "fixed",
            "before": before,
            "after": {"span_start": new_start, "span_end": new_end},
            "matched_span": matched_span,
            "note": "exact match",
        })
    elif matched_span and matched_span.lower() in sentence.lower():
        # case-insensitive match — find actual position in original sentence
        idx = sentence.lower().find(matched_span.lower())
        new_end = idx + len(matched_span)
        row["span_start"] = idx
        row["span_end"]   = new_end
        span_fixes.append({
            "idiom_id": iid, "status": "fixed",
            "before": before,
            "after": {"span_start": idx, "span_end": new_end},
            "matched_span": matched_span,
            "note": "case-insensitive match",
        })
    else:
        # Last resort: try longest suffix of matched_span words that appears in sentence
        # (handles cases like extra words inserted in sentence, e.g. "nasi [yang] sudah menjadi bubur")
        words = matched_span.split()
        best_idx, best_text = None, None
        for start_w in range(len(words)):
            candidate = " ".join(words[start_w:])
            idx = sentence.lower().find(candidate.lower())
            if idx != -1:
                best_idx = idx
                best_text = candidate
                break
        if best_idx is not None and len(best_text) >= len(matched_span) // 2:
            new_end = best_idx + len(best_text)
            row["span_start"] = best_idx
            row["span_end"]   = new_end
            span_fixes.append({
                "idiom_id": iid, "status": "fixed_partial",
                "before": before,
                "after": {"span_start": best_idx, "span_end": new_end},
                "matched_span": matched_span,
                "note": f"partial suffix match on '{best_text}'",
            })
        else:
            span_fixes.append({
                "idiom_id": iid, "status": "flagged",
                "before": before, "after": before,
                "matched_span": matched_span,
                "reason": "matched_span not found in sentence (broken idiom phrase or wrong matched_span)",
            })


# ---------------------------------------------------------------------------
# Write outputs
# ---------------------------------------------------------------------------

print("Writing output files...", flush=True)

write_jsonl(TEST_OUT,  test_new)
write_jsonl(DEV_OUT,   dev_new)
write_jsonl(POOL_OUT,  pool_new)

print(f"  {TEST_OUT}")
print(f"  {DEV_OUT}")
print(f"  {POOL_OUT}")
print(f"  {INVENTORY_OUT}")


# ---------------------------------------------------------------------------
# Step 6 — write diff report
# ---------------------------------------------------------------------------

print("Step 6: writing diff report...", flush=True)

def label_change_summary(remap_log):
    counts = defaultdict(int)
    for r in remap_log:
        counts[f"{r['old_label']}→{r['new_label']}"] += 1
    return dict(counts)

def remap_table(remap_log):
    lines = []
    lines.append("| idiom_id | old_sense→new_sense | old_label→new_label | sentence (60 chars) |")
    lines.append("|----------|---------------------|---------------------|---------------------|")
    for r in remap_log:
        lines.append(
            f"| {r['idiom_id']} | {r['old_sense']}→{r['new_sense']} "
            f"| {r['old_label']}→{r['new_label']} "
            f"| {r['sentence'].replace('|', '/')} |"
        )
    return "\n".join(lines)

# sense-records dropped sample (up to 10)
sample_dropped = dup_groups_sample[:10]

# label change counts
test_changes = label_change_summary(test_remap)
dev_changes  = label_change_summary(dev_remap)
pool_changes = label_change_summary(pool_remap)

def count_idiom_to_literal(change_dict):
    return change_dict.get("idiomatic→literal", 0)

# row-count sanity
sanity = [
    ("inventory (kept+dropped == original)",
     len(kept) + len(dropped) == len(inventory),
     len(inventory), len(kept) + len(dropped)),
    ("test row count",
     len(test_new) == len(test_rows),
     len(test_rows), len(test_new)),
    ("dev row count",
     len(dev_new) == len(dev_rows),
     len(dev_rows), len(dev_new)),
    ("pool row count",
     len(pool_new) == len(pool_rows),
     len(pool_rows), len(pool_new)),
]

report_lines = []
report_lines.append("# DEDUP_REBUILD_DIFF — English idiom deduplication report")
report_lines.append(f"\nGenerated: 2026-06-17\n")

report_lines.append("## 1. Inventory sense-records dropped")
report_lines.append(f"\nTotal dropped: **{len(dropped)}** (duplicate senses with copied examples)")
report_lines.append(f"Total kept: **{len(kept)}**\n")
report_lines.append("### Sample of duplicated (idiom_id, dup_sense→canonical_sense) (up to 10):")
report_lines.append("| idiom_id | dup_sense | canonical_sense |")
report_lines.append("|----------|-----------|-----------------|")
for (iid, dup_sn, can_sn) in sample_dropped:
    report_lines.append(f"| {iid} | {dup_sn} | {can_sn} |")

report_lines.append("\n## 2. test.jsonl remap")
report_lines.append(f"\nRows remapped (label corrected): **{len(test_remap)}**")
report_lines.append(f"Label changes: {dict(test_changes)}")
report_lines.append(f"idiomatic→literal: **{count_idiom_to_literal(test_changes)}**\n")
if test_remap:
    report_lines.append(remap_table(test_remap))
else:
    report_lines.append("_No rows remapped._")

report_lines.append("\n## 3. dev.jsonl remap")
report_lines.append(f"\nRows remapped (label corrected): **{len(dev_remap)}**")
report_lines.append(f"Label changes: {dict(dev_changes)}")
report_lines.append(f"idiomatic→literal: **{count_idiom_to_literal(dev_changes)}**\n")
if dev_remap:
    report_lines.append(remap_table(dev_remap))
else:
    report_lines.append("_No rows remapped._")

report_lines.append("\n## 4. annotation_pool.jsonl remap")
report_lines.append(f"\nRows remapped (label corrected): **{len(pool_remap)}**")
report_lines.append(f"Label changes: {dict(pool_changes)}")
report_lines.append(f"idiomatic→literal: **{count_idiom_to_literal(pool_changes)}**\n")
if pool_remap:
    report_lines.append(remap_table(pool_remap))
else:
    report_lines.append("_No rows remapped._")

report_lines.append("\n## 5. Indonesian span fixes (id_0004, id_0022, id_0026)")
report_lines.append("\n| idiom_id | status | before (start,end) | after (start,end) | matched_span |")
report_lines.append("|----------|--------|--------------------|-------------------|--------------|")
for f in span_fixes:
    ms = f.get("matched_span", "")
    report_lines.append(
        f"| {f['idiom_id']} | {f['status']} "
        f"| ({f['before']['span_start']},{f['before']['span_end']}) "
        f"| ({f['after']['span_start']},{f['after']['span_end']}) "
        f"| {ms} |"
    )

report_lines.append("\n## 6. Row-count sanity")
report_lines.append("\n| File | Original | Output | PASS/FAIL |")
report_lines.append("|------|----------|--------|-----------|")
for name, ok, orig, out in sanity:
    pf = "PASS" if ok else "FAIL"
    report_lines.append(f"| {name} | {orig} | {out} | {pf} |")

# Inventory shrinks — separate note
inv_ok = (len(kept) + len(dropped) == len(inventory))
report_lines.append(
    f"\nNote: inventory output has {len(kept)} rows (original {len(inventory)}, "
    f"dropped {len(dropped)} duplicate senses) — shrinkage is expected."
)

report_lines.append("\n## 7. Output file paths")
report_lines.append(f"- `{INVENTORY_OUT}`")
report_lines.append(f"- `{TEST_OUT}`")
report_lines.append(f"- `{DEV_OUT}`")
report_lines.append(f"- `{POOL_OUT}`")
report_lines.append(f"- `{REPORT}` (this file)")

with open(REPORT, "w", encoding="utf-8") as f:
    f.write("\n".join(report_lines) + "\n")

print(f"  Report: {REPORT}", flush=True)


# ---------------------------------------------------------------------------
# Summary to stdout (counts only, no raw data)
# ---------------------------------------------------------------------------

print("\n=== SUMMARY ===")
print(f"Inventory sense-records dropped: {len(dropped)}")
print(f"test  remapped: {len(test_remap)}  idiomatic->literal: {count_idiom_to_literal(test_changes)}")
print(f"dev   remapped: {len(dev_remap)}  idiomatic->literal: {count_idiom_to_literal(dev_changes)}")
print(f"pool  remapped: {len(pool_remap)}  idiomatic->literal: {count_idiom_to_literal(pool_changes)}")
print("Span fixes:")
for f in span_fixes:
    print(f"  {f['idiom_id']}: {f['status']}  before={f['before']}  after={f['after']}")
print("Row-count sanity:")
all_pass = True
for name, ok, orig, out in sanity:
    pf = "PASS" if ok else "FAIL"
    all_pass = all_pass and ok
    print(f"  {name}: {pf} ({orig} -> {out})")
print(f"Overall sanity: {'PASS' if all_pass else 'FAIL'}")
print("Done.")
