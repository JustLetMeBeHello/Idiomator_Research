"""
summarize_language_ablation_matrix.py

Collects results from run_language_ablation_matrix.sh into paper-ready CSVs.

Outputs:
  - ablation_summary.csv
      Long-form rows: combo, system, metric, language, value.

  - stability_summary.csv
      One row per combo/system with mean Joint F1, std dev across target
      languages, worst-language F1, and best-worst gap.

  - transfer_matrix.csv
      Compact Joint F1 matrix: train combo x target language x system.

These tables are designed for the IdiomBERT paper's core ablation claim:
whether QA-style multi-task/span-pointer systems generalize more evenly than
pipeline or BIO-style systems under smaller multilingual training subsets.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from statistics import mean, pstdev


COMBOS = [
    "en", "es", "hi", "te",
    "en_es", "en_hi", "en_te", "es_hi", "es_te", "hi_te",
    "en_es_hi", "en_es_te", "en_hi_te", "es_hi_te",
    "en_es_hi_te",
]

COMBO_LANGS = {
    "en": ["English"],
    "es": ["Spanish"],
    "hi": ["Hindi"],
    "te": ["Telugu"],
    "en_es": ["English", "Spanish"],
    "en_hi": ["English", "Hindi"],
    "en_te": ["English", "Telugu"],
    "es_hi": ["Spanish", "Hindi"],
    "es_te": ["Spanish", "Telugu"],
    "hi_te": ["Hindi", "Telugu"],
    "en_es_hi": ["English", "Spanish", "Hindi"],
    "en_es_te": ["English", "Spanish", "Telugu"],
    "en_hi_te": ["English", "Hindi", "Telugu"],
    "es_hi_te": ["Spanish", "Hindi", "Telugu"],
    "en_es_hi_te": ["English", "Spanish", "Hindi", "Telugu"],
}

TARGET_LANGS = ["English", "Spanish", "Hindi", "Telugu"]

SYSTEM_LABELS = {
    "system_a_mbert_pipeline": "A_pipeline",
    "system_e_joint_end_to_end": "E_joint",
    "system_f_sequential_phase1_ph2": "F_sequential",
    "system_g_bio_tagger": "G_bio",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--eval_dir", default="results/language_ablation_matrix")
    p.add_argument("--output_dir", default="results/language_ablation_matrix")
    return p.parse_args()


def get_nested(obj, *keys):
    cur = obj
    for key in keys:
        if not isinstance(cur, dict):
            return None
        cur = cur.get(key)
    return cur


def joint_value(lang_entry):
    if isinstance(lang_entry, dict):
        return lang_entry.get("macro_f1")
    if isinstance(lang_entry, (int, float)):
        return lang_entry
    return None


def metric_rows(combo: str, results: dict) -> list[dict]:
    rows = []
    train_langs = "+".join(COMBO_LANGS[combo])

    for raw_system, system in SYSTEM_LABELS.items():
        block = results.get(raw_system)
        if not block:
            continue

        # Joint F1 per language.
        joint = block.get("joint_f1")
        if isinstance(joint, dict):
            for lang in TARGET_LANGS + ["Overall"]:
                val = joint_value(joint.get(lang))
                if val is not None:
                    rows.append({
                        "combo": combo,
                        "train_langs": train_langs,
                        "system": system,
                        "metric": "joint_f1",
                        "language": lang,
                        "value": val,
                    })
            overall = joint.get("Overall")
            if isinstance(overall, dict) and overall.get("macro_avg_f1") is not None:
                rows.append({
                    "combo": combo,
                    "train_langs": train_langs,
                    "system": system,
                    "metric": "joint_f1_macro_avg",
                    "language": "Overall",
                    "value": overall["macro_avg_f1"],
                })

        # Classification F1 per language.
        cls = block.get("cls_f1")
        if isinstance(cls, dict):
            for lang, val in cls.items():
                rows.append({
                    "combo": combo,
                    "train_langs": train_langs,
                    "system": system,
                    "metric": "classification_macro_f1",
                    "language": lang,
                    "value": val,
                })

        # E2E span overlap F1 per language.
        span = get_nested(block, "span_e2e", "overlap")
        if isinstance(span, dict):
            for lang, val in span.items():
                rows.append({
                    "combo": combo,
                    "train_langs": train_langs,
                    "system": system,
                    "metric": "e2e_span_overlap_f1",
                    "language": lang,
                    "value": val,
                })

        if block.get("joint_acc") is not None:
            rows.append({
                "combo": combo,
                "train_langs": train_langs,
                "system": system,
                "metric": "joint_accuracy",
                "language": "Overall",
                "value": block["joint_acc"],
            })

    return rows


def build_stability(rows: list[dict]) -> list[dict]:
    by_key = {}
    for row in rows:
        if row["metric"] != "joint_f1" or row["language"] not in TARGET_LANGS:
            continue
        by_key.setdefault((row["combo"], row["train_langs"], row["system"]), {})[
            row["language"]
        ] = float(row["value"])

    out = []
    for (combo, train_langs, system), vals_by_lang in sorted(by_key.items()):
        vals = [vals_by_lang[l] for l in TARGET_LANGS if l in vals_by_lang]
        if not vals:
            continue
        worst_lang = min(vals_by_lang, key=vals_by_lang.get)
        best_lang = max(vals_by_lang, key=vals_by_lang.get)
        out.append({
            "combo": combo,
            "train_langs": train_langs,
            "system": system,
            "mean_joint_f1": round(mean(vals), 4),
            "std_joint_f1": round(pstdev(vals), 4),
            "worst_language": worst_lang,
            "worst_language_joint_f1": vals_by_lang[worst_lang],
            "best_language": best_lang,
            "best_language_joint_f1": vals_by_lang[best_lang],
            "best_minus_worst_gap": round(vals_by_lang[best_lang] - vals_by_lang[worst_lang], 4),
            "stability_score_mean_minus_std": round(mean(vals) - pstdev(vals), 4),
        })
    return out


def build_transfer(rows: list[dict]) -> list[dict]:
    out = []
    for row in rows:
        if row["metric"] == "joint_f1" and row["language"] in TARGET_LANGS:
            out.append({
                "combo": row["combo"],
                "train_langs": row["train_langs"],
                "system": row["system"],
                "target_language": row["language"],
                "joint_f1": row["value"],
                "target_seen_in_training": row["language"] in COMBO_LANGS[row["combo"]],
            })
    return out


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    eval_dir = Path(args.eval_dir)
    output_dir = Path(args.output_dir)

    rows = []
    missing = []
    for combo in COMBOS:
        path = eval_dir / combo / "pipeline_eval_results.json"
        if not path.exists():
            missing.append(combo)
            continue
        with path.open(encoding="utf-8") as f:
            results = json.load(f)
        rows.extend(metric_rows(combo, results))

    stability = build_stability(rows)
    transfer = build_transfer(rows)

    write_csv(output_dir / "ablation_summary.csv", rows)
    write_csv(output_dir / "stability_summary.csv", stability)
    write_csv(output_dir / "transfer_matrix.csv", transfer)

    print(f"Wrote {len(rows)} metric rows")
    print(f"Wrote {len(stability)} stability rows")
    print(f"Wrote {len(transfer)} transfer rows")
    if missing:
        print("Missing combos:", ", ".join(missing))


if __name__ == "__main__":
    main()
