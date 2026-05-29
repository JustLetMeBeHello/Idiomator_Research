"""Read System G g_only results + QA system results across all 15 combos.

Two Drive directories:
  G (corrected):  Idiomator_Research/results/language_ablation_matrix/<combo>/g_only/
  QA systems:     IdiomBERT_Ablations/results/<combo>/

Steps 4-5 of fix_system_g_combos.py permanently failed (no main
pipeline_eval_results.json per combo). This script reads both dirs directly
and merges them for per-combo comparison.

Usage (Colab, from repo root):
    python Ablations/summarize_g_only.py \
        --g_dir   /content/drive/MyDrive/Idiomator_Research/results/language_ablation_matrix \
        --qa_dir  /content/drive/MyDrive/IdiomBERT_Ablations/results

Usage (local — pass explicit paths if Drive is mounted or results are downloaded):
    python Ablations/summarize_g_only.py --g_dir /path/g --qa_dir /path/qa

Without --qa_dir: outputs G-only summary.

Outputs:
    <g_dir>/g_only_summary.csv          — G metrics per combo
    <g_dir>/g_vs_qa_comparison.csv      — G vs D/E/F per combo (if --qa_dir given)
    Printed tables to stdout
"""

import argparse
import csv
import json
from pathlib import Path

COMBOS = [
    "en", "es", "hi", "te",
    "en_es", "en_hi", "en_te", "es_hi", "es_te", "hi_te",
    "en_es_hi", "en_es_te", "en_hi_te", "es_hi_te",
    "en_es_hi_te",
]

# QA system keys in pipeline_eval_results.json
QA_SYSTEMS = {
    "D": "system_d_mbert_s1_joint_span",
    "E": "system_e_joint_end_to_end",
    "F": "system_f_sequential_phase1_ph2",
    "A": "system_a_mbert_pipeline",
}

G_CSV_FIELDS = [
    "combo",
    "span_all_overlap",
    "span_e2e_exact",
    "span_e2e_overlap",
    "span_e2e_en", "span_e2e_es", "span_e2e_hi", "span_e2e_te",
    "joint_f1",
    "joint_acc",
    "stability_mean", "stability_std", "stability_score",
    "indo_span_overlap",
    "indo_span_exact",
]

CMP_CSV_FIELDS = (
    ["combo"]
    + ["g_span_e2e", "g_span_all", "g_indo", "g_joint_f1", "g_stability"]
    + [f"{s}_span_e2e" for s in QA_SYSTEMS]
    + [f"{s}_joint_f1" for s in QA_SYSTEMS]
    + [f"{s}_stability" for s in QA_SYSTEMS]
    + [f"g_vs_{s}_e2e_delta" for s in QA_SYSTEMS]
)


def load_json(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def extract_g(combo: str, data: dict) -> dict:
    g = data["system_g_bio_tagger"]
    row = {"combo": combo}
    row["span_all_overlap"] = g.get("span_all", {}).get("overlap", {}).get("Overall")
    row["span_e2e_exact"]   = g.get("span_e2e", {}).get("exact",   {}).get("Overall")
    row["span_e2e_overlap"] = g.get("span_e2e", {}).get("overlap", {}).get("Overall")
    for lang, short in [("English","en"),("Spanish","es"),("Hindi","hi"),("Telugu","te")]:
        row[f"span_e2e_{short}"] = g.get("span_e2e", {}).get("overlap", {}).get(lang)
    jf1 = g.get("joint_f1")
    row["joint_f1"]  = jf1.get("Overall") if isinstance(jf1, dict) else jf1
    row["joint_acc"] = g.get("joint_acc")
    stab = g.get("stability", {})
    row["stability_mean"]  = stab.get("mean_joint")
    row["stability_std"]   = stab.get("std_joint")
    row["stability_score"] = stab.get("stability")
    indo = g.get("indonesian_ci", {})
    row["indo_span_overlap"] = (indo.get("span_overlap") or [None])[0]
    row["indo_span_exact"]   = (indo.get("span_exact")   or [None])[0]
    return row


def extract_qa_system(label: str, key: str, data: dict) -> dict:
    """Extract span_e2e overlap, joint_f1, stability for one QA system."""
    sys = data.get(key)
    if sys is None:
        return {f"{label}_span_e2e": None, f"{label}_joint_f1": None, f"{label}_stability": None}
    e2e = sys.get("span_e2e", {}).get("overlap", {}).get("Overall") or \
          sys.get("span_correct_id", {}).get("overlap", {}).get("Overall")
    jf1 = sys.get("joint_f1")
    if isinstance(jf1, dict):
        overall = jf1.get("Overall")
        if isinstance(overall, dict):
            # QA systems: joint_f1["Overall"]["macro_avg_f1"]
            joint_f1 = overall.get("macro_avg_f1")
        elif overall is not None:
            # G: joint_f1["Overall"] is a scalar
            joint_f1 = overall
        else:
            joint_f1 = jf1.get("macro_avg_f1")
    else:
        joint_f1 = jf1
    stab = sys.get("stability", {}).get("stability")
    return {f"{label}_span_e2e": e2e, f"{label}_joint_f1": joint_f1, f"{label}_stability": stab}


def fmt(v, decimals=4) -> str:
    if v is None:
        return "  —   "
    return f"{v:.{decimals}f}"


def delta_str(v) -> str:
    if v is None:
        return "  —   "
    sign = "+" if v >= 0 else ""
    return f"{sign}{v:.4f}"


def print_g_table(rows: list[dict]) -> None:
    hdr = f"{'combo':<14} {'e2e_ovlp':>8} {'all_ovlp':>8} {'indo_ovlp':>9} {'stability':>9} {'joint_f1':>8}"
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        print(
            f"{r['combo']:<14} {fmt(r['span_e2e_overlap']):>8} "
            f"{fmt(r['span_all_overlap']):>8} "
            f"{fmt(r['indo_span_overlap']):>9} "
            f"{fmt(r['stability_score']):>9} "
            f"{fmt(r['joint_f1']):>8}"
        )


def print_cmp_table(rows: list[dict]) -> None:
    print(f"\n{'combo':<14} {'G_e2e':>7} {'D_e2e':>7} {'E_e2e':>7} {'G_vs_D':>8} {'G_joint':>8} {'D_joint':>8} {'G_stab':>7} {'D_stab':>7}")
    print("-" * 80)
    for r in rows:
        print(
            f"{r['combo']:<14} "
            f"{fmt(r['g_span_e2e']):>7} "
            f"{fmt(r.get('D_span_e2e')):>7} "
            f"{fmt(r.get('E_span_e2e')):>7} "
            f"{delta_str(r.get('g_vs_D_e2e_delta')):>8} "
            f"{fmt(r['g_joint_f1']):>8} "
            f"{fmt(r.get('D_joint_f1')):>8} "
            f"{fmt(r['g_stability']):>7} "
            f"{fmt(r.get('D_stability')):>7}"
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--g_dir",  default=None,
        help="Path to language_ablation_matrix/ containing g_only/ subdirs. "
             "Defaults to results/language_ablation_matrix/ relative to repo root.")
    parser.add_argument("--qa_dir", default=None,
        help="Path to IdiomBERT_Ablations/results/ containing per-combo QA JSONs. "
             "Drive path: /content/drive/MyDrive/IdiomBERT_Ablations/results")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent.parent
    g_base  = Path(args.g_dir)  if args.g_dir  else repo_root / "results" / "language_ablation_matrix"
    qa_base = Path(args.qa_dir) if args.qa_dir else None

    g_rows, cmp_rows = [], []
    g_missing, qa_missing = [], []

    for combo in COMBOS:
        g_path = g_base / combo / "g_only" / "pipeline_eval_results.json"
        if not g_path.exists():
            print(f"MISSING G: {g_path}")
            g_missing.append(combo)
            continue
        g_data = load_json(g_path)
        g_row  = extract_g(combo, g_data)
        g_rows.append(g_row)

        if qa_base is not None:
            qa_path = qa_base / combo / "pipeline_eval_results.json"
            cmp = {
                "combo":       combo,
                "g_span_e2e":  g_row["span_e2e_overlap"],
                "g_span_all":  g_row["span_all_overlap"],
                "g_indo":      g_row["indo_span_overlap"],
                "g_joint_f1":  g_row["joint_f1"],
                "g_stability": g_row["stability_score"],
            }
            if qa_path.exists():
                qa_data = load_json(qa_path)
                for label, key in QA_SYSTEMS.items():
                    cmp.update(extract_qa_system(label, key, qa_data))
                for label in QA_SYSTEMS:
                    g_e2e = g_row["span_e2e_overlap"]
                    qa_e2e = cmp.get(f"{label}_span_e2e")
                    cmp[f"g_vs_{label}_e2e_delta"] = (
                        round(g_e2e - qa_e2e, 4)
                        if g_e2e is not None and qa_e2e is not None else None
                    )
            else:
                print(f"MISSING QA: {qa_path}")
                qa_missing.append(combo)
                for label in QA_SYSTEMS:
                    cmp[f"{label}_span_e2e"]    = None
                    cmp[f"{label}_joint_f1"]    = None
                    cmp[f"{label}_stability"]   = None
                    cmp[f"g_vs_{label}_e2e_delta"] = None
            cmp_rows.append(cmp)

    if not g_rows:
        print("No g_only results found. Run fix_system_g_combos.py on Colab first.")
        return

    print(f"\nSystem G ablation — {len(g_rows)}/{len(COMBOS)} combos\n")
    print_g_table(g_rows)
    if g_missing:
        print(f"\nMissing G combos: {', '.join(g_missing)}")

    out_csv = g_base / "g_only_summary.csv"
    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=G_CSV_FIELDS)
        writer.writeheader()
        writer.writerows(g_rows)
    print(f"\nG summary CSV: {out_csv}")

    if cmp_rows:
        print(f"\n\nG vs QA comparison ({len(cmp_rows)} combos):")
        print_cmp_table(cmp_rows)
        if qa_missing:
            print(f"\nMissing QA combos: {', '.join(qa_missing)}")

        cmp_csv = g_base / "g_vs_qa_comparison.csv"
        with open(cmp_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=CMP_CSV_FIELDS)
            writer.writeheader()
            writer.writerows(cmp_rows)
        print(f"Comparison CSV: {cmp_csv}")


if __name__ == "__main__":
    main()
