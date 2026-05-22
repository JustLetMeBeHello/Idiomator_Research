"""
merge_ablations.py

Merges language ablation results produced by multiple Colab accounts into a
single unified output directory, then validates completeness.

Each Colab account downloads its portion of the ablation matrix as a zip and
you point this script at the extracted folders. The script merges them into one
canonical directory tree matching the structure expected by Full_evaluation.py
and the run_stage*.py runners.

Expected source structure (one folder per Colab account):
    colab_A/
        models/language_ablation_matrix/
            en/
                stage1_mbert/
                    metrics.json
                    test_predictions.jsonl
                    config.json          (optional)
                    best_model/          (optional — large, skip by default)
                stage2_mbert/
                    ...
                joint_mbert/
                    ...
                sequential/
                    phase1/ ...
                    phase2/ ...
                bio_tagger/
                    ...
            en_es/
                ...
    colab_B/
        models/language_ablation_matrix/
            hi/
                ...
            te/
                ...

Output structure (what Full_evaluation.py expects):
    merged/
        models/language_ablation_matrix/
            en/
                stage1_mbert/
                stage2_mbert/
                joint_mbert/
                sequential/
                bio_tagger/
            en_es/
                ...
            ...  (all 15 combos)

Usage:
    # Basic — point at any number of source roots
    python merge_ablations.py \\
        --sources colab_A colab_B colab_C \\
        --output  merged

    # If your friends zipped just the ablation matrix folder (not the full models/ tree)
    python merge_ablations.py \\
        --sources colab_A/models/language_ablation_matrix \\
                  colab_B/models/language_ablation_matrix \\
        --sources_are_matrix_roots \\
        --output merged

    # Dry run — see what would be copied without touching anything
    python merge_ablations.py \\
        --sources colab_A colab_B \\
        --output  merged \\
        --dry_run

    # Skip copying large best_model/ checkpoint folders (saves disk space)
    python merge_ablations.py \\
        --sources colab_A colab_B \\
        --output  merged \\
        --skip_checkpoints

    # After merging, validate completeness against expected combos + systems
    python merge_ablations.py \\
        --sources colab_A colab_B \\
        --output  merged \\
        --validate_only        # skip copy, just check what's in --output already

Notes:
  - If two sources both contain results for the same combo+system, the script
    keeps whichever has a higher test_overlap_f1 in metrics.json (not just the
    last one). Use --conflict=overwrite to always take the latest source instead.
  - The script never deletes anything from --output. It only adds or updates.
  - Run with --dry_run first to verify paths before copying.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from collections import defaultdict
from pathlib import Path


# ── All expected combos and systems (must match run_stage*.py) ────────────────

COMBOS = [
    "en",
    "es",
    "hi",
    "te",
    "en_es",
    "en_hi",
    "en_te",
    "es_hi",
    "es_te",
    "hi_te",
    "en_es_hi",
    "en_es_te",
    "en_hi_te",
    "es_hi_te",
    "en_es_hi_te",
]

# Systems that live directly under combo/
FLAT_SYSTEMS = [
    "stage1_mbert",
    "stage2_mbert",
    "joint_mbert",
    "bio_tagger",
]

# Systems with a nested structure under combo/
NESTED_SYSTEMS = {
    "sequential": ["phase1", "phase2"],
}

# Files that constitute a valid completed run (both must exist)
REQUIRED_FILES = ["metrics.json", "test_predictions.jsonl"]

# Files that are useful but not required for evaluation
OPTIONAL_FILES = ["config.json", "training_args.json"]

# Directories to skip when --skip_checkpoints is set
CHECKPOINT_DIRS = {"best_model", "checkpoint"}


# ── Args ──────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Merge language ablation results from multiple Colab accounts."
    )
    p.add_argument(
        "--sources", nargs="+", required=True,
        help="Root directories from each Colab account (or matrix roots if "
             "--sources_are_matrix_roots is set)."
    )
    p.add_argument(
        "--output", default="merged",
        help="Output directory. Will be created if it doesn't exist. "
             "Default: merged/"
    )
    p.add_argument(
        "--sources_are_matrix_roots", action="store_true",
        help="If set, --sources point directly at language_ablation_matrix/ "
             "folders rather than the project root."
    )
    p.add_argument(
        "--ablation_subpath",
        default="models/language_ablation_matrix",
        help="Path from project root to the ablation matrix folder. "
             "Default: models/language_ablation_matrix"
    )
    p.add_argument(
        "--skip_checkpoints", action="store_true",
        help="Skip best_model/ and checkpoint/ subdirectories. "
             "Keeps metrics.json and test_predictions.jsonl only. "
             "Recommended if disk space is limited."
    )
    p.add_argument(
        "--conflict", choices=["best_f1", "overwrite", "skip"], default="best_f1",
        help="What to do when the same combo+system exists in multiple sources:\n"
             "  best_f1   — keep whichever has higher test_overlap_f1 (default)\n"
             "  overwrite — always take the latest source (last --sources wins)\n"
             "  skip      — keep whatever is already in --output, skip duplicates"
    )
    p.add_argument(
        "--dry_run", action="store_true",
        help="Print what would be copied/skipped without touching the filesystem."
    )
    p.add_argument(
        "--validate_only", action="store_true",
        help="Skip merging entirely. Just validate completeness of --output."
    )
    p.add_argument(
        "--summary_json", default=None,
        help="If set, write a JSON summary of what was merged/missing to this path."
    )
    return p.parse_args()


# ── Utilities ─────────────────────────────────────────────────────────────────

def resolve_matrix_root(source: str, args: argparse.Namespace) -> Path:
    """Return the language_ablation_matrix/ path for a given source root."""
    src = Path(source)
    if args.sources_are_matrix_roots:
        return src
    return src / args.ablation_subpath


def get_overlap_f1(system_dir: Path) -> float | None:
    """Read test_overlap_f1 from metrics.json, return None if unavailable."""
    metrics_path = system_dir / "metrics.json"
    if not metrics_path.exists():
        return None
    try:
        with metrics_path.open(encoding="utf-8") as f:
            m = json.load(f)
        # Support both top-level and nested keys
        return float(
            m.get("test_overlap_f1")
            or m.get("overlap_f1")
            or m.get("dev_overlap_f1")
            or 0.0
        )
    except (json.JSONDecodeError, ValueError, OSError):
        return None


def is_complete(system_dir: Path) -> bool:
    """Return True iff all required files exist in system_dir."""
    return all((system_dir / f).exists() for f in REQUIRED_FILES)


def copy_system_dir(
    src_dir: Path,
    dst_dir: Path,
    skip_checkpoints: bool,
    dry_run: bool,
) -> list[str]:
    """
    Copy src_dir → dst_dir, optionally skipping checkpoint subdirectories.
    Returns list of copied file paths (relative to dst_dir).
    """
    copied = []
    if not dry_run:
        dst_dir.mkdir(parents=True, exist_ok=True)

    for item in src_dir.rglob("*"):
        # Skip checkpoint dirs if requested
        if skip_checkpoints:
            parts = item.relative_to(src_dir).parts
            if any(p in CHECKPOINT_DIRS for p in parts):
                continue

        if item.is_file():
            rel   = item.relative_to(src_dir)
            dst   = dst_dir / rel
            if not dry_run:
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(item, dst)
            copied.append(str(rel))

    return copied


# ── Discovery ─────────────────────────────────────────────────────────────────

def discover_systems(matrix_root: Path) -> dict[str, dict[str, Path]]:
    """
    Scan matrix_root for completed system runs.
    Returns {combo: {system_key: system_dir_path}}.
    system_key for nested systems is e.g. "sequential/phase1".
    """
    found: dict[str, dict[str, Path]] = defaultdict(dict)

    if not matrix_root.exists():
        return found

    for combo_dir in sorted(matrix_root.iterdir()):
        if not combo_dir.is_dir():
            continue
        combo = combo_dir.name
        if combo not in COMBOS:
            continue

        # Flat systems
        for sys_name in FLAT_SYSTEMS:
            sys_dir = combo_dir / sys_name
            if sys_dir.is_dir() and is_complete(sys_dir):
                found[combo][sys_name] = sys_dir

        # Nested systems (e.g. sequential/phase1, sequential/phase2)
        for parent_name, sub_names in NESTED_SYSTEMS.items():
            for sub in sub_names:
                sys_dir = combo_dir / parent_name / sub
                key     = f"{parent_name}/{sub}"
                if sys_dir.is_dir() and is_complete(sys_dir):
                    found[combo][key] = sys_dir

    return found


# ── Merge ─────────────────────────────────────────────────────────────────────

def merge_sources(
    matrix_roots: list[Path],
    output_matrix: Path,
    args: argparse.Namespace,
) -> dict:
    """
    Merge all sources into output_matrix.
    Returns a summary dict.
    """
    summary = {
        "sources":       [str(r) for r in matrix_roots],
        "output":        str(output_matrix),
        "copied":        [],
        "skipped_conflict": [],
        "errors":        [],
    }

    # Discover what each source has
    source_inventories = []
    for root in matrix_roots:
        inv = discover_systems(root)
        source_inventories.append((root, inv))
        total = sum(len(v) for v in inv.values())
        print(f"  {root}: {total} completed system runs found across "
              f"{len(inv)} combos")

    # For each combo × system, decide which source to copy from
    # Build a map: (combo, system_key) → list of (source_root, sys_dir, f1)
    candidates: dict[tuple[str, str], list[tuple[Path, Path, float]]] = defaultdict(list)

    for root, inv in source_inventories:
        for combo, systems in inv.items():
            for sys_key, sys_dir in systems.items():
                f1 = get_overlap_f1(sys_dir) or 0.0
                candidates[(combo, sys_key)].append((root, sys_dir, f1))

    for (combo, sys_key), srcs in sorted(candidates.items()):
        # Determine destination path
        if "/" in sys_key:
            # nested: sequential/phase1 → combo/sequential/phase1
            dst_dir = output_matrix / combo / sys_key
        else:
            dst_dir = output_matrix / combo / sys_key

        already_exists = is_complete(dst_dir)

        # Conflict resolution
        if len(srcs) == 1:
            chosen_root, chosen_dir, chosen_f1 = srcs[0]
        else:
            if args.conflict == "best_f1":
                chosen_root, chosen_dir, chosen_f1 = max(srcs, key=lambda x: x[2])
            elif args.conflict == "overwrite":
                chosen_root, chosen_dir, chosen_f1 = srcs[-1]
            else:  # skip
                if already_exists:
                    summary["skipped_conflict"].append(
                        f"{combo}/{sys_key} — kept existing, "
                        f"{len(srcs)} sources available"
                    )
                    print(f"  ↷ skip  {combo}/{sys_key}  (already in output, conflict=skip)")
                    continue
                chosen_root, chosen_dir, chosen_f1 = srcs[0]

        if already_exists:
            existing_f1 = get_overlap_f1(dst_dir) or 0.0
            if args.conflict == "best_f1" and existing_f1 >= chosen_f1:
                summary["skipped_conflict"].append(
                    f"{combo}/{sys_key} — kept existing "
                    f"(f1={existing_f1:.4f} >= incoming f1={chosen_f1:.4f})"
                )
                print(f"  ↷ keep  {combo}/{sys_key}  "
                      f"existing f1={existing_f1:.4f} >= incoming f1={chosen_f1:.4f}")
                continue
            elif args.conflict == "skip":
                summary["skipped_conflict"].append(f"{combo}/{sys_key} — kept existing")
                print(f"  ↷ skip  {combo}/{sys_key}  (already in output)")
                continue

        action = "dry_run" if args.dry_run else "copy"
        if len(srcs) > 1:
            note = f" (chose best f1={chosen_f1:.4f} from {chosen_root.name})" \
                   if args.conflict == "best_f1" else ""
        else:
            note = ""

        print(f"  {'[DRY]' if args.dry_run else '→'} "
              f"{combo}/{sys_key}  f1={chosen_f1:.4f}{note}")

        try:
            copied_files = copy_system_dir(
                chosen_dir, dst_dir,
                skip_checkpoints=args.skip_checkpoints,
                dry_run=args.dry_run,
            )
            summary["copied"].append({
                "combo":     combo,
                "system":    sys_key,
                "source":    str(chosen_root),
                "f1":        chosen_f1,
                "files":     copied_files,
            })
        except Exception as e:
            msg = f"ERROR copying {combo}/{sys_key} from {chosen_root}: {e}"
            print(f"  ✗ {msg}", file=sys.stderr)
            summary["errors"].append(msg)

    return summary


# ── Validation ────────────────────────────────────────────────────────────────

def validate_output(output_matrix: Path) -> dict:
    """
    Check completeness of the merged output against all expected combos × systems.
    Returns a validation report dict.
    """
    all_systems = FLAT_SYSTEMS + [
        f"{parent}/{sub}"
        for parent, subs in NESTED_SYSTEMS.items()
        for sub in subs
    ]

    present  = defaultdict(set)   # combo → set of system keys
    missing  = defaultdict(list)  # combo → list of missing system keys
    f1_table = defaultdict(dict)  # combo → {system_key: f1}

    for combo in COMBOS:
        for sys_key in all_systems:
            if "/" in sys_key:
                sys_dir = output_matrix / combo / sys_key
            else:
                sys_dir = output_matrix / combo / sys_key

            if is_complete(sys_dir):
                present[combo].add(sys_key)
                f1 = get_overlap_f1(sys_dir)
                f1_table[combo][sys_key] = f1
            else:
                missing[combo].append(sys_key)

    total_expected = len(COMBOS) * len(all_systems)
    total_present  = sum(len(v) for v in present.values())
    total_missing  = sum(len(v) for v in missing.values())

    print(f"\n{'═'*70}")
    print(f"VALIDATION REPORT")
    print(f"{'═'*70}")
    print(f"  Expected:  {total_expected} runs  ({len(COMBOS)} combos × {len(all_systems)} systems)")
    print(f"  Present:   {total_present}")
    print(f"  Missing:   {total_missing}")
    print(f"{'─'*70}")

    if total_missing == 0:
        print("  ✓ All runs present. Ready for Full_evaluation.py.")
    else:
        print("  ✗ Missing runs:")
        for combo in COMBOS:
            if missing[combo]:
                print(f"    {combo}:")
                for sys_key in missing[combo]:
                    print(f"      - {sys_key}")

    # Print F1 summary table for present runs
    print(f"\n{'─'*70}")
    print("  Overlap F1 summary (test_overlap_f1 from metrics.json):")
    print(f"  {'Combo':<18}" + "".join(f"  {s[:12]:<13}" for s in all_systems))
    print(f"  {'-'*17}" + "".join(f"  {'-'*12}" for _ in all_systems))

    for combo in COMBOS:
        row = f"  {combo:<18}"
        for sys_key in all_systems:
            f1 = f1_table[combo].get(sys_key)
            row += f"  {f'{f1:.4f}':<13}" if f1 is not None else f"  {'—':<13}"
        print(row)

    return {
        "total_expected": total_expected,
        "total_present":  total_present,
        "total_missing":  total_missing,
        "missing":        {k: v for k, v in missing.items() if v},
        "f1_table":       {k: v for k, v in f1_table.items()},
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()

    output_matrix = Path(args.output) / args.ablation_subpath
    if not args.validate_only and not args.dry_run:
        output_matrix.mkdir(parents=True, exist_ok=True)

    if args.validate_only:
        print(f"Validating: {output_matrix}")
        report = validate_output(output_matrix)
        if args.summary_json:
            Path(args.summary_json).write_text(
                json.dumps(report, indent=2), encoding="utf-8"
            )
        return

    # Resolve all source matrix roots
    matrix_roots = []
    for src in args.sources:
        root = resolve_matrix_root(src, args)
        if not root.exists():
            print(f"  ⚠ Source not found, skipping: {root}", file=sys.stderr)
            continue
        matrix_roots.append(root)

    if not matrix_roots:
        print("No valid source directories found. Exiting.", file=sys.stderr)
        sys.exit(1)

    print(f"{'='*70}")
    print(f"Merge ablation results")
    print(f"{'='*70}")
    print(f"Sources  ({len(matrix_roots)}):")
    for r in matrix_roots:
        print(f"  {r}")
    print(f"Output:  {output_matrix}")
    print(f"Conflict: {args.conflict}")
    print(f"Skip checkpoints: {args.skip_checkpoints}")
    print(f"Dry run: {args.dry_run}")
    print(f"{'─'*70}\n")

    summary = merge_sources(matrix_roots, output_matrix, args)

    print(f"\n{'─'*70}")
    print(f"Merge complete.")
    print(f"  Copied:   {len(summary['copied'])} system runs")
    print(f"  Skipped:  {len(summary['skipped_conflict'])} (conflict resolution)")
    print(f"  Errors:   {len(summary['errors'])}")

    if summary["errors"]:
        print("\n  Errors encountered:")
        for e in summary["errors"]:
            print(f"    ✗ {e}")

    # Always validate after merging so you know what's still missing
    print()
    validation = validate_output(output_matrix)
    summary["validation"] = validation

    if args.summary_json:
        out_path = Path(args.summary_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"\nSummary written → {out_path}")

    if validation["total_missing"] > 0:
        print(f"\n  {validation['total_missing']} runs still missing. "
              f"Run the remaining combos and merge again.")
        sys.exit(0)   # not an error — just incomplete


if __name__ == "__main__":
    main()
