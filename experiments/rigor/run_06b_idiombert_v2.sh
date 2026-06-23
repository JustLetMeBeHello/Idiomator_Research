#!/usr/bin/env bash
# ── Experiment 06 runner: IdiomBERT-v2 8-cell SCL/HNR/LI ablation ───────────
#
# REVISION_PLAN Exp06 — biggest remaining GPU commitment (~8 cells x ~100 min
# T4/A100). Wraps run_06_idiombert_v2.py in the repo's canonical launch
# harness: Drive-persistence gate + per-cell skip/resume, same pattern as
# run_16b_word_tagger.sh / run_17b_bio_cls_joint.sh.
#
# WHAT EXP06 IS
# -------------
# Toggles 3 MWE-specific training innovations on the QA-style Joint baseline
# (Train_Join.py's architecture): SCL (span contrastive loss), HNR (hard-
# negative reweighting), LI (lateral inhibition at decode). 8 cells = every
# subset of {SCL, HNR, LI}. This is the only path to Main-tier for IdiomBERT
# per the 2026-06-19 council verdict — everything else on the rigor queue is
# polish, not a tier-mover.
#
# TWO PRE-EXISTING BUGS FIXED 2026-06-22 (before this runner could even be
# written) — both predate this runner and were caught by colab-preflight's
# own checklist:
#   1. Train_Join import path was stale from the 2026-06-19 reorg
#      (flat -> training/ + experiments/{rigor,ablations}/); REPO_ROOT
#      resolved one directory short of repo root, ModuleNotFoundError.
#   2. Saved predictions used pred_cls/pred_token_start/pred_token_end
#      (token indices) instead of Train_Join.py's pred_idiomaticity/
#      pred_span_start/pred_span_end (CHAR offsets) — silently would NOT
#      have registered through Full_evaluation.py --joint_preds. Now
#      converts via token_to_char_span and matches the schema exactly.
# Verified via a real local smoke test (1 epoch, English only, baseline
# cell) before this runner was written, not just py_compile.
#
# OUTPUTS  models/idiombert_v2/<cell>/{config.json,metrics.json,test_predictions.jsonl}
# Register with Full_evaluation's --joint_preds (the command is printed per
# cell at the end) — directly comparable to System D/E's Joint F1 (same
# encoder + loss-weight convention, only the 3 toggle flags differ).
#
# ── SAVE + RESUME ON TIMEOUT (read this) ────────────────────────────────────
# SAVE:   with DRIVE_OUT set, each cell's output dir is symlinked to Drive and
#         the run HARD-FAILS before training if the link is not Drive-backed
#         and writable. console.log streams via the notebook's tee.
# RESUME: 8 independent jobs (one per cell). A cell whose test_predictions.jsonl
#         already exists is SKIPPED. Re-run the same command after any timeout —
#         it resumes at the first incomplete cell.
# LIMIT:  no intra-cell resume (best_model.pt is the only checkpoint and stays
#         until the cell finishes — unlike E3/E4 there's no "delete after done"
#         step here since this script saves a single best_model.pt, not a full
#         HF directory; Drive quota impact is smaller). A timeout DURING a cell
#         restarts that cell from epoch 1. Worst-case loss = one cell (~100 min).
#
# Usage:
#   # always dry-run first (prints plan + persistence gate, trains nothing):
#   DRIVE_OUT=/content/drive/MyDrive/IdiomatorRigor bash experiments/rigor/run_06b_idiombert_v2.sh --dry-run
#   # full 8-cell matrix (re-run the same line after any timeout to resume):
#   DRIVE_OUT=/content/drive/MyDrive/IdiomatorRigor bash experiments/rigor/run_06b_idiombert_v2.sh
#   # one cell only:
#   CELLS=scl DRIVE_OUT=... bash experiments/rigor/run_06b_idiombert_v2.sh
#   # FORCE=1 retrains even completed cells (use after a code change, never to
#   # "make sure" — a stale skip is not success, a fresh artifact is):
#   FORCE=1 DRIVE_OUT=... bash experiments/rigor/run_06b_idiombert_v2.sh
# ──────────────────────────────────────────────────────────────────────────────

set -euo pipefail
cd "$(git -C "$(dirname "$0")" rev-parse --show-toplevel)"

DRY_RUN=0
[[ "${1:-}" == "--dry-run" ]] && DRY_RUN=1

# 8-cell matrix: cell_name -> flags. baseline has no flags.
# Plain function + case, not `declare -A` — macOS ships bash 3.2 (no assoc
# arrays); this must dry-run locally on a Mac, not just on Colab's bash 5.
cell_flags () {
    case "$1" in
        baseline)    echo "" ;;
        scl)         echo "--use_scl" ;;
        hnr)         echo "--use_hnr" ;;
        li)          echo "--use_li" ;;
        scl_hnr)     echo "--use_scl --use_hnr" ;;
        scl_li)      echo "--use_scl --use_li" ;;
        hnr_li)      echo "--use_hnr --use_li" ;;
        scl_hnr_li)  echo "--use_scl --use_hnr --use_li" ;;
        *)           return 1 ;;
    esac
}
ALL_CELLS="baseline scl hnr li scl_hnr scl_li hnr_li scl_hnr_li"
CELLS="${CELLS:-$ALL_CELLS}"
FORCE="${FORCE:-0}"
LANGS=(English Spanish Hindi Telugu)
TEST_LANGS=(English Spanish Hindi Telugu Indonesian)
TRAINER=experiments/rigor/run_06_idiombert_v2.py
LR=2e-5            # matches Train_Join.py default — only the 3 toggle flags vary
BATCH=32
SEED=42            # single-seed per cell (8 cells already ~8x the GPU budget of a 3-seed run)

echo "════════════════════════════════════════════════════════════════════════"
echo "Exp06 IdiomBERT-v2 ablation   cells=[${CELLS}]   force=${FORCE}   dry_run=${DRY_RUN}"
echo "  encoder: bert-base-multilingual-cased (same as Train_Join.py / System D/E)"
echo "  lr: ${LR}   batch: ${BATCH}   seed: ${SEED}"
echo "  langs:      ${LANGS[*]}"
echo "  test_langs: ${TEST_LANGS[*]}"
echo "════════════════════════════════════════════════════════════════════════"

# ── Per-cell persistence gate (CLAUDE.md hard rule / colab-preflight) ───────
gate_dir () {
    local d="$1"
    if [[ -n "${DRIVE_OUT:-}" ]]; then
        local name="${d#models/}"
        mkdir -p "$DRIVE_OUT/$name"
        rm -rf "$d"
        mkdir -p "$(dirname "$d")"
        ln -s "$DRIVE_OUT/$name" "$d"
        python - "$DRIVE_OUT" "$d" <<'PY'
import os, sys
drive = os.path.realpath(sys.argv[1]); d = sys.argv[2]
assert os.path.islink(d), f"{d} is not a symlink — output would be ephemeral"
tgt = os.path.realpath(d)
assert tgt.startswith(drive), f"{d} -> {tgt} not under Drive ({drive})"
probe = os.path.join(d, ".persist_probe")
open(probe, "w").write("ok")
assert open(probe).read() == "ok", f"readback failed at {probe}"
os.remove(probe)
print(f"  ✓ persistence gate passed — {d} is Drive-backed and writable")
PY
    else
        echo "  ⚠ DRIVE_OUT unset — ${d}/ is LOCAL (OK locally; EPHEMERAL on Colab)"
        mkdir -p "$d"
    fi
}

# ── Skip a cell iff its predictions exist AND FORCE!=1 ──────────────────────
should_run () {
    local preds="$1/test_predictions.jsonl"
    if [[ "$FORCE" != "1" && -f "$preds" ]]; then
        echo "  ✓ skip — $preds exists (set FORCE=1 to retrain)"
        return 1
    fi
    return 0
}

for CELL in $CELLS; do
    OUT="models/idiombert_v2/${CELL}"
    if ! FLAGS="$(cell_flags "$CELL")"; then
        echo "  ✗ unknown cell '${CELL}' — valid: ${ALL_CELLS}"
        exit 1
    fi
    echo
    echo "── Exp06 cell=${CELL}  flags=[${FLAGS}]  → ${OUT} ───────────────────────────────────"
    if ! should_run "$OUT"; then
        continue
    fi

    if [[ "$DRY_RUN" == "1" ]]; then
        printf '   [dry-run] '
        printf '%q ' python "$TRAINER" \
            --output_dir "$OUT" \
            --model_name bert-base-multilingual-cased \
            --langs "${LANGS[@]}" --test_langs "${TEST_LANGS[@]}" \
            --lr "$LR" --batch_size "$BATCH" --seed "$SEED" $FLAGS
        echo
        continue
    fi

    gate_dir "$OUT"
    python "$TRAINER" \
        --output_dir "$OUT" \
        --model_name bert-base-multilingual-cased \
        --langs "${LANGS[@]}" --test_langs "${TEST_LANGS[@]}" \
        --lr "$LR" --batch_size "$BATCH" --seed "$SEED" $FLAGS

    if [[ -f "$OUT/test_predictions.jsonl" ]]; then
        echo "  ✓ cell ${CELL} done"
    fi
done

echo
echo "════════════════════════════════════════════════════════════════════════"
[[ "$DRY_RUN" == "1" ]] && { echo "Dry-run only — nothing trained."; exit 0; }
echo "Exp06 training complete. Register each cell as a system row with:"
echo
for CELL in $CELLS; do
cat <<EOF
  python Evaluation/Full_evaluation.py \\
      --joint_preds models/idiombert_v2/${CELL}/test_predictions.jsonl \\
      --output_dir  results/exp06_${CELL}
EOF
done
echo
echo "Then compare each cell's Joint F1 against System E (baseline cell should"
echo "roughly match System E — same architecture/loss-weights, sanity check)"
echo "and against D/A — the SCL/HNR/LI ablation matrix for the v2 preprint."
echo "Run experiments/rigor/check_metric_drift.py before any paper build."
echo "════════════════════════════════════════════════════════════════════════"
