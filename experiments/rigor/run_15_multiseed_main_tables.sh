#!/usr/bin/env bash
# ── Experiment E2: multi-seed the main 9-system A–G comparison tables ────────
#
# REVISION_PLAN E2. The main Joint-F1 tables (Systems A–G) are currently
# SINGLE-SEED (42). E2 re-runs the trainable mBERT systems at two more seeds
# (123, 7) so the A–G ranking — and the stability-as-discriminator metric built
# on it — carries variance bars instead of a single point. The encoder study
# (C1: XLM-R / RemBERT) is ALREADY 3-seeded; this script is ONLY the A–G main
# tables, not C1.
#
# What gets retrained per seed (mBERT = bert-base-multilingual-cased):
#   Stage 1 cls      → Stage_1_training.py   (feeds System A AND System D)
#   Stage 2 span     → Stage_2_training.py   (System A pipeline span head)
#   Joint end-to-end → Train_Join.py         (System E; also D's span head)
#   Sequential P1+P2 → Train_Sequential.py   (System F, phase 0 = both phases)
#   BIO tagger       → BiO_Task_mBERT_train.py (System G)
#
# NOT retrained (no seed dependence):
#   Systems B/B4/C/C4 = GPT-4o (API; deterministic-ish, no GPU training).
#   System D          = eval-time recombination of Stage-1 cls + Joint span;
#                       produced by Full_evaluation, not a separate training.
#
# Hyperparameters: every trainer is invoked at its DEFAULT args, which ARE the
# canonical seed-42 main-table config (tuned lr / epochs / loss weights are
# baked into each trainer's argparse defaults). The ONLY variables changed are
# --seed and --output_dir. Do NOT pass tuning flags here — E2's validity
# depends on holding everything except the seed fixed.
#
# Run order is sequential on one GPU. On a T4 the full 5-job seed costs roughly
# 3–5 GPU-hr; budget two seeds ≈ one long A100 session or several T4 blocks.
#
# Usage (run once per seed):
#   SEED=123 DRIVE_OUT=/content/drive/MyDrive/IdiomatorRigor bash experiments/rigor/run_15_multiseed_main_tables.sh
#   SEED=7   DRIVE_OUT=/content/drive/MyDrive/IdiomatorRigor bash experiments/rigor/run_15_multiseed_main_tables.sh
#
#   FORCE=1   re-trains even if a job's test_predictions.jsonl already exists
#             (idempotency hard rule: a bare existing file is NOT trusted as
#             success after a code change — use FORCE to overwrite stale runs).
#   --dry-run (first arg) prints the plan + persistence gate, trains nothing.
# ──────────────────────────────────────────────────────────────────────────────

set -euo pipefail
cd "$(git -C "$(dirname "$0")" rev-parse --show-toplevel)"

DRY_RUN=0
[[ "${1:-}" == "--dry-run" ]] && DRY_RUN=1

SEED="${SEED:-42}"
FORCE="${FORCE:-0}"
if [[ "$SEED" == "42" ]]; then
    echo "⚠ SEED=42 is already canonical (the single-seed main tables). E2 wants 123 and 7."
    echo "  Set SEED=123 or SEED=7. Continuing only if you really meant to re-run seed 42."
fi

LANGS=(English Spanish Hindi Telugu)
TEST_LANGS=(English Spanish Hindi Telugu Indonesian)   # Indonesian = held-out table

OUT_ROOT="models/main_s${SEED}"
STAGE1_DIR="$OUT_ROOT/stage1_mbert"
STAGE2_DIR="$OUT_ROOT/stage2_mbert"
JOINT_DIR="$OUT_ROOT/joint_mbert"
SEQ_DIR="$OUT_ROOT/sequential_mbert"   # trainer writes phase1/ and phase2/ underneath
BIO_DIR="$OUT_ROOT/bio_tagger"

ALL_DIRS=("$STAGE1_DIR" "$STAGE2_DIR" "$JOINT_DIR" "$SEQ_DIR" "$BIO_DIR")

echo "════════════════════════════════════════════════════════════════════════"
echo "E2 multi-seed main A–G tables   seed=${SEED}   force=${FORCE}   dry_run=${DRY_RUN}"
echo "  langs:      ${LANGS[*]}"
echo "  test_langs: ${TEST_LANGS[*]}"
echo "  out root:   ${OUT_ROOT}"
echo "════════════════════════════════════════════════════════════════════════"

# ── Persistence gate (CLAUDE.md hard rule) ──────────────────────────────────
# /content is ephemeral on Colab; only Drive survives a disconnect. With
# DRIVE_OUT set, symlink each output dir to Drive and HARD-FAIL before training
# if the link is not Drive-backed and writable (readback probe).
if [[ -n "${DRIVE_OUT:-}" ]]; then
    for d in "${ALL_DIRS[@]}"; do
        name="${d#models/}"
        mkdir -p "$DRIVE_OUT/$name"
        rm -rf "$d"
        mkdir -p "$(dirname "$d")"
        ln -s "$DRIVE_OUT/$name" "$d"
    done
    python - "$DRIVE_OUT" "${ALL_DIRS[@]}" <<'PY'
import os, sys
drive = os.path.realpath(sys.argv[1])
for d in sys.argv[2:]:
    assert os.path.islink(d), f"{d} is not a symlink — output would be ephemeral"
    tgt = os.path.realpath(d)
    assert tgt.startswith(drive), f"{d} -> {tgt} not under Drive ({drive})"
    probe = os.path.join(d, ".persist_probe")
    open(probe, "w").write("ok")
    assert open(probe).read() == "ok", f"readback failed at {probe}"
    os.remove(probe)
print("✓ persistence gate passed — all E2 outputs are Drive-backed and writable")
PY
else
    echo "⚠ DRIVE_OUT unset — outputs go to local ${OUT_ROOT}/ (OK locally; EPHEMERAL on Colab)"
    for d in "${ALL_DIRS[@]}"; do mkdir -p "$d"; done
fi

# ── Helper: skip a job iff its predictions exist AND FORCE!=1 ────────────────
# Never trust a bare existing file after a code change (idempotency rule):
# FORCE=1 bypasses the skip so stale outputs are overwritten.
should_run () {
    local preds="$1/test_predictions.jsonl"
    if [[ "$FORCE" != "1" && -f "$preds" ]]; then
        echo "  ✓ skip — $preds exists (set FORCE=1 to retrain)"
        return 1
    fi
    return 0
}

run () {   # run "<label>" <cmd...>
    local label="$1"; shift
    echo
    echo "── ${label}  (seed=${SEED}) ──────────────────────────────────────────"
    if [[ "$DRY_RUN" == "1" ]]; then
        printf '   [dry-run] %q ' "$@"; echo
        return 0
    fi
    "$@"
}

# ── A/D) Stage 1 classifier ─────────────────────────────────────────────────
if should_run "$STAGE1_DIR"; then
    run "System A/D — Stage 1 cls" \
        python training/Stage_1_training.py \
            --output_dir "$STAGE1_DIR" \
            --langs "${LANGS[@]}" --test_langs "${TEST_LANGS[@]}" \
            --seed "$SEED"
fi

# ── A) Stage 2 span ─────────────────────────────────────────────────────────
if should_run "$STAGE2_DIR"; then
    run "System A — Stage 2 span" \
        python training/Stage_2_training.py \
            --output_dir "$STAGE2_DIR" \
            --langs "${LANGS[@]}" --test_langs "${TEST_LANGS[@]}" \
            --seed "$SEED"
fi

# ── E) Joint end-to-end (also supplies System D span head) ──────────────────
if should_run "$JOINT_DIR"; then
    run "System E — Joint end-to-end" \
        python training/Train_Join.py \
            --output_dir "$JOINT_DIR" \
            --langs "${LANGS[@]}" --test_langs "${TEST_LANGS[@]}" \
            --seed "$SEED"
fi

# ── F) Sequential (phase 0 = train both phase1 + phase2) ────────────────────
if should_run "$SEQ_DIR/phase2"; then
    run "System F — Sequential P1+P2" \
        python training/Train_Sequential.py \
            --output_dir "$SEQ_DIR" --phase 0 \
            --langs "${LANGS[@]}" --test_langs "${TEST_LANGS[@]}" \
            --seed "$SEED"
fi

# ── G) BIO tagger ───────────────────────────────────────────────────────────
if should_run "$BIO_DIR"; then
    run "System G — BIO tagger" \
        python experiments/ablations/BiO_Task_mBERT_train.py \
            --output_dir "$BIO_DIR" \
            --langs "${LANGS[@]}" --test_langs "${TEST_LANGS[@]}" \
            --seed "$SEED"
fi

echo
echo "════════════════════════════════════════════════════════════════════════"
echo "E2 seed=${SEED} training complete. Register A–G rows for this seed with:"
echo
cat <<EOF
  python Evaluation/Full_evaluation.py \\
      --stage1_mbert  ${STAGE1_DIR}/test_predictions.jsonl \\
      --stage2_mbert  ${STAGE2_DIR}/test_predictions.jsonl \\
      --joint_preds   ${JOINT_DIR}/test_predictions.jsonl \\
      --span2_joint   ${JOINT_DIR}/test_predictions.jsonl \\
      --seq_phase1    ${SEQ_DIR}/phase1/test_predictions.jsonl \\
      --seq_phase2    ${SEQ_DIR}/phase2/test_predictions.jsonl \\
      --bio_preds     ${BIO_DIR}/test_predictions.jsonl \\
      --output_dir    results/pipeline_eval_s${SEED}
EOF
echo
echo "Then aggregate the 3 seeds (42/123/7) → per-system mean ± std Joint-F1 and"
echo "feed the variance bars into the stability ranking (REVISION_PLAN N3)."
echo "Run experiments/rigor/check_metric_drift.py before any paper build."
echo "════════════════════════════════════════════════════════════════════════"
