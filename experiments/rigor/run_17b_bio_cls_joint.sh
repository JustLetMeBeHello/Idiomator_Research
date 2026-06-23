#!/usr/bin/env bash
# ── Experiment E4 runner: BIO span head + CLS head jointly trained, 3 seeds ──
#
# REVISION_PLAN E4 (was Exp07 in the older Rigor Experiments table). Wraps the
# E4 trainer (run_17_bio_cls_joint.py) in the repo's canonical launch harness:
# Drive-persistence gate + per-seed skip/resume, matching run_16b_word_tagger.sh
# / run_15_multiseed_main_tables.sh exactly.
#
# WHAT E4 IS / WHY MATCHED LOSS WEIGHTS MATTER
# ---------------------------------------------
# Systems D/E/F pair mBERT with QA-style start/end span heads. System G is a
# BIO tagger with NO classifier (structurally penalized on Joint F1 — see
# key_numbers.md). E4 is the missing cell: mBERT + BIO span head + CLS head,
# JOINTLY trained, directly comparable to D/E on Joint F1. Loss weights
# (cls=0.3, bio=1.9) are passed EXPLICITLY below, identical to Train_Join.py's
# defaults, so a future change to either trainer's argparse defaults can't
# silently de-fair the D/E-vs-E4 comparison.
#
# OUTPUTS  models/bio_cls_joint_mbert_s{42,123,7}/{metrics.json,test_predictions.jsonl}
# E4 saves preds in Train_Join.py's schema → register with Full_evaluation's
# --joint_preds (the command is printed per seed at the end).
#
# ── SAVE + RESUME ON TIMEOUT (read this) ────────────────────────────────────
# SAVE:   with DRIVE_OUT set, each seed's output dir is symlinked to Drive and the
#         run HARD-FAILS before training if the link is not Drive-backed+writable.
#         metrics.json + test_predictions.jsonl land on Drive; console.log streams
#         via the notebook's tee, so a partial log survives a mid-epoch kill.
# RESUME: 3 independent jobs (one per seed). A seed whose test_predictions.jsonl
#         already exists is SKIPPED. If Colab times out, RE-RUN THE SAME COMMAND —
#         it skips finished seeds and continues at the first incomplete one.
# LIMIT:  no intra-job resume. best_model/ is deleted after each seed completes
#         (repo checkpoint-deletion policy + Drive quota), so a timeout DURING a
#         seed restarts THAT seed from epoch 1. Worst-case loss = one seed.
#
# Usage:
#   # always dry-run first (prints plan + persistence gate, trains nothing):
#   DRIVE_OUT=/content/drive/MyDrive/IdiomatorRigor bash experiments/rigor/run_17b_bio_cls_joint.sh --dry-run
#   # full E4 (re-run the same line after any timeout to resume):
#   DRIVE_OUT=/content/drive/MyDrive/IdiomatorRigor bash experiments/rigor/run_17b_bio_cls_joint.sh
#   # one seed only:
#   SEEDS=42 DRIVE_OUT=... bash experiments/rigor/run_17b_bio_cls_joint.sh
#   # FORCE=1 retrains even completed seeds (use after a code change, never to
#   # "make sure" — a stale skip is not success, a fresh artifact is):
#   FORCE=1 DRIVE_OUT=... bash experiments/rigor/run_17b_bio_cls_joint.sh
# ──────────────────────────────────────────────────────────────────────────────

set -euo pipefail
cd "$(git -C "$(dirname "$0")" rev-parse --show-toplevel)"

DRY_RUN=0
[[ "${1:-}" == "--dry-run" ]] && DRY_RUN=1

SEEDS="${SEEDS:-42 123 7}"
FORCE="${FORCE:-0}"
LANGS=(English Spanish Hindi Telugu)
TEST_LANGS=(English Spanish Hindi Telugu Indonesian)   # Indonesian = held-out
TRAINER=experiments/rigor/run_17_bio_cls_joint.py
CLS_W=0.3            # identical to Train_Join.py default — same loss-weight convention as System E
BIO_W=1.9             # identical to Train_Join.py span_loss_weight default
LR=2e-5               # matches System D/E/F's default encoder LR (Train_Join.py default)
BATCH=32

echo "════════════════════════════════════════════════════════════════════════"
echo "E4 BIO+CLS joint   seeds=[${SEEDS}]   force=${FORCE}   dry_run=${DRY_RUN}"
echo "  encoder:    bert-base-multilingual-cased (same as Systems D/E/F)"
echo "  cls_w:      ${CLS_W}   bio_w: ${BIO_W}   lr: ${LR}   batch: ${BATCH}"
echo "  langs:      ${LANGS[*]}"
echo "  test_langs: ${TEST_LANGS[*]}"
echo "════════════════════════════════════════════════════════════════════════"

# ── Per-seed persistence gate (CLAUDE.md hard rule) ─────────────────────────
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

# ── Skip a seed iff its predictions exist AND FORCE!=1 ──────────────────────
should_run () {
    local preds="$1/test_predictions.jsonl"
    if [[ "$FORCE" != "1" && -f "$preds" ]]; then
        echo "  ✓ skip — $preds exists (set FORCE=1 to retrain)"
        return 1
    fi
    return 0
}

for SEED in $SEEDS; do
    OUT="models/bio_cls_joint_mbert_s${SEED}"
    echo
    echo "── E4 seed=${SEED}  → ${OUT} ───────────────────────────────────────────"
    if ! should_run "$OUT"; then
        continue
    fi

    if [[ "$DRY_RUN" == "1" ]]; then
        printf '   [dry-run] '
        printf '%q ' python "$TRAINER" \
            --output_dir "$OUT" \
            --model_name bert-base-multilingual-cased \
            --langs "${LANGS[@]}" --test_langs "${TEST_LANGS[@]}" \
            --cls_loss_weight "$CLS_W" --bio_loss_weight "$BIO_W" \
            --lr "$LR" --batch_size "$BATCH" --seed "$SEED"
        echo
        continue
    fi

    gate_dir "$OUT"
    python "$TRAINER" \
        --output_dir "$OUT" \
        --model_name bert-base-multilingual-cased \
        --langs "${LANGS[@]}" --test_langs "${TEST_LANGS[@]}" \
        --cls_loss_weight "$CLS_W" --bio_loss_weight "$BIO_W" \
        --lr "$LR" --batch_size "$BATCH" --seed "$SEED"

    # Checkpoint-deletion policy: keep metrics.json + test_predictions.jsonl,
    # drop the ~700MB encoder once preds are confirmed written (Drive quota).
    if [[ -f "$OUT/test_predictions.jsonl" ]]; then
        rm -rf "$OUT/best_model"
        echo "  ✓ seed ${SEED} done — best_model/ removed (preds + metrics kept)"
    fi
done

echo
echo "════════════════════════════════════════════════════════════════════════"
[[ "$DRY_RUN" == "1" ]] && { echo "Dry-run only — nothing trained."; exit 0; }
echo "E4 training complete. Register each seed as a system row with:"
echo
for SEED in $SEEDS; do
cat <<EOF
  python Evaluation/Full_evaluation.py \\
      --joint_preds models/bio_cls_joint_mbert_s${SEED}/test_predictions.jsonl \\
      --output_dir  results/e4_bio_cls_joint_s${SEED}
EOF
done
echo
echo "Then aggregate the 3 seeds → per-language mean ± std Joint F1 and compare"
echo "directly against System D/E's Joint F1 (the 'simpler BIO architecture, no"
echo "QA pointer needed' test). Run experiments/rigor/check_metric_drift.py"
echo "before any paper build."
echo "════════════════════════════════════════════════════════════════════════"
