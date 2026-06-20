#!/usr/bin/env bash
# ── Experiment 07: 2nd SentencePiece encoder (default RemBERT) ──────────────
#
# PURPOSE. Lock the tokenizer×architecture interaction for the IdiomBERT Main
# claim. The clean 3-encoder/3-seed flip matrix (Main_Readiness_IdiomBERT.ipynb)
# killed the "flip" headline and left a real, seed-stable finding:
#
#     QA ≈ BIO under WordPiece (mBERT, MuRIL)      — near parity
#     QA wins BIG under SentencePiece (XLM-R)      — large, tight gap
#
# The blocker to Main is sample size on the SentencePiece side: n=1 (XLM-R
# only). The claim currently reads "XLM-R amplifies QA," not "SentencePiece
# amplifies QA." This experiment adds a SECOND SentencePiece encoder to decide:
#
#     2nd SP encoder shows the large QA gain too → "SentencePiece amplifies QA"
#     2nd SP encoder does NOT                    → XLM-R-specific; claim narrows
#
# ENCODER CHOICE. Default = google/rembert (set via MODEL/ENC_SHORT env vars).
# The original target was microsoft/mdeberta-v3-base, but its disentangled
# attention is numerically unstable to fine-tune on a T4 (NaN loss within
# epoch 1, unrecoverable — see git history). RemBERT uses STANDARD attention,
# fine-tunes cleanly, and — being a different architecture AND a different SP
# tokenizer from XLM-R — makes the "it's the tokenizer, not the model" argument
# STRONGER, not weaker.
#
# TWO PROOFS (the notebook runs the cheap one first):
#   (1) MECHANISM, no GPU — tokenizer reachability. SentencePiece pushes gold
#       spans off word boundaries → high "subtok%" (QA-only-reachable). If the
#       2nd SP tokenizer patterns with XLM-R (not mBERT/MuRIL), that alone
#       predicts QA>BIO for SentencePiece. Runs in seconds, see notebook.
#   (2) EMPIRICAL, this script — train QA (joint) and BIO under the 2nd encoder
#       across 3 seeds and measure the QA−BIO exact gap per language.
#
# COMPATIBILITY GATE (hard). training/Train_Join.py and experiments/ablations/BiO_Task_mBERT_train.py
# pass token_type_ids unconditionally into the encoder. A new encoder may have a
# different type_vocab_size / tokenizer / offset behaviour. Do NOT launch the 6
# full runs blind — run `--dry-run` first (1 epoch English, joint+bio) and
# confirm both produce metrics.json + test_predictions.jsonl.
#
# HYPERPARAMETERS.
#   Joint (QA): epochs 7, bs $BATCH (16 for RemBERT; matrix used 32), lr 1e-5,
#               cls_loss_weight 0.3, span_loss_weight 1.9
#   BIO:        epochs 6, bs $BATCH, lr 3.27e-5 (System G default, untuned for SP
#               — kept fixed for a clean apples-to-apples vs XLM-R BIO)
#   bs note: QA−BIO gap is measured WITHIN an encoder (both at the same bs), and
#            the cross-encoder comparison is of gaps, so bs 16 vs 32 is not a
#            confound for the claim.
#   Seeds: 42 123 7   Langs: EN ES HI TE (train+test)  +ID test-only
#
# METRIC INTEGRITY. These are FRESH runs — NOT canonical, NOT in
# key_numbers.md. They must clear Evaluation/Full_evaluation.py before any
# paper use. The XLM-R numbers compared against live in the Drive flip dir and
# are likewise fresh. Do not paste any value here into a draft.
#
# Run from Research_And_Training/ root.
#   Dry-run (compat gate):  bash experiments/rigor/run_07_mdeberta_sp_replication.sh --dry-run
#   Full matrix:            bash experiments/rigor/run_07_mdeberta_sp_replication.sh
# ──────────────────────────────────────────────────────────────────────────────

set -euo pipefail
cd "$(git -C "$(dirname "$0")" rev-parse --show-toplevel)"

# Encoder is parameterised so this same script works for any 2nd SentencePiece
# encoder. Default = google/rembert (stable standard-attention SP encoder).
# mdeberta-v3 was the original target but its disentangled attention is too
# numerically unstable to fine-tune on a T4 (NaN loss); see git history.
# Override for a different encoder, e.g.:
#   MODEL=facebook/xlm-v-base ENC_SHORT=xlmv bash .../run_07_mdeberta_sp_replication.sh
MODEL="${MODEL:-google/rembert}"
ENC_SHORT="${ENC_SHORT:-rembert}"
BATCH="${BATCH:-16}"      # rembert ~576M params; 16 fits T4. (XLM-R matrix used 32.)
GRAD_ACCUM="${GRAD_ACCUM:-1}"  # gradient accumulation steps; effective batch = BATCH × GRAD_ACCUM
JOINT="training/Train_Join.py"
BIO="experiments/ablations/BiO_Task_mBERT_train.py"
LANGS="English Spanish Hindi Telugu"
TEST_LANGS="English Spanish Hindi Telugu Indonesian"

DRY_RUN=0
[[ "${1:-}" == "--dry-run" ]] && DRY_RUN=1

# ── Persistence gate (CLAUDE.md hard rule) ──────────────────────────────────
# /content is ephemeral on Colab; only Drive survives a disconnect. Point
# DRIVE_OUT at a mounted-Drive dir to force outputs through to Drive, then
# HARD-FAIL before training if the link is not actually Drive-backed/writable.
# Output dirs are created per run below; here we just gate the base path.
if [[ -n "${DRIVE_OUT:-}" ]]; then
    OUT_BASE="$DRIVE_OUT/$ENC_SHORT"
    mkdir -p "$OUT_BASE"
    python - "$DRIVE_OUT" <<'PY'
import os, sys
drive = os.path.realpath(sys.argv[1])
assert "drive" in drive.lower(), f"DRIVE_OUT does not look like a Drive path: {drive}"
probe = os.path.join(drive, ".persist_probe")
open(probe, "w").write("ok")
assert open(probe).read() == "ok", f"readback failed at {probe}"
os.remove(probe)
print(f"✓ persistence gate passed — outputs are Drive-backed and writable under {drive}")
PY
else
    OUT_BASE="models/$ENC_SHORT"
    mkdir -p "$OUT_BASE"
    echo "⚠ DRIVE_OUT unset — outputs go to local $OUT_BASE (OK locally; EPHEMERAL on Colab)"
fi

run_joint () {  # $1=outdir  $2=epochs  $3=seed
    python -u "$JOINT" \
        --model_name "$MODEL" --output_dir "$1" \
        --langs $LANGS --test_langs $TEST_LANGS \
        --epochs "$2" --batch_size "$BATCH" --grad_accum_steps "$GRAD_ACCUM" --lr "${LR_JOINT:-3e-6}" \
        --cls_loss_weight 0.3 --span_loss_weight 1.9 --seed "$3" \
        2>&1 | tee -a "$1/console.log"
}
run_bio () {    # $1=outdir  $2=epochs  $3=seed
    python -u "$BIO" \
        --model_name "$MODEL" --output_dir "$1" \
        --langs $LANGS --test_langs $TEST_LANGS \
        --epochs "$2" --batch_size "$BATCH" --lr "${LR_BIO:-2e-6}" --o_weight 0.104 --seed "$3" \
        2>&1 | tee -a "$1/console.log"
}

assert_outputs () {  # $1=outdir  $2=label — fail fast if the run produced nothing usable
    local d="$1" label="$2"
    if [[ ! -s "$d/test_predictions.jsonl" ]]; then
        echo "✗ $label: missing/empty test_predictions.jsonl in $d" >&2
        echo "  mDeBERTa compatibility likely broke (token_type_ids / SentencePiece tokenizer / offsets)." >&2
        echo "  Inspect $d/console.log before launching the full matrix." >&2
        exit 1
    fi
    echo "✓ $label: produced test_predictions.jsonl ($(wc -l < "$d/test_predictions.jsonl") rows)"
}

if [[ "$DRY_RUN" == "1" ]]; then
    echo "═══ DRY-RUN — $ENC_SHORT ($MODEL) compatibility gate (1 epoch, English only) ═══"
    DJ="$OUT_BASE/_dryrun_joint"; DB="$OUT_BASE/_dryrun_bio"
    mkdir -p "$DJ" "$DB"
    echo; echo "── dry A) Joint (QA) ──"
    python -u "$JOINT" --model_name "$MODEL" --output_dir "$DJ" \
        --langs English --test_langs English \
        --epochs 1 --batch_size 8 --lr "${LR_JOINT:-3e-6}" \
        --cls_loss_weight 0.3 --span_loss_weight 1.9 --seed 42 \
        2>&1 | tee -a "$DJ/console.log"
    assert_outputs "$DJ" "dry-joint"
    echo; echo "── dry B) BIO ──"
    python -u "$BIO" --model_name "$MODEL" --output_dir "$DB" \
        --langs English --test_langs English \
        --epochs 1 --batch_size 8 --lr "${LR_BIO:-2e-6}" --o_weight 0.104 --seed 42 \
        2>&1 | tee -a "$DB/console.log"
    assert_outputs "$DB" "dry-bio"
    echo
    echo "✓ DRY-RUN PASSED — $ENC_SHORT is compatible with both trainers."
    echo "  Now launch the full matrix (no --dry-run). Clean the dry dirs first:"
    echo "    rm -rf $DJ $DB"
    exit 0
fi

echo "═══ FULL MATRIX — $ENC_SHORT ($MODEL) QA vs BIO × seeds {42,123,7} ═══"
for SEED in 42 123 7; do
    JD="$OUT_BASE/${ENC_SHORT}_joint_s${SEED}"
    BD="$OUT_BASE/${ENC_SHORT}_bio_s${SEED}"
    mkdir -p "$JD" "$BD"

    echo; echo "── Joint (QA)  seed $SEED ──────────────────────────────────────"
    if [[ -s "$JD/metrics.json" && "${FORCE:-0}" != "1" ]]; then
        echo "skip (metrics.json exists; set FORCE=1 to overwrite)"
    else
        rm -rf "$JD/best_model"          # stale weights from a prior partial run
        run_joint "$JD" 7 "$SEED"
        assert_outputs "$JD" "joint s$SEED"
        rm -rf "$JD/best_model"          # checkpoint-deletion policy + Drive quota
    fi

    echo; echo "── BIO  seed $SEED ─────────────────────────────────────────────"
    if [[ -s "$BD/metrics.json" && "${FORCE:-0}" != "1" ]]; then
        echo "skip (metrics.json exists; set FORCE=1 to overwrite)"
    else
        rm -rf "$BD/best_model"
        run_bio "$BD" 6 "$SEED"
        assert_outputs "$BD" "bio s$SEED"
        rm -rf "$BD/best_model"
    fi
done

echo
echo "═══ Experiment 07 complete ═══"
echo "Aggregate the QA−BIO gap (live, vs XLM-R/mBERT/MuRIL) in the notebook."
echo "REMINDER: fresh runs — NOT canonical. Clear Full_evaluation.py before any paper use."
