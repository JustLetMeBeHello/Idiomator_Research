#!/usr/bin/env bash
# ── Experiment 02: XLM-R replication of QA-vs-BIO ────────────────────────────
#
# Tests whether the Joint-vs-BIO comparison survives an encoder swap from
# mBERT to XLM-R.
#
# CORRECTED 2026-05-28: the original premise here — "mBERT BIO fails on Telugu
# (exact=0.00)" — was a decoder/encoder bug, NOT a real result. Post-fix,
# System G mBERT BIO scores Telugu exact=0.77 / overlap F1=0.88 (see
# memory/key_numbers.md). Pure BIO is a competitive multilingual span
# extractor; the QA/staged advantage is in joint classify+extract, not span
# extraction. So this run checks whether the Joint-F1 gap holds under a second
# encoder — it is NOT testing a BIO span-extraction failure.
#
# Two trainings, run sequentially on the same GPU:
#   A) Joint mBERT (System E) → encoder = xlm-roberta-base
#   B) BIO Tagger (System G)  → encoder = xlm-roberta-base
#
# Tokenizer note: XLM-R uses SentencePiece vs mBERT WordPiece. Compare XLM-R
# BIO Telugu span exact against System G mBERT's corrected 0.77 to see whether
# the tokenizer family shifts span quality either way.
#
# Hyperparameters: identical to the main-paper Systems E and G, except lr
# dropped from 2e-5 → 1e-5 for the Joint head (XLM-R standard practice).
# BIO lr kept at the System G default (3.27e-5) since LR-sensitivity for
# BIO has not been re-tuned for XLM-R and a clean apples-to-apples is the
# goal of this experiment.
#
# Run from Research_And_Training/ root.
# ──────────────────────────────────────────────────────────────────────────────

set -euo pipefail
cd "$(dirname "$0")/.."

# Seed: default 42; override with SEED=123 or SEED=7 for multi-seed canonical runs.
# Seed 42 is already canonical (rigor_xlmr_joint_s42 in pipeline_eval_results.json).
# Run seeds 123 and 7 with:
#   SEED=123 DRIVE_OUT=... bash run_02_xlmr_qa_vs_bio.sh
#   SEED=7   DRIVE_OUT=... bash run_02_xlmr_qa_vs_bio.sh
SEED="${SEED:-42}"
JOINT_DIR="rigor_xlmr_joint_s${SEED}"
BIO_DIR="rigor_bio_xlmr_s${SEED}"

# ── Persistence gate (CLAUDE.md hard rule) ──────────────────────────────────
# /content is ephemeral on Colab; only Drive survives a disconnect. Set
# DRIVE_OUT to a mounted-Drive path to force rigor outputs through to Drive
# via symlink, then HARD-FAIL before training if the link is not actually
# Drive-backed and writable. Leave DRIVE_OUT unset only for local runs.
if [[ -n "${DRIVE_OUT:-}" ]]; then
    for name in "$JOINT_DIR" "$BIO_DIR"; do
        mkdir -p "$DRIVE_OUT/$name"
        rm -rf "models/$name"
        ln -s "$DRIVE_OUT/$name" "models/$name"
    done
    python - "$DRIVE_OUT" "$JOINT_DIR" "$BIO_DIR" <<'PY'
import os, sys
drive = os.path.realpath(sys.argv[1])
for name in sys.argv[2:]:
    p = os.path.join("models", name)
    assert os.path.islink(p), f"{p} is not a symlink — output would be ephemeral"
    tgt = os.path.realpath(p)
    assert tgt.startswith(drive), f"{p} -> {tgt} not under Drive ({drive})"
    probe = os.path.join(p, ".persist_probe")
    open(probe, "w").write("ok")
    assert open(probe).read() == "ok", f"readback failed at {probe}"
    os.remove(probe)
print("✓ persistence gate passed — rigor outputs are Drive-backed and writable")
PY
else
    echo "⚠ DRIVE_OUT unset — outputs go to local models/ (OK locally; EPHEMERAL on Colab)"
    mkdir -p "models/$JOINT_DIR" "models/$BIO_DIR"
fi

echo
echo "── A) System E + XLM-R (Joint) seed=${SEED} ──────────────────────────"
python Train_Join.py \
    --model_name xlm-roberta-base \
    --output_dir "models/$JOINT_DIR" \
    --langs English Spanish Hindi Telugu \
    --test_langs English Spanish Hindi Telugu Indonesian \
    --epochs 7 \
    --batch_size 32 \
    --lr 1e-5 \
    --cls_loss_weight 0.3 \
    --span_loss_weight 1.9 \
    --seed "$SEED"

echo
echo "── B) System G + XLM-R (BIO) seed=${SEED} ────────────────────────────"
python Ablations/BiO_Task_mBERT_train.py \
    --model_name xlm-roberta-base \
    --output_dir "models/$BIO_DIR" \
    --langs English Spanish Hindi Telugu \
    --test_langs English Spanish Hindi Telugu Indonesian \
    --epochs 6 \
    --batch_size 32 \
    --lr 3.27e-5 \
    --o_weight 0.104 \
    --seed "$SEED"

echo
echo "── Experiment 02 complete (seed=${SEED}) ──────────────────────────────"
echo "Run Evaluation/Full_evaluation.py pointing at models/$JOINT_DIR and"
echo "models/$BIO_DIR to produce canonical rows for pipeline_eval_results.json."
echo "mBERT baselines (computed live):"
python Additional_Rigor_Experiments/mbert_baselines.py || \
    echo "  (run Evaluation/Full_evaluation.py to populate the results json)"
echo
echo "NOTE: 'mBERT BIO Telugu exact = 0.00' was a decoder/encoder bug, since"  # metric-drift:ok
echo "corrected in the eval json (System G TE exact = 0.77). mBERT BIO does NOT"
echo "fail on Telugu. Frame this run as: does the Joint-vs-BIO gap survive an"
echo "XLM-R encoder swap? — not as a BIO span-extraction failure test."
