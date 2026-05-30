#!/usr/bin/env bash
# ── Experiment 01: BIO + MuRIL Telugu recovery ───────────────────────────────
#
# PERMANENTLY SKIPPED (do not run). Its premise — "Telugu BIO exact match =
# 0.0000" — was a decoder/encoder bug, not a real result. Post-fix, System G
# mBERT BIO scores Telugu exact = 0.77 (see memory/key_numbers.md), so the
# MuRIL recovery this script was built to test has nothing left to recover.
# Kept only for provenance.
#
# Original (now-invalid) premise: whether the Telugu BIO "failure" was
# architectural or tokenizer-driven (mBERT WordPiece fragments Telugu abugida).
#
# Method: identical System G setup, only encoder swapped.
#   mBERT (bert-base-multilingual-cased) → MuRIL (google/muril-base-cased)
# MuRIL ships an Indic-aware SentencePiece tokenizer with native Telugu
# script coverage, isolating the tokenizer-vs-architecture question.
#
# Expected outcomes (decision-influencing):
#   - TE exact match > 0.20 → tokenizer was the cause. MultiIdiom gains
#     a Main-worthy "cross-lingual recipe" story; IdiomBERT's QA-vs-BIO
#     claim weakens to "BIO + WordPiece fails on non-Latin abugida."
#   - TE exact match ≈ 0.00 → architecture is the cause. IdiomBERT's
#     "QA-style is necessary" claim hardens.
#
# Run from Research_And_Training/ root.
# Resolved from script location so it works whether invoked via bash, sh,
# or `chmod +x`.
# ──────────────────────────────────────────────────────────────────────────────

set -euo pipefail
cd "$(dirname "$0")/.."

mkdir -p models/rigor_bio_muril_full

python Ablations/BiO_Task_mBERT_train.py \
    --model_name google/muril-base-cased \
    --output_dir models/rigor_bio_muril_full \
    --langs English Spanish Hindi Telugu \
    --test_langs English Spanish Hindi Telugu Indonesian \
    --epochs 6 \
    --batch_size 32 \
    --lr 3.27e-5 \
    --o_weight 0.104 \
    --seed 42

echo
echo "── Experiment 01 complete ─────────────────────────────────────────────"
echo "Compare TE exact match to System G mBERT baseline (0.77, post decoder fix)."
echo "Predictions: models/rigor_bio_muril_full/test_predictions.jsonl"
echo "Run Evaluation/Full_evaluation.py to fold into the system table."
