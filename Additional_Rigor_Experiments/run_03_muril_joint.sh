#!/usr/bin/env bash
# ── Experiment 03: MuRIL Joint (System E) on full training ───────────────────
#
# Closes the "first publicly-released Telugu idiom resource but didn't use
# the Indic-aware encoder" gap that the domain expert flagged as
# indefensible at Main venues in 2026.
#
# Method: System E architecture (Joint mBERT design — cls head + start head +
# end head, shared encoder, asymmetric span-weighted loss), encoder swap
# only.
#   mBERT (bert-base-multilingual-cased) → MuRIL (google/muril-base-cased)
#
# Caveats acknowledged in advance:
#   - MuRIL was pretrained on Indic + English text; Spanish performance
#     may drop. Report it honestly — that asymmetry is the point.
#   - MuRIL's SentencePiece tokenizer handles Telugu script natively, so
#     this is the natural Indic-language encoder for the "first Telugu
#     resource" framing in MultiIdiom.
#
# Hyperparameters: identical to main-paper System E (lr=2e-5, 7 epochs,
# batch=32, β=0.3, γ=1.9). Loss weights stay at the main-paper value
# because re-tuning per encoder would confound the encoder-swap experiment.
#
# Run from Research_And_Training/ root.
# ──────────────────────────────────────────────────────────────────────────────

set -euo pipefail
cd "$(dirname "$0")/.."

mkdir -p models/rigor_joint_muril_full

python Train_Join.py \
    --model_name google/muril-base-cased \
    --output_dir models/rigor_joint_muril_full \
    --langs English Spanish Hindi Telugu \
    --test_langs English Spanish Hindi Telugu Indonesian \
    --epochs 7 \
    --batch_size 32 \
    --lr 2e-5 \
    --cls_loss_weight 0.3 \
    --span_loss_weight 1.9 \
    --seed 42

echo
echo "── Experiment 03 complete ─────────────────────────────────────────────"
echo "Compare MuRIL Joint vs System E mBERT baselines (computed live, never hardcoded):"
python Additional_Rigor_Experiments/mbert_baselines.py --system E || \
    echo "  (run Evaluation/Full_evaluation.py to populate the results json)"
echo "Expectation: MuRIL helps HI/TE, may drop ES (acceptable), Indonesian likely drops."
echo
echo "Add as row 'E (MuRIL)' to Tables 2, 4, 6, 7."
