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

MODEL_DIR="rigor_joint_muril_full"

# ── Persistence gate (CLAUDE.md hard rule) ──────────────────────────────────
# Set DRIVE_OUT=/content/drive/MyDrive/IdiomatorRigor when running on Colab.
if [[ -n "${DRIVE_OUT:-}" ]]; then
    mkdir -p "$DRIVE_OUT/$MODEL_DIR"
    rm -rf "models/$MODEL_DIR"
    ln -s "$DRIVE_OUT/$MODEL_DIR" "models/$MODEL_DIR"
    python - "$DRIVE_OUT" "$MODEL_DIR" <<'PY'
import os, sys
drive = os.path.realpath(sys.argv[1])
for name in sys.argv[2:]:
    p = os.path.join("models", name)
    assert os.path.islink(p), f"{p} is not a symlink — output would be ephemeral"
    assert os.path.isdir(p), f"{p} symlink target missing on Drive"
    print(f"  ✓ {p} -> {os.path.realpath(p)}")
PY
else
    mkdir -p "models/$MODEL_DIR"
fi

python Train_Join.py \
    --model_name google/muril-base-cased \
    --output_dir "models/$MODEL_DIR" \
    --langs English Spanish Hindi Telugu \
    --test_langs English Spanish Hindi Telugu Indonesian \
    --epochs 7 \
    --batch_size 32 \
    --lr 2e-5 \
    --cls_loss_weight 0.5 \
    --span_loss_weight 1.9 \
    --seed 42
# NOTE: cls_loss_weight raised 0.3→0.5 for MuRIL. At 0.3 the cls head never
# escapes random-coin-flip (stuck at ln(2)=0.693 all 7 epochs) because the
# span gradient dominates. All other HP identical to System E mBERT.
# Paper footnote: "MuRIL required cls_loss_weight=0.5 to achieve cls convergence."

echo
echo "── Experiment 03 complete ─────────────────────────────────────────────"
echo "Compare MuRIL Joint vs System E mBERT baselines (computed live, never hardcoded):"
python Additional_Rigor_Experiments/mbert_baselines.py --system E || \
    echo "  (run Evaluation/Full_evaluation.py to populate the results json)"
echo "Expectation: MuRIL helps HI/TE, may drop ES (acceptable), Indonesian likely drops."
echo
echo "Add as row 'E (MuRIL)' to Tables 2, 4, 6, 7."
