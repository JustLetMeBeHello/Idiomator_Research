#!/bin/bash
# ablations_stage1.sh
#
# Runs all Stage 1 language ablation experiments with locked best config:
#   lr=1e-5, epochs=7, batch_size=32
#
# Usage in Colab:
#   %%bash
#   cd /content/drive/MyDrive/Idiomator_Research
#   export WANDB_API_KEY=your_key_here
#   bash Google_Colab/ablations_stage1.sh

set -e

# ── Config ────────────────────────────────────────────────────────────────────

DRIVE_ROOT="${DRIVE_ROOT:-/content/drive/MyDrive/Idiomator_Research}"
SCRIPT="$DRIVE_ROOT/Stage_1_training.py"
MODELS_DIR="$DRIVE_ROOT/models/ablations"
RESULTS_DIR="$DRIVE_ROOT/results"
PYTHON="${PYTHON:-python3}"

# Locked hyperparameters from sweep
LR=1e-5
EPOCHS=7
BATCH_SIZE=32

mkdir -p "$MODELS_DIR" "$RESULTS_DIR"

# ── Ablation runs ─────────────────────────────────────────────────────────────

declare -a RUN_NAMES=(
    "mbert_hi_te"
    "mbert_en_te"
    "mbert_en_hi"
    "mbert_en"
    "mbert_hi"
    "mbert_te"
    "bert_en_monolingual"
)

declare -a RUN_LANGS=(
    "Hindi Telugu"
    "English Telugu"
    "English Hindi"
    "English"
    "Hindi"
    "Telugu"
    "English"
)

declare -a RUN_MODELS=(
    "bert-base-multilingual-cased"
    "bert-base-multilingual-cased"
    "bert-base-multilingual-cased"
    "bert-base-multilingual-cased"
    "bert-base-multilingual-cased"
    "bert-base-multilingual-cased"
    "bert-base-uncased"
)

TOTAL=${#RUN_NAMES[@]}

for i in "${!RUN_NAMES[@]}"; do
    RUN_NAME="${RUN_NAMES[$i]}"
    LANGS="${RUN_LANGS[$i]}"
    MODEL="${RUN_MODELS[$i]}"
    OUTPUT_DIR="$MODELS_DIR/$RUN_NAME"
    RUN_NUM=$((i + 1))

    echo ""
    echo "════════════════════════════════════════════════════"
    echo "  Ablation $RUN_NUM / $TOTAL : $RUN_NAME"
    echo "  langs=$LANGS"
    echo "  model=$MODEL"
    echo "  lr=$LR  epochs=$EPOCHS  bs=$BATCH_SIZE"
    echo "  output → $OUTPUT_DIR"
    echo "════════════════════════════════════════════════════"

    # Skip if already completed
    if [ -f "$OUTPUT_DIR/metrics.json" ]; then
        echo "  Already completed — skipping."
        continue
    fi

    $PYTHON "$SCRIPT" \
        --model_name    "$MODEL" \
        --data_dir      "$DRIVE_ROOT/idioms_structured/Splits" \
        --output_dir    "$OUTPUT_DIR" \
        --langs         $LANGS \
        --lr            "$LR" \
        --epochs        "$EPOCHS" \
        --batch_size    "$BATCH_SIZE" \
        --use_wandb

    echo "  ✓ Ablation $RUN_NUM complete."
done

echo ""
echo "════════════════════════════════════════════════════"
echo "  All ablations complete!"
echo "════════════════════════════════════════════════════"

# Collect results
$PYTHON "$DRIVE_ROOT/collect_results.py" \
    --models_dir "$MODELS_DIR" \
    --results_dir "$RESULTS_DIR" \
    --output stage1_ablation_results.csv

echo "Results → $RESULTS_DIR/stage1_ablation_results.csv"