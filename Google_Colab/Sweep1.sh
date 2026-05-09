#!/bin/bash
# sweep_stage1.sh
#
# Hyperparameter sweep for Stage 1 idiomaticity classifier.
# Runs a grid search over learning rate x epochs, evaluates on dev,
# saves only the best checkpoint per config, logs all results to CSV.
#
# Usage (in Colab cell):
#   !bash /content/drive/MyDrive/Idiomator_Training/scripts/sweep_stage1.sh
#
# Or with custom drive root:
#   !DRIVE_ROOT=/content/drive/MyDrive/Idiomator_Training bash scripts/sweep_stage1.sh

set -e

# ── Paths ─────────────────────────────────────────────────────────────────────

DRIVE_ROOT="${DRIVE_ROOT:-/content/drive/MyDrive/Idiomator_Training}"
DATA_DIR="$DRIVE_ROOT/data/Splits"
SCRIPTS_DIR="$DRIVE_ROOT/scripts"
RESULTS_DIR="$DRIVE_ROOT/results"
MODELS_DIR="$DRIVE_ROOT/models/sweep"
PYTHON="${PYTHON:-python}"

mkdir -p "$RESULTS_DIR" "$MODELS_DIR"

# ── Sweep config ──────────────────────────────────────────────────────────────

LANGS="English Hindi Telugu"
MODEL="bert-base-multilingual-cased"
BATCH_SIZE=32
MAX_LEN=128
WARMUP_RATIO=0.1
SEED=42

LEARNING_RATES=(1e-5 2e-5 3e-5 5e-5)
EPOCHS_LIST=(3 5 7)

# ── Results CSV ───────────────────────────────────────────────────────────────

RESULTS_CSV="$RESULTS_DIR/sweep_stage1_results.csv"

# Write header if file doesn't exist
if [ ! -f "$RESULTS_CSV" ]; then
    echo "run_id,model,langs,lr,epochs,batch_size,best_dev_f1,best_epoch,test_macro_f1,en_f1,hi_f1,te_f1,output_dir" > "$RESULTS_CSV"
    echo "Created results CSV: $RESULTS_CSV"
fi

# ── Sweep ─────────────────────────────────────────────────────────────────────

RUN_ID=0

for LR in "${LEARNING_RATES[@]}"; do
    for EPOCHS in "${EPOCHS_LIST[@]}"; do

        RUN_ID=$((RUN_ID + 1))
        RUN_NAME="run${RUN_ID}_lr${LR}_ep${EPOCHS}"
        OUTPUT_DIR="$MODELS_DIR/$RUN_NAME"

        echo ""
        echo "════════════════════════════════════════════════════"
        echo "  Run $RUN_ID / $((${#LEARNING_RATES[@]} * ${#EPOCHS_LIST[@]}))"
        echo "  lr=$LR  epochs=$EPOCHS"
        echo "  output → $OUTPUT_DIR"
        echo "════════════════════════════════════════════════════"

        # Skip if already completed
        if [ -f "$OUTPUT_DIR/metrics.json" ]; then
            echo "  Already completed — skipping."
            continue
        fi

        # Train
        $PYTHON "$SCRIPTS_DIR/train_stage1_classifier.py" \
            --model_name    "$MODEL" \
            --data_dir      "$DATA_DIR" \
            --output_dir    "$OUTPUT_DIR" \
            --langs         $LANGS \
            --epochs        "$EPOCHS" \
            --batch_size    "$BATCH_SIZE" \
            --lr            "$LR" \
            --max_len       "$MAX_LEN" \
            --warmup_ratio  "$WARMUP_RATIO" \
            --seed          "$SEED"

        # Parse metrics.json and append to CSV
        $PYTHON - <<EOF
import json, csv, os, sys

metrics_path = "$OUTPUT_DIR/metrics.json"
results_csv  = "$RESULTS_CSV"

if not os.path.exists(metrics_path):
    print(f"  ✗ metrics.json not found at {metrics_path}")
    sys.exit(1)

m = json.load(open(metrics_path))

# Parse per-language F1 from test_predictions.jsonl
from collections import defaultdict
from sklearn.metrics import f1_score

preds_path = "$OUTPUT_DIR/test_predictions.jsonl"
lang_data  = defaultdict(lambda: {'preds': [], 'labels': []})
label2id   = {'literal': 0, 'idiomatic': 1}

for line in open(preds_path, encoding='utf-8'):
    r = json.loads(line)
    lang_data[r['language']]['preds'].append(label2id[r['pred_idiomaticity']])
    lang_data[r['language']]['labels'].append(label2id[r['idiomaticity']])

lang_f1 = {}
for lang, data in lang_data.items():
    lang_f1[lang] = round(f1_score(data['labels'], data['preds'], average='macro'), 4)

row = {
    'run_id':        "$RUN_NAME",
    'model':         m.get('model', ''),
    'langs':         '+'.join(m.get('langs', [])),
    'lr':            "$LR",
    'epochs':        "$EPOCHS",
    'batch_size':    "$BATCH_SIZE",
    'best_dev_f1':   round(m.get('best_dev_f1', 0), 4),
    'best_epoch':    m.get('best_epoch', ''),
    'test_macro_f1': round(m.get('test_macro_f1', 0), 4),
    'en_f1':         lang_f1.get('English', ''),
    'hi_f1':         lang_f1.get('Hindi', ''),
    'te_f1':         lang_f1.get('Telugu', ''),
    'output_dir':    "$OUTPUT_DIR",
}

with open(results_csv, 'a', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=row.keys())
    writer.writerow(row)

print(f"  ✓ Logged to CSV: dev_f1={row['best_dev_f1']}  test_f1={row['test_macro_f1']}")
print(f"    EN={row['en_f1']}  HI={row['hi_f1']}  TE={row['te_f1']}")
EOF

        # Remove non-best checkpoints to save Drive space
        # Keep only best_model/ and metrics/predictions
        find "$OUTPUT_DIR" -maxdepth 1 -name "checkpoint-*" -type d -exec rm -rf {} + 2>/dev/null || true

        echo "  ✓ Run $RUN_ID complete."

    done
done

echo ""
echo "════════════════════════════════════════════════════"
echo "  Sweep complete! Results → $RESULTS_CSV"
echo "════════════════════════════════════════════════════"

# Print summary table
$PYTHON - <<EOF
import csv

results_csv = "$RESULTS_CSV"
rows = list(csv.DictReader(open(results_csv)))

if not rows:
    print("No results yet.")
else:
    rows.sort(key=lambda r: float(r['best_dev_f1']), reverse=True)
    print(f"\n{'Run':<25} {'LR':<8} {'EP':<4} {'Dev F1':<10} {'Test F1':<10} {'EN':<8} {'HI':<8} {'TE':<8}")
    print("-" * 85)
    for r in rows:
        print(f"{r['run_id']:<25} {r['lr']:<8} {r['epochs']:<4} {r['best_dev_f1']:<10} {r['test_macro_f1']:<10} {r['en_f1']:<8} {r['hi_f1']:<8} {r['te_f1']:<8}")
    print(f"\nBest config: lr={rows[0]['lr']}  epochs={rows[0]['epochs']}  dev_f1={rows[0]['best_dev_f1']}")
EOF