#!/usr/bin/env bash
# run_language_ablation_matrix.sh
#
# Colab-safe runner for the IdiomBERT language-combination ablation matrix.
# It trains/evaluates the reproducible model families for every non-empty
# subset of English, Spanish, Hindi, and Telugu:
#
#   A: mBERT two-stage pipeline     (Stage 1 classifier + Stage 2 QA span)
#   E: Joint mBERT                  (classification + QA span in one model)
#   F: Sequential mBERT             (classification-dominant -> span-dominant)
#   G: BIO tagger                   (token-level sequence-labeling baseline)
#
# Usage in Colab:
#   from google.colab import drive
#   drive.mount('/content/drive')
#
#   %%bash
#   cd /content/drive/MyDrive/Idiomator_Research/Research_And_Training
#   bash Google_Colab/run_language_ablation_matrix.sh
#
# Resume behavior:
#   Every model run is skipped if its expected metrics/predictions already exist.
#   Re-running after a Colab timeout continues where it left off.

set -euo pipefail

ROOT="${ROOT:-$(pwd)}"
PYTHON="${PYTHON:-python3}"
DATA_DIR="${DATA_DIR:-idioms_structured/Splits}"
MODEL_NAME="${MODEL_NAME:-bert-base-multilingual-cased}"
DEVICE="${DEVICE:-cuda}"
TEST_LANGS="${TEST_LANGS:-English Spanish Hindi Telugu}"

ABLATION_DIR="${ABLATION_DIR:-models/language_ablation_matrix}"
EVAL_DIR="${EVAL_DIR:-results/language_ablation_matrix}"
LOG_DIR="${LOG_DIR:-results/language_ablation_matrix/logs}"

# Google Drive space saver:
#   0 = keep only paper artifacts (metrics, predictions, summaries, logs)
#       and delete bulky best_model checkpoints after each completed run.
#   1 = keep all model checkpoints.
KEEP_CHECKPOINTS="${KEEP_CHECKPOINTS:-0}"

# Locked hyperparameters from the paper/full-condition runs.
STAGE1_LR="${STAGE1_LR:-3e-5}"
STAGE1_EPOCHS="${STAGE1_EPOCHS:-7}"
STAGE1_BATCH="${STAGE1_BATCH:-32}"

STAGE2_LR="${STAGE2_LR:-1e-5}"
STAGE2_EPOCHS="${STAGE2_EPOCHS:-7}"
STAGE2_BATCH="${STAGE2_BATCH:-32}"

JOINT_LR="${JOINT_LR:-2e-5}"
JOINT_EPOCHS="${JOINT_EPOCHS:-7}"
JOINT_BATCH="${JOINT_BATCH:-32}"
JOINT_CLS_WEIGHT="${JOINT_CLS_WEIGHT:-0.3}"
JOINT_SPAN_WEIGHT="${JOINT_SPAN_WEIGHT:-1.9}"

SEQ_P1_LR="${SEQ_P1_LR:-1e-5}"
SEQ_P1_EPOCHS="${SEQ_P1_EPOCHS:-7}"
SEQ_P1_BATCH="${SEQ_P1_BATCH:-32}"
SEQ_P2_LR="${SEQ_P2_LR:-3e-5}"
SEQ_P2_EPOCHS="${SEQ_P2_EPOCHS:-5}"
SEQ_P2_BATCH="${SEQ_P2_BATCH:-16}"
SEQ_UNFREEZE_TOP="${SEQ_UNFREEZE_TOP:-3}"

BIO_LR="${BIO_LR:-3.27e-5}"
BIO_EPOCHS="${BIO_EPOCHS:-6}"
BIO_BATCH="${BIO_BATCH:-32}"
BIO_DROPOUT="${BIO_DROPOUT:-0.239431179668018881}"
BIO_O_WEIGHT="${BIO_O_WEIGHT:-0.104}"

mkdir -p "$ABLATION_DIR" "$EVAL_DIR" "$LOG_DIR"

cd "$ROOT"

cleanup_checkpoints() {
  if [[ "$KEEP_CHECKPOINTS" == "1" ]]; then
    return
  fi

  local run_id="$1"
  local base="$ABLATION_DIR/$run_id"

  # These folders contain the large HuggingFace checkpoints. The metrics and
  # test_predictions.jsonl files remain in place, so completed runs still skip
  # correctly and all paper tables can be regenerated.
  rm -rf "$base/stage1_mbert/best_model"
  rm -rf "$base/stage2_mbert/best_model"
  rm -rf "$base/joint_mbert/best_model"
  rm -rf "$base/sequential_mbert/phase1/best_model"
  rm -rf "$base/sequential_mbert/phase2/best_model"
  rm -rf "$base/bio_tagger/best_model"
}

run_cmd() {
  local name="$1"
  shift
  echo ""
  echo "================================================================================"
  echo "$name"
  echo "================================================================================"
  "$@" 2>&1 | tee "$LOG_DIR/${name}.log"
}

train_stage1() {
  local run_id="$1"
  shift
  local out="$ABLATION_DIR/$run_id/stage1_mbert"
  if [[ -f "$out/metrics.json" && -f "$out/test_predictions.jsonl" ]]; then
    echo "✓ Stage 1 exists: $out"
    return
  fi
  run_cmd "${run_id}__stage1_mbert" \
    "$PYTHON" Base_Pipeline/Stage_1_training.py \
      --model_name "$MODEL_NAME" \
      --data_dir "$DATA_DIR" \
      --output_dir "$out" \
      --langs "$@" \
      --test_langs $TEST_LANGS \
      --epochs "$STAGE1_EPOCHS" \
      --batch_size "$STAGE1_BATCH" \
      --lr "$STAGE1_LR" \
      --device "$DEVICE"
}

train_stage2() {
  local run_id="$1"
  shift
  local out="$ABLATION_DIR/$run_id/stage2_mbert"
  if [[ -f "$out/metrics.json" && -f "$out/test_predictions.jsonl" ]]; then
    echo "✓ Stage 2 exists: $out"
    return
  fi
  run_cmd "${run_id}__stage2_mbert" \
    "$PYTHON" Base_Pipeline/Stage_2_training.py \
      --model_name "$MODEL_NAME" \
      --data_dir "$DATA_DIR" \
      --output_dir "$out" \
      --langs "$@" \
      --test_langs $TEST_LANGS \
      --epochs "$STAGE2_EPOCHS" \
      --batch_size "$STAGE2_BATCH" \
      --lr "$STAGE2_LR" \
      --device "$DEVICE"
}

train_joint() {
  local run_id="$1"
  shift
  local out="$ABLATION_DIR/$run_id/joint_mbert"
  if [[ -f "$out/metrics.json" && -f "$out/test_predictions.jsonl" ]]; then
    echo "✓ Joint exists: $out"
    return
  fi
  run_cmd "${run_id}__joint_mbert" \
    "$PYTHON" Train_Join.py \
      --model_name "$MODEL_NAME" \
      --data_dir "$DATA_DIR" \
      --output_dir "$out" \
      --langs "$@" \
      --test_langs $TEST_LANGS \
      --epochs "$JOINT_EPOCHS" \
      --batch_size "$JOINT_BATCH" \
      --lr "$JOINT_LR" \
      --cls_loss_weight "$JOINT_CLS_WEIGHT" \
      --span_loss_weight "$JOINT_SPAN_WEIGHT" \
      --device "$DEVICE"
}

train_sequential() {
  local run_id="$1"
  shift
  local out="$ABLATION_DIR/$run_id/sequential_mbert"
  if [[ -f "$out/phase1/metrics.json" && -f "$out/phase2/metrics.json" \
        && -f "$out/phase1/test_predictions.jsonl" && -f "$out/phase2/test_predictions.jsonl" ]]; then
    echo "✓ Sequential exists: $out"
    return
  fi
  run_cmd "${run_id}__sequential_mbert" \
    "$PYTHON" Train_Sequential.py \
      --model_name "$MODEL_NAME" \
      --data_dir "$DATA_DIR" \
      --output_dir "$out" \
      --langs "$@" \
      --test_langs $TEST_LANGS \
      --p1_epochs "$SEQ_P1_EPOCHS" \
      --p1_batch_size "$SEQ_P1_BATCH" \
      --p1_lr "$SEQ_P1_LR" \
      --p2_epochs "$SEQ_P2_EPOCHS" \
      --p2_batch_size "$SEQ_P2_BATCH" \
      --p2_lr "$SEQ_P2_LR" \
      --unfreeze_top_layers "$SEQ_UNFREEZE_TOP" \
      --device "$DEVICE"
}

train_bio() {
  local run_id="$1"
  shift
  local out="$ABLATION_DIR/$run_id/bio_tagger"
  if [[ -f "$out/metrics.json" && -f "$out/test_predictions.jsonl" ]]; then
    echo "✓ BIO exists: $out"
    return
  fi
  run_cmd "${run_id}__bio_tagger" \
    "$PYTHON" Ablations/BiO_Task_mBERT_train.py \
      --model_name "$MODEL_NAME" \
      --data_dir "$DATA_DIR" \
      --output_dir "$out" \
      --langs "$@" \
      --test_langs $TEST_LANGS \
      --epochs "$BIO_EPOCHS" \
      --batch_size "$BIO_BATCH" \
      --lr "$BIO_LR" \
      --dropout "$BIO_DROPOUT" \
      --o_weight "$BIO_O_WEIGHT" \
      --device "$DEVICE"
}

evaluate_combo() {
  local run_id="$1"
  local out="$EVAL_DIR/$run_id"
  if [[ -f "$out/pipeline_eval_results.json" ]]; then
    echo "✓ Evaluation exists: $out"
    return
  fi
  mkdir -p "$out"
  run_cmd "${run_id}__full_eval" \
    "$PYTHON" Evaluation/Full_evaluation.py \
      --stage1_mbert "$ABLATION_DIR/$run_id/stage1_mbert/test_predictions.jsonl" \
      --stage2_mbert "$ABLATION_DIR/$run_id/stage2_mbert/test_predictions.jsonl" \
      --stage1_gpt "$ABLATION_DIR/$run_id/missing_gpt_stage1.jsonl" \
      --stage2_gpt "$ABLATION_DIR/$run_id/missing_gpt_stage2.jsonl" \
      --single_gpt "$ABLATION_DIR/$run_id/missing_gpt_single.jsonl" \
      --joint_preds "$ABLATION_DIR/$run_id/joint_mbert/test_predictions.jsonl" \
      --span2_joint "$ABLATION_DIR/$run_id/missing_joint_span_only.jsonl" \
      --seq_phase1 "$ABLATION_DIR/$run_id/sequential_mbert/phase1/test_predictions.jsonl" \
      --seq_phase2 "$ABLATION_DIR/$run_id/sequential_mbert/phase2/test_predictions.jsonl" \
      --bio_preds "$ABLATION_DIR/$run_id/bio_tagger/test_predictions.jsonl" \
      --output_dir "$out"
}

run_combo() {
  local run_id="$1"
  shift
  echo ""
  echo "################################################################################"
  echo "LANGUAGE COMBINATION: $run_id ($*)"
  echo "################################################################################"
  train_stage1 "$run_id" "$@"
  train_stage2 "$run_id" "$@"
  train_joint "$run_id" "$@"
  train_sequential "$run_id" "$@"
  train_bio "$run_id" "$@"
  evaluate_combo "$run_id"
  cleanup_checkpoints "$run_id"
}

run_combo "en" English
run_combo "es" Spanish
run_combo "hi" Hindi
run_combo "te" Telugu

run_combo "en_es" English Spanish
run_combo "en_hi" English Hindi
run_combo "en_te" English Telugu
run_combo "es_hi" Spanish Hindi
run_combo "es_te" Spanish Telugu
run_combo "hi_te" Hindi Telugu

run_combo "en_es_hi" English Spanish Hindi
run_combo "en_es_te" English Spanish Telugu
run_combo "en_hi_te" English Hindi Telugu
run_combo "es_hi_te" Spanish Hindi Telugu

run_combo "en_es_hi_te" English Spanish Hindi Telugu

"$PYTHON" Google_Colab/summarize_language_ablation_matrix.py \
  --eval_dir "$EVAL_DIR" \
  --output_dir "$EVAL_DIR"

echo ""
echo "Done. Summary files:"
echo "  $EVAL_DIR/ablation_summary.csv"
echo "  $EVAL_DIR/stability_summary.csv"
echo "  $EVAL_DIR/transfer_matrix.csv"
