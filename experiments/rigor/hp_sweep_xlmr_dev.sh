#!/usr/bin/env bash
# ── Optional: XLM-R learning-rate sweep on DEV only ──────────────────────────
#
# DO NOT run this by default. It's the safety valve if a reviewer asks
# "did you tune XLM-R fairly?" — three extra training runs over the dev
# set, no test set touched (so it doesn't leak into the headline numbers).
#
# Rationale: the main paper's protocol fixes hyperparameters to isolate the
# variable under study. Per-encoder tuning would confound encoder choice
# with hyperparameter search. But if a reviewer specifically questions the
# 1e-5 default for XLM-R Joint, this sweep produces a one-line response:
# "We performed a 3-point dev-only LR sweep over {5e-6, 1e-5, 2e-5}; the
# best dev score was at LR={X}, matching/differing-by-Y% from our chosen
# default. Final test numbers reported are with the published default."
#
# Each run trains on full EN+ES+HI+TE for 7 epochs. Skip test eval to save
# time — we only care about dev performance for HP selection.
#
# Run from Research_And_Training/ root.
# ──────────────────────────────────────────────────────────────────────────────

set -euo pipefail
cd "$(git -C "$(dirname "$0")" rev-parse --show-toplevel)"

for LR in 5e-6 1e-5 2e-5; do
    OUT="models/rigor_xlmr_hp_sweep/lr_${LR}"
    mkdir -p "$OUT"
    echo
    echo "── XLM-R Joint sweep: lr=$LR ─────────────────────────────────────────"

    python training/Train_Join.py \
        --model_name xlm-roberta-base \
        --output_dir "$OUT" \
        --langs English Spanish Hindi Telugu \
        --test_langs English Spanish Hindi Telugu \
        --epochs 7 \
        --batch_size 32 \
        --lr "$LR" \
        --cls_loss_weight 0.3 \
        --span_loss_weight 1.9 \
        --seed 42
done

echo
echo "── HP sweep complete ──────────────────────────────────────────────────"
echo "Inspect dev metrics across LRs:"
echo "  for d in models/rigor_xlmr_hp_sweep/lr_*; do"
echo "    echo \"\$d:\""
echo "    jq '.dev_metrics // .best_dev_score' \$d/metrics.json"
echo "  done"
echo
echo "Report-back template for reviewer response:"
echo "  '3-point dev-only LR sweep {5e-6, 1e-5, 2e-5} for XLM-R Joint. Best"
echo "   dev Joint F1 at LR=X. Final test numbers reported with the encoder"
echo "   release-paper default (1e-5).'"
