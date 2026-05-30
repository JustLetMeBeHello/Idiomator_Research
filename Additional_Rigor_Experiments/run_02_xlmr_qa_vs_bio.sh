#!/usr/bin/env bash
# ── Experiment 02: XLM-R replication of QA-vs-BIO ────────────────────────────
#
# Verifies the QA-style-beats-BIO finding survives an encoder swap. Single
# biggest ceiling-mover identified by the council — if QA still beats BIO
# on Telugu under XLM-R, the architectural claim generalizes beyond mBERT
# and IdiomBERT becomes Main-eligible at NAACL 2027.
#
# Two trainings, run sequentially on the same GPU:
#   A) Joint mBERT (System E) → encoder = xlm-roberta-base
#   B) BIO Tagger (System G)  → encoder = xlm-roberta-base
#
# Important tokenizer note: XLM-R uses SentencePiece. The BIO run will
# probably *partially* recover on Telugu even before MuRIL, which is itself
# a publishable finding ("tokenizer dominates labeling-scheme choice").
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

# ── Persistence gate (CLAUDE.md hard rule) ──────────────────────────────────
# /content is ephemeral on Colab; only Drive survives a disconnect. Set
# DRIVE_OUT to a mounted-Drive path to force rigor outputs through to Drive
# via symlink, then HARD-FAIL before training if the link is not actually
# Drive-backed and writable. Leave DRIVE_OUT unset only for local runs.
if [[ -n "${DRIVE_OUT:-}" ]]; then
    for name in rigor_joint_xlmr_full rigor_bio_xlmr_full; do
        mkdir -p "$DRIVE_OUT/$name"
        rm -rf "models/$name"            # drop stale local dir/link
        ln -s "$DRIVE_OUT/$name" "models/$name"
    done
    # Gate: islink + target-under-Drive + write/readback proof, else exit 1.
    python - "$DRIVE_OUT" <<'PY'
import os, sys
drive = os.path.realpath(sys.argv[1])
for name in ("rigor_joint_xlmr_full", "rigor_bio_xlmr_full"):
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
    mkdir -p models/rigor_joint_xlmr_full models/rigor_bio_xlmr_full
fi

echo
echo "── A) System E + XLM-R (Joint) ────────────────────────────────────────"
python Train_Join.py \
    --model_name xlm-roberta-base \
    --output_dir models/rigor_joint_xlmr_full \
    --langs English Spanish Hindi Telugu \
    --test_langs English Spanish Hindi Telugu Indonesian \
    --epochs 7 \
    --batch_size 32 \
    --lr 1e-5 \
    --cls_loss_weight 0.3 \
    --span_loss_weight 1.9 \
    --seed 42

echo
echo "── B) System G + XLM-R (BIO) ──────────────────────────────────────────"
python Ablations/BiO_Task_mBERT_train.py \
    --model_name xlm-roberta-base \
    --output_dir models/rigor_bio_xlmr_full \
    --langs English Spanish Hindi Telugu \
    --test_langs English Spanish Hindi Telugu Indonesian \
    --epochs 6 \
    --batch_size 32 \
    --lr 3.27e-5 \
    --o_weight 0.104 \
    --seed 42

echo
echo "── Experiment 02 complete ─────────────────────────────────────────────"
echo "Key comparisons (vs IdiomBERT main paper Table 3):"
echo "  - Joint XLM-R Joint F1 (TE) vs System E mBERT (0.7903)"
echo "  - BIO XLM-R exact match (TE) vs System G mBERT (0.0000)"
echo "  - Indonesian zero-shot Joint F1 vs System E (0.7685)"
echo
echo "If BIO + XLM-R still fails on Telugu → architectural claim holds across"
echo "two encoders, two tokenizer families. Strong Main-track narrative."
