#!/usr/bin/env bash
# ── Experiment E1: RemBERT 3 seeds (42/123/7) at BATCH 32 ───────────────────
#
# REVISION_PLAN E1. The C1 headline encoder study compares RemBERT against the
# other encoders, but RemBERT was trained at batch_size 16 while everything else
# used 32 — a batch confound sitting inside the flagship claim. E1 removes it:
# re-train RemBERT (QA-joint + BIO) at batch 32, 3 seeds, with RemBERT's OWN
# learning rates (LR_JOINT=3e-6 / LR_BIO=2e-6 — NOT XLM-R's higher LRs), so the
# ONLY changed variable vs the existing batch-16 RemBERT is the batch size.
#
# This is a thin wrapper over run_07_mdeberta_sp_replication.sh — it bakes in the
# E1 config and delegates ALL launch gating (Drive persistence gate, per-job
# skip/resume, dry-run, output assertions) to that tested runner. No logic is
# duplicated here.
#
# Outputs land under <root>/rembert32/rembert32_{joint,bio}_s{seed}. The "32"
# suffix keeps E1 distinct from the batch-16 RemBERT on disk. To fold E1 into the
# C1 strip-gap analysis, run run_08 / run_08b / run_13 with REMBERT_ENC=rembert32
# (they read this layout via that env var).
#
# ── SAVE + RESUME ON TIMEOUT (read this) ────────────────────────────────────
# SAVE:   with DRIVE_OUT set, run_07's persistence gate symlinks every output dir
#         to Drive and HARD-FAILS before training if a dir is not Drive-backed
#         and writable. Each finished (seed,system) writes metrics.json +
#         test_predictions.jsonl to Drive; console.log streams incrementally
#         (tee -a), so a partial log survives a mid-epoch kill.
# RESUME: the matrix is 6 independent jobs (3 seeds × {joint,bio}). run_07 SKIPS
#         any job whose metrics.json already exists. So if the Colab session
#         times out, just RE-RUN THIS SAME COMMAND — it skips the finished jobs
#         and continues at the first incomplete one. The matrix self-heals across
#         as many sessions as it takes.
# LIMIT:  there is NO mid-training (intra-job) resume — the trainers have no
#         --resume_from_checkpoint and best_model/ is deleted after each job
#         (repo checkpoint-deletion policy + Drive quota). A timeout DURING a job
#         restarts THAT job from epoch 1 on re-run. Worst-case loss = one job
#         (~40–60 min on a T4), never the whole matrix. Budget ~5 GPU-hr total.
#
# Usage:
#   # always smoke-test the migration first (1 epoch, English only):
#   DRIVE_OUT=/content/drive/MyDrive/IdiomatorRigor bash experiments/rigor/run_07b_e1_rembert_batch32.sh --dry-run
#   # full E1 (re-run the same line after any timeout to resume):
#   DRIVE_OUT=/content/drive/MyDrive/IdiomatorRigor bash experiments/rigor/run_07b_e1_rembert_batch32.sh
#   # FORCE=1 re-trains even completed jobs (use after a code change, never to
#   # "make sure" — a stale skip is not success, a fresh artifact is):
#   FORCE=1 DRIVE_OUT=... bash experiments/rigor/run_07b_e1_rembert_batch32.sh
# ──────────────────────────────────────────────────────────────────────────────

set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# E1 config — RemBERT, batch 32, RemBERT's own LRs (run_07 defaults LR_JOINT=3e-6
# / LR_BIO=2e-6 already match; set explicitly so the config is self-documenting).
export MODEL="${MODEL:-google/rembert}"
export ENC_SHORT="${ENC_SHORT:-rembert32}"
export BATCH="${BATCH:-32}"
export LR_JOINT="${LR_JOINT:-3e-6}"
export LR_BIO="${LR_BIO:-2e-6}"

echo "E1 = RemBERT @ batch ${BATCH}, seeds 42/123/7, LR_JOINT=${LR_JOINT} LR_BIO=${LR_BIO}"
echo "Delegating to run_07 (persistence gate + per-job skip/resume)..."
echo

exec bash "$HERE/run_07_mdeberta_sp_replication.sh" "$@"
