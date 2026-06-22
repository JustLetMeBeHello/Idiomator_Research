#!/usr/bin/env bash
# ── Experiment E3 runner: WordPiece word-level tagger, 3 seeds (42/123/7) ────
#
# REVISION_PLAN E3. Wraps the E3 trainer (run_16_wordpiece_word_tagger.py) in the
# repo's canonical launch harness: Drive-persistence gate + per-seed skip/resume,
# matching run_15_multiseed_main_tables.sh exactly. The trainer itself is done;
# this file only gates + loops it so Colab timeouts self-heal and outputs survive
# a disconnect (the bug that lost the System-G Telugu rerun THREE times — the
# trainer wrote to ephemeral /content, nothing was Drive-backed).
#
# WHAT E3 IS / WHY MATCHED LR MATTERS
# -----------------------------------
# C1 claims a plain WORD-LEVEL tagger on a WordPiece encoder matches the QA/Joint
# span pointer once the SentencePiece offset artifact is removed — currently an
# INFERENCE in the paper, never built. E3 builds + measures it. The whole point
# is a FAIR comparison, so this runs at lr=3.27e-5 (the canonical main-table LR,
# identical to System G / the joint models) and effective batch 32 — NOT a
# mismatched LR like E1 accidentally used. lr/batch are passed EXPLICITLY below so
# a future change to the trainer's argparse defaults can't silently de-fair this.
#
# OUTPUTS  models/word_tagger_mbert_s{42,123,7}/{metrics.json,test_predictions.jsonl}
# E3 saves preds in System-G schema → register it with Full_evaluation's
# --bio_preds (the command is printed per seed at the end).
#
# ── SAVE + RESUME ON TIMEOUT (read this) ────────────────────────────────────
# SAVE:   with DRIVE_OUT set, each seed's output dir is symlinked to Drive and the
#         run HARD-FAILS before training if the link is not Drive-backed+writable.
#         metrics.json + test_predictions.jsonl land on Drive; console.log streams
#         via the notebook's tee, so a partial log survives a mid-epoch kill.
# RESUME: 3 independent jobs (one per seed). A seed whose test_predictions.jsonl
#         already exists is SKIPPED. If Colab times out, RE-RUN THE SAME COMMAND —
#         it skips finished seeds and continues at the first incomplete one.
# LIMIT:  no intra-job resume. best_model/ is deleted after each seed completes
#         (repo checkpoint-deletion policy + Drive quota), so a timeout DURING a
#         seed restarts THAT seed from epoch 1. Worst-case loss = one seed.
#
# Usage:
#   # always dry-run first (prints plan + persistence gate, trains nothing):
#   DRIVE_OUT=/content/drive/MyDrive/IdiomatorRigor bash experiments/rigor/run_16b_word_tagger.sh --dry-run
#   # full E3 (re-run the same line after any timeout to resume):
#   DRIVE_OUT=/content/drive/MyDrive/IdiomatorRigor bash experiments/rigor/run_16b_word_tagger.sh
#   # one seed only:
#   SEEDS=42 DRIVE_OUT=... bash experiments/rigor/run_16b_word_tagger.sh
#   # FORCE=1 retrains even completed seeds (use after a code change, never to
#   # "make sure" — a stale skip is not success, a fresh artifact is):
#   FORCE=1 DRIVE_OUT=... bash experiments/rigor/run_16b_word_tagger.sh
# ──────────────────────────────────────────────────────────────────────────────

set -euo pipefail
cd "$(git -C "$(dirname "$0")" rev-parse --show-toplevel)"

DRY_RUN=0
[[ "${1:-}" == "--dry-run" ]] && DRY_RUN=1

SEEDS="${SEEDS:-42 123 7}"
FORCE="${FORCE:-0}"
LANGS=(English Spanish Hindi Telugu)
TEST_LANGS=(English Spanish Hindi Telugu Indonesian)   # Indonesian = held-out
TRAINER=experiments/rigor/run_16_wordpiece_word_tagger.py
LR=3.27e-5          # canonical main-table LR — fair vs System G / joint
BATCH=32            # effective batch 32 (mBERT/WordPiece fits physically on T4)

echo "════════════════════════════════════════════════════════════════════════"
echo "E3 word-level tagger   seeds=[${SEEDS}]   force=${FORCE}   dry_run=${DRY_RUN}"
echo "  encoder:    bert-base-multilingual-cased (WordPiece — by design)"
echo "  lr:         ${LR}   batch: ${BATCH}"
echo "  langs:      ${LANGS[*]}"
echo "  test_langs: ${TEST_LANGS[*]}"
echo "════════════════════════════════════════════════════════════════════════"

# ── Per-seed persistence gate (CLAUDE.md hard rule) ─────────────────────────
# /content is ephemeral on Colab; only Drive survives a disconnect. With
# DRIVE_OUT set, symlink the seed's output dir to Drive and HARD-FAIL before
# training if the link is not Drive-backed and writable (readback probe).
gate_dir () {
    local d="$1"
    if [[ -n "${DRIVE_OUT:-}" ]]; then
        local name="${d#models/}"
        mkdir -p "$DRIVE_OUT/$name"
        rm -rf "$d"
        mkdir -p "$(dirname "$d")"
        ln -s "$DRIVE_OUT/$name" "$d"
        python - "$DRIVE_OUT" "$d" <<'PY'
import os, sys
drive = os.path.realpath(sys.argv[1]); d = sys.argv[2]
assert os.path.islink(d), f"{d} is not a symlink — output would be ephemeral"
tgt = os.path.realpath(d)
assert tgt.startswith(drive), f"{d} -> {tgt} not under Drive ({drive})"
probe = os.path.join(d, ".persist_probe")
open(probe, "w").write("ok")
assert open(probe).read() == "ok", f"readback failed at {probe}"
os.remove(probe)
print(f"  ✓ persistence gate passed — {d} is Drive-backed and writable")
PY
    else
        echo "  ⚠ DRIVE_OUT unset — ${d}/ is LOCAL (OK locally; EPHEMERAL on Colab)"
        mkdir -p "$d"
    fi
}

# ── Skip a seed iff its predictions exist AND FORCE!=1 ──────────────────────
should_run () {
    local preds="$1/test_predictions.jsonl"
    if [[ "$FORCE" != "1" && -f "$preds" ]]; then
        echo "  ✓ skip — $preds exists (set FORCE=1 to retrain)"
        return 1
    fi
    return 0
}

for SEED in $SEEDS; do
    OUT="models/word_tagger_mbert_s${SEED}"
    echo
    echo "── E3 seed=${SEED}  → ${OUT} ───────────────────────────────────────────"
    if ! should_run "$OUT"; then
        continue
    fi

    if [[ "$DRY_RUN" == "1" ]]; then
        printf '   [dry-run] '
        printf '%q ' python "$TRAINER" \
            --output_dir "$OUT" \
            --model_name bert-base-multilingual-cased \
            --langs "${LANGS[@]}" --test_langs "${TEST_LANGS[@]}" \
            --lr "$LR" --batch_size "$BATCH" --seed "$SEED"
        echo
        continue
    fi

    gate_dir "$OUT"
    python "$TRAINER" \
        --output_dir "$OUT" \
        --model_name bert-base-multilingual-cased \
        --langs "${LANGS[@]}" --test_langs "${TEST_LANGS[@]}" \
        --lr "$LR" --batch_size "$BATCH" --seed "$SEED"

    # Checkpoint-deletion policy: keep metrics.json + test_predictions.jsonl,
    # drop the ~700MB encoder once preds are confirmed written (Drive quota).
    if [[ -f "$OUT/test_predictions.jsonl" ]]; then
        rm -rf "$OUT/best_model"
        echo "  ✓ seed ${SEED} done — best_model/ removed (preds + metrics kept)"
    fi
done

echo
echo "════════════════════════════════════════════════════════════════════════"
[[ "$DRY_RUN" == "1" ]] && { echo "Dry-run only — nothing trained."; exit 0; }
echo "E3 training complete. Register each seed as a system row with:"
echo
for SEED in $SEEDS; do
cat <<EOF
  python Evaluation/Full_evaluation.py \\
      --bio_preds   models/word_tagger_mbert_s${SEED}/test_predictions.jsonl \\
      --output_dir  results/e3_word_tagger_s${SEED}
EOF
done
echo
echo "Then aggregate the 3 seeds → per-language mean ± std span EM / overlap-F1 and"
echo "compare against the QA/Joint span EM (the C1 'word-level tagger suffices' test)."
echo "Run experiments/rigor/check_metric_drift.py before any paper build."
echo "════════════════════════════════════════════════════════════════════════"
