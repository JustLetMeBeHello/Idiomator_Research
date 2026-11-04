# PREREG 6.22 — Does Li-style MRC beat a *tuned* BIO tagger on CoNLL-2003? (Paper B, main-venue test)

Written 2026-11-03, BEFORE any run. Predictions and decision rules below are fixed; do not edit them after results land.

## Why
Li et al. 2020 (ACL, "A Unified MRC Framework for NER", arXiv 1910.11476) report MRC > BERT-Tagger on flat NER (CoNLL03 +0.24 F1, OntoNotes5 +1.95) with no seeds/CIs, BIO numbers copied from Devlin et al. 2018, and no reported O-weight or decoding tuning for BIO. Paper B's single-span results say part of "pointer beats tagging" is BIO's sensitivity to label distribution (O-weight halves the gap on ABSA; the gap tracks span density). CoNLL-03 is sparse (entity tokens ≈ 0.153 of tokens, ≈ ABSA's 0.16), so the same account predicts a baseline-tuning effect here. This is the field's own benchmark and metric.

## Scope limits (state in the paper)
- CoNLL-03 is pre-tokenized: entities followed by punctuation (22%) are separate tokens, so the SentencePiece trailing-punctuation artifact CANNOT occur here by construction. This test is about the loss/imbalance part of the decomposition only.
- Reimplementation, not Li's code: absolute F1 is not comparable to the published table; only the within-study BIO-vs-MRC comparison is claimed. MRC arm uses flat-NER start/end matching (each start paired with the nearest end ≥ start), not their matching classifier. Li's BERT size was not verified from the paper text read.
- Learning rate fixed at 3e-5 for all arms (the factor under test is the imbalance knob). Equal-budget LR sweep is a follow-up on one encoder if a reviewer needs it.

## Design
- Data: `tomaarsen/conll2003` via HF datasets-server (train 14,041 / dev 3,250 / test 3,453 sentences), sentence-level, no document context.
- Metric: entity-level micro-F1, exact span + type (CoNLL/seqeval style), same as Li. Paired bootstrap over test sentences (10k), seeds pooled.
- Encoders (one per tokenizer family): `bert-base-cased` (WordPiece, Li's family), `roberta-base` (byte-BPE), `xlm-roberta-base` (SentencePiece).
- Seeds 42 / 123 / 7. Config selection on dev with seed 42 only; finals use the selected config.
- Arms per encoder:
  - A. BIO-std: first-subword token CE, O-weight 1.0, argmax decode.
  - B. BIO-tuned: O-weight ∈ {1.0, 0.5, 0.2, 0.1} selected on dev F1, argmax decode.
  - C. BIO-tuned + valid-BIO decode: arm B's model, Viterbi with BIO transition constraints (no new parameters).
  - D. MRC (Li-style): one fixed annotation-guideline query per type (PER/ORG/LOC/MISC), query+sentence concatenated, two per-token start/end heads (2-way CE), positive-class weight ∈ {1, 5} selected on dev (equal tuning effort to B).
- Runs: sweeps 3 enc × (4 BIO + 2 MRC) = 18 (seed 42); finals add seeds 123/7 for A (ow 1.0), B/C (selected ow; reuses A when ow=1.0) and D, i.e. up to 18 more (C is a decode of B, no extra training). 33–36 runs total (corrected at build time; design unchanged). MRC costs ~4× a BIO run (4 queries per sentence). Runtime unmeasured: a 1-epoch dry-run times it before launch.

## Predictions (fixed now)
- P1: A < D by ≥ 0.5 F1 on at least one encoder (replicates Li's direction).
- P2: C closes ≥ 50% of the A→D gap on every encoder where P1 holds.
- P3: the tuned gap (D − C) does not depend on tokenizer family (artifact is absent by construction here).

## Decision rules
- D − C has 95% CI inside ±0.3 F1 (TOST, Δ=0.3) on all 3 encoders → "the reported pointer/MRC advantage is baseline tuning": main-venue framing (baselines in pointer-vs-tagging comparisons are under-tuned; decomposition = artifact on raw-offset eval + imbalance/decoding everywhere).
- D − C > 0.5 F1 with CI excluding 0 on ≥ 2 encoders → Li's result stands on CoNLL; the decomposition is limited to single-span raw-text tasks; Paper B stays a Findings-level measurement paper, reported honestly.
- Anything else → report as partial; no main-venue claim.

## Build list
1. `run_39_conll_qa_vs_bio.py`: `--mode bio|mrc`, data via datasets-server, O-weight / pos-weight flags, Viterbi decode, F1, per-run `metrics.json` + `test_predictions.jsonl`, `--selftest` (decode constraints, flat matching, F1 on toy sentences), `--force`.
2. `notebooks/C1_PaperB_Batch5_conll.ipynb`: preflight, 1-epoch timing dry-run, sweep, finals, readback, disconnect.
3. CPU aggregator: paired bootstrap + TOST on test predictions (A/B/C vs D).
4. Commit this file BEFORE launching; Colab clones from GitHub.
