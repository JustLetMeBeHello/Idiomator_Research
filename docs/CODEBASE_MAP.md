# CODEBASE_MAP — IdiomBERT / Research_And_Training

> Auto-generated 2026-06-11; rigor runners updated 2026-06-19; ablation matrix section corrected 2026-06-20; N2 3-seed status updated 2026-06-20. Read this before grepping the repo.
> Trust it unless it conflicts with something you can see in the actual file.
> Update touched sections after any session that adds/renames files or functions.

---

## Architecture overview

Two-stage pipeline for multilingual idiom detection + span extraction:

```
Raw data (idioms_structured/)
        ↓ Sampler.py
idioms_structured/Splits/{train,dev,test}.jsonl   ← canonical splits used by ALL training
        ↓
┌──────────────────────────────────────────────────────┐
│  Systems A–G (7 systems evaluated in Full_evaluation)│
│  A: Stage1_mBERT  → Stage2_mBERT                    │
│  B: GPT Stage1    → GPT Stage2                       │
│  B4/C4: GPT 4-shot variants                          │
│  C: GPT single-stage                                 │
│  D: Stage1_mBERT  → Joint span head (best Joint F1) │
│  E: Joint end-to-end (Train_Join.py)                 │
│  F: Sequential two-phase (Train_Sequential.py)       │
│  G: BIO tagger, span-only (no classifier)            │
└──────────────────────────────────────────────────────┘
        ↓
Evaluation/Full_evaluation.py → results/pipeline_eval/pipeline_eval_results.json
```

**Languages:** EN, ES, HI, TE (in-distribution) + ID (zero-shot held-out, n=328)
**Test set:** 956 examples (EN 254, ES 254, HI 62, TE 62). Indonesian separate.
**Best system:** D — Joint F1 0.7515, Stability 0.7061

---

## Data

### `idioms_structured/Splits/` — canonical training data
- `train.jsonl / dev.jsonl / test.jsonl` — used by ALL training scripts
- `split_stats.json` — exact counts per language/split (trust this over inline comments)
- `loss_weights.json` — per-language × idiomaticity cell weights (β=0.3, γ=1.9, tuned on EN+HI+TE dev only)
- `Update_indonesian.py` — one-off script to patch Indonesian examples

**Actual training counts** (from split_stats.json):
| Lang | Train | Dev | Test |
|------|-------|-----|------|
| EN   | 2034  | 254 | 254  |
| ES   | 2034  | 254 | 254  |
| HI   | 508   | 62  | 62   |
| TE   | 506   | 61  | 62   |

### `idioms_structured/Span_tagged_data/` — raw language source files
- `English/Final_English_MERGED_normalized.jsonl`
- `Spanish/Final_Spanish_MERGED.jsonl`
- `Hindi/Final_Hindi_MERGED.jsonl`
- `Telugu/Final_Telugu_MERGED.jsonl`

### `Sampler.py` — dataset preparation
- Produces train/dev/test from raw sources using 80/10/10 unseen-idiom split
- Sampling: HI/TE use ALL examples; EN/ES capped at 2× (HI+TE) to prevent gradient dominance
- **Run once to rebuild splits** — do not re-run unless the source files changed

---

## Training scripts

### `Base_Pipeline/Stage_1_training.py` — System A/D Stage 1 (classifier)
- Model: `bert-base-multilingual-cased` (mBERT) default
- Task: binary sequence classification (idiomatic=1 / literal=0)
- Key args: `--output_dir`, `--langs`, `--model_name`
- Output: `models/<output_dir>/test_predictions.jsonl` + `best_model/`
- Called by language ablation matrix for the "stage1" system slot

### `Base_Pipeline/Stage_2_training.py` — System A/D Stage 2 (QA span)
- Model: mBERT
- Task: QA-style span extraction (SQuAD-style start/end token prediction)
- Key args: `--output_dir`, `--langs`, `--model_name`
- Output: `models/<output_dir>/test_predictions.jsonl`
- **Note:** uses ALL examples (idiomatic + literal) — span exists for both

### `Train_Join.py` — System E (joint end-to-end)
- One encoder, 3 heads: cls + span_start + span_end
- Loss = cls_weight * cls_loss + span_weight * (start + end) / 2
- Output: `models/en_es_hi_te/joint_mbert/`
- Key args: `--cls_loss_weight`, `--span_loss_weight`, `--langs`

### `Train_Sequential.py` — System F (sequential two-phase)
- Phase 1: cls-dominant (cls=0.7, span=0.3) → saves on best dev cls F1
- Phase 2: span-dominant (cls=0.3, span=0.7), freezes bottom encoder layers, init from Phase 1
- Outputs: `models/sequential/phase1/`, `models/sequential/phase2/`
- Joint evaluation: run `Joint_Evaluation.py` pointing at both phase outputs

### `Ablations/BiO_Task_mBERT_train.py` — System G (BIO tagger, span-only)
- Token-level BIO sequence labeling (no idiomaticity classification head)
- Fine-tuned model in: `models/bio_tagger_en_hi_te/best_model/` (~676 MB)
- **Critical:** this model is also deployed in Language_Learning_BaseWebsite as `Backend/checkpoints_system_g/`
- System G "Joint F1" is structurally penalized (no classifier) — exclude from Joint F1 comparisons or annotate

---

## Evaluation

### `Evaluation/Full_evaluation.py` — canonical eval (use this, not Ablations/Full_evaluation.py)
- Chains all 7 systems (A–G, B4/C4) in one run
- Reads all `test_predictions.jsonl` files from `models/`
- Outputs: `results/pipeline_eval/pipeline_eval_results.json` (CANONICAL metrics)
- Key args: `--xlmr_seed` (42/123/7) for Exp02 multi-seed, `--output_dir`
- Reports: Classification Macro F1, E2E Span Overlap F1, Joint F1, Stability, Indonesian bootstrap CIs

### `Ablations/Full_evaluation.py` — OLDER version
- Kept for ablation-specific runs. For paper metrics, use `Evaluation/Full_evaluation.py`.

### `Joint_Evaluation.py` — System F joint scoring
- Combines phase1 (cls) + phase2 (span) predictions for Sequential system
- Use after `Train_Sequential.py` finishes both phases

### `results/pipeline_eval/pipeline_eval_results.json` — CANONICAL metric store
- Single source of truth for all table numbers in the paper
- Updated by: `Evaluation/Full_evaluation.py`
- Contains keys like `system_d_mbert_s1_joint_span` (System D), `rigor_xlmr_joint_s42` (Exp02 seed 42), etc. — verify exact key names against the json before scripting against them.

---

## Language ablation matrix (15 training combos × 7 systems)

**CANONICAL OUTPUT (current, 2026-06-20): `models/results_Full_Pipeline/`** — one `pipeline_eval_results.json` per combo (all 7 systems each), plus `ablation_summary.csv` (1005 rows) / `stability_summary.csv` / `transfer_matrix.csv` rollups, `logs/<combo>__eval.log`, `job_checkpoints/<combo>__eval.json`. Systems A/D/E/F are clean; B/C are correctly static across combos (GPT isn't retrained per combo); **System G has a Telugu-specific collapse (exact=0, overlap 0.27–0.30) in every combo** — root cause found 2026-06-20: `run_bio.py`'s skip-if-output-exists check had no `--force` flag, so a bad early result was silently re-served across all 15 combos (not a decoder bug — that path is shared+patched and confirmed fine). All 5 `Language_Ablations/run_*.py` runners patched with `--force`. **Pending Colab verification** (`run_bio.py --only_combo te --force`) before trusting G's ablation rows. Detail: `memory/ablation_matrix_status.md`.

Superseded one earlier generation — moved to `notebooks/colab/_deprecated/` 2026-06-20, confirmed unreferenced by any current script before moving:
- `run_language_ablation_matrix.py`, `run_language_ablation_matrix.sh`, `fix_system_g_combos.py`, `Fix_System_G.ipynb` — May-29 generation, only ever produced System-G-only results in `g_only/` subdirs (main per-combo json never existed for this generation).

Current (live) scripts — don't move these without re-checking the call graph:
- `notebooks/colab/Language_Ablations/run_stage1.py run_stage2.py run_joint.py run_bio.py run_sequential.py run_eval.py` — Jun-19 per-system runners. `run_eval.py` calls `Evaluation/Full_evaluation.py` per combo, then `notebooks/colab/summarize_language_ablation_matrix.py` (the May-29-era summarizer — still live, NOT part of the deprecated set, kept in place) to roll up into `results_Full_Pipeline/`.
- `notebooks/colab/Language_Ablations/merge_ablations.py` — separate manual utility, not called by `run_eval.py`. Merges partial ablation outputs from multiple Colab accounts (each account does a subset of the 15 combos, downloads its `models/language_ablation_matrix/` tree, this script combines them) — explains why `models/language_ablation_matrix/<combo>/` is only partially populated locally (other accounts' shares not yet merged in here).

**If "where are the ablation results" comes up again: it's `models/results_Full_Pipeline/`, not `results/language_ablation_matrix/` (that path doesn't exist) and not `models/language_ablation_matrix/` (that's raw per-job checkpoints, not the merged eval).**

---

## Additional Rigor Experiments (`Additional_Rigor_Experiments/`)

These are numbered experiments adding statistical rigor to paper claims.

| Script | Purpose | Status |
|--------|---------|--------|
| `run_02_xlmr_qa_vs_bio.sh` | XLM-R QA vs BIO multi-seed (Exp02, 3 seeds: 42/123/7) | DONE, all 3 seeds — registered 2026-06-09 in key_numbers.md |
| `run_04_llama3_baseline.py` | Llama-3.3-70B via Groq baseline (no GPU, ~$0.25) | not yet run |
| `run_05_decoder_rerun.py` | Fixed UTF-8 char-aware decoder rerun (produced System G corrected metrics) | done |
| `run_06_idiombert_v2.py` | IdiomBERT v2 (planned) | not yet run |
| `run_07b_e1_rembert_batch32.sh` | **E1** — RemBERT @ batch 32, 3 seeds; thin wrapper over run_07 (bakes MODEL/ENC_SHORT=rembert32/BATCH=32/LRs); inherits run_07 Drive gate + per-job skip/resume | NEW 2026-06-19; smoke-test passed on Colab |
| `run_08_extended_gold.py` | SP QA>BIO gap under 3 gold normalizations (Scenario A). `REMBERT_ENC` env override (default `rembert`) repoints the rembert source to E1's `rembert32/` layout; run_08b+run_13 inherit it via import | DONE, seed-stable (42/123/7); REMBERT_ENC added 2026-06-19 |
| `run_08b_strip_ci.py` | Bootstrap 95% CI on the SP-family strip gap (macro stat); reuses run_08 evaluate_encoder+SOURCES. Fills §7 [FILL] | NEW, unrun (needs preds on Colab) |
| `run_09_semeval_eval.py` | SemEval benchmark eval | not yet run |
| `run_10_id10m_eval.py` | ID10M zero-shot eval | done (EN BIO=0.615, ES BIO=0.590) |
| `run_12_id10m_systemg_eval.py` | ID10M eval specifically for System G | done |
| `run_13_tost_equivalence.py` | **N1** — TOST equivalence test for C1 (replaces "CI contains reference"); reuses run_08 evaluate_encoder; default Δ=0.02; `--reference wp|zero` | NEW 2026-06-19; unrun (needs Colab preds-root) |
| `run_14_holm_bh_correction.py` | **N2** — Holm-Bonferroni + BH correction on per-language mBERT-vs-GPT-4o Joint-F1 family (paired bootstrap p); no scipy dep | DONE all 3 seeds (42/123/7) independently, 2026-06-20 — 0/4 survive correction at any seed, robust null |
| `run_14b_seed_majority_holm_bh.py` | **N2 (3-seed combined)** — majority-vote ensembles the 3 seeds' mBERT predictions per sentence (right if ≥2/3 agree) before the same bootstrap+Holm/BH; the statistically correct way to use 3 seeds (naive row-concat is pseudo-replication + GPT is seed-deterministic so pooling it fakes a tighter CI) | NEW 2026-06-20; ran — still 0/4 survive even ensembled |
| `run_15_multiseed_main_tables.sh` | **E2** — multi-seed (123/7) driver for mBERT A–G tables (Stage_1/Stage_2/Train_Join/Train_Sequential/BiO at default=seed-42 hparams); Drive gate + FORCE=1 + --dry-run | **DONE 2026-06-20** — seeds 123/7 trained+registered (commit `0ff2b09`), joined w/ seed-42; 3-seed mean±std + per-language breakdown written into `IdiomBERT_Submission_Ready_v8.docx` same day |
| `run_16_wordpiece_word_tagger.py` | **E3** — WordPiece WORD-level POS-style tagger (mean-pool subwords→word, one B/I/O/word, word-boundary decode); plugs into Full_evaluation via `--bio_preds` | NEW 2026-06-19; smoke-tested locally, GPU run pending |
| `check_metric_drift.py` | Detects if metrics in json drifted from expected values | run before any paper build |
| `sample_error_analysis.py` | Sample ~25 error examples for §9 error table | not yet run for paper |

### Key run_08 result (Scenario A — LOCKED)
SP QA>BIO gap is tokenizer artifact (trailing-punct attachment):
- SP family mean: original=+0.138, extend=−0.090, **strip=+0.025**
- WP family mean (mBERT+MuRIL, n=2): original=+0.030, extend=+0.031, strip=+0.029 (flat)
- mBERT alone (WordPiece, the paper's reference control): original=+0.032, extend=+0.029, strip=+0.032 (flat). Paper's +0.032 is mBERT-only, not the family mean — MuRIL was dropped from the paper entirely (seed collapse, see "MuRIL all seeds collapsed" note in key_numbers.md), so +0.032 vs +0.029 is not a drift bug, just two different baselines.

### `DeBERTa_SentencePiece_Replication.ipynb` — Exp07 notebook
- Expects predictions under `<preds-root>/rembert/` and `<preds-root>/flip/{xlmr,mbert,muril}_<system>_s<seed>/`
- `flip-2/` directory contains the flip-architecture SP predictions

---

## Annotation tool (`Annotation_tool/`)

### `main.py` — IdiomBank Validator web app
- Flask or FastAPI annotation web app for native-speaker validation
- Exports: `annotator_*.jsonl` files

### `compute_iaa.py` — Inter-Annotator Agreement
- Input: two annotator export JSONL files
- Output: % Agreement + Cohen's Kappa for idiomaticity label + span correctness
- Key args: `--lang Telugu`, `--out iaa_results.json`, `--latex`
- **HARD BLOCKER for MultiIdiom paper** — needs 2 annotators per language (EN+TE minimum)

### `build_annotation_pool.py` — prepares examples for annotators

---

## Models directory structure

```
models/
├── en_es_hi_te/          ← main 4-language training
│   ├── stage1_mbert/     → System A/D stage1 predictions
│   ├── stage2_mbert/     → System A stage2 predictions
│   ├── joint_mbert/      → System D/E predictions
│   ├── bio_tagger/       → System G predictions (all langs)
│   └── sequential_mbert/ → System F phase1 + phase2
├── bio_tagger_en_hi_te/  ← deployed BIO model (also in BaseWebsite)
│   └── best_model/       → BertForTokenClassification (~676 MB)
├── stage1_mbert_en_hi_te/ ← EN+HI+TE only stage1
├── gpt_baseline_stage1/  → GPT System B stage1 preds
├── gpt_baseline_stage2/  → GPT System B stage2 preds
├── gpt_single_stage/     → GPT System C preds
├── rigor_xlmr_joint_s42/ → XLM-R Exp02 seed 42 (in pipeline_eval_results.json)
└── rigor_bio_xlmr_s42/   → XLM-R BIO seed 42
```

---

## Key numbers (snapshot — verify in `memory/key_numbers.md` before citing)

| System | Joint F1 | Stability |
|--------|----------|-----------|
| D | **0.7515** | 0.7061 |
| A | 0.7486 | 0.7049 |
| F | 0.7440 | 0.6843 |
| E | 0.7381 | 0.6774 |
| C | 0.6977 | 0.6148 |
| G | 0.4750 | 0.4658 (span-only; structurally penalized) |

Classification F1 (full training): A/D=0.7823, E=0.7760, F=0.7741

---

## State / blockers (as of 2026-06-11)

| Item | Status |
|------|--------|
| `git push` (3 commits ahead of origin) | **BLOCKED Colab** — do first |
| Exp02 seeds 123+7 (Colab) | DONE — registered 2026-06-09 |
| Commit v8.docx to Paper_Drafts | Pending (untracked) |
| IAA (EN+TE annotators) | **HARD BLOCKER** for MultiIdiom paper |
| §9.2 ID10M insert into v8.docx | Pending (check `/tmp/idiombert_v7_unpacked/`) |
| run_04 Llama-3 baseline | Not run (no GPU needed, Groq) |
| Error analysis §9 (~25 rows) | Not run |
| HF checkpoint push | Not done (needs `huggingface-cli login`) |

---

## agentic_os/

FastAPI dashboard reading research memory files — NOT training code.
- `backend/parsers/key_numbers.py` — parses `memory/key_numbers.md`
- `backend/parsers/project_overview.py` — parses `memory/project_overview.md`
- `backend/sources/results.py` — reads `results/` CSVs for dashboard display
- Not relevant to training or paper pipeline; ignore unless debugging the dashboard.

---

## Common workflows

**Start a new Colab run:**
1. `git push` from local first (3 commits behind)
2. Open notebook in Colab → Runtime → GPU
3. Run mount + clone cells (force fresh clone if repo already exists)
4. Use `--force` flag if any training code changed since last run

**Update paper numbers:**
1. Run `Additional_Rigor_Experiments/check_metric_drift.py`
2. Read `results/pipeline_eval/pipeline_eval_results.json` for current values
3. Cross-check against `memory/key_numbers.md`
4. If discrepancy: trust `pipeline_eval_results.json` (most recent eval run)

**Add a new rigor experiment:**
1. Script goes in `Additional_Rigor_Experiments/run_NN_<name>.py`
2. Output keys in `pipeline_eval_results.json` use pattern `rigor_<name>_s{seed}`
3. Add `--<name>_preds` arg to `Evaluation/Full_evaluation.py` if it needs to appear in main table
