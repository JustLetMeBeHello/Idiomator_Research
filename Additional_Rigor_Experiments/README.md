# Additional Rigor Experiments

Four targeted experiments added in response to council deliberation (2026-05-25)
to lift IdiomBERT + MultiIdiom from Findings-ceiling toward Main-eligible at
NAACL 2027 (August 2026 ARR cycle).

Each experiment is a thin wrapper over the existing trainers — same code,
same hyperparameters, **only the encoder or LLM API is swapped**. The ablation
matrix in the main paper (15 training combinations × 3 systems × mBERT) is
**not** re-run for new encoders. Only the cells that defend each new claim
are added.

## Why these four

| # | Experiment | Defends the claim |
|---|------------|-------------------|
| 01 | BIO + MuRIL on Telugu | Is BIO failure architectural or tokenizer-driven? **Run first — decision-influencing.** |
| 02 | XLM-R replication of QA-vs-BIO | Does the QA > BIO finding survive an encoder swap? Single biggest ceiling-mover. |
| 03 | MuRIL Joint (System E) on full training | Closes the "first Telugu resource but no Indic encoder" gap. |
| 04 | Llama-3.3-70B baseline | Removes the single-LLM (GPT-4o only) reviewer attack. Auto-reject signal at Main in 2026 if absent. |

## Run order

1. **`run_01_bio_muril_recovery.sh`** — RUN FIRST. The result determines which paper carries the Main shot:
   - If Telugu BIO exact match recovers (e.g., > 0.20) → MultiIdiom becomes Main-eligible with a "tokenizer-matched cross-lingual recipe" story. IdiomBERT's "BIO fails" claim weakens to "BIO + WordPiece fails for non-Latin abugida."
   - If Telugu BIO still ≈ 0.000 with MuRIL → IdiomBERT's architectural claim hardens. The QA-style necessity claim becomes Main-defensible.
2. **`run_02_xlmr_qa_vs_bio.sh`** — parallel to 01, same GPU.
3. **`run_03_muril_joint.sh`** — after 01/02.
4. **`run_04_llama3_baseline.py`** — runs off-GPU via Together AI / DeepInfra API. Can run concurrently with 01-03.

## Colab quickstart

**Use the notebook**: open `Run_In_Colab.ipynb` directly in Colab. It handles
repo clone, dependency install, Drive mount (so outputs survive disconnects),
API key loading from Colab Secrets, experiment selection, and a multi-session
parallel plan for free-tier T4 users.

```
File → Open notebook → GitHub → JustLetMeBeHello/Idiomator_Research
→ Research_And_Training/Additional_Rigor_Experiments/Run_In_Colab.ipynb
```

The notebook supports two modes:
- **Sequential** (`RUN = 'all'`): all four experiments back-to-back, ~3.5 hr on A100.
- **Parallel across sessions**: open the notebook in 4 browser tabs, set
  `RUN = '01' / '02' / '03' / '04'` in each — outputs don't collide because
  each writes to its own Drive subdirectory under `MyDrive/IdiomatorRigor/`.

## Outputs

All runs save under `models/rigor_<experiment>/` to keep them separate from
main-paper artifacts:

```
models/
  rigor_bio_muril_full/        ← experiment 01
  rigor_joint_xlmr_full/       ← experiment 02 (joint side)
  rigor_bio_xlmr_full/         ← experiment 02 (BIO side)
  rigor_joint_muril_full/      ← experiment 03
  rigor_llama3_pipeline/       ← experiment 04 (pipeline mode)
  rigor_llama3_single/         ← experiment 04 (single-stage mode)
```

Each directory contains `test_predictions.jsonl` and a metrics JSON matching
the existing trainers' output format, so they plug into `Joint_Evaluation.py`
and `Evaluation/Full_evaluation.py` without modification.

## Expected GPU time per experiment

| Experiment | A100 | T4 (free Colab) | Notes |
|------------|------|------------------|-------|
| 01 BIO + MuRIL | ~45 min | ~2.5 hr | full training (EN+ES+HI+TE), 6 epochs |
| 02 XLM-R (joint + BIO) | ~90 min | ~5 hr | two trainings, run sequentially in the script |
| 03 MuRIL Joint | ~50 min | ~2.5 hr | full training, 7 epochs |
| 04 Llama-3.3-70B | ~30 min | n/a (API) | Together AI inference, ~$15-30 budget |

## Cost summary

GPU: Colab Pro+ ($50/mo) covers all four with margin.
LLM API: ~$15-30 total for Llama-3.3 inference on common test (956) + Indonesian (328).

## On hyperparameter tuning

**Each new encoder uses its published default learning rate**, not a re-tuned
value. This is deliberate.

| Encoder | LR | Source |
|---------|----|--------|
| mBERT (main paper) | 2e-5 / 3.27e-5 (BIO) | Already in paper |
| XLM-R | 1e-5 | Conneau et al. 2020 standard |
| MuRIL | 2e-5 | Khanuja et al. 2021 standard, same family as mBERT |
| Llama-3.3-70B | n/a | API inference, no training |

**Reasoning**: the main paper's protocol (Sec 7) fixes hyperparameters across
all 15 training combinations to "ensure observed differences reflect training
data composition rather than optimization." The same principle applies to
encoder swaps — per-encoder tuning would confound the encoder choice with a
hyperparameter search.

**Paper-side framing**: *"We use each encoder's published learning rate from
its release paper; all other hyperparameters (loss weights, batch, epochs,
dropout) are fixed to the main-paper Joint mBERT system, following our
protocol of holding optimization constant to isolate the variable under study."*

**Safety valve**: `hp_sweep_xlmr_dev.sh` runs a 3-point LR sweep {5e-6, 1e-5,
2e-5} on **dev only** (no test contamination) for the most-questioned cell
(XLM-R Joint). Do not run by default — only if a reviewer specifically
challenges the 1e-5 choice. The output gives you a one-line reviewer response.

## What this does NOT do

- Does **not** re-run the 15-combination ablation for new encoders. Only full
  training cells are added — defending the specific new claim, not regenerating
  the entire matrix. If a reviewer asks for replication, run experiment 03 at
  EN+ES+TE and HI+TE-only (~2 more runs).
- Does **not** add IndoBERT for Indonesian. IndoBERT is monolingual Indonesian
  so it cannot replace mBERT in EN+ES+HI+TE training. A separate IndoBERT-on-ID
  fine-tuning run as a ceiling baseline is left out for scope; consider adding
  if a reviewer asks.
- Does **not** populate the error analysis tables (Tables 8-9 in IdiomBERT)
  or the IAA table (Table 4 in MultiIdiom). Those are tracked separately.
- Does **not** perform the GPT-5 contamination check (Wiktionary vs. test
  idiom overlap). Consider adding `05_contamination_check.py` if reviewers ask.

## Pending decisions to revisit after results land

1. Drop or soften "stability is the decisive discriminator" framing in
   IdiomBERT Section 7 regardless of these results — the council was
   unanimous that the post-hoc metric-selection critique is unfixable as
   currently worded.
2. After experiment 04, also re-prompt with seed sweep (3-5 exemplar
   resamples) for the 4-shot variants. Council flagged C4's collapse as
   possibly single-draw noise.
3. Anonymize `github.com/JustLetMeBeHello/...` in IdiomBERT Section 9
   before ARR submission — double-blind violation as written.
