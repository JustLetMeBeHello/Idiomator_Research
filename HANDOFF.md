# Handoff — 2026-05-25 — Council venue deliberation + rigor experiments setup (preprint-first)

## TL;DR
User ran `/council` on the two paper drafts (MultiIdiom + IdiomBERT) for venue/combine strategy. Council converged: **SPLIT the papers (unanimous), target NAACL 2027 via August 2026 ARR**, Findings is the realistic ceiling with current scope, Main is conditional on adding XLM-R replication + MuRIL/IndoBERT + Llama-3 baselines + a BIO+MuRIL Telugu recovery experiment. Created `Additional_Rigor_Experiments/` with 7 files (4 experiments + HP sweep + Colab notebook + README) implementing the council's load-bearing fixes. **User is now optimizing for a preprint** (not venue submission first) — anonymization constraints relax, GitHub URL stays in, companion citations can be de-anonymized via arXiv IDs.

## Code state
- Repo: `/Users/shishirmaddineni/Desktop/Idiomator_Research/Research_And_Training`, branch: `main`, up to date with `origin/main`
- Uncommitted changes: **YES** — `Additional_Rigor_Experiments/` is entirely untracked; some pre-existing annotation-tool diffs from prior sessions also unstaged
- Files created this session (all under `Additional_Rigor_Experiments/`):
  - [`README.md`](Research_And_Training/Additional_Rigor_Experiments/README.md) — overview, run order, hyperparameter rationale, Colab quickstart
  - [`run_01_bio_muril_recovery.sh`](Research_And_Training/Additional_Rigor_Experiments/run_01_bio_muril_recovery.sh) — System G (BIO) + `google/muril-base-cased` on full training. **Decision-influencing — run FIRST.**
  - [`run_02_xlmr_qa_vs_bio.sh`](Research_And_Training/Additional_Rigor_Experiments/run_02_xlmr_qa_vs_bio.sh) — System E + System G with `xlm-roberta-base` (lr=1e-5 for joint, sequential trainings)
  - [`run_03_muril_joint.sh`](Research_And_Training/Additional_Rigor_Experiments/run_03_muril_joint.sh) — System E + MuRIL on full training
  - [`run_04_llama3_baseline.py`](Research_And_Training/Additional_Rigor_Experiments/run_04_llama3_baseline.py) — Llama-3.3-70B single-stage. Provider registry for Together/DeepInfra/Groq/OpenRouter/local-vLLM. Output format matches `Ablations/GPT-Baseline.py` exactly so it plugs into `Evaluation/Full_evaluation.py` unchanged.
  - [`hp_sweep_xlmr_dev.sh`](Research_And_Training/Additional_Rigor_Experiments/hp_sweep_xlmr_dev.sh) — optional safety valve: 3-point LR sweep {5e-6, 1e-5, 2e-5} on dev only for XLM-R Joint. Do not run by default.
  - [`Run_In_Colab.ipynb`](Research_And_Training/Additional_Rigor_Experiments/Run_In_Colab.ipynb) — Colab notebook with Drive symlink, API-key loader, `RUN = '01'/'02'/'03'/'04'/'all'` selector, multi-session + multi-account fan-out instructions
- Auto-memory updated: `project_idiomator_research.md` created in `~/.claude/projects/-Users-shishirmaddineni-Desktop-Personal-Language-Automation/memory/`, MEMORY.md index updated
- Build/test status: **none of the experiment scripts have been run yet.** Code is reviewed against existing trainers but not executed. Recommend a `--dry_run`-style 10-example smoke test on experiment 01 before committing to a full A100 hour.

## Decisions made
- **Split the two papers** (do not combine) — Why: Council unanimous (4/4 panelists). Combining forces 8-page cuts that gut either the 15-combo ablation or the IAA/validation methodology; concentrates reviewer-attack surface (silver-data + small-n CIs apply to both, but isolable when separate); MWE field convention (PARSEME, ID10M, AStitchInLanguageModels) is split.
- **Target August 2026 ARR → NAACL 2027 (May 2027 venue)** — Why: User stated August ARR cycle. EACL 2027 (Aug 27, 2026 commit) is unreachable from August ARR — reviews not ready in time. EMNLP 2026 commit (~Sep) also unreachable. NAACL 2027 commit window (~Nov-Dec 2026) is the natural fit. **NOTE: User pivoted mid-session to preprint-first** — venue submission becomes secondary.
- **Optimize for preprint (arXiv) before venue** — Why: User stated this in `/handoff` args. Implications: (a) GitHub URL in IdiomBERT Sec 9 does NOT need anonymizing for the preprint; (b) "Anonymous, 2026" companion-paper citations can be replaced with arXiv IDs once both are posted; (c) arXiv has no page limit so all 15-combo ablation cells + appendices can stay in; (d) anonymized version still needed later for ARR August commit.
- **Use published per-encoder learning rates, do NOT re-tune per encoder** — Why: Main paper Sec 7 already fixes hyperparameters across 15 combos to "isolate training data composition." Same logic applies to encoder swaps — per-encoder tuning would confound encoder choice with HP search. XLM-R gets 1e-5 (Conneau et al. 2020 default), MuRIL gets 2e-5 (Khanuja et al. 2021 default, same family as mBERT). Safety valve: `hp_sweep_xlmr_dev.sh` if a reviewer specifically pushes back.
- **Run BIO+MuRIL recovery first** — Why: Decision-influencing. If Telugu BIO recovers with MuRIL tokenizer → MultiIdiom gains a Main-worthy "cross-lingual tokenizer recipe" story and IdiomBERT's "QA-style is necessary" claim weakens. If still 0.000 → IdiomBERT's architectural claim hardens. Single experiment flips which paper has the Main shot.
- **Llama-3.3-70B as second LLM baseline (not just GPT-4o)** — Why: Council + dev-scan agree single-LLM comparison is auto-reject signal at Main venues in 2026. Open-weights second LLM defuses the cherry-pick critique.

## Rejected approaches
- **Combine the two papers into one** — Why not: 4/4 panelists rejected. Page budget would force cutting either the 15-combo ablation (IdiomBERT's distinct contribution) or the IAA/validation infrastructure (MultiIdiom's distinct contribution).
- **TACL combined submission** — Why not: 9-15 month R&R timeline misses every reasonable 2026 venue; by EMNLP 2027 the LLM baselines would be stale; council assessed acceptance probability <20%.
- **LREC-COLING 2026 for MultiIdiom** — Why not: Submission deadlines closed Oct 2025; next LREC is 2028 (biennial).
- **NAACL/EACL 2026** — Why not: Commitment windows close before mid-June; both venues already happened or finalized for this cycle.
- **EMNLP 2026 via June ARR** — Why not (now): Was viable from June ARR but user chose August ARR, which can't reach EMNLP 2026 commit.
- **Per-encoder full hyperparameter sweep** — Why not: Confounds encoder swap with HP search; multiplies time budget 4×; contradicts main paper's "fixed HP" protocol.
- **15-combination ablation re-run for new encoders** — Why not: The ablation matrix defends "how training composition affects mBERT" — a different question from "does the QA-vs-BIO finding survive encoder swap?" New encoders only need full-training cells to defend the new claims.
- **IndoBERT for Indonesian as 4th rigor experiment** — Why not: IndoBERT is monolingual Indonesian; cannot replace mBERT in EN+ES+HI+TE training; clean experimental design unclear. Skipped unless a reviewer asks.

## Open questions
- **BIO+MuRIL outcome (experiment 01) — undetermined.** This single result flips paper framing. Run it FIRST and revisit narrative before writing.
- **Llama-3.3-70B provider choice** — Together AI ($15-30 budget, paid) vs Groq (free tier, rate-limited but enough for 956+328 calls) vs DeepInfra. Notebook supports all four via `--provider` flag; user hasn't picked yet.
- **Multi-account fan-out vs solo Pro+** — User asked about both. If solo: Colab Pro+ ($50/mo, A100) → sequential `RUN='all'` in ~3.5 hr. If fan-out: 3-4 collaborators with free T4s, shared Drive folder, ~same wall-clock. User hasn't decided.
- **Preprint version: anonymize or not?** Standard arXiv is de-anonymized. ARR submission later will need an anonymized version. Recommend maintaining two branches/versions OR just the de-anon preprint now and re-anonymize before Aug ARR.
- **IAA / error analysis** — User confirmed they're personally doing TE + EN + HI validation with 2 annotators each and error-analysis tables ~mid-June. Not in the rigor-experiments scope; tracked separately.
- **Whether `Train_Join.py` works cleanly with XLM-R/MuRIL out of the box** — Both encoders use SentencePiece (different tokenizer offset behavior than WordPiece). Code looks tokenizer-agnostic via AutoTokenizer/AutoModel, but the character-offset → token-position mapping in the span head may need verification. Watch the first ~50 training steps for offset-mapping warnings.
- **Citations to add to both papers' Related Work** — MuRIL (Khanuja et al. 2021), XLM-R (Conneau et al. 2020), IndoBERT (Wilie et al. 2020), IndicBERT (Kakwani et al. 2020), Lin et al. 2019 "Choosing Transfer Languages," Constant et al. 2017 MWE survey. Currently uncited.
- **Synthetic data quality audit** — Council recommended sampling 200 training examples per language for a native-speaker correctness rating, adding as MultiIdiom Sec 5 subsection. Not yet started.

## Next steps
1. **Run experiment 01 (BIO + MuRIL) FIRST in Colab to get the decision-influencing result.** Open `Additional_Rigor_Experiments/Run_In_Colab.ipynb` in Colab → A100 runtime → `RUN = '01'` → run all. ~45 min on A100. Compare TE exact match to System G mBERT baseline (0.0000) to determine paper framing.
2. **Pick Llama-3 provider** (Together AI for stability, Groq for free) and set the corresponding API key in Colab Secrets. The notebook reads `TOGETHER_API_KEY` / `GROQ_API_KEY` / `DEEPINFRA_API_KEY` automatically.
3. **Decide fan-out plan**: solo Pro+ or recruit 3-4 collaborators. If fan-out: create shared Drive folder `IdiomatorRigor/` with the `claims.md` template (template in the notebook's multi-account section).
4. **After experiment 01 lands, run 02 + 03 + 04 in parallel** (different Colab sessions / accounts).
5. **Pull results to local repo**: `rsync -av ~/Google\ Drive/My\ Drive/IdiomatorRigor/ ~/Desktop/Idiomator_Research/Research_And_Training/models/` then `python Evaluation/Full_evaluation.py`. Add new System rows to IdiomBERT Tables 4, 6, 7 and MultiIdiom Table 3.
6. **Reframe "stability is decisive discriminator"** in IdiomBERT Sec 7 to the council's wording: *"No fine-tuned mBERT system shows a statistically significant Joint F1 advantage at full training (within-cluster spread 0.013, smaller than HI/TE bootstrap CI widths from Table 7). We report stability scores as a complementary view rather than a decisive discriminator."*
7. **Add missing citations** to both papers' Related Work (list under Open Questions).
8. **For preprint**: keep GitHub URL in IdiomBERT Sec 9 as-is; replace "Anonymous, 2026" companion citations with the arXiv IDs once both are posted. Make a separate anonymized branch for the ARR August submission.
9. **Optional**: ask current chat or future chat to create `05_contamination_check.py` (Wiktionary-vs-test-idiom overlap) and `06_seed_sweep_llama3_4shot.py` (addresses C4 single-draw concern). Council flagged both; neither blocks the preprint.

## Pick-up prompt
Continuing the IdiomBERT + MultiIdiom preprint preparation. Prior session ran `/council` on both paper drafts and converged on: SPLIT the papers, target NAACL 2027 via August 2026 ARR, but user is now optimizing for an arXiv preprint first (venue submission secondary). Created `Additional_Rigor_Experiments/` directory with 7 files implementing the council's load-bearing fixes (BIO+MuRIL Telugu recovery, XLM-R replication of QA-vs-BIO, MuRIL Joint, Llama-3.3-70B baseline, optional HP sweep, Colab notebook, README). **Nothing has been run yet.** Immediate next step: run experiment 01 (BIO + MuRIL Telugu recovery) in Colab — it's decision-influencing and determines which paper carries the Main story. Open `Research_And_Training/Additional_Rigor_Experiments/Run_In_Colab.ipynb` in Colab, set `RUN = '01'`, ~45 min on A100. Read `Research_And_Training/HANDOFF.md` (top-most section) for full context including council convergence reasoning, decisions/rejections, all open questions, and the rest of the experiment plan.

---

# Handoff — 2026-05-24 — Annotation tool: progress bug fix, Railway deploy, help panel, English language

## TL;DR
Fixed a critical progress-counter bug in the IdiomBank Validator (results were keyed by `meaning_id` which is shared across multiple sentences; re-keyed to `_idx` so each sentence counts separately). Added Railway deployment support (PostgreSQL fallback, frontend served from FastAPI, `requirements.txt`, `Procfile`). Added a `?` help drawer with full annotation guide built into the tool. Added English as a 5th language (254 examples, 222/254 with definitions). Tool is now ready for remote annotators — just needs Railway deployment and a git commit.

## Code state
- Repo: `/Users/shishirmaddineni/Desktop/Idiomator_Research/Research_And_Training`, branch: `main`
- Uncommitted changes: **YES** — all changes below are unstaged
- Files touched this session:
  - `Annotation_tool/index.html` — (1) Fixed progress bug: `results` re-keyed from `meaning_id` to `e._idx` (numeric index assigned on load) across 8 call sites; (2) Added `?` help drawer with annotation guide for all 3 dimensions; (3) Changed `API` constant to auto-detect local vs deployed (`port===5500` → localhost:8000, else same-origin); (4) Added `English` to `LANG_META`
  - `Annotation_tool/main.py` — Added `GET /` serving `index.html` via `FileResponse`; added `GET /api/health`; reads `DATABASE_URL` env var (Railway PostgreSQL) falling back to SQLite; fixes `postgres://` → `postgresql://` URL rewrite; skips `check_same_thread` for non-SQLite
  - `Annotation_tool/requirements.txt` — Created (fastapi, uvicorn, sqlalchemy, psycopg2-binary, pydantic)
  - `Annotation_tool/Procfile` — Created (`web: uvicorn main:app --host 0.0.0.0 --port $PORT`)
  - `Annotation_tool/data/English/test.jsonl` — Created: 254 English examples enriched with definitions from `Merged_Meanings_English_FINAL.jsonl`; 32/254 examples have no definition (sense question is hidden for those)
- Build/test status: frontend verified in browser preview (login → lang grid shows all 5 languages → annotation flow tested); FastAPI running locally at port 8000; Railway NOT yet deployed

## Decisions made
- **Key `results` by `_idx` not `meaning_id`** — Why: Telugu test has 62 rows but only 22 unique `meaning_id` values (avg ~3 sentences per sense). Keying by `meaning_id` made 1 submitted annotation count as 3 in the progress bar, and auto-skipped the other 2 sentences sharing that meaning_id. `_idx` (assigned via `forEach((e,i) => e._idx = i)` on load) is stable, unique per row, and survives filtering.
- **Serve frontend from FastAPI** — Why: Railway is one dyno — splitting frontend to Vercel creates a separate deploy. Serving `index.html` from `GET /` keeps everything in one Railway service with zero CORS complexity.
- **PostgreSQL via env var, SQLite fallback** — Why: SQLite filesystem on Railway resets on each deploy (ephemeral). Railway auto-injects `DATABASE_URL` when a PostgreSQL addon is attached; locally we fall back to SQLite so no local setup changes needed.
- **Help guide built into the tool (drawer), not separate PDF/page** — Why: user preference; keeps context in one place; annotators don't need to manage a second tab.
- **English added as 5th language** — Why: user requested it. Definitions sourced from `Merged_Meanings_English_FINAL.jsonl` (18,773 definitions loaded; 222/254 matched by `meaning_id`).

## Rejected approaches
- **Vercel for frontend** — Why not: user chose Railway for simplicity (one full-stack deployment).
- **ngrok for remote access** — Why not: user wanted a proper always-on deployment, not machine-dependent tunneling.
- **Key `results` by `sentence`** — Why not: would work (sentences are unique), but `_idx` is cleaner and more robust (doesn't depend on sentence content, no escaping issues as a dict key).

## Open questions
- **Railway not yet deployed** — The `requirements.txt`, `Procfile`, and `main.py` changes are all done but nothing is on Railway yet. The git repo needs a commit, a Railway project created, and PostgreSQL addon attached.
- **32 English examples with no definition** — These missing definitions in `data/English/test.jsonl` silently suppress the sense context question (Question ③). Low priority but worth flagging to annotators.
- **compute_iaa.py doesn't cover `sense_correct`** — The IAA script (`Annotation_tool/compute_iaa.py`) only computes kappa for `idiomaticity_verdict` and `span_correct`. The new `sense_correct` dimension is not yet included. Needs a 3rd dimension added before running the IAA analysis.
- **IAA table (Section 5.1) still empty** — Requires two annotators to complete Telugu + English annotation and export their JSONL files, then run `compute_iaa.py`. No annotators have started yet.
- **Canonical submission docx** — From previous session: unclear whether `Paper_Drafts/MultiIdiom_Dataset_Paper.docx` or `Downloads/MultiIdiom_Submission_Ready.docx` is the true submission draft. Still unresolved.

## Next steps
1. **Commit and push the annotation tool**: `cd Annotation_tool && git add index.html main.py requirements.txt Procfile data/English/` then commit with message "Add Railway deploy support, help drawer, English language, fix progress counter"
2. **Deploy to Railway**: New Project → Deploy from GitHub → attach PostgreSQL addon → confirm `DATABASE_URL` is set → verify `GET /` serves the UI
3. **Add `sense_correct` to `compute_iaa.py`**: Around line 118 in `compute_iaa.py`, add a 3rd dimension block after `span_boundary` for `sense_correct` (filter only records where `sense_correct` is non-null before computing kappa, since it's optional)
4. **Send tool URL to Telugu annotators**: Include the Railway URL + instruction to click `?` for the guide. Both annotators annotate all 62 Telugu examples, export `{name}_Telugu_results.jsonl`, send back.
5. **Run IAA**: `python compute_iaa.py annotator1.jsonl annotator2.jsonl --lang Telugu --latex` → paste kappa values into Section 5.1 placeholder

## Pick-up prompt
Continuing the IdiomBank Validator annotation tool (`Research_And_Training/Annotation_tool/`). This session fixed a progress-counter bug (re-keyed `results` dict from `meaning_id` to `_idx`), added Railway deployment support (`requirements.txt`, `Procfile`, PostgreSQL env-var fallback in `main.py`, frontend served from `GET /`), added an in-tool annotation guide (help drawer), and added English as a 5th language (`data/English/test.jsonl`, 254 examples). All changes are **uncommitted**. Immediate next step: commit and push, then deploy to Railway. After that: add `sense_correct` as a 3rd dimension to `compute_iaa.py` (currently only covers idiomaticity + span_correct), then get Telugu annotators started. Read HANDOFF.md at `Research_And_Training/HANDOFF.md` for full context.

---

# Handoff — 2026-05-24 — Telugu test count discrepancy fix

## TL;DR
Investigated a discrepancy between the paper's claimed Telugu test count (n=73) and the actual data. Found that 73 is a phantom number that doesn't correspond to any dataset version. Corrected both paper drafts to n=62 (the canonical count backed by all current pipeline artifacts).

## Code state
- Repo: `/Users/shishirmaddineni/Desktop/Idiomator_Research/Research_And_Training`, branch: `main`
- Uncommitted changes: none (working tree clean)
- Files touched this session:
  - `~/Downloads/MultiIdiom_Submission_Ready.docx` — Table 2 Telugu Test 73→62, Test Idioms 10→11; Section 5 "All 73"→"All 62" (note: this is in Downloads, not the canonical Paper_Drafts location)
  - `~/Desktop/Idiomator_Research/Paper_Drafts/MultiIdiom_Dataset_Paper.docx` — Table 2 Telugu Test 73→62, Test Idioms 10→11 (Section 5 did not have a hardcoded count in this version)
- Build/test status: not applicable (paper editing session, no code changes)

## Decisions made
- **Use n=62, not n=70 or n=73** — Why: 62 is the count confirmed by four independent artifacts: `idioms_structured/Splits/split_stats.json` (Telugu test: 62), `idioms_structured/Splits/test.jsonl` (62 Telugu rows), `Annotation_tool/telugu_test.jsonl` (62 lines, the file actually sent to annotators), and `Sampler.py` running with `random.seed(42)` on the current source file deterministically produces 62. The 70-example count is from the `Retired_data/` split (commit `32c38ef`, predates Spanish addition and Hindi rebalancing). The number 73 does not correspond to any version of the data.
- **Test Idioms corrected to 11, not 10** — Why: `split_stats.json` records `"unique_idioms": {"test": 11}` for Telugu. The paper had 10, which was also wrong.
- **Did not touch Section 5 of `MultiIdiom_Dataset_Paper.docx`** — Why: that version's Section 5 describes the Telugu validation process without stating a specific count ("two native Telugu L1 speakers…both annotators independently validate"), so no numeric fix was needed there.

## Rejected approaches
- **Rebuilding to get 73 examples** — Why not: there is no path to a 73-example split. The source (`Final_Telugu_MERGED.jsonl`) has 103 unique idioms; at 80/10/10 that's ~11 test idioms; at N_PER_CLASS=31.8 that's 62 examples. 73 is not derivable from any version of the pipeline.
- **Reverting to the 70-example Retired_data split** — Why not: that split predates Spanish/Hindi rebalancing and is explicitly archived. All model training and evaluation was run on the 62-example split.

## Open questions
- **Downloads copy vs. Paper_Drafts copy**: `MultiIdiom_Submission_Ready.docx` in `~/Downloads/` appears to be a separate (possibly more polished) version of the paper. Unclear which is the true submission draft. The user should consolidate — either copy the fixes from `MultiIdiom_Dataset_Paper.docx` into a single canonical file, or clarify which version gets submitted.
- **IAA table still missing**: Section 5.1 in both files has a `[FILL IN: Insert Telugu and English IAA table…]` placeholder. This cannot be left blank at submission. Requires running `Annotation_tool/compute_iaa.py` with the two annotators' output files (annotator1.jsonl, annotator2.jsonl not yet located).
- **Hindi and Indonesian test sets unvalidated**: noted in paper as silver-standard. No action needed before submission but should be flagged in limitations (already done in current draft).
- **`~/Downloads/MultiIdiom_Submission_Ready.docx`**: This file was edited this session (same fixes applied). Its origin/relationship to `Paper_Drafts/MultiIdiom_Dataset_Paper.docx` is unclear — they appear to be slightly different drafts of the same paper, with the Submission_Ready version having more polished Section 5 language.

## Next steps
1. **Decide canonical submission file**: Clarify whether `Paper_Drafts/MultiIdiom_Dataset_Paper.docx` or `Downloads/MultiIdiom_Submission_Ready.docx` is the true submission draft. Merge into one file.
2. **Fill the IAA table (Section 5.1)**: Locate annotator output files and run `Research_And_Training/Annotation_tool/compute_iaa.py`. The script expects `annotator1.jsonl` and `annotator2.jsonl` filtered to Telugu. The placeholder in the paper is at the paragraph after "Table 4 reports agreement figures" in Section 5.1.
3. **Verify all other numeric claims in Table 2**: While investigating, confirmed Telugu row is now correct. The other rows (English 7608, Spanish 2476, Hindi 142, Indonesian 33 test examples; test idiom counts 964/319/18/5) were not verified against the actual split files this session — worth a quick cross-check against `split_stats.json` before submission.
4. **Update key_numbers.md in memory**: Add the confirmed Telugu test count (62, 11 unique idioms) so future sessions don't re-investigate this.

## Pick-up prompt
Continuing work on the MultiIdiom dataset paper (`Paper_Drafts/MultiIdiom_Dataset_Paper.docx`). Last session resolved a Telugu test-count discrepancy: the paper claimed n=73 but the canonical split has n=62 (confirmed via `idioms_structured/Splits/split_stats.json` and `Sampler.py` with seed 42). Both paper drafts were corrected (Test=62, Test Idioms=11). The immediate next task is filling the IAA table placeholder in Section 5.1 — locate the two annotator output files and run `Annotation_tool/compute_iaa.py`. There is also an open question about which of two docx files (`Paper_Drafts/MultiIdiom_Dataset_Paper.docx` vs `~/Downloads/MultiIdiom_Submission_Ready.docx`) is the true submission draft. Read HANDOFF.md at `Research_And_Training/HANDOFF.md` for full context.
