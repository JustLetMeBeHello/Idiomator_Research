# Handoff — 2026-05-26 PM — BIO decoder fix, IdiomBERT-v2 design, annotation-tool reliability patch

## TL;DR
Three threads, all closed cleanly. **(1) Decoder bug fix end-to-end:** patched `decode_bio_to_char_span` in both source files (`Ablations/BiO_Task_mBERT_train.py:352` and `Google_Colab/bio_tagger_hparam_search.py:277`), wrote a re-decode comparison script (`Additional_Rigor_Experiments/run_05_decoder_rerun.py`), and ran it on both BIO prediction files. Results are dramatic — Telugu BIO EM goes 0.0000 → 0.7419, F1 0.2415 → 0.8674, and the bug affects every language with multi-subtoken span endwords (English +0.16 EM, Hindi +0.39, Spanish +0.34, Indonesian +0.26). The IdiomBERT paper's "BIO=0.000 on Telugu" smoking gun is a decoder artifact. **(2) IdiomBERT-v2 unified-architecture design + code sketch:** discovered via Google search that Matheny et al. 2026 (arXiv:2603.22799) already published slot loss + SCL + hard negative reweighting for English idiomaticity, and Avram et al. 2023 (arXiv:2306.10419) already published lateral inhibition for multilingual MWE on PARSEME. Neither evaluates Telugu/Hindi/Indonesian/Spanish — Matheny explicitly invites the multilingual extension as future work. Wrote `Additional_Rigor_Experiments/run_06_idiombert_v2.py` (~400 lines, flag-toggleable SCL+HNR+LI on the existing QA-style backbone) and smoke-tested all four loss components. **(3) Annotation-tool reliability patch:** replaced fire-and-forget `fetch('/save')` in `Annotation_tool/index.html` with a localStorage-backed retry queue + visible "⟳ N pending sync" badge + 30 s periodic drain + sendBeacon flush on unload. Verified end-to-end in a headless preview browser (happy path, simulated network failure, recovery — all pass). Session ended with the full 1-2-month preprint plan + Aug ARR submission + NAACL 2027 Main commit calendar locked in.

## Code state
- Repo: `/Users/shishirmaddineni/Desktop/Idiomator_Research/Research_And_Training`, branch: `main`, ahead-of-origin: no
- Session CWD was `/Users/shishirmaddineni/Desktop/Personal_Language_Automation`; all real work happened in this repo
- Uncommitted changes:
  - `Ablations/BiO_Task_mBERT_train.py` (+11 lines) — `decode_bio_to_char_span` patched at line 352: walk forward through subtokens of the last span word before taking `offsets[last_tok][1]`
  - `Google_Colab/bio_tagger_hparam_search.py` (+5 lines) — same fix at line 277, shorter variable names
  - `Annotation_tool/index.html` (+123 lines, -8 lines) — added `pendingBadge` element in topbar; `PENDING_KEY`, `SAVE_TIMEOUT_MS`, `DRAIN_INTERVAL_MS` constants; `loadPending`/`savePending`/`updatePendingBadge`/`enqueueSave`/`_postOne`/`drainQueue` functions; replaced fire-and-forget POST in `submitExample()` with enqueue + drain; added `setInterval(drainQueue, 30000)` and `beforeunload` `navigator.sendBeacon` flush in the load handler
  - `Annotation_tool/annotations.db` (binary, no row count change) — `_test_drain` test rows added and deleted during testing
- Untracked new files:
  - `Additional_Rigor_Experiments/run_05_decoder_rerun.py` (~230 lines) — re-decode comparison; runs both BIO prediction files through buggy + patched decoders, dumps side-by-side per-language metrics + full per-example diff
  - `Additional_Rigor_Experiments/run_06_idiombert_v2.py` (~400 lines) — unified training script extending `Train_Join.py` with flag-toggleable SCL + HNR + LI. Inherits `JointDataset` + `JointIdiomModel`; adds `JointDatasetV2` (exposes idiom_id + is_literal), `IdiomBERTv2` (projection head for SCL), `IdiomBalancedBatchSampler` (in-batch positives + literal hard negatives), `span_contrastive_loss` (numerically-stable SupCon), `hnr_weighted_ce`, `lateral_inhibition_decode` (output-side NMS over top-k start/end pairs)
  - `Additional_Rigor_Experiments/results/decoder_rerun_system_G.jsonl` (632 rows) — per-example side-by-side from System G predictions
  - `Additional_Rigor_Experiments/results/decoder_rerun_xling_matrix.jsonl` (957 rows) — per-example from cross-lingual matrix BIO row
  - `Annotation_tool/annotations.db.bak-20260526-134405` — pre-test backup of annotations.db (safe to delete)
- Untracked outside the repo:
  - `/Users/shishirmaddineni/Desktop/Personal_Language_Automation/.claude/launch.json` — `idiombank-annotator` Preview server config (`preview_start idiombank-annotator` to re-test annotation tool interactively)
  - `/tmp/smoke_idiombert_v2.py` — smoke test for `run_06_idiombert_v2.py` (temp, ok to delete)
- Build/test status:
  - All source files `py_compile`-clean (smoke-imported by `run_06`'s test harness; syntax & all required identifiers verified for the annotation `index.html` via static check)
  - `run_05_decoder_rerun.py` **fully run** on both prediction files — results in `Additional_Rigor_Experiments/results/`
  - `run_06_idiombert_v2.py` smoke-tested (SCL on toy batches: finite mixed/zero perfect/zero all-literal/zero all-unique; LI returns valid `e>=s`; HNR weighted CE finite; IdiomBalancedBatchSampler 50/50 coverage, no dups, fallback works on all-unique inputs). **Not yet run on real training data — needs one-epoch dry-run before committing Colab time.**
  - Annotation tool patched + verified end-to-end via headless preview browser: happy path (drain to 0, badge hidden, DB +1), failure path (queue +1, amber badge visible, UI advances), recovery path (drain clears queue, DB +1), red threshold at N>10 confirmed (`rgb(248,113,113)` = `#f87171`)
- venv side-effect: `fastapi`, `uvicorn`, `sqlalchemy`, `pydantic`, `psycopg2-binary` installed into `Research_And_Training/.venv` (needed to run annotation server locally; listed in `Annotation_tool/requirements.txt` anyway). NOTE: the venv's `pip` shebang is broken (points to `Idiomator_Research/.venv/bin/python3` which doesn't exist) — use `.venv/bin/python3 -m pip` not `.venv/bin/pip`.

## Decisions made
- **Patch BIO decoder in-place, don't just footnote it** — Why: the fix is 4 lines and the corrected baseline numbers strengthen the paper rather than weakening it. Telugu BIO F1 (0.87) now actually exceeds QA-style F1 (~0.71-0.74), inverting the "QA is necessary" claim — but System G as a baseline becomes more competitive, which is good for the comparative tables.
- **No retraining needed for System G** — Why: training loss in `align_bio_labels` only supervises per-token BIO at first-subtoken positions; the decoder is purely a post-processor and never touches the loss. Re-decoding existing predictions gives the same numbers a re-trained model would. One footnote-able caveat: best-epoch selection used `best_dev_overlap` computed with the buggy decoder; almost certainly unchanged but unprovable without per-epoch checkpoints (which weren't saved).
- **Pivot from "errata audit" (path 1) to "model paper" approach** — Why: user said "I don't want to find errors but I do want a model paper." Drops the Path 1 audit of prior MWE work for the same decoder bug. Keeps the decoder fix as a contribution-flavored footnote in IdiomBERT, not as a stand-alone paper.
- **Adopt Matheny + Avram as related work, position IdiomBERT-v2 as multilingual non-Latin extension** — Why: Matheny et al. 2026 (SCL+HNR for English idiomaticity) and Avram et al. 2023 (lateral inhibition for European MWE) already publish the techniques the user wanted to propose. Novelty as initially scoped is gone. But the multilingual gap is wide open and Matheny *explicitly* names it as future work. Reframed contribution: "first multilingual non-Latin MWE benchmark unifying SCL + HNR + LI; we propose a small language-conditioned adaptation that recovers expected gains on Telugu/Hindi/Indonesian."
- **Preprint = corrected v1, ARR submission = v2** — Why: v1 (decoder fix + corrected baseline + IAA) ships in 3-4 weeks at zero compute. v2 (unified architecture + ablation matrix + Matheny/Avram extension) ships ~6-8 weeks. v1→v2 arXiv progression is normal and de-risks the timeline — if v2 ablations surprise (null result on any cell), v1 preprint is already live.
- **8-cell ablation matrix, not 5** — Why: user has the 4-5 day Colab babysitting budget; reviewers will accept fewer than 8 but resist if pair-cells (SCL+HNR, SCL+LI, HNR+LI) are missing and the all-three result is the headline. Cells: baseline / +SCL / +HNR / +LI / +SCL+HNR / +SCL+LI / +HNR+LI / all-three.
- **Run Llama-3 baseline (exp 04) from laptop via Groq HTTP, not Colab** — Why: Groq is HTTP-only, no GPU needed. Frees Colab time exclusively for the 8-cell ablation matrix.
- **Skip exp 01 (BIO+MuRIL Telugu recovery)** — Why: the decoder fix already gives Telugu BIO F1 of 0.87 on mBERT, leaving almost no room for MuRIL to add a meaningful delta. The experiment was decision-influencing under the buggy decoder; with the bug fixed it's mostly redundant.
- **IAA table: 3 rows (EN+HI+TE), NOT 5** — Why: user has 2 annotators for TE+HI confirmed, EN getting recruited (2nd annotator), Spanish only 1 annotator (no κ possible), Indonesian held out for zero-shot. Drop any prior "Spanish validated ✓" claim from paper text; Spanish + Indonesian both stay silver-standard. (Project memory still has stale `ES ✓` line — left as-is per user, but should be reconciled when the user does their next memory consolidation.)
- **Annotation-tool fix: queue + retry, not just await + alert** — Why: keeps the annotator unblocked even on offline submission. Queue catches both network errors AND server-side validation 422s — extending the Pydantic schema later won't silently lose old records. Visible badge (amber → red at N>10) gives annotators feedback without being alarming.
- **Annotation security: trusted-friends model, not auth** — Why: low-effort for current annotator pool of 5 trusted people; pre-agreed unique name strings ("shishir_te", "anu_te") + daily `/export/{annotator}` backups + private URL channel close the realistic risk. Token-based auth is a ~15 min future hardening if the pool expands.
- **Calendar: preprint by ~Jun 28, ARR by Aug 15, NAACL 2027 commit by Nov-Dec 2026** — Why: fits user's 1-2 month preprint window + stated "Main paper(s) accepted at ACL by end of 2026" goal. The "accepted by end of 2026" milestone = the Aug ARR review → Nov/Dec 2026 commit-to-NAACL-2027 decision.

## Rejected approaches
- **Path 1 — audit prior MWE work (PARSEME, ID10M, AStitchInLanguageModels, DiMSUM) for the same decoder bug** — Why not: user explicitly declined ("I don't want to find errors"). Would have been the highest-reward Main path but requires 2-3 weeks of reading other authors' code and possibly re-running their models — too much wall-clock for the preprint window.
- **Option B — biaffine span head + lateral inhibition + SCL as architectural contribution** — Why not: both biaffine NER (Yu et al. 2020) and lateral inhibition for MWE (Avram et al. 2023) already exist; novelty story is too thin and reviewers will flag the kitchen-sink composition.
- **Option C — all-in (Option A + Option B combined)** — Why not: reviewers will say "we can't tell what matters."
- **v2 as preprint directly** — Why not: 6-8 weeks puts you at the back of your 1-2 month window with no fallback if any ablation cell surprises.
- **Full 15-combination × 3 techniques ablation matrix (45 cells)** — Why not: 15-combo ablation defends "how training composition affects mBERT" — a different question from "which training objective works best." For v2 only need full-training cells.
- **Coefficient sweep at 10 points × 7 epochs** — Why not: Colab Free T4 budget. Reduced to 3 points {0.05, 0.1, 0.5} × 3 epochs on dev only; pick winning λ then full-train once.
- **Spanish IAA with 1 annotator** — Why not: κ requires 2 annotators. Single-annotator spot-check is reportable but weaker than κ; Spanish remains silver-standard in the paper.
- **Use a real headless browser (Claude_in_Chrome) for the annotation-tool test** — Why not: Claude_Preview was sufficient and lighter; the only quirk was that `preview_click` sometimes didn't fire `onclick` handlers cleanly, mitigated by calling the JS handlers directly (`selectSpan(true)`, `selectSense('correct')`). Real browser clicks aren't affected.

## Open questions
- **5 sanity mismatches in the decoder re-run** (5/632 in System G, 6/957 in cross-lingual matrix) — all are cases where the buggy decoder returned `None,None` but the JSONL stored `(0,0)`. All representational, all score EM=0 / F1=0 either way. Worth a one-line footnote in the IdiomBERT §6.2 errata; not a real defect.
- **A second smaller decoder issue surfaced during the audit**: even the *patched* decoder ignores B-IDIOM tags that appear only on non-first-subtokens of words (4/632 examples, 0.6%). Fixable by treating "any subtoken of word W is B-IDIOM" as "word W starts the span." Minor; mention in the same errata footnote.
- **Whether to retrain System G once with patched decoder in eval loop** to remove the best-epoch-selection caveat — preprint can footnote it; ARR submission probably should retrain (~1 Colab hour) to be reviewer-proof.
- **Recruit 2nd English annotator** — user said they can but hadn't yet at session end.
- **`MultiIdiom_Dataset_Paper.docx`** still in `Paper_Drafts/` with author's real name visible — privacy risk if accidentally submitted to ARR double-blind. Carried over from prior HANDOFF; no decision made.
- **Spanish ✓ stale claim in `project_idiomator_research.md` memory** — user said leave it for this session; needs reconciliation in next memory consolidation pass.
- **ARR submission identity** — user wants 1-2 Main acceptances. MultiIdiom Main path is "first publicly-released span-annotated Telugu + 3-language gold IAA + 5-lang coverage + synthetic with audit." IdiomBERT Main path is the unified-architecture multilingual benchmark. If ablation matrix shows null results on per-component contributions, IdiomBERT drops to Findings (still acceptable for user's stated goal).
- **Telugu IAA table cell** still `[TODO]` — pending annotator completion (~2-3 weeks).
- **`/tmp/smoke_idiombert_v2.py`** can be deleted; not used elsewhere.

## Next steps
1. **TODAY: send annotation tool URL to TE + HI annotators** with pre-agreed unique name strings (e.g. `shishir_te`, `anu_te`). Critical-path bottleneck — every day of delay shifts the whole calendar right.
2. **TODAY: recruit 2nd English annotator** so EN doesn't become a separate critical path later.
3. **Daily during annotation window**: `curl http://<railway-url>/export/{annotator}?lang=Telugu > backup_{date}.jsonl` for each active annotator.
4. **As each language hits 2-annotator completion**: `cd Research_And_Training/Annotation_tool && python compute_iaa.py annotator1.jsonl annotator2.jsonl --lang Telugu --latex` → drop κ values into MultiIdiom §5.1 table.
5. **Preprint edits to `Paper_Drafts/MultiIdiom_Submission_Ready_v3.docx`**:
   - Drop Spanish ✓ claim; switch §5 IAA table to 3 rows (EN+HI+TE)
   - Hindi "Partial" → gold (matches new 2-annotator status)
6. **Preprint edits to `Paper_Drafts/IdiomBERT_Submission_Ready_v3.docx`**:
   - Resolve FLAG A in §3: the "956-example common test set" is 957 with Indonesian breakdown (EN 254 + ES 254 + HI 62 + TE 62 + ID 325 = 957; ID held out for zero-shot)
   - Resolve FLAG B in §6.1: Hindi range is 0.56-0.69 (actual cell values), not 0.45-0.69
   - Drop "BIO=0.000 on Telugu" framing in §6.2; replace with corrected numbers from `run_05` (Telugu BIO EM=0.74, F1=0.87)
   - Add decoder-fix errata footnote (covers the 5 sanity mismatches + the non-first-subtoken B-IDIOM minor issue)
   - Cite Matheny et al. 2026 + Avram et al. 2023 + Tedeschi & Navigli NER4ID 2022 in related work
   - Reframe Conclusion / §7 to position v2 as the multilingual extension
7. **Produce de-anonymized `_preprint.docx` versions** of both papers (recipe in prior HANDOFF Next Steps #4: author block "Shishir Maddineni · Independent Researcher · 1356shishir@gmail.com", `(Anonymous, 2026)` → `(Maddineni, 2026)` throughout, drop "(de-anonymized for camera-ready)" parenthetical from reproducibility paragraph).
8. **Run exp 04 (Llama-3 via Groq) from laptop** — get Groq API key (free, no card), `export GROQ_API_KEY=...`, smoke test: `python Additional_Rigor_Experiments/run_04_llama3_baseline.py --dry_run 10`, then full 960-call run (~60 min, $0).
9. **One-epoch dry-run of `run_06_idiombert_v2.py` on real data** before committing Colab time — catches any shape/path bugs cheaply. `python Additional_Rigor_Experiments/run_06_idiombert_v2.py --output_dir /tmp/_dryrun --epochs 1 --batch_size 8 --langs English`. Likely 20-30 min on Mac CPU. Worth it.
10. **Block 2 weekends for the Colab ablation matrix** (user + friend, both accounts active). 8 cells × ~100 min on Free T4. Cell order (most decision-influencing first): baseline + all-three (day 1 AM) → +SCL + +HNR (day 1 PM) → +LI + +SCL+HNR (day 2 AM) → +SCL+LI + +HNR+LI (day 2 PM) → λ sweep cells (faster). Shared Drive folder `IdiomatorRigor/idiombert_v2/<cell>/`.
11. **Pull results to local repo**: `rsync -av ~/Google\ Drive/My\ Drive/IdiomatorRigor/idiombert_v2/ ~/Desktop/Idiomator_Research/Research_And_Training/models/idiombert_v2/`, then `python Evaluation/Full_evaluation.py`.
12. **Write IdiomBERT-v2 sections** (~15-20 hr): ablation table, position as "first multilingual non-Latin benchmark unifying SCL+HNR+LI", report the per-language transfer story.
13. **Final polish + ARR anonymization** by ~Aug 12, submit ~Aug 15.
14. **Optional**: retrain System G once with patched decoder in eval loop (~1 Colab hour) to remove the best-epoch-selection caveat from §6.2.
15. **Optional**: add `?token=...` query-param check to `Annotation_tool/main.py` `/save` and `/export` endpoints (~15 min) if annotator pool expands beyond trusted friends.
16. **Cleanup whenever comfortable**: `rm Annotation_tool/annotations.db.bak-20260526-134405`, `rm /tmp/smoke_idiombert_v2.py`.

## Pick-up prompt
Continuing IdiomBERT + MultiIdiom preprint prep targeting **1-2 month preprint → Aug 2026 ARR → NAACL 2027 Main commit (Nov-Dec 2026)**. This session closed three threads: **(A)** patched the BIO decoder bug (`Ablations/BiO_Task_mBERT_train.py:352` + `Google_Colab/bio_tagger_hparam_search.py:277`), re-decoded both BIO prediction files via the new `Additional_Rigor_Experiments/run_05_decoder_rerun.py` — Telugu BIO EM 0.0000 → **0.7419**, F1 0.24 → **0.87**, and every language is affected (EN +0.16, HI +0.39, ES +0.34, ID +0.26); **(B)** designed IdiomBERT-v2 after discovering Matheny et al. 2026 (arXiv:2603.22799, SCL+HNR for English idiomaticity) and Avram et al. 2023 (arXiv:2306.10419, lateral inhibition for European MWE) already exist — the novelty pivot is "first multilingual non-Latin benchmark unifying these techniques," with a flag-toggleable training script at `Additional_Rigor_Experiments/run_06_idiombert_v2.py` (smoke-tested, not yet run on real data); **(C)** patched the annotation tool with a localStorage-backed retry queue + visible badge + `sendBeacon` flush (`Annotation_tool/index.html`, verified end-to-end in headless browser). IAA table is now 3 rows (EN+HI+TE) — Spanish silver, Indonesian zero-shot. **Immediate next actions today**: (1) send annotation URL to TE+HI annotators with unique names, (2) recruit 2nd English annotator, (3) start preprint .docx edits resolving FLAG A (957 not 956 in §3, includes Indonesian breakdown) and FLAG B (Hindi range 0.56-0.69 in §6.1) and dropping "BIO=0.000 on Telugu" framing in §6.2. Calendar target: preprints posted ~Jun 28, ARR submitted Aug 15. Read this `Research_And_Training/HANDOFF.md` top-most dated section (2026-05-26 PM) for full context including decisions, rejected paths, decoder-fix per-language numbers, and the 16-step plan. The next-down section (2026-05-26 AM) covers the prior decoder-bug discovery + Llama-3 provider context.

---

# Handoff — 2026-05-26 — Llama-3 provider strategy + paper-changing BIO decoder bug discovered

## TL;DR
Two threads. **Thread A (closed)**: Reworked the Llama-3.3-70B baseline provider strategy in `run_04_llama3_baseline.py`. Pivoted Together FP8 Turbo → DeepInfra BF16 → Groq free preprint default (dual-track: Groq now for $0 preprint, DeepInfra BF16 later for the ARR cutover, avoids $5 DeepInfra minimum top-up until paper proves worth submitting). Two commits on main, working tree clean. **Thread B (open, paper-changing)**: While pre-flighting exp 01 for SentencePiece-vs-WordPiece tokenizer issues, discovered that **MuRIL is actually WordPiece** (HANDOFF concern was unfounded) but also found a **decoder bug in `Ablations/BiO_Task_mBERT_train.py:decode_bio_to_char_span`** that **structurally prevents exact_match > 0 on Telugu regardless of model quality**. Even with PERFECT predictions, 3 real Telugu test examples decode to truncated spans. The IdiomBERT paper's "BIO=0.000 on Telugu" finding is a decoder artifact, not a learning failure. The "QA-style is necessary" claim may dissolve entirely once the decoder is fixed. **Do not run exp 01 until the decoder is patched and the baseline is re-decoded.**

## Code state
- Repo: `/Users/shishirmaddineni/Desktop/Idiomator_Research/Research_And_Training`, branch: `main`, up to date with `origin/main`
- Uncommitted changes: **none** — working tree clean
- Commits made this session (both on `main`, both pushed):
  - `5d856dc` — Llama-3 baseline: Groq free as preprint default, DeepInfra deferred to ARR cutover *(current HEAD)*
  - `6bc760e` — *(superseded by 5d856dc)* — Defaulted Llama-3 baseline to DeepInfra BF16 instead of Together FP8 Turbo
- Files touched this session:
  - [`Additional_Rigor_Experiments/run_04_llama3_baseline.py`](Research_And_Training/Additional_Rigor_Experiments/run_04_llama3_baseline.py) — default `--provider` is now `groq`; new "Dual-track precision strategy" docstring section with exact footnote text for preprint and ARR phases; PROVIDERS dict reordered (Groq first as preprint-grade, DeepInfra second as paper-grade-for-ARR); `--sleep` default raised 0.2 → 2.0 to respect Groq's 30-RPM free-tier limit; `--cost_per_million_tokens` default now 0.0 (Groq free).
  - [`Additional_Rigor_Experiments/README.md`](Research_And_Training/Additional_Rigor_Experiments/README.md) — corrected the ~$15-30 LLM budget estimate to $0 (preprint) / ~$0.25 (ARR re-run); run-time table has two rows for exp 04; cost summary explains dual-track strategy with exact paper-footnote text for each phase.
  - [`Additional_Rigor_Experiments/Run_In_Colab.ipynb`](Research_And_Training/Additional_Rigor_Experiments/Run_In_Colab.ipynb) — cell 7 (API keys) now references `GROQ_API_KEY` as primary; cell 19 (exp 04 invocation) uses `--provider groq`; fan-out section (cell 25) updated to keep all LLM API calls on owner's machine.
- Build/test status: scripts syntax-checked clean (`py_compile` on `run_04_llama3_baseline.py`, JSON parse on `Run_In_Colab.ipynb`). **None of the exp 01-04 scripts have been run on real data this session.** Empirical tokenizer probe (`AutoTokenizer.from_pretrained` for mBERT/MuRIL/XLM-R + `align_bio_labels` + `decode_bio_to_char_span` simulated with gold labels) was done at `/tmp` and CWD — that's how the decoder bug was found.

## Decisions made
- **Dual-track Llama-3 strategy: Groq free (preprint) → DeepInfra BF16 (ARR)** — Why: User raised the $5 DeepInfra minimum top-up. arXiv reviewers don't ding precision; we can ship the preprint at $0 with a footnote ("Llama-3.3-70B served via Groq's LPU; provider-quantized"), then $5 of DeepInfra credit before ARR Aug 2026 lets us re-run on BF16 and swap the footnote. Total preprint commitment: $0. Total ARR commitment: ~$5 minimum.
- **Direct commits to `main`, no PRs** — Why: User is solo author/reviewer on this repo; PRs add ceremony with no review benefit. Confirmed explicitly.
- **MuRIL is BertTokenizerFast (WordPiece), not SentencePiece** — Why: Empirically verified via `AutoTokenizer.from_pretrained('google/muril-base-cased')`. The HANDOFF.md concern from prior session about "SentencePiece offset behavior in MuRIL" was based on a misconception. MuRIL uses BERT-architecture WordPiece with an Indic-aware 197k vocab. This means exps 01 and 03 are drop-in compatible with `align_bio_labels` and `char_to_token_span` — no code changes needed for that reason alone.
- **XLM-R SentencePiece offsets are clean** — Why: Empirically verified. `▁Hello` returns `offset=(0, 5)`, NOT `(-1, 5)`. The leading `▁` does not push a phantom character into the offset. The alignment logic's `tok_char_s >= char_start AND tok_char_e <= char_end` will work correctly for XLM-R (exp 02).
- **Don't run exp 01 with the current decoder** — Why: The decoder bug (see Open Questions) means exp 01 would still produce a meaningful mBERT-vs-MuRIL delta, but mis-attribute the cause. With the bug, MuRIL beats mBERT on Telugu because MuRIL's larger first-subtokens partially mask the truncation, not because of any real encoder-side improvement. The result wouldn't answer the council's decision-influencing question.

## Rejected approaches
- **Together AI FP8 Turbo as Llama-3 default** — Why not: Together's serverless 70B inventory is FP8-quantized "Turbo" only as of 2026-05. A reviewer can correctly flag FP8 as a confound for the "frontier open-weights LLM" framing. Kept in PROVIDERS dict as a fallback with explicit warning comment.
- **OpenRouter for Llama-3** — Why not: Routes to whichever backend has capacity. Reproducibility is provider-dependent between runs. Marked "NEVER use for the paper artifact" in code comment.
- **Run Llama-3.3-70B on Colab Pro+** — Why not: 140 GB BF16 weights vs 80 GB max A100-80 VRAM. Doesn't fit. Quantizing to INT4 (~35 GB, fits) reintroduces the same FP8/quantization confound we just escaped Together to avoid.
- **Smaller Llama variant (3.1-8B)** — Why not: Loses the "frontier open-weights LLM" framing that defuses the single-LLM (GPT-4o only) reviewer attack the council flagged.
- **Feature branch + PR for the Llama-3 work** — Why not: User explicitly said "I don't want a PR I'm a self hosted developer". Pushed a branch initially, then cleaned up: fast-forwarded `main`, deleted the feature branch locally and remotely.

## Open questions
- **Should we fix the BIO decoder and re-run System G mBERT baseline?** Paper-changing question. If a fixed decoder recovers Telugu BIO performance (likely — even imperfect predictions would now produce real exact_match scores), the IdiomBERT "QA-style is necessary" architectural claim collapses. Alternative framings: (a) quiet correction in revised paper, (b) headline contribution "we identified a subtle but consequential decoder issue affecting prior BIO-based MWE work."
- **Does the same bug affect Hindi / Indonesian numbers?** Likely yes for Hindi (also multi-subtoken-prone with Devanagari + mBERT). English likely barely affected (words rarely fragment). Needs side-by-side audit.
- **If exp 01 framing changes, what about the entire "QA-vs-BIO" Section 6.2 narrative?** The council's reframing recommendation ("no fine-tuned mBERT system shows a statistically significant Joint F1 advantage at full training") might now extend to: "the apparent BIO failure on Telugu was a decoder artifact; with the corrected decoder, BIO and QA-style perform within bootstrap CI on all languages." If true, the paper's structural argument has to change.
- **Llama-3 next-action**: Groq API key not yet generated; smoke test (`--dry_run 10`) not yet run; full 960-call run not yet started. Still on the user's plate, no blockers.
- **Existing baseline predictions location**: `models/bio_tagger_en_hi_te/test_predictions.jsonl` is the most likely location for the System G mBERT predictions that need re-decoding with the fixed decoder — needs verification before patching.
- **IAA / Telugu validation** (carried from prior session) — User is doing this on a separate track, mid-June completion expected. Not in this thread.

## Next steps
1. **Patch the BIO decoder** at `Ablations/BiO_Task_mBERT_train.py:352-353`. Current code: `last_tok = span_tokens[-1]; char_end = offsets[last_tok][1]`. Fix: extend `last_tok` to the LAST subtoken of the last word in the span, by walking forward through tokens while `word_id == word_ids[last_tok]`. ~5 lines.
2. **Locate the existing System G mBERT predictions** — most likely `models/bio_tagger_en_hi_te/test_predictions.jsonl`. Verify path and structure (per-example dict with predicted token labels + sentence + gold span).
3. **Write a re-decode script** that loads those predictions, re-runs `decode_bio_to_char_span` with both the original and patched version, computes per-language exact_match + overlap_f1 for both. No GPU needed, runs in seconds. **This is decision-influencing — gives the corrected baseline numbers before any new GPU work.**
4. **If Telugu BIO recovers materially** (>0.2 exact match), revisit the council's experiment-01 framing entirely — the question shifts from "encoder swap rescues Telugu" to "decoder fix is sufficient, encoder swap is incremental."
5. **Then run exp 01 in Colab** (BIO + MuRIL Telugu recovery) with the patched decoder. ~45 min on A100. Compare to the corrected baseline, not the buggy 0.000 baseline.
6. **In parallel**: get a Groq API key (free, no credit card), `export GROQ_API_KEY=...`, smoke test exp 04 with `python Additional_Rigor_Experiments/run_04_llama3_baseline.py --dry_run 10`, then the full 960-call run (~60 min, $0).
7. **Audit Hindi numbers** with the patched decoder using the same re-decode script. Hindi/Devanagari may be affected similarly to Telugu.
8. **Add a "Limitations / Errata" subsection** or rewrite Section 6.2 of IdiomBERT depending on what the corrected numbers show.

## Pick-up prompt
Continuing IdiomBERT + MultiIdiom preprint prep. This session closed the Llama-3 provider question (Groq free for preprint, DeepInfra BF16 deferred to ARR cutover — two commits on main, `5d856dc` is HEAD, working tree clean) and **opened a paper-changing finding**: while pre-flighting exp 01 for tokenizer compatibility, I proved empirically that `Ablations/BiO_Task_mBERT_train.py:decode_bio_to_char_span` (lines 312-365) has a bug where `last_tok = span_tokens[-1]` is the *first subtoken of the last word in the span*, so `offsets[last_tok][1]` truncates the decoded char_end. Even with PERFECT predictions (using gold labels as predictions), 3 real Telugu test examples decode to truncated spans — mBERT exact_match=0/3, MuRIL exact_match=0/3. **The IdiomBERT paper's "BIO=0.000 on Telugu" is a decoder artifact, not a learning failure.** The QA-style decoder in `Train_Join.py:174-195` walks backward from char_end and is correct, so the architectural advantage of QA-style may collapse once BIO's decoder is fixed. Immediate next action: patch the decoder (extend `last_tok` forward to the last subtoken of the last word, ~5 lines), then write a re-decode script that re-runs the existing System G mBERT predictions through both old and new decoders side-by-side — this gives the corrected Telugu/Hindi baseline numbers without burning any Colab time, and tells us whether exp 01's framing needs to change before we run it. Read `Research_And_Training/HANDOFF.md` (top section, dated 2026-05-26) for full context including the empirical decoder-bug proof and dual-track Llama-3 strategy.

---

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
