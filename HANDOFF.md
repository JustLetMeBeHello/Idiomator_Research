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
