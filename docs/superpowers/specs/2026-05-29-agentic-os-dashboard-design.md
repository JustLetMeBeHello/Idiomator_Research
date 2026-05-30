# Personal Agentic OS — Design Spec (v1 slice)

**Date:** 2026-05-29
**Status:** Approved, ready for implementation plan
**Scope:** Vertical slice — OS shell + fully-wired Research folder + Next-Step engine. Other folders stubbed.

---

## 1. Goal

A local "Personal Agentic OS": a desktop-metaphor web dashboard that acts as a command center over the
IdiomBERT / MultiIdiom research. Top level is the OS shell; specialized functions live in folders ("rooms").
Research is the one real folder in v1; Papers / Agents / OS Core are clickable stubs.

A **Next-Step engine** runs across the whole OS: it derives an ordered backlog from real project state,
always surfaces the single next action, and auto-advances when a task is completed.

## 2. Decisions (locked during brainstorming)

| Decision | Choice |
|---|---|
| Build slice | Shell + Research (real data), other folders stubbed |
| Stack | Python FastAPI backend + React/vite frontend |
| Shell aesthetic | Desktop OS — folder icons → draggable/resizable windows |
| Metric source of truth | Both; `key_numbers.md` wins, live JSON for drill-down, flag mismatches |
| Stub depth | Shell + placeholder text (planned-panel sketch, no fake data) |
| Next-step task source | Derived from project state (no manual list) |
| Task completion persistence | Write back to memory/project source files (read-back verified) |

## 3. Non-Goals (YAGNI v1)

No auth, no database, no write-back beyond task-status line edits, no real agent spawning, no live Colab
API polling (experiment status read from memory files). No cloud deploy — runs locally.

## 4. Architecture

```
agentic_os/                          # new dir at repo root
  backend/
    main.py                          # FastAPI app, CORS, startup config validation
    config.py                        # REPO_ROOT, MEMORY_DIR env vars (absolute), validated on boot
    sources/
      folders.py                     # folder registry (live vs stub)
      memory.py                      # read ~/.claude/.../memory/*.md
      results.py                     # read results/pipeline_eval/pipeline_eval_results.json
      git_activity.py                # git log -> activity feed
    parsers/
      key_numbers.py                 # markdown tables -> typed dicts
      project_overview.py            # exp queue + blockers -> typed dicts
    tasks/
      derive.py                      # build ordered backlog from project state
      writeback.py                   # surgical line-edit + backup + read-back verify
    tests/
      fixtures/                      # copied md/json samples
      test_parsers.py
      test_tasks_roundtrip.py
  frontend/                          # vite + React + TypeScript
    src/
      api.ts                         # typed fetch client
      os/                            # Desktop, Window, Taskbar, IconGrid, NextStepBar
      folders/
        research/                    # 6 panels
        papers/  agents/  oscore/    # stub rooms
  README.md                          # how to run (backend + frontend)
```

## 5. Backend

### 5.1 Config (`config.py`)
- `REPO_ROOT` (default: this repo abs path), `MEMORY_DIR` (default: the memory dir abs path).
- Validated on startup: each path must exist + be readable. Fail loud (raise, refuse to serve) if not.

### 5.2 Data sources (confirmed shapes)
- `results/pipeline_eval/pipeline_eval_results.json` = dict keyed by 9 systems
  (`system_a_mbert_pipeline` ... `system_g_bio_tagger`), each →
  `{cls_f1, span_standalone|span_all, span_e2e, span_correct_id, joint_acc, joint_f1, stability, small_lang_ci|indonesian_ci}`.
- Curated metrics live in `memory/key_numbers.md` as markdown tables/lines (Joint F1, stability,
  classification F1, ablation matrix 15-combo table, Indonesian zero-shot).
- Ablation matrix has **no local JSON** (`results/language_ablation_matrix/` is Drive-only) → ablation
  panel sources from the `key_numbers.md` table exclusively.
- Experiment queue + blockers live in `memory/project_overview.md`.

### 5.3 Endpoints

| Method | Path | Returns |
|---|---|---|
| GET | `/api/folders` | `[{id,name,icon,status:"live"|"stub"}]` |
| GET | `/api/research/systems` | 9 systems × metrics. key_numbers primary, live JSON merged, `mismatch:true` when curated ≠ JSON. G annotated "span-only, not comparable." |
| GET | `/api/research/ablation` | parsed 15-combo matrix (span F1 / Indo held-out / stability) from key_numbers.md |
| GET | `/api/research/experiments` | exps 01–07 with status + notes + run-order |
| GET | `/api/research/blockers` | IdiomBERT + MultiIdiom blocking TODOs (IAA = hard blocker flag) |
| GET | `/api/activity` | recent git commits |
| GET | `/api/next` | the single current next step (+why +exact command/file) |
| GET | `/api/tasks` | full ordered backlog with statuses |
| POST | `/api/tasks/{id}/done` | mark complete → write-back → re-derive → return new `/api/next` |

Each source file read fresh per request (no cache — files small, live truth matters).

### 5.4 Error handling
- Missing/unreadable source → endpoint returns HTTP 200 with `{error, detail}` body; panel renders an
  error card, never blank/crash.
- Parser can't find an expected table/key → explicit error naming the file. **Never** return a silent
  empty result (CLAUDE.md: never invent metrics — fail loud instead of guessing).

## 6. Next-Step Engine

### 6.1 Ordering rules (encoded from CLAUDE.md + memory)
1. **IAA table** (MultiIdiom hard blocker) floats to top until done.
2. Rigor exps by run-order: `02+03 (parallel) → 04 (anytime) → 06 → 07`. Exp 01 permanently skipped;
   05 already done.
3. Exp 06 gate: dry-run (`run_06_idiombert_v2.py --output_dir /tmp/_dryrun --epochs 1 --batch_size 8 --langs English`)
   emitted as its own pre-task before the real 06 run.
4. Remaining paper blockers: error-analysis table (§9), HF checkpoint IDs, Indonesian CI verify.

### 6.2 Task model
`{id, title, source_file, why, next_action (command or file ref), status, blocked_by}`.

### 6.3 Write-back safety (HARD rules — per CLAUDE.md persistence)
- Edit **only** the specific status token / checkbox line in the source md. Match the exact line;
  never regenerate the whole file.
- Backup `file.md` → `file.md.bak.<ts>` before any write.
- Read-back: re-open + re-parse the file, assert the task now reads as done. If assertion fails →
  restore backup, return error. (Never trust the write; re-read the persisted artifact.)
- `--dry-run` mode for testing that touches no real files.

## 7. Frontend

### 7.1 OS shell
- Desktop with folder icons; click opens a draggable/resizable window ("room").
- Taskbar lists open windows. Window state (position/size/open) in React state (not persisted v1).
- **Persistent Next-Step bar** pinned top of desktop, visible across all folders: task title + why +
  action button. On complete → toast "Done. Next: <title>".

### 7.2 Research folder — 6 panels
1. **Systems board** — sortable table (Joint F1, stability, cls_f1, per-system). key_numbers values,
   ⚠ on mismatch vs live JSON. G annotated "span-only, not comparable."
2. **Ablation matrix** — 15-combo heatmap (span F1 / Indo / stability).
3. **Experiment queue** — exps 01–07 kanban (skip/done/not-run) + run-order hint.
4. **Blockers** — IdiomBERT + MultiIdiom TODO list; IAA flagged hard-blocker.
5. **Activity** — git commit feed.
6. **Tasks** — ordered backlog, ✓ to complete; auto-refills + animates next-to-top on completion.

### 7.3 Stub folders
Papers / Agents / OS Core — room opens, routing works, shows planned-panel sketch + "coming soon."
No fake data.

## 8. Data flow
Frontend → `/api/*` → source module reads file fresh → parser converts md/json → typed JSON → panel.
Completion: panel → `POST /api/tasks/{id}/done` → writeback (backup→edit→read-back-verify) →
derive recomputes → response carries new `/api/next` → Next-Step bar + Tasks panel update.

## 9. Testing
- **pytest** on parsers + sources against committed md/json fixtures (parser is the fragile bit).
- **Task round-trip:** parse → mark done → write to fixture copy → re-parse → assert done; backup/restore
  path tested. Write-back tests run against fixtures ONLY, never real memory files.
- **Frontend:** smoke-render each panel against mocked api.
- Verify via preview tools (server up, panels render, no console errors) before claiming done.

## 10. Build order (for the plan)
1. Backend config + read-only sources/parsers + their tests.
2. Read-only endpoints (folders, research/*, activity).
3. Frontend shell (desktop, window, taskbar, icon grid) + Research panels 1–5 wired.
4. Next-Step engine: derive (read-only) → `/api/next` + `/api/tasks` → Tasks panel + Next-Step bar.
5. Write-back (backup + read-back verify) + round-trip tests → `POST .../done`.
6. Stub folders. README. Full preview verification.
