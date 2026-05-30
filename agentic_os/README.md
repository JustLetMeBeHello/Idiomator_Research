# Personal Agentic OS

Local desktop-metaphor dashboard over the IdiomBERT / MultiIdiom research. A FastAPI
backend reads the repo + memory files live; a React/vite frontend renders a desktop
with draggable windows ("rooms"). The **Research** room is fully wired; Papers / Agents
/ OS Core are stubs.

## Run

Backend (from repo root):

    /Users/shishirmaddineni/Desktop/Idiomator_Research/Research_And_Training/.venv/bin/python \
      -m uvicorn agentic_os.backend.main:app --reload --port 8011

Frontend:

    cd agentic_os/frontend
    npm install
    npm run dev   # http://localhost:5173

The backend reads every source file fresh per request (no cache). It fails loud if a
required source path is missing — it never invents a metric.

## Tests

    # backend (run from repo root)
    PYTHONPATH=. .venv/bin/python -m pytest agentic_os/backend/tests -v

    # frontend
    cd agentic_os/frontend && npm run test

## Folders

- **Research** — live. Six panels:
  1. Systems board — 9 systems × Joint F1 / stability / cls F1. Curated values from
     `key_numbers.md` win; live JSON merged for drill-down; a ⚠ marks any mismatch.
     System G is annotated "span-only, not comparable".
  2. Ablation matrix — 15 language-combo heatmap (span F1 / Indonesian held-out / stability).
  3. Experiment queue — rigor experiments 01–07 with status.
  4. Blockers — IdiomBERT + MultiIdiom blocking TODOs; the IAA table is flagged HARD.
  5. Tasks — the ordered Next-Step backlog (see below); ✓ marks an experiment done.
  6. Activity — recent git commits.
- **Papers / Agents / OS Core** — stubs (room opens, shows planned panels + "coming soon").

## Next-Step engine

Derives an ordered backlog from `project_overview.md`:

1. **IAA table** (MultiIdiom hard blocker) floats to the top until done.
2. Rigor experiments in run order: `02 / 03 → 04 → 06-dryrun → 06 → 07`
   (01 permanently skipped; 05 already done; the 06 dry-run is emitted as its own pre-task).
3. Remaining paper blockers.

The persistent **Next-Step bar** (pinned top of the desktop) always shows the single
current next action. Marking an experiment done performs a **backed-up, read-back-verified**
edit of `project_overview.md`: the file is copied to `*.bak.<ts>` first, the one status
cell is surgically flipped to `✅ Done`, then the file is re-read and re-parsed to confirm
the change persisted — if verification fails, the backup is restored and an error is raised.
Only experiment tasks are writable in v1; blockers are read-only prose.

## Data sources

- `results/pipeline_eval/pipeline_eval_results.json` — live per-system metrics (deeply nested).
- `key_numbers.md` (memory) — curated metrics + 15-combo ablation table (source of truth).
- `project_overview.md` (memory) — experiment queue + blocker lists.

Paths are configured in `agentic_os/backend/config.py` (override via `AOS_REPO_ROOT` /
`AOS_MEMORY_DIR` env vars).
