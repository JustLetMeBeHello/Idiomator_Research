Moved here 2026-06-20 — May-29 ablation-matrix orchestrator generation, superseded by `Language_Ablations/run_*.py` + `models/results_Full_Pipeline/`.

Confirmed unreferenced by any current script before moving (`grep -rl` across `experiments/` for each filename, current pipeline only).

Do not move back without re-checking call graph — `summarize_language_ablation_matrix.py` and `merge_ablations.py` stayed in place (one sibling file, one parent dir up) because they're still called by `Language_Ablations/run_eval.py` / invoked by hand respectively. See `docs/CODEBASE_MAP.md` "Language ablation matrix" section and `memory/ablation_matrix_status.md`.
