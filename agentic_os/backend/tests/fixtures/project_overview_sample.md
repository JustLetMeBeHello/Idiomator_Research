## Rigor Experiments (`Additional_Rigor_Experiments/`)

| # | Script | What | Status | Notes |
|---|--------|------|--------|-------|
| 01 | `run_01_bio_muril_recovery.sh` | BIO + MuRIL on Telugu | **SKIP** | Decoder fix. |
| 02 | `run_02_xlmr_qa_vs_bio.sh` | XLM-R Joint + BIO vs mBERT | ❌ Not run | Single biggest ceiling-mover. |
| 03 | `run_03_muril_joint.sh` | MuRIL Joint (System E) full training | ❌ Not run | After 02. |
| 04 | `run_04_llama3_baseline.py` | Llama-3.3-70B baseline | ❌ Not run | API only. |
| 05 | `run_05_decoder_rerun.py` | Decoder bug comparison table | ✅ Done | CPU only. |
| 06 | `run_06_idiombert_v2.py` | IdiomBERT-v2 8-cell ablation | ❌ Not run | Last. |
| 07 | TBD `run_07_bio_joint.py` | BIO Joint | ❌ Not planned yet | After 06. |

**Blocking TODOs:**
1. **IAA table (§5.1) — HARD BLOCKER.** Needs 2 annotators per language.
2. Error analysis table (§9) — user doing manual analysis.
3. Indonesian CI bug — Systems A–F return wrong key.
