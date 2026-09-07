# scratch/ — fundamentals curriculum

Learning scaffold, not part of the paper pipeline. Nothing here is imported by
`training/`, `Evaluation/`, or `experiments/`.

**Method:** always work against an oracle. Write each function blind, then assert
your output matches the known-correct one (HF's tokenizer, HF's model logits,
`Evaluation/Full_evaluation.py`). Don't read the reference implementation first.

## Setup

```bash
cd scratch
python3 -m venv .venv && source .venv/bin/activate
pip install torch transformers
python3 -c "from transformers import AutoTokenizer, AutoModel; \
  AutoTokenizer.from_pretrained('bert-base-multilingual-cased').save_pretrained('vocab/mbert'); \
  AutoModel.from_pretrained('bert-base-multilingual-cased').save_pretrained('vocab/mbert')"
python3 common/data.py     # builds the 200-example subset
```

## Tiers, in recommended order

| Tier | What | Needs | Oracle |
|------|------|-------|--------|
| 2 `tier2_tokenizer/` | WordPiece + char/subword offset alignment + span decode | stdlib | HF tokenizer output |
| 4 `tier4_stats/` | paired bootstrap, Holm, BH, TOST | stdlib | `experiments/rigor/run_14*` |
| 1 `tier1_encoder/` | mBERT forward pass from scratch | torch (CPU ok) | `load_state_dict(strict=True)` + `allclose(atol=1e-4)` |
| 3 `tier3_heads/` | cls / QA-span / BIO heads + joint loss | torch (GPU) | Systems A/E/G numbers within seed noise |

Tier 2 underwrites the run_08 tokenizer-artifact claim; Tier 4 underwrites the
run_13 TOST / run_14 correction results. Those two are the ones that touch live
paper claims — do them first.

## Status

- `common/data.py` — DONE. Builds balanced 200-example subsets (25 per
  language x class) from `data/idioms_structured/Splits/`, seed 42. Verified
  `sentence[span_start:span_end] == matched_span` on all 600 rows, all 4 languages.
- `tier2_tokenizer/wordpiece.py`, `tier2_tokenizer/alignment.py` — stubs, all
  `NotImplementedError`. Start here.
- Tiers 1, 3, 4 — not scaffolded yet.
