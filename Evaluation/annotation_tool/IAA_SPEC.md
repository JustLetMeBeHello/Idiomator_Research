# IAA Spec v2 — corrected inter-annotator agreement

Run with `compute_iaa_v2.py`. This supersedes the kappa-only output of
`compute_iaa.py` for anything that goes in MultiIdiom §5.1 or IdiomBERT §5.

## Why the old IAA is not enough

`compute_iaa.py` reports Cohen's κ on two dimensions:
1. `idiomaticity_verdict` (idiomatic / literal) — **keep, it's fine.**
2. `span_correct` (True/False, "is the pipeline span correct?") — **insufficient.**

`span_correct` κ does **not** measure whether two annotators agree on the span
*boundary*. It is a verification judgment against the pipeline's own span, so it:
- is blind to the ±1-token trailing-punctuation differences IdiomBERT's C1
  ("annotation artifact") claim is entirely about; and
- is **anchored**: when both annotators accept the pipeline span, their effective
  gold spans are identical *by construction* → agreement is inflated.

Net: IdiomBERT's exact-match span metric (where the headline +0.13→+0.03 collapse
lives) is currently **unvalidated**. A high `span_correct` κ would not fix this.

## What v2 measures

Reconstructs each annotator's **effective gold span** (char offsets):
- `span_correct == True`  → they accepted the pipeline span `(span_start, span_end)`.
- `span_correct == False` → locate their `span_correction` text in the sentence
  (sentence pulled from `data/<Lang>/annotation_pool.jsonl` by `meaning_id`).

Then reports, each with **95% bootstrap CIs** (10k resamples, seed 42):

| Metric | Meaning | Use |
|---|---|---|
| **A. Idiomaticity κ** | label agreement | MultiIdiom headline reliability |
| **B. Span-acceptance κ** | old `span_correct`, *relabeled honestly* | report as-is, do not call it boundary agreement |
| **C-all. mean IoU + exact-boundary rate** | boundary agreement over all items | **anchored upper bound** — do not headline |
| **C-unanchored. mean IoU + exact-boundary rate** | boundary agreement on items where ≥1 annotator corrected | **the real signal** — headline this |

## Decision rule (what the number means for the papers)

Let `d = 1 − exact_boundary_rate(unanchored)` (boundary disagreement).

- `d ≤ 0.03` → boundary agreement tighter than the artifact deltas (0.01–0.03).
  Exact-match span eval is defensible; **C1 can be headlined on human gold (EN+TE).**
- `d > 0.03` → human annotators disagree on boundaries by *more* than the effect
  IdiomBERT reports. Exact-match is measuring annotation noise. **Switch span
  reporting to IoU-thresholded (overlap), state exact-match is unreliable at this
  IAA, and reframe C1 to "the gap collapses into the noise floor"** (the salvage
  framing). This is the most likely outcome — plan for it.

The script prints this verdict inline per language.

## English caveat (blocker)

Cohen's κ needs **two** annotators. EN currently has one → **no EN κ is
computable.** Options:
- recruit a 2nd EN annotator on a ~50–100 example overlap subset (preferred), or
- drop the EN κ cell and report TE only (state EN was single-validated).

Do **not** leave a `[TODO]` in MultiIdiom Table 4 EN row — it's structurally
impossible to fill as designed.

## How to run

```bash
cd Research_And_Training/Annotation_tool
# collect two annotator exports first (browser "Export results.jsonl" or
# GET /export/{annotator}); name them e.g. te_ann1.jsonl te_ann2.jsonl
python compute_iaa_v2.py te_ann1.jsonl te_ann2.jsonl --lang Telugu --latex --out iaa_te.json
```

Required export fields (already in the validator schema): `meaning_id`,
`language`, `idiomaticity_verdict`, `span_start`, `span_end`, `matched_span`,
`span_correct`, `span_correction`. Sentences are joined from `--pool-dir`
(default `./data`).

## What to put in the papers

- **MultiIdiom §5.1 / IdiomBERT §5:** one table — Idiomaticity κ (with CI), and
  boundary agreement on the **unanchored** subset (IoU + exact, with CIs). State
  n and the anchoring-correction explicitly.
- Report **every κ with its bootstrap CI.** At n=62 a bare point κ is uninformative.
- If `d > 0.03`, add one sentence: "exact-match span agreement (d=…) is comparable
  to the architecture deltas we study, so we report overlap-based span metrics and
  treat exact-match as a lower bound."
