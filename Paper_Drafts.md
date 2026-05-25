# Paper Drafts — IdiomBERT & MultiIdiom

Drop-in drafts for the remaining ARR TODOs. Copy each section into LaTeX as needed; placeholders are clearly marked `[TODO: ...]`.

---

## 1. Error analysis table scaffold (IdiomBERT, Section 9)

### Suggested error categories (use these or rename to match your taxonomy)

| Code | Category | Definition |
|------|----------|------------|
| **SC** | Sense confusion | Idiomaticity label correct; wrong sense selected within a multi-sense idiom. |
| **SB** | Span boundary | Idiom detected, but predicted span differs from gold by ≥1 token (over- or under-extension). |
| **FP-L** | Literal → idiomatic (FP) | Literal use of an expression incorrectly labeled idiomatic. |
| **FN-I** | Idiomatic → literal (FN) | Figurative use labeled literal — typically transparent or novel idioms. |
| **SM** | Span missed entirely | Idiomatic use detected (or not), but the span itself was not produced. |
| **CL** | Cross-lingual transfer | Error attributable to low-resource transfer (HI/TE/ID); system pattern matches an EN/ES analogue. |
| **AM** | Ambiguous gold | Annotation pool itself is borderline — example flagged for re-annotation rather than as a model error. |

### LaTeX table (drop-in)

```latex
\begin{table*}[t]
\centering
\small
\begin{tabular}{@{}llp{4.8cm}llcc p{3.5cm}@{}}
\toprule
\textbf{\#} & \textbf{Lang} & \textbf{Sentence (idiom in bold)} & \textbf{Sys} & \textbf{Gold} & \textbf{Pred} & \textbf{Cat} & \textbf{Notes} \\
\midrule
1  & EN & [TODO: sentence with \textbf{idiom}]               & B  & idio & lit  & FN-I & [TODO] \\
2  & ES & [TODO]                                              & D  & lit  & idio & FP-L & [TODO] \\
3  & HI & [TODO]                                              & G  & idio & idio & SB   & off by 2 tokens \\
4  & TE & [TODO]                                              & A  & idio & idio & SC   & wrong sense (#2 vs #1) \\
% [TODO: add 16–36 more rows — target 20–40 total, ~5 per category]
\bottomrule
\end{tabular}
\caption{Error analysis on a stratified sample of \textbf{[TODO: N]} misclassifications across systems and languages. Categories: SC=sense confusion, SB=span boundary, FP-L=literal misread as idiomatic, FN-I=idiomatic misread as literal, SM=span missed, CL=cross-lingual transfer, AM=ambiguous gold.}
\label{tab:error-analysis}
\end{table*}
```

### How to populate (suggested workflow)

1. From each best/worst system pair, pull ~50 misclassified examples per language from the test set.
2. Sample 4–6 per category × 5 categories ≈ 20–30 rows (skip AM if no borderline cases surface).
3. Aim for balanced language coverage (≥3 rows per language).
4. The Categories table above goes near the table; the LaTeX table itself ends Section 9.

---

## 2. Annotator demographics & proficiency (MultiIdiom, Section 5.1 or Appendix A)

> **Annotator profile.** All idiomaticity, span, and sense-level annotations were produced by native speakers of the target language. Annotators were [TODO: N total, e.g., "five (two for English, two for Hindi, two for Telugu, one for Spanish, one for Indonesian")]; each language pack was annotated by speakers who had used the language as a primary medium of communication from childhood. No formal linguistic training was required, but annotators were briefed with the guide reproduced in Appendix B and completed [TODO: ~5–10] practice items before beginning the validation pass. Annotators were [TODO: compensated at $X/hour | uncompensated co-authors | volunteer collaborators], and annotation sessions averaged [TODO: ~X seconds] per item as measured by the tool's per-example timer. No personally identifying information was collected.

---

## 3. Reproducibility statement (IdiomBERT, Limitations)

> **Reproducibility.** Source code, training scripts, and the evaluation harness are released at <https://github.com/JustLetMeBeHello/Idiomator_Research> (de-anonymized for camera-ready). All experiments use `random.seed(42)` (see [`Sampler.py:30`](Sampler.py:30)). The evaluation reported in Tables [TODO: refs] is produced by [`Evaluation/Full_evaluation.py`](Evaluation/Full_evaluation.py); raw per-system metrics are stored at [`results/pipeline_eval/pipeline_eval_results.json`](results/pipeline_eval/pipeline_eval_results.json). Train/dev/test splits are the JSONL files at [`idioms_structured/Splits/`](idioms_structured/Splits/), produced by [`Sampler.py`](Sampler.py) with class-balanced sampling at the `idiom_id` level (80/10/10, idiom-disjoint splits). Model checkpoints are released on HuggingFace: [TODO: HF model IDs — one per system A/D/E/F/G + 4-shot variants for B/C]. Indonesian is held out for zero-shot evaluation and was not used during training of any system. The annotation tool used to validate the test sets is at [`Annotation_tool/`](Annotation_tool/) and is deployable as a single FastAPI service; the validated annotation pools are at [`Annotation_tool/data/{lang}/annotation_pool.jsonl`](Annotation_tool/data/), regenerable from train/dev/test via [`Annotation_tool/build_annotation_pool.py`](Annotation_tool/build_annotation_pool.py) (same seed=42).

---

## 4. Appendix B — Annotator instructions (MultiIdiom)

> **Appendix B: Annotator instructions.** The following guide was shown to all human validators in the IdiomBank Validator tool. Annotators could re-open it at any time during a session.
>
> ### Task overview
> Each item presents a sentence containing a candidate idiomatic expression. The expression is pre-highlighted in the sentence, and a pre-filled idiomaticity label, span, and (where applicable) sense definition are shown. The annotator's job is to **confirm or correct** each of three labels.
>
> ### ① Idiomaticity label
> Decide whether the highlighted expression is used **figuratively** (idiomatic) or **literally** (literal) in this sentence.
>
> - *Idiomatic.* "She **kicked the bucket** last night." — figurative use; means "she died."
> - *Literal.* "He **kicked the bucket** across the yard." — physical action.
>
> If the pre-filled label is wrong, click the opposing button; the override is recorded.
>
> ### ② Span annotation
> The highlighted tokens should cover **exactly** the idiomatic expression — no more, no fewer.
>
> - *Correct.* "She [**kicked the bucket**] last night." — span covers the full idiom.
> - *Wrong.* "She [**kicked the**] bucket last night." — span is too short.
>
> When marking *Wrong*, the annotator types the exact phrase that should have been highlighted, copied verbatim from the sentence.
>
> ### ③ Sense context
> This question appears only when a sense definition is shown (multi-sense idioms). The annotator checks whether the sentence uses the **specific sense** described.
>
> - *Correct.* Sentence clearly uses the idiom in the stated sense.
> - *Uncertain.* Sense is ambiguous in context; annotator may add a free-text note.
> - *Wrong.* Sentence uses the idiom in a different sense than described; annotator notes the intended sense.
>
> ### General guidance
> - Annotators are instructed to go with first instinct; if uncertain after ~30 seconds, mark *Uncertain* and add a note.
> - Annotators may navigate backward (Prev) to revise any previous answer in a session.
> - Progress is auto-saved per annotator; sessions can be resumed by re-entering the same name.
> - A free-text Notes field is available on every item for cases that do not fit the structured options.
> - Validated annotations are exported as JSONL via an Export button on the completion screen.

---

## 5. Section 5.1 IAA stub (MultiIdiom)

> **Inter-annotator agreement.** To assess the reliability of the validated test sets, we computed pairwise inter-annotator agreement on a [TODO: n_overlap]-example overlap subset for three languages spanning three typological families: English (Germanic), Hindi (Indo-Aryan), and Telugu (Dravidian). Agreement was measured along two dimensions: (a) the binary idiomaticity label (idiomatic vs. literal), and (b) span boundary correctness (treated as a binary correct/incorrect judgment with respect to the pipeline's predicted span).
>
> Table [TODO: ref] reports Cohen's $\kappa$ and raw percent agreement for each dimension and language.
>
> | Language | n (overlap) | Idiomaticity $\kappa$ | Idiomaticity % agr. | Span $\kappa$ | Span % agr. |
> |---|---|---|---|---|---|
> | English | [TODO] | [TODO] | [TODO] | [TODO] | [TODO] |
> | Hindi   | [TODO] | [TODO] | [TODO] | [TODO] | [TODO] |
> | Telugu  | [TODO] | [TODO] | [TODO] | [TODO] | [TODO] |
>
> [TODO: 2–3 sentence interpretation. Standard framing if all $\kappa \geq 0.6$: "Agreement falls in the *substantial* to *near-perfect* range (Landis & Koch, 1977) across all three languages on both dimensions, supporting the reliability of the validated test sets." If any language is below 0.6, discuss likely cause — e.g., higher polysemy in Telugu idioms, span boundary ambiguity around postpositions in Hindi.] Spanish and Indonesian test sets remain single-annotator (silver-standard) and we report pipeline-quality estimates rather than $\kappa$ for those languages.

### Compute IAA script reminder

Run after both annotators have exported their JSONL from the Validator's completion screen (or `/export/{annotator}` endpoint):

```bash
# All languages, JSON results + LaTeX table for the paper:
python Annotation_tool/compute_iaa.py \
  annotator_A.jsonl annotator_B.jsonl \
  --latex --out iaa_results.json

# Or one language at a time:
python Annotation_tool/compute_iaa.py \
  annotator_A.jsonl annotator_B.jsonl \
  --lang Telugu --latex
```

The script (Cohen's κ + % agreement on **idiomaticity** and **span boundary** separately, per language) handles all the math. Note: the auto-generated LaTeX caption hard-codes a Telugu/English scope from a previous version — edit the caption in [`compute_iaa.py:204`](Annotation_tool/compute_iaa.py:204) to reflect HI/TE/EN, or replace the caption by hand in LaTeX.
