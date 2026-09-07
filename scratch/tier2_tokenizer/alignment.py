"""
Tier 2b — character offsets <-> subword indices, and span decoding.

Stdlib only.

WHY THIS TIER EXISTS
    Your data stores spans as CHARACTER offsets (span_start/span_end into
    `sentence`). Your QA head predicts SUBWORD indices. Every span number in the
    paper passes through the two conversions below. run_16's word-level tagger
    mean-pools subwords back to words, which is a third conversion on top.

    A bug in any of these looks exactly like a modelling result. That is how the
    trailing-punct artifact got mistaken for a real QA>BIO advantage until run_08
    pinned it down.
"""


def tokenize_with_offsets(text, vocab, tokenize_fn):
    """
    Tokenize AND return, for each subword, the (char_start, char_end) it covers
    in the ORIGINAL text.

    Returns: (tokens, offsets) where offsets[i] == (start, end) and
             text[start:end] is the surface form of tokens[i].

    This is the hard part of Tier 2. Things that will bite you:
      - '##' prefixes are not in the original text, so you cannot use len(token)
      - accent stripping / lowercasing change string length in general
      - whitespace between tokens is not covered by any offset
      - the same surface substring can appear many times; you must track a
        cursor, not use str.find from 0

    Verify with: text[offsets[i][0]:offsets[i][1]] reconstructs the token
    (modulo '##' and any normalisation).

    TODO: implement.
    """
    raise NotImplementedError


def char_span_to_subword_span(char_start, char_end, offsets):
    """
    Map a gold character span onto inclusive subword indices (tok_start, tok_end).

    Definition to use: the smallest subword range whose combined char coverage
    CONTAINS [char_start, char_end).

    Edge cases you must decide on explicitly (write your choice in a comment,
    because it changes your numbers):
      - gold span starts mid-subword (common in Telugu/Hindi agglutination)
      - gold span ends just before a punctuation mark that got glued to the
        final subword  <-- THIS IS THE TRAILING-PUNCT CASE
      - gold span covers whitespace at either edge

    Return None if the span cannot be mapped at all.

    TODO: implement.
    """
    raise NotImplementedError


def subword_span_to_char_span(tok_start, tok_end, offsets):
    """
    Inverse: inclusive subword indices -> (char_start, char_end) in the original
    text. This is what turns QA head logits back into a predicted span string.

    TODO: implement.
    """
    raise NotImplementedError


def decode_qa_logits(start_logits, end_logits, offsets, max_answer_len=20,
                     n_best=20):
    """
    Turn start/end logits into a character span.

    Standard SQuAD-style decode:
      - take the n_best highest start indices and n_best highest end indices
      - consider all (s, e) pairs with s <= e and (e - s + 1) <= max_answer_len
      - score = start_logits[s] + end_logits[e]; take the argmax
      - map the winning subword pair back through subword_span_to_char_span

    Pure Python — start_logits/end_logits are plain lists of floats, so this
    stays testable without torch.

    TODO: implement.
    """
    raise NotImplementedError


def decode_bio_tags(tags, offsets):
    """
    Turn a BIO tag sequence (list of 'B'/'I'/'O', one per subword) into a
    character span.

    Decide and document: what do you do with 'I' without a preceding 'B'?
    What about multiple disjoint B...I runs? System G's decoder made choices
    here, and run_05 was a rerun to fix a UTF-8 char-awareness bug in exactly
    this function's real counterpart.

    Return (char_start, char_end) for the selected run, or None if all 'O'.

    TODO: implement.
    """
    raise NotImplementedError


def overlap_f1(pred_start, pred_end, gold_start, gold_end):
    """
    Character-level overlap F1 between two spans — mirrors compute_overlap_f1 in
    Evaluation/Full_evaluation.py. Implement it here from the definition, then
    diff against the real one.

    precision = |overlap| / |pred|,  recall = |overlap| / |gold|

    TODO: implement.
    """
    raise NotImplementedError
