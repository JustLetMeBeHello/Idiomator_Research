"""
Tier 2a — WordPiece tokenization, by hand.

Stdlib only. No transformers, no torch.

WHY THIS TIER EXISTS
    run_08's locked Scenario A result says the SentencePiece QA>BIO gap is a
    tokenizer artifact (trailing-punctuation attachment). Right now that is a
    conclusion inferred from metrics. After this tier you should be able to
    PREDICT it from the algorithm, which is what makes it defensible.

WHAT TO IMPLEMENT
    Fill in every function marked TODO. Do not import transformers to check
    yourself mid-way — write it, then run test_tier2.py, then compare against
    the real tokenizer at the very end (see compare_to_hf.py).
"""

import unicodedata

UNK = "[UNK]"
MAX_INPUT_CHARS_PER_WORD = 100


def load_vocab(path):
    """
    Read a BERT vocab.txt into {token: id}. One token per line, id = line index.

    Get the file with:
        python3 -c "from transformers import AutoTokenizer as T; \
            T.from_pretrained('bert-base-multilingual-cased').save_pretrained('scratch/vocab/mbert')"
    """
    vocab = {}
    with open(path, encoding="utf-8") as fh:
        for i, line in enumerate(fh):
            vocab[line.rstrip("\n")] = i
    return vocab


# ─────────────────────────────────────────────────────────────────────────────
# Step 1: BasicTokenizer — whitespace/punctuation splitting, before subwords
# ─────────────────────────────────────────────────────────────────────────────

def is_punctuation(char):
    """
    True if char is punctuation by BERT's definition.

    NOTE: BERT treats all non-alphanumeric ASCII as punctuation, PLUS anything
    Unicode categorises as P*. This is broader than str.ispunct-style intuition
    and it is the seed of the trailing-punct behaviour you are chasing.

    TODO: implement.
    """
    raise NotImplementedError


def is_control(char):
    """True for control characters (Unicode C*), excluding \\t \\n \\r. TODO."""
    raise NotImplementedError


def is_whitespace(char):
    """True for space/tab/newline/return and Unicode Zs. TODO."""
    raise NotImplementedError


def strip_accents(text):
    """
    NFD-normalise and drop combining marks (Unicode Mn).

    mBERT is the *cased* checkpoint, so accent stripping is OFF for it. Implement
    it anyway and keep it behind a flag — then run the Spanish subset with it on
    and off and look at what happens to your span offsets. That is the lesson.

    TODO: implement.
    """
    raise NotImplementedError


def basic_tokenize(text, do_lower_case=False, do_strip_accents=False):
    """
    Split text into whitespace/punctuation-delimited tokens.

    Punctuation becomes its OWN token. Returns a list of strings.

    TODO: implement.
    """
    raise NotImplementedError


# ─────────────────────────────────────────────────────────────────────────────
# Step 2: WordpieceTokenizer — greedy longest-match-first
# ─────────────────────────────────────────────────────────────────────────────

def wordpiece_tokenize_word(word, vocab, unk=UNK,
                            max_chars=MAX_INPUT_CHARS_PER_WORD):
    """
    Greedy longest-match-first over ONE word (output of basic_tokenize).

    The algorithm:
      - if len(word) > max_chars: return [unk]
      - start at position 0; find the LONGEST substring word[start:end] that is
        in vocab (prefixed with '##' when start > 0); emit it; start = end
      - if no substring matches at some start position, the WHOLE word is [UNK]
        (not just the failing piece — this all-or-nothing rule surprises people)

    Returns a list of subword strings.

    TODO: implement.
    """
    raise NotImplementedError


def tokenize(text, vocab, do_lower_case=False, do_strip_accents=False):
    """
    Full pipeline: basic_tokenize -> wordpiece per word -> flat list of subwords.

    TODO: implement.
    """
    raise NotImplementedError


def convert_tokens_to_ids(tokens, vocab, unk=UNK):
    """Map subword strings to vocab ids, falling back to unk. TODO."""
    raise NotImplementedError
