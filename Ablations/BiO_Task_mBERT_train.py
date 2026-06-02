"""
train_bio_tagger.py

Token-level BIO tagger for MWE span extraction.
Each token is labelled: B-IDIOM (span start), I-IDIOM (span continuation), O (outside).

Architecture : mBERT encoder → linear token classification head
Loss         : CrossEntropyLoss with inverse-frequency class weights (O dominates)
Input        : raw sentence only — no idiom hint
Output       : per-token BIO label → decoded to character offsets for evaluation

Evaluation (matching Stage 2 / Joint format):
  - Exact span match
  - Overlap F1 (character-level)
  - Per-language breakdown

Usage:
    python train_bio_tagger.py \\
        --output_dir models/bio_tagger_en_hi_te \\
        --langs English Hindi Telugu

    # English only
    python train_bio_tagger.py \\
        --output_dir models/bio_tagger_en \\
        --langs English

    # Tune class weights
    python train_bio_tagger.py \\
        --output_dir models/bio_tagger_en_hi_te \\
        --o_weight 0.1
"""

import json
import argparse
import numpy as np
from pathlib import Path
from collections import defaultdict, Counter

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    AutoModel,
    get_linear_schedule_with_warmup,
)
from torch.optim import AdamW
from sklearn.metrics import classification_report
from tqdm import tqdm

try:
    import wandb
    WANDB = True
except ImportError:
    WANDB = False
    print("wandb not installed — skipping. pip install wandb to enable.")


# ── BIO label scheme ──────────────────────────────────────────────────────────

LABEL2ID = {'O': 0, 'B-IDIOM': 1, 'I-IDIOM': 2}
ID2LABEL  = {0: 'O', 1: 'B-IDIOM', 2: 'I-IDIOM'}
NUM_LABELS = 3
IGNORE_IDX = -100  # ignored by CrossEntropyLoss (used for special tokens + padding)


# ── Args ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--model_name',    default='bert-base-multilingual-cased')
    p.add_argument('--data_dir',      default='idioms_structured/Splits')
    p.add_argument('--dropout', type=float, default=.239431179668018881)
    p.add_argument('--output_dir',    default='models/bio_tagger_en_hi_te')
    p.add_argument('--langs',         nargs='+', default=['English', 'Hindi', 'Telugu','Spanish'])
    p.add_argument('--test_langs',    nargs='+', default=None,
                   help='Languages to evaluate on. Defaults to --langs. Use all target languages for cross-lingual ablations.')
    p.add_argument('--epochs',        type=int,   default=6)
    p.add_argument('--batch_size',    type=int,   default=32)
    p.add_argument('--lr',            type=float, default=3.27e-05)
    p.add_argument('--max_len',       type=int,   default=128)
    p.add_argument('--warmup_ratio',  type=float, default=0.096)
    p.add_argument('--seed',          type=int,   default=42)
    p.add_argument('--o_weight',      type=float, default=0.104,
                   help='Loss weight for O class (B/I are weighted 1.0). '
                        'Lower values focus training on span tokens.')
    p.add_argument('--use_wandb',     action='store_true')
    p.add_argument('--device',        default=None)
    return p.parse_args()


# ── Device ────────────────────────────────────────────────────────────────────

def get_device(forced=None):
    if forced:
        return torch.device(forced)
    if torch.cuda.is_available():
        return torch.device('cuda')
    if torch.backends.mps.is_available():
        print("MPS detected — precompiling shaders...")
        device = torch.device('mps')
        _ = torch.zeros(1, device=device) + 1
        print("MPS shaders ready.")
        return device
    return torch.device('cpu')


# ── Data loading ──────────────────────────────────────────────────────────────

def load_split(data_dir, split_name, langs):
    path = Path(data_dir) / f'{split_name}.jsonl'
    if not path.exists():
        raise FileNotFoundError(f"Split file not found: {path}")
    langs_set = set(langs)
    examples = []
    for line in open(path, encoding='utf-8'):
        r = json.loads(line)
        if r['language'] in langs_set:
            examples.append(r)
    print(f"  {split_name}: {len(examples)} examples")
    return examples


# ── Language loss weights ─────────────────────────────────────────────────────

def compute_lang_loss_weights(train_examples):
    """
    Compute per-example loss weights using inverse-frequency over
    (language, idiomaticity) cells — identical logic to Stage 1.

    Returns a dict mapping (language, idiomaticity) → scalar weight,
    normalised so the minimum cell weight is 1.0.
    """
    cell_counts = Counter(
        (ex['language'], ex['idiomaticity']) for ex in train_examples
    )
    total   = sum(cell_counts.values())
    n_cells = len(cell_counts)

    cell_weights = {
        cell: total / (n_cells * count)
        for cell, count in cell_counts.items()
    }

    # Normalise: min weight → 1.0
    min_w = min(cell_weights.values())
    cell_weights = {cell: w / min_w for cell, w in cell_weights.items()}

    print("\nPer (language, idiomaticity) cell weights:")
    for (lang, idio), w in sorted(cell_weights.items()):
        count = cell_counts[(lang, idio)]
        print(f"  {lang:<10} {idio:<10}: n={count:5d}  weight={w:.4f}")

    return cell_weights


# ── BIO label alignment ───────────────────────────────────────────────────────

def align_bio_labels(offsets, word_ids, char_start, char_end, max_len):
    """
    Convert character-level span offsets to per-token BIO labels.

    Special tokens ([CLS], [SEP]) and padding get IGNORE_IDX so they
    don't contribute to the loss. Wordpiece subtokens that aren't the
    first subtoken of a word also get IGNORE_IDX — we only supervise
    on the first subtoken of each word (standard NER practice).

    Args:
        offsets:    list of (char_s, char_e) tuples from offset_mapping
        word_ids:   list of word indices (None for special/padding tokens)
        char_start: gold span start (character offset)
        char_end:   gold span end   (character offset)
        max_len:    sequence length to pad/truncate to

    Returns a list of label IDs of length max_len.
    """

    if char_start is None or char_end is None:
        return None

    labels  = []
    prev_word_id = None

    for token_idx, (offset, word_id) in enumerate(zip(offsets, word_ids)):
        # Special token ([CLS], [SEP]) or padding
        if word_id is None:
            labels.append(IGNORE_IDX)
            prev_word_id = word_id
            continue

        # Non-first subtoken of a word → ignore
        if word_id == prev_word_id:
            labels.append(IGNORE_IDX)
            prev_word_id = word_id
            continue

        # First subtoken of a word — overlap check (handles SP leading-space offset)
        tok_char_s, tok_char_e = offset
        in_span = min(tok_char_e, char_end) > max(tok_char_s, char_start)

        if in_span:
            # B-IDIOM for the very first span token, I-IDIOM for the rest
            if len([l for l in labels if l == LABEL2ID['B-IDIOM']]) == 0:
                labels.append(LABEL2ID['B-IDIOM'])
            else:
                labels.append(LABEL2ID['I-IDIOM'])
        else:
            labels.append(LABEL2ID['O'])

        prev_word_id = word_id

    # Pad to max_len
    labels += [IGNORE_IDX] * (max_len - len(labels))
    return labels[:max_len]


# ── Dataset ───────────────────────────────────────────────────────────────────

class BIODataset(Dataset):
    def __init__(self, examples, tokenizer, max_len, cell_weights=None):
        self.valid_examples  = []
        self.input_ids       = []
        self.attention_masks = []
        self.token_type_ids  = []
        self.bio_labels      = []
        self.example_weights = []   # per-example loss scaling

        skipped = 0
        for ex in examples:
            sentence   = ex['sentence']
            char_start = ex['span_start']
            char_end   = ex['span_end']

            encoding = tokenizer(
                sentence,
                max_length=max_len,
                padding='max_length',
                truncation=True,
                return_offsets_mapping=True,
                return_tensors='pt',
            )

            labels = align_bio_labels(
                encoding['offset_mapping'][0].tolist(),
                encoding.word_ids(batch_index=0),
                char_start, char_end, max_len
            )

            # Skip if span was None or no B-IDIOM token survived (span fully outside max_len)
            if labels is None or LABEL2ID['B-IDIOM'] not in labels:
                skipped += 1
                continue

            seq_len = encoding['input_ids'].shape[1]
            tid = encoding.get('token_type_ids')

            self.valid_examples.append(ex)
            self.input_ids.append(encoding['input_ids'].squeeze(0))
            self.attention_masks.append(encoding['attention_mask'].squeeze(0))
            self.token_type_ids.append(
                tid.squeeze(0) if tid is not None
                else torch.zeros(seq_len, dtype=torch.long)
            )
            self.bio_labels.append(torch.tensor(labels, dtype=torch.long))

            # Per-example weight from (language, idiomaticity) cell
            w = 1.0
            if cell_weights is not None:
                w = cell_weights.get((ex['language'], ex['idiomaticity']), 1.0)
            self.example_weights.append(w)

        if skipped:
            print(f"  Skipped {skipped} examples (span outside max_len or alignment failed)")

    def __len__(self):
        return len(self.valid_examples)

    def __getitem__(self, idx):
        return {
            'input_ids':       self.input_ids[idx],
            'attention_mask':  self.attention_masks[idx],
            'token_type_ids':  self.token_type_ids[idx],
            'bio_labels':      self.bio_labels[idx],
            'example_weight':  torch.tensor(self.example_weights[idx], dtype=torch.float),
        }


# ── Model ─────────────────────────────────────────────────────────────────────

class BIOTagger(torch.nn.Module):
    """mBERT + single linear head projecting each token to 3 BIO classes."""

    def __init__(self, model_name, num_labels=NUM_LABELS, dropout=0.1):
        super().__init__()
        self.bert    = AutoModel.from_pretrained(model_name)
        hidden_size  = self.bert.config.hidden_size
        self.dropout = torch.nn.Dropout(dropout)
        self.head    = torch.nn.Linear(hidden_size, num_labels)

    def forward(self, input_ids, attention_mask, token_type_ids):
        outputs  = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        seq_out  = self.dropout(outputs.last_hidden_state.float())  # cast: mDeBERTa-v3 can emit fp16 on GPU
        logits   = self.head(seq_out)                        # [B, T, 3]
        return logits


# ── Span decoding ─────────────────────────────────────────────────────────────

def decode_bio_to_char_span(bio_preds, encoding, sentence, max_len):
    """
    Convert a sequence of predicted BIO label IDs back to character offsets.

    Strategy:
      1. Find the first B-IDIOM token.
      2. Extend through consecutive I-IDIOM tokens.
      3. Map token offsets → character offsets.
      4. If no B-IDIOM found, return (None, None).
    """
    offsets  = encoding['offset_mapping']   # list of (char_s, char_e)
    word_ids = encoding.word_ids()          # works on BatchEncoding (not plain dict)

    # Collect (token_idx, label) for first-subtoken positions only
    first_subtokens = []
    seen_words = set()
    for i, (label, wid) in enumerate(zip(bio_preds, word_ids)):
        if wid is None:
            continue
        if wid in seen_words:
            continue
        seen_words.add(wid)
        first_subtokens.append((i, label))

    # Find span: B followed by I*
    span_tokens = []
    in_span = False
    for tok_idx, label in first_subtokens:
        if label == LABEL2ID['B-IDIOM']:
            span_tokens = [tok_idx]
            in_span = True
        elif label == LABEL2ID['I-IDIOM'] and in_span:
            span_tokens.append(tok_idx)
        else:
            if in_span:
                break   # span ended

    if not span_tokens:
        return None, None

    first_tok = span_tokens[0]
    last_tok  = span_tokens[-1]

    # last_tok is the FIRST subtoken of the last span word. Walk forward
    # through any continuation subtokens of that same word so char_end
    # lands at the end of the full word, not at the end of its first
    # subtoken. Without this, multi-subtoken endwords (very common for
    # non-Latin scripts) get truncated mid-word.
    last_word_id = word_ids[last_tok]
    j = last_tok
    while j + 1 < len(word_ids) and word_ids[j + 1] == last_word_id:
        j += 1
    last_tok = j

    if first_tok >= len(offsets) or last_tok >= len(offsets):
        return None, None

    char_start = offsets[first_tok][0]
    char_end   = offsets[last_tok][1]

    # Sanity check: char_end must be within sentence bounds
    if char_start is None or char_end is None:
        return None, None
    char_end = min(char_end, len(sentence))

    # SentencePiece tokenizers (XLM-R, RemBERT, mDeBERTa) prepend ▁ to
    # word-initial tokens, shifting the token's char offset left by one
    # into the preceding space. Without this strip, every word-initial
    # span boundary decodes one char early — exact_match collapses while
    # overlap_f1 stays high (off-by-1 pattern). Skip leading whitespace.
    while char_start < char_end and sentence[char_start].isspace():
        char_start += 1
    return int(char_start), int(char_end)


# ── Metrics ───────────────────────────────────────────────────────────────────

def compute_overlap_f1(pred_s, pred_e, gold_s, gold_e):
    if pred_s is None or pred_e is None:
        return 0.0
    pred_set = set(range(pred_s, pred_e))
    gold_set = set(range(gold_s, gold_e))
    if not pred_set or not gold_set:
        return 0.0
    overlap = len(pred_set & gold_set)
    if overlap == 0:
        return 0.0
    p = overlap / len(pred_set)
    r = overlap / len(gold_set)
    return 2 * p * r / (p + r)


# ── Evaluation ────────────────────────────────────────────────────────────────

def evaluate(model, loader, tokenizer, examples, device, split_name, max_len):
    model.eval()

    lang_exact = defaultdict(list)
    lang_f1    = defaultdict(list)

    # Also track token-level BIO accuracy for debugging
    all_true_labels = []
    all_pred_labels = []

    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(loader, desc=f'Eval {split_name}', leave=False)):
            input_ids      = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            token_type_ids = batch['token_type_ids'].to(device)
            bio_labels     = batch['bio_labels']          # [B, T], CPU

            logits = model(input_ids, attention_mask, token_type_ids)  # [B, T, 3]
            preds  = torch.argmax(logits, dim=-1).cpu()                # [B, T]

            batch_start = batch_idx * loader.batch_size
            for i in range(len(preds)):
                ex_idx = batch_start + i
                if ex_idx >= len(examples):
                    break

                ex       = examples[ex_idx]
                sentence = ex['sentence']
                lang     = ex['language']
                gold_s   = ex['span_start']
                gold_e   = ex['span_end']

                pred_bio = preds[i].tolist()   # list of label IDs, length max_len

                # Re-encode for offset_mapping (needed for decode)
                enc = tokenizer(
                    sentence,
                    max_length=max_len,
                    truncation=True,
                    return_offsets_mapping=True,
                )

                pred_char_s, pred_char_e = decode_bio_to_char_span(
                    pred_bio, enc, sentence, max_len
                )

                exact   = int(pred_char_s == gold_s and pred_char_e == gold_e) \
                          if pred_char_s is not None else 0
                overlap = compute_overlap_f1(pred_char_s, pred_char_e, gold_s, gold_e)

                lang_exact[lang].append(exact)
                lang_f1[lang].append(overlap)

                # Token-level labels (ignoring IGNORE_IDX positions)
                true_bio = bio_labels[i].tolist()
                for t, p in zip(true_bio, pred_bio):
                    if t != IGNORE_IDX:
                        all_true_labels.append(t)
                        all_pred_labels.append(p)

    # Span-level results
    all_exact   = [v for vals in lang_exact.values() for v in vals]
    all_overlap = [v for vals in lang_f1.values()    for v in vals]

    print(f"\n── {split_name} Span Results ──")
    print(f"  {'Language':<12} {'Exact':<10} {'Overlap F1':<12} {'N':<6}")
    print(f"  {'-'*42}")
    for lang in sorted(lang_exact.keys()):
        em = np.mean(lang_exact[lang])
        f1 = np.mean(lang_f1[lang])
        print(f"  {lang:<12} {em:<10.4f} {f1:<12.4f} {len(lang_exact[lang]):<6}")
    print(f"  {'Overall':<12} {np.mean(all_exact):<10.4f} {np.mean(all_overlap):<12.4f} {len(all_exact):<6}")

    # Token-level BIO report
    print(f"\n── {split_name} Token BIO Report ──")
    print(classification_report(
        all_true_labels, all_pred_labels,
        target_names=[ID2LABEL[i] for i in range(NUM_LABELS)],
        digits=4, zero_division=0,
    ))

    return np.mean(all_exact), np.mean(all_overlap)


# ── Save / load ───────────────────────────────────────────────────────────────

def save_model(model, tokenizer, output_dir):
    best = Path(output_dir) / 'best_model'
    best.mkdir(parents=True, exist_ok=True)
    model.bert.save_pretrained(best, safe_serialization=True)
    weight_files = list(best.glob("*.safetensors")) + list(best.glob("pytorch_model.bin"))
    if not weight_files:
        raise RuntimeError(
            f"Encoder save_pretrained() wrote no weights to {best}. "
            f"Dir: {[p.name for p in best.iterdir()]}"
        )
    print(f"  Saved encoder ({sum(p.stat().st_size for p in weight_files)/1e6:.1f} MB) + head + tokenizer")
    tokenizer.save_pretrained(best)
    torch.save(model.head.state_dict(), best / 'bio_head.pt')


def load_best_model(model_name, output_dir, device):
    best = Path(output_dir) / 'best_model'
    if not best.exists():
        # No checkpoint was saved — dev overlap never improved above 0.
        # Happens in short dry-runs (1 epoch). Fall back to the base HF model
        # so test-eval can run and confirm code compatibility.
        print(f"  ⚠ best_model/ not found (dev never improved). "
              f"Loading base {model_name} for compatibility test eval.")
        model = BIOTagger(model_name)
    else:
        model = BIOTagger(str(best))
        model.head.load_state_dict(torch.load(best / 'bio_head.pt', map_location=device, weights_only=True))
    return model.to(device)


# ── Train ─────────────────────────────────────────────────────────────────────

def train(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device     = get_device(args.device)
    print(f"Device: {device}")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    json.dump(vars(args), open(output_dir / 'config.json', 'w'), indent=2)
    print(f"Languages : {args.langs}")
    print(f"O weight  : {args.o_weight}  (B/I weight: 1.0)")

    # Load splits
    print("\nLoading splits...")
    train_examples = load_split(args.data_dir, 'train', args.langs)
    dev_examples   = load_split(args.data_dir, 'dev',   args.langs)
    test_langs = args.test_langs or args.langs
    test_examples  = load_split(args.data_dir, 'test',  test_langs)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    print("\nTokenizing + aligning BIO labels...")
    cell_weights = compute_lang_loss_weights(train_examples)
    train_ds = BIODataset(train_examples, tokenizer, args.max_len, cell_weights=cell_weights)
    dev_ds   = BIODataset(dev_examples,   tokenizer, args.max_len)
    test_ds  = BIODataset(test_examples,  tokenizer, args.max_len)

    print(f"Train: {len(train_ds)} | Dev: {len(dev_ds)} | Test: {len(test_ds)}")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    dev_loader   = DataLoader(dev_ds,   batch_size=args.batch_size)
    test_loader  = DataLoader(test_ds,  batch_size=args.batch_size)

    model = BIOTagger(args.model_name).to(device)

    # Class weights: downweight O so the model focuses on B/I tokens.
    # reduction='none' so we can additionally scale by per-example language weight.
    class_weights = torch.tensor(
        [args.o_weight, 1.0, 1.0], dtype=torch.float
    ).to(device)
    criterion = torch.nn.CrossEntropyLoss(
        weight=class_weights, ignore_index=IGNORE_IDX, reduction='none'
    )

    optimizer    = AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    total_steps  = len(train_loader) * args.epochs
    warmup_steps = int(total_steps * args.warmup_ratio)
    scheduler    = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    if WANDB and args.use_wandb:
        wandb.init(project='idiom-bio-tagger', config=vars(args),
                   name=Path(args.output_dir).name)

    best_dev_overlap = 0.0
    best_epoch       = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}", unit='batch')
        for batch in pbar:
            input_ids       = batch['input_ids'].to(device)
            attention_mask  = batch['attention_mask'].to(device)
            token_type_ids  = batch['token_type_ids'].to(device)
            bio_labels      = batch['bio_labels'].to(device)       # [B, T]
            example_weights = batch['example_weight'].to(device)   # [B]

            logits = model(input_ids, attention_mask, token_type_ids)  # [B, T, 3]

            # Per-token loss: [B*T]
            token_loss = criterion(logits.view(-1, NUM_LABELS), bio_labels.view(-1))

            # Reshape to [B, T], then average over non-ignored tokens per example
            token_loss = token_loss.view(bio_labels.shape[0], -1)       # [B, T]
            valid_mask = (bio_labels != IGNORE_IDX).float()              # [B, T]
            denom      = valid_mask.sum(dim=1).clamp(min=1)              # [B]
            per_example_loss = (token_loss * valid_mask).sum(dim=1) / denom  # [B]

            # Scale by language cell weights and average over batch
            loss = (per_example_loss * example_weights).mean()

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            total_loss += loss.item()
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})

        avg_loss = total_loss / len(train_loader)
        print(f"\nEpoch {epoch} avg loss: {avg_loss:.4f}")

        dev_exact, dev_overlap = evaluate(
            model, dev_loader, tokenizer, dev_ds.valid_examples,
            device, f'Dev (epoch {epoch})', args.max_len
        )

        if WANDB and args.use_wandb:
            wandb.log({
                'epoch':       epoch,
                'train_loss':  avg_loss,
                'dev_exact':   dev_exact,
                'dev_overlap': dev_overlap,
            })

        if dev_overlap > best_dev_overlap:
            best_dev_overlap = dev_overlap
            best_epoch       = epoch
            save_model(model, tokenizer, output_dir)
            print(f"  ✓ New best model saved (dev overlap F1: {best_dev_overlap:.4f})")

    print(f"\nBest dev overlap F1: {best_dev_overlap:.4f} at epoch {best_epoch}")

    # Final test evaluation
    print("\nLoading best model for test evaluation...")
    best_model = load_best_model(args.model_name, output_dir, device)

    test_exact, test_overlap = evaluate(
        best_model, test_loader, tokenizer, test_ds.valid_examples,
        device, 'Test (final)', args.max_len
    )

    # Save predictions (matching Stage 2 output format for Full_evaluation.py)
    best_model.eval()
    preds_out = []
    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            input_ids      = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            token_type_ids = batch['token_type_ids'].to(device)

            logits = best_model(input_ids, attention_mask, token_type_ids)
            preds  = torch.argmax(logits, dim=-1).cpu()

            batch_start = batch_idx * test_loader.batch_size
            for i in range(len(preds)):
                ex_idx = batch_start + i
                if ex_idx >= len(test_ds.valid_examples):
                    break

                ex       = test_ds.valid_examples[ex_idx]
                sentence = ex['sentence']
                gold_s   = ex['span_start']
                gold_e   = ex['span_end']

                enc = tokenizer(
                    sentence, max_length=args.max_len,
                    truncation=True, return_offsets_mapping=True,
                )
                pred_bio = preds[i].tolist()
                pred_char_s, pred_char_e = decode_bio_to_char_span(
                    pred_bio, enc, sentence, args.max_len
                )

                # Fall back to (0, 0) if decoding failed (matches Stage 2 convention)
                if pred_char_s is None:
                    pred_char_s, pred_char_e = 0, 0

                exact   = bool(pred_char_s == gold_s and pred_char_e == gold_e)
                overlap = compute_overlap_f1(pred_char_s, pred_char_e, gold_s, gold_e)

                preds_out.append({
                    **ex,
                    'pred_span_start':   pred_char_s,
                    'pred_span_end':     pred_char_e,
                    'pred_matched_span': sentence[pred_char_s:pred_char_e],
                    'pred_bio_tags':     [ID2LABEL.get(l, 'O') for l in pred_bio
                                          if l != IGNORE_IDX],
                    'span_exact_match':  exact,
                    'span_overlap_f1':   round(overlap, 4),
                })

    preds_path = output_dir / 'test_predictions.jsonl'
    with open(preds_path, 'w', encoding='utf-8') as f:
        for p in preds_out:
            f.write(json.dumps(p, ensure_ascii=False) + '\n')
    print(f"Predictions saved → {preds_path}")

    metrics = {
        'model':             args.model_name,
        'langs':             args.langs,
        'test_langs':        test_langs,
        'best_epoch':        best_epoch,
        'best_dev_overlap':  best_dev_overlap,
        'test_exact_match':  round(float(test_exact),   4),
        'test_overlap_f1':   round(float(test_overlap), 4),
        'train_size':        len(train_ds),
        'dev_size':          len(dev_ds),
        'test_size':         len(test_ds),
        'o_weight':          args.o_weight,
    }
    json.dump(metrics, open(output_dir / 'metrics.json', 'w'), indent=2)
    print(f"Metrics saved → {output_dir / 'metrics.json'}")

    if WANDB and args.use_wandb:
        wandb.log({'test_exact': test_exact, 'test_overlap': test_overlap})
        wandb.finish()


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == '__main__':
    args = parse_args()
    train(args)