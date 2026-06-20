"""
Stage_2_training.py

Stage 2: Fine-tune mBERT (or monolingual BERT) for MWE span extraction.
Input  : raw sentence (no idiom hint)
Output : start + end character offsets of the multi-word expression

Task framing: QA-style (SQuAD), predicting start/end token indices.
Trains on ALL examples (idiomatic + literal) since every example has a
valid MWE span — the model just learns to find the expression regardless
of whether it's used idiomatically or literally.

Evaluation:
  - Exact span match (predicted span == gold span exactly)
  - Partial overlap F1 (character-level overlap between predicted and gold)

Fixes vs previous version:
  - TypeError crash: char_start/char_end could be None for examples where
    span_start/span_end is missing or null in the data. Both char_to_token_span
    and SpanDataset.__init__ now guard against this explicitly before any
    comparison, so null-span examples are skipped cleanly with a diagnostic
    count broken down by language.
  - char_to_token_span now takes int(char_start/char_end) defensively even
    after the None check, in case values arrive as float or string.
  - SpanDataset reports null-span skips separately from alignment-failure
    skips so you can distinguish a data issue from a tokenizer issue.
  - evaluate() uses dataset.valid_examples directly rather than the raw
    examples list, so batch indexing is always correct even after skips.

Usage:
    # mBERT multilingual
    python Stage_2_training.py \
        --output_dir models/stage2_mbert_en_hi_te \
        --langs English Hindi Telugu

    # Hindi + Telugu only
    python Stage_2_training.py \
        --output_dir models/stage2_mbert_hi_te \
        --langs Hindi Telugu

    # Monolingual BERT English
    python Stage_2_training.py \
        --model_name bert-base-uncased \
        --output_dir models/stage2_bert_en \
        --langs English
"""

import json
import argparse
import numpy as np
from pathlib import Path
from collections import defaultdict

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    AutoModel,
    get_linear_schedule_with_warmup,
)
from torch.optim import AdamW
from tqdm import tqdm

try:
    import wandb
    WANDB = True
except ImportError:
    WANDB = False
    print("wandb not installed — skipping. pip install wandb to enable.")


# ── Config ────────────────────────────────────────────────────────────────────

LOW_RESOURCE_LANGS = {'Hindi', 'Telugu'}


# ── Args ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--model_name',   default='bert-base-multilingual-cased')
    p.add_argument('--data_dir',     default='data/idioms_structured/Splits')
    p.add_argument('--output_dir',   default='models/stage2_mbert_en_hi_te')
    p.add_argument('--langs',        nargs='+', default=['English', 'Hindi', 'Telugu', 'Spanish'])
    p.add_argument('--test_langs',   nargs='+', default=None,
                   help='Languages to evaluate on. Defaults to --langs. '
                        'Pass all target languages for cross-lingual ablations.')
    p.add_argument('--epochs',       type=int,   default=7)
    p.add_argument('--batch_size',   type=int,   default=32)
    p.add_argument('--lr',           type=float, default=1e-5)
    p.add_argument('--max_len',      type=int,   default=128)
    p.add_argument('--warmup_ratio', type=float, default=0.1)
    p.add_argument('--seed',         type=int,   default=42)
    p.add_argument('--use_wandb',    action='store_true')
    p.add_argument('--device',       default=None)
    return p.parse_args()


# ── Device ────────────────────────────────────────────────────────────────────

def get_device(forced=None):
    if forced:
        device = torch.device(forced)
    elif torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        print("MPS detected — precompiling shaders...")
        device = torch.device('mps')
        _x = torch.zeros(1, device=device) + 1
        del _x
        print("MPS shaders ready.")
    else:
        device = torch.device('cpu')
    print(f"Device: {device}")
    return device


# ── Data loading ──────────────────────────────────────────────────────────────

def load_split_examples(split_path, langs):
    """Load examples for specific languages from a split JSONL file."""
    if not split_path.exists():
        print(f"  ⚠ Split file not found: {split_path}")
        return []
    langs_set = set(langs)
    examples = []
    for line in open(split_path, encoding='utf-8'):
        ex = json.loads(line)
        if ex['language'] in langs_set:
            examples.append(ex)
    return examples


def build_dataset_for_split(split_name, data_dir, langs, seed=42):
    """Load examples for a given split from the JSONL split files."""
    split_path = Path(data_dir) / f'{split_name}.jsonl'
    print(f"Loading {split_name} from {split_path}...")
    return load_split_examples(split_path, langs)


# ── Tokenization & span alignment ─────────────────────────────────────────────

def char_to_token_span(encoding, char_start, char_end, sentence):
    """
    Convert character offsets to token indices using HuggingFace's
    char_to_token() method. Returns (token_start, token_end) or (None, None)
    if alignment fails.

    FIX: char_start and char_end are now validated as non-None integers before
    any comparison. The previous version crashed with:
        TypeError: '>=' not supported between instances of 'int' and 'NoneType'
    when span_start/span_end were null in the source data.
    """
    # ── Guard: reject None or non-integer span values ─────────────────────────
    if char_start is None or char_end is None:
        return None, None
    try:
        char_start = int(char_start)
        char_end   = int(char_end)
    except (TypeError, ValueError):
        return None, None

    if char_start < 0 or char_end <= char_start or char_end > len(sentence):
        return None, None
    # ──────────────────────────────────────────────────────────────────────────

    # Find first token that covers or follows char_start
    token_start = None
    for i in range(char_start, len(sentence)):
        t = encoding.char_to_token(i)
        if t is not None:
            token_start = t
            break

    # Find last token that covers or precedes char_end
    token_end = None
    for i in range(char_end - 1, char_start - 1, -1):
        t = encoding.char_to_token(i)
        if t is not None:
            token_end = t
            break

    if token_start is None or token_end is None:
        return None, None
    if token_start > token_end:
        token_end = token_start
    return token_start, token_end


# ── Dataset ───────────────────────────────────────────────────────────────────

class SpanDataset(Dataset):
    def __init__(self, examples, tokenizer, max_len):
        self.valid_examples  = []
        self.input_ids       = []
        self.attention_masks = []
        self.token_type_ids  = []
        self.start_positions = []
        self.end_positions   = []

        # Track skips separately: null spans (data issue) vs alignment failures
        # (tokenizer issue) so the two are distinguishable in logs.
        null_span_skips      = 0
        null_span_by_lang    = defaultdict(int)
        alignment_skips      = 0
        alignment_by_lang    = defaultdict(int)

        for ex in examples:
            sentence   = ex.get('sentence', '')
            char_start = ex.get('span_start')
            char_end   = ex.get('span_end')
            lang       = ex.get('language', 'Unknown')

            # ── FIX: skip null-span examples before touching char offsets ─────
            if char_start is None or char_end is None:
                null_span_skips += 1
                null_span_by_lang[lang] += 1
                continue
            try:
                char_start = int(char_start)
                char_end   = int(char_end)
            except (TypeError, ValueError):
                null_span_skips += 1
                null_span_by_lang[lang] += 1
                continue
            # ──────────────────────────────────────────────────────────────────

            encoding = tokenizer(
                sentence,
                max_length=max_len,
                padding='max_length',
                truncation=True,
                return_offsets_mapping=False,
                return_tensors='pt',
            )

            # Encode without padding to get char_to_token mapping
            enc_map = tokenizer(
                sentence,
                max_length=max_len,
                truncation=True,
                return_offsets_mapping=True,
            )

            token_start, token_end = char_to_token_span(
                enc_map, char_start, char_end, sentence
            )

            if token_start is None or token_end is None:
                alignment_skips += 1
                alignment_by_lang[lang] += 1
                continue

            # Clamp to max_len - 1
            seq_len     = encoding['input_ids'].shape[1]
            token_start = min(token_start, seq_len - 1)
            token_end   = min(token_end,   seq_len - 1)

            self.valid_examples.append(ex)
            self.input_ids.append(encoding['input_ids'].squeeze(0))
            self.attention_masks.append(encoding['attention_mask'].squeeze(0))
            tid = encoding.get('token_type_ids')
            self.token_type_ids.append(
                tid.squeeze(0) if tid is not None
                else torch.zeros(seq_len, dtype=torch.long)
            )
            self.start_positions.append(token_start)
            self.end_positions.append(token_end)

        # ── Diagnostic reporting ──────────────────────────────────────────────
        if null_span_skips:
            print(f"  ⚠ Skipped {null_span_skips} examples with null/missing span_start or span_end")
            print(f"    This is a DATA issue — check upstream pipeline for these languages:")
            for lang, count in sorted(null_span_by_lang.items()):
                print(f"      {lang}: {count} examples")

        if alignment_skips:
            print(f"  ⚠ Skipped {alignment_skips} examples where char→token alignment failed")
            print(f"    This is a TOKENIZER issue (e.g. subword boundary mismatch):")
            for lang, count in sorted(alignment_by_lang.items()):
                print(f"      {lang}: {count} examples")

        total_skipped = null_span_skips + alignment_skips
        if total_skipped:
            print(f"  Total skipped: {total_skipped} / {len(examples)} "
                  f"({100 * total_skipped / max(len(examples), 1):.1f}%)")
        # ──────────────────────────────────────────────────────────────────────

    def __len__(self):
        return len(self.valid_examples)

    def __getitem__(self, idx):
        return {
            'input_ids':       self.input_ids[idx],
            'attention_mask':  self.attention_masks[idx],
            'token_type_ids':  self.token_type_ids[idx],
            'start_positions': torch.tensor(self.start_positions[idx], dtype=torch.long),
            'end_positions':   torch.tensor(self.end_positions[idx],   dtype=torch.long),
        }


# ── Model ─────────────────────────────────────────────────────────────────────

class SpanExtractor(torch.nn.Module):
    """mBERT + two linear heads for start/end token prediction (QA-style)."""

    def __init__(self, model_name):
        super().__init__()
        self.bert       = AutoModel.from_pretrained(model_name)
        hidden_size     = self.bert.config.hidden_size
        self.start_head = torch.nn.Linear(hidden_size, 1)
        self.end_head   = torch.nn.Linear(hidden_size, 1)

    def forward(self, input_ids, attention_mask, token_type_ids):
        outputs    = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        seq_output = outputs.last_hidden_state  # [batch, seq_len, hidden]

        start_logits = self.start_head(seq_output).squeeze(-1)  # [batch, seq_len]
        end_logits   = self.end_head(seq_output).squeeze(-1)    # [batch, seq_len]

        # Mask padding tokens
        mask = attention_mask.bool()
        start_logits = start_logits.masked_fill(~mask, float('-inf'))
        end_logits   = end_logits.masked_fill(~mask,   float('-inf'))

        return start_logits, end_logits


# ── Evaluation ────────────────────────────────────────────────────────────────

def token_to_char_span(tokenizer, sentence, token_start, token_end, max_len):
    """Convert predicted token indices back to character offsets."""
    enc     = tokenizer(sentence, max_length=max_len, truncation=True,
                        return_offsets_mapping=True)
    offsets = enc['offset_mapping']

    if token_start >= len(offsets) or token_end >= len(offsets):
        return None, None

    char_start = offsets[token_start][0]
    char_end   = offsets[token_end][1]
    return char_start, char_end


def compute_overlap_f1(pred_start, pred_end, gold_start, gold_end):
    """Character-level overlap F1 between predicted and gold spans."""
    pred_set = set(range(pred_start, pred_end + 1))
    gold_set = set(range(gold_start, gold_end + 1))

    if not pred_set or not gold_set:
        return 0.0

    overlap = len(pred_set & gold_set)
    if overlap == 0:
        return 0.0

    precision = overlap / len(pred_set)
    recall    = overlap / len(gold_set)
    return 2 * precision * recall / (precision + recall)


def evaluate(model, loader, tokenizer, dataset, device, split_name, max_len):
    """
    Evaluate span extraction on a DataLoader.
    Uses dataset.valid_examples (post-skip list) for correct index alignment.
    """
    model.eval()

    exact_matches = 0
    overlap_f1s   = []
    lang_exact    = defaultdict(list)
    lang_f1       = defaultdict(list)
    total         = 0

    # Build a flat index over valid_examples aligned to loader order
    examples = dataset.valid_examples

    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(loader, desc=f'Eval {split_name}', leave=False)):
            input_ids      = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            token_type_ids = batch['token_type_ids'].to(device)

            start_logits, end_logits = model(input_ids, attention_mask, token_type_ids)

            pred_starts = torch.argmax(start_logits, dim=-1).cpu().numpy()
            pred_ends   = torch.argmax(end_logits,   dim=-1).cpu().numpy()
            gold_starts = batch['start_positions'].numpy()
            gold_ends   = batch['end_positions'].numpy()

            batch_start = batch_idx * loader.batch_size
            for i in range(len(pred_starts)):
                ex_idx = batch_start + i
                if ex_idx >= len(examples):
                    break

                ex   = examples[ex_idx]
                lang = ex['language']

                pred_s = int(pred_starts[i])
                pred_e = int(pred_ends[i])
                gold_s = int(gold_starts[i])
                gold_e = int(gold_ends[i])

                # Ensure pred_end >= pred_start
                if pred_e < pred_s:
                    pred_e = pred_s

                exact = int(pred_s == gold_s and pred_e == gold_e)
                exact_matches += exact
                lang_exact[lang].append(exact)

                f1 = compute_overlap_f1(pred_s, pred_e, gold_s, gold_e)
                overlap_f1s.append(f1)
                lang_f1[lang].append(f1)

                total += 1

    exact_match = exact_matches / total if total > 0 else 0.0
    avg_overlap = float(np.mean(overlap_f1s)) if overlap_f1s else 0.0

    print(f"\n── {split_name} results ──")
    print(f"  Exact match:    {exact_match:.4f}")
    print(f"  Overlap F1:     {avg_overlap:.4f}")
    print(f"  Total examples: {total}")

    for lang in sorted(lang_exact.keys()):
        le = float(np.mean(lang_exact[lang]))
        lf = float(np.mean(lang_f1[lang]))
        print(f"  {lang:<12}: exact={le:.4f}  overlap_f1={lf:.4f}  "
              f"({len(lang_exact[lang])} examples)")

    return exact_match, avg_overlap


# ── Train ─────────────────────────────────────────────────────────────────────

def train(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device     = get_device(args.device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    config = vars(args)
    json.dump(config, open(output_dir / 'config.json', 'w'), indent=2)
    print(f"Languages (train): {args.langs}")

    # Build raw example lists
    train_examples = build_dataset_for_split('train', args.data_dir, args.langs, args.seed)
    dev_examples   = build_dataset_for_split('dev',   args.data_dir, args.langs, args.seed)
    test_langs     = args.test_langs or args.langs
    test_examples  = build_dataset_for_split('test',  args.data_dir, test_langs, args.seed)

    print(f"Languages (test):  {test_langs}")
    print(f"Raw counts — train: {len(train_examples)} | "
          f"dev: {len(dev_examples)} | test: {len(test_examples)}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    print("Tokenizing train...")
    train_ds = SpanDataset(train_examples, tokenizer, args.max_len)
    print("Tokenizing dev...")
    dev_ds   = SpanDataset(dev_examples,   tokenizer, args.max_len)
    print("Tokenizing test...")
    test_ds  = SpanDataset(test_examples,  tokenizer, args.max_len)

    print(f"After tokenization — train: {len(train_ds)} | "
          f"dev: {len(dev_ds)} | test: {len(test_ds)}")

    if len(train_ds) == 0:
        raise RuntimeError(
            "Training set is empty after tokenization. "
            "Check --langs and --data_dir, and inspect the null-span warnings above."
        )

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    dev_loader   = DataLoader(dev_ds,   batch_size=args.batch_size)
    test_loader  = DataLoader(test_ds,  batch_size=args.batch_size)

    model     = SpanExtractor(args.model_name).to(device)
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)

    total_steps  = len(train_loader) * args.epochs
    warmup_steps = int(total_steps * args.warmup_ratio)
    scheduler    = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    criterion = torch.nn.CrossEntropyLoss()

    if WANDB and args.use_wandb:
        wandb.init(project='idiomator-app', config=config,
                   name=f"stage2_{Path(args.output_dir).name}")

    best_overlap = 0.0
    best_epoch   = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}", unit='batch')
        for batch in pbar:
            input_ids       = batch['input_ids'].to(device)
            attention_mask  = batch['attention_mask'].to(device)
            token_type_ids  = batch['token_type_ids'].to(device)
            start_positions = batch['start_positions'].to(device)
            end_positions   = batch['end_positions'].to(device)

            start_logits, end_logits = model(input_ids, attention_mask, token_type_ids)

            start_loss = criterion(start_logits, start_positions)
            end_loss   = criterion(end_logits,   end_positions)
            loss       = (start_loss + end_loss) / 2

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            total_loss += loss.item()
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})

        avg_loss = total_loss / len(train_loader)
        print(f"\nEpoch {epoch} avg loss: {avg_loss:.4f}")

        exact, overlap = evaluate(
            model, dev_loader, tokenizer, dev_ds,
            device, f'Dev (epoch {epoch})', args.max_len
        )

        if WANDB and args.use_wandb:
            wandb.log({'epoch': epoch, 'train_loss': avg_loss,
                       'dev_exact': exact, 'dev_overlap_f1': overlap})

        # Optimise for overlap F1 (more lenient, better for low-resource)
        if overlap > best_overlap:
            best_overlap = overlap
            best_epoch   = epoch
            model.bert.save_pretrained(output_dir / 'best_model')
            tokenizer.save_pretrained(output_dir / 'best_model')
            torch.save(
                {'start_head': model.start_head.state_dict(),
                 'end_head':   model.end_head.state_dict()},
                output_dir / 'best_model' / 'span_heads.pt'
            )
            print(f"  ✓ New best model saved (dev overlap F1: {best_overlap:.4f})")

    print(f"\nBest dev overlap F1: {best_overlap:.4f} at epoch {best_epoch}")

    # ── Final test evaluation ─────────────────────────────────────────────────
    print("\nLoading best model for test evaluation...")
    best_model = SpanExtractor(args.model_name)
    best_model.bert = AutoModel.from_pretrained(output_dir / 'best_model')
    heads = torch.load(output_dir / 'best_model' / 'span_heads.pt', map_location='cpu')
    best_model.start_head.load_state_dict(heads['start_head'])
    best_model.end_head.load_state_dict(heads['end_head'])
    best_model = best_model.to(device)

    test_exact, test_overlap = evaluate(
        best_model, test_loader, tokenizer, test_ds,
        device, 'Test (final)', args.max_len
    )

    # ── Save predictions ──────────────────────────────────────────────────────
    best_model.eval()
    preds_out = []

    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            input_ids      = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            token_type_ids = batch['token_type_ids'].to(device)

            start_logits, end_logits = best_model(input_ids, attention_mask, token_type_ids)
            pred_starts = torch.argmax(start_logits, dim=-1).cpu().numpy()
            pred_ends   = torch.argmax(end_logits,   dim=-1).cpu().numpy()
            gold_starts = batch['start_positions'].numpy()
            gold_ends   = batch['end_positions'].numpy()

            batch_start = batch_idx * test_loader.batch_size
            for i in range(len(pred_starts)):
                ex_idx = batch_start + i
                if ex_idx >= len(test_ds.valid_examples):
                    break
                ex = test_ds.valid_examples[ex_idx]

                pred_s = int(pred_starts[i])
                pred_e = int(pred_ends[i])
                if pred_e < pred_s:
                    pred_e = pred_s

                gold_s = int(gold_starts[i])
                gold_e = int(gold_ends[i])

                pred_char_s, pred_char_e = token_to_char_span(
                    tokenizer, ex['sentence'], pred_s, pred_e, args.max_len
                )

                pred_span = ex['sentence'][pred_char_s:pred_char_e] \
                    if pred_char_s is not None else ''

                preds_out.append({
                    **ex,
                    'pred_span_start':   pred_char_s,
                    'pred_span_end':     pred_char_e,
                    'pred_matched_span': pred_span,
                    'exact_match':       bool(pred_s == gold_s and pred_e == gold_e),
                    'overlap_f1':        round(compute_overlap_f1(pred_s, pred_e, gold_s, gold_e), 4),
                })

    preds_path = output_dir / 'test_predictions.jsonl'
    with open(preds_path, 'w', encoding='utf-8') as f:
        for pred in preds_out:
            f.write(json.dumps(pred, ensure_ascii=False) + '\n')
    print(f"Predictions saved → {preds_path}  ({len(preds_out)} examples)")

    metrics = {
        'best_dev_overlap_f1': best_overlap,
        'best_epoch':          best_epoch,
        'test_exact_match':    test_exact,
        'test_overlap_f1':     test_overlap,
        'model':               args.model_name,
        'langs':               args.langs,
        'test_langs':          test_langs,
        'train_size':          len(train_ds),
        'dev_size':            len(dev_ds),
        'test_size':           len(test_ds),
    }
    json.dump(metrics, open(output_dir / 'metrics.json', 'w'), indent=2)
    print(f"Metrics saved → {output_dir / 'metrics.json'}")

    if WANDB and args.use_wandb:
        wandb.log({'test_exact_match': test_exact, 'test_overlap_f1': test_overlap})
        wandb.finish()


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == '__main__':
    args = parse_args()
    train(args)