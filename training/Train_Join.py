"""
train_joint.py

Joint single-stage mBERT model for idiom classification + MWE span extraction.
One encoder, three heads, one forward pass:
  - Classification head  : idiomatic vs literal (Macro F1)
  - Span start head      : token-level start position (Exact / Overlap F1)
  - Span end head        : token-level end position   (Exact / Overlap F1)

Loss = cls_weight * cls_loss + span_weight * (start_loss + end_loss) / 2

Data pipeline:
  - All languages (English, Hindi, Telugu) use pre-built splits:
    data/idioms_structured/Splits/train.jsonl
    data/idioms_structured/Splits/dev.jsonl
    data/idioms_structured/Splits/test.jsonl
  - Loss weighted by (language × idiomaticity) cell frequency

Usage:
    # mBERT — all three languages
    python train_joint.py \\
        --output_dir models/joint_mbert_en_hi_te \\
        --langs English Hindi Telugu

    # Hindi + Telugu only
    python train_joint.py \\
        --output_dir models/joint_mbert_hi_te \\
        --langs Hindi Telugu

    # Tune loss balance
    python train_joint.py \\
        --output_dir models/joint_mbert_en_hi_te \\
        --cls_loss_weight 1.0 \\
        --span_loss_weight 1.0 \\
        --langs English Hindi Telugu
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
from sklearn.metrics import classification_report, f1_score
from tqdm import tqdm

try:
    import wandb
    WANDB = True
except ImportError:
    WANDB = False
    print("wandb not installed — skipping. pip install wandb to enable.")


# ── Config ────────────────────────────────────────────────────────────────────

LABEL2ID = {'literal': 0, 'idiomatic': 1}
ID2LABEL  = {0: 'literal', 1: 'idiomatic'}

LOW_RESOURCE_LANGS = {'Hindi', 'Telugu'}

# ── Args ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--model_name',      default='bert-base-multilingual-cased')
    p.add_argument('--data_dir',        default='data/idioms_structured/Splits')
    p.add_argument('--dropout', type=float, default=0.1)
    p.add_argument('--output_dir',      default='models/Spanish_Addition/joint_mbert_en_hi_te')
    p.add_argument('--langs', nargs='+', default=['English', 'Spanish', 'Hindi', 'Telugu'], # Added Spanish
    help='Languages to include in training/eval'
)
    p.add_argument('--test_langs', nargs='+', default=None,
    help='Languages to evaluate on. Defaults to --langs. Use all target languages for cross-lingual ablations.'
)
    p.add_argument('--epochs',          type=int,   default=7)
    p.add_argument('--batch_size',      type=int,   default=32)
    p.add_argument('--lr',              type=float, default=2e-05)
    p.add_argument('--max_len',         type=int,   default=128)
    p.add_argument('--warmup_ratio',    type=float, default=0.05)
    p.add_argument('--seed',            type=int,   default=42)
    p.add_argument('--grad_accum_steps', type=int, default=1,
               help='Gradient accumulation steps. Effective batch = batch_size × grad_accum_steps.')
    p.add_argument('--cls_loss_weight', type=float, default=0.3,
                   help='Weight on classification loss term')
    p.add_argument('--span_loss_weight',type=float, default=1.9,
                   help='Weight on span extraction loss term')
    p.add_argument('--use_wandb',       action='store_true')
    p.add_argument('--device',          default=None)
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
    langs_set = set(langs)
    return [
        json.loads(l) for l in open(split_path, encoding='utf-8')
        if json.loads(l)['language'] in langs_set
    ]


def build_dataset_for_split(split_name, data_dir, langs, seed=42):
    """
    Load examples directly from pre-built split files.
    All languages (including Hindi and Telugu) use data/idioms_structured/Splits/
    train.jsonl, dev.jsonl, test.jsonl — no on-the-fly re-splitting.
    """
    split_path = Path(data_dir) / f'{split_name}.jsonl'
    return load_split_examples(split_path, langs)


# ── Loss weights (from Stage 1, unchanged) ────────────────────────────────────

def compute_cls_loss_weights(train_examples, device):
    """
    Inverse-frequency weights over (language, idiomaticity) cells,
    marginalized to per-class weights for CrossEntropyLoss.
    Identical to Stage 1 logic.
    """
    cell_counts = Counter((ex['language'], ex['idiomaticity']) for ex in train_examples)
    total   = sum(cell_counts.values())
    n_cells = len(cell_counts)

    cell_weights = {cell: total / (n_cells * count) for cell, count in cell_counts.items()}

    class_weighted = defaultdict(float)
    class_counts   = defaultdict(int)
    for ex in train_examples:
        cls = ex['idiomaticity']
        class_weighted[cls] += cell_weights[(ex['language'], cls)]
        class_counts[cls]   += 1

    literal_w   = class_weighted['literal']   / class_counts['literal']
    idiomatic_w = class_weighted['idiomatic'] / class_counts['idiomatic']

    min_w       = min(literal_w, idiomatic_w)
    literal_w  /= min_w
    idiomatic_w /= min_w

    weights = torch.tensor([literal_w, idiomatic_w], dtype=torch.float).to(device)
    print(f"\nCls loss weights — literal: {weights[0]:.4f}, idiomatic: {weights[1]:.4f}")
    return weights


# ── Span alignment helpers (from Stage 2, unchanged) ─────────────────────────

def char_to_token_span(encoding, char_start, char_end, sentence):
    if char_start is None or char_end is None:
        return None, None
    token_start = None
    for i in range(len(sentence)):
        t = encoding.char_to_token(i)
        if t is not None and i >= char_start:
            token_start = t
            break

    token_end = None
    for i in range(char_end - 1, -1, -1):
        t = encoding.char_to_token(i)
        if t is not None:
            token_end = t
            break

    if token_start is None or token_end is None:
        return None, None
    if token_start > token_end:
        token_end = token_start
    return token_start, token_end


def token_to_char_span(tokenizer, sentence, token_start, token_end, max_len):
    enc = tokenizer(sentence, max_length=max_len, truncation=True,
                    return_offsets_mapping=True)
    offsets = enc['offset_mapping']
    if token_start >= len(offsets) or token_end >= len(offsets):
        return None, None
    cs, ce = offsets[token_start][0], offsets[token_end][1]
    # SentencePiece tokenizers (XLM-R, RemBERT, mDeBERTa) prepend ▁ to
    # word-initial tokens, shifting the token's char offset left by one
    # into the preceding space. Without this strip, every word-initial
    # span boundary decodes one char early under SP — exact_match collapses
    # while overlap_f1 stays high. WordPiece tokenizers are unaffected
    # (no whitespace at decoded char_start).
    while cs < ce and sentence[cs].isspace():
        cs += 1
    return cs, ce


def compute_overlap_f1(pred_start, pred_end, gold_start, gold_end):
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


# ── Dataset ───────────────────────────────────────────────────────────────────

class JointDataset(Dataset):
    """
    Single dataset that provides inputs for both classification and span tasks.
    Every example contributes to both loss terms.
    Span alignment failures are skipped (same behaviour as Stage 2).
    """

    def __init__(self, examples, tokenizer, max_len):
        self.valid_examples  = []
        self.input_ids       = []
        self.attention_masks = []
        self.token_type_ids  = []
        self.cls_labels      = []   # 0=literal, 1=idiomatic
        self.start_positions = []
        self.end_positions   = []

        skipped = 0
        for ex in examples:
            sentence   = ex['sentence']
            char_start = ex['span_start']
            char_end   = ex['span_end']

            # Padded encoding for model input
            encoding = tokenizer(
                sentence,
                max_length=max_len,
                padding='max_length',
                truncation=True,
                return_tensors='pt',
            )

            # Unpadded encoding for char→token mapping
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
                skipped += 1
                continue

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
            self.cls_labels.append(LABEL2ID[ex['idiomaticity']])
            self.start_positions.append(token_start)
            self.end_positions.append(token_end)

        if skipped:
            print(f"  Skipped {skipped} examples with failed span alignment")

    def __len__(self):
        return len(self.valid_examples)

    def __getitem__(self, idx):
        return {
            'input_ids':       self.input_ids[idx],
            'attention_mask':  self.attention_masks[idx],
            'token_type_ids':  self.token_type_ids[idx],
            'cls_labels':      torch.tensor(self.cls_labels[idx],      dtype=torch.long),
            'start_positions': torch.tensor(self.start_positions[idx], dtype=torch.long),
            'end_positions':   torch.tensor(self.end_positions[idx],   dtype=torch.long),
        }


# ── Model ─────────────────────────────────────────────────────────────────────

class JointIdiomModel(torch.nn.Module):
    """
    Single mBERT encoder with three task heads:
      [CLS] token  → classification head  (literal / idiomatic)
      All tokens   → start head           (span start logits)
      All tokens   → end head             (span end logits)

    All three heads share the same encoder weights and are updated
    jointly in every training step.
    """

    def __init__(self, model_name):
        super().__init__()
        self.bert        = AutoModel.from_pretrained(model_name)
        hidden_size      = self.bert.config.hidden_size

        self.cls_head    = torch.nn.Linear(hidden_size, 2)   # literal / idiomatic
        self.start_head  = torch.nn.Linear(hidden_size, 1)   # span start
        self.end_head    = torch.nn.Linear(hidden_size, 1)   # span end

    def forward(self, input_ids, attention_mask, token_type_ids):
        outputs    = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        seq_output = outputs.last_hidden_state.float()  # cast: mDeBERTa-v3 can emit fp16 on GPU
        cls_output = seq_output[:, 0, :]            # [batch, hidden]  ([CLS] token)

        cls_logits   = self.cls_head(cls_output)            # [batch, 2]
        start_logits = self.start_head(seq_output).squeeze(-1)  # [batch, seq_len]
        end_logits   = self.end_head(seq_output).squeeze(-1)    # [batch, seq_len]

        # Mask padding tokens for span heads
        mask         = attention_mask.bool()
        start_logits = start_logits.masked_fill(~mask, float('-inf'))
        end_logits   = end_logits.masked_fill(~mask,   float('-inf'))

        return cls_logits, start_logits, end_logits


# ── Save / load helpers ───────────────────────────────────────────────────────

def save_model(model, tokenizer, output_dir):
    model.bert.save_pretrained(output_dir / 'best_model')
    tokenizer.save_pretrained(output_dir / 'best_model')
    torch.save({
        'cls_head':   model.cls_head.state_dict(),
        'start_head': model.start_head.state_dict(),
        'end_head':   model.end_head.state_dict(),
    }, output_dir / 'best_model' / 'task_heads.pt')


def load_best_model(model_name, output_dir, device):
    model      = JointIdiomModel(model_name)
    model.bert = AutoModel.from_pretrained(output_dir / 'best_model')
    heads      = torch.load(
        output_dir / 'best_model' / 'task_heads.pt', map_location='cpu', weights_only=True
    )
    model.cls_head.load_state_dict(heads['cls_head'])
    model.start_head.load_state_dict(heads['start_head'])
    model.end_head.load_state_dict(heads['end_head'])
    return model.to(device)


# ── Evaluation ────────────────────────────────────────────────────────────────

def evaluate(model, loader, tokenizer, examples, device, split_name, max_len):
    model.eval()

    all_cls_preds, all_cls_labels = [], []
    exact_matches = 0
    overlap_f1s   = []
    lang_exact    = defaultdict(list)
    lang_f1       = defaultdict(list)
    lang_cls_preds  = defaultdict(list)
    lang_cls_labels = defaultdict(list)
    total = 0

    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(loader, desc=f'Eval {split_name}', leave=False)):
            input_ids      = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            token_type_ids = batch['token_type_ids'].to(device)

            cls_logits, start_logits, end_logits = model(
                input_ids, attention_mask, token_type_ids
            )

            cls_preds   = torch.argmax(cls_logits,   dim=-1).cpu().numpy()
            pred_starts = torch.argmax(start_logits, dim=-1).cpu().numpy()
            pred_ends   = torch.argmax(end_logits,   dim=-1).cpu().numpy()
            gold_cls    = batch['cls_labels'].numpy()
            gold_starts = batch['start_positions'].numpy()
            gold_ends   = batch['end_positions'].numpy()

            batch_start = batch_idx * loader.batch_size
            for i in range(len(cls_preds)):
                ex_idx = batch_start + i
                if ex_idx >= len(examples):
                    break

                ex   = examples[ex_idx]
                lang = ex['language']

                # Classification
                all_cls_preds.append(int(cls_preds[i]))
                all_cls_labels.append(int(gold_cls[i]))
                lang_cls_preds[lang].append(int(cls_preds[i]))
                lang_cls_labels[lang].append(int(gold_cls[i]))

                # Span
                pred_s = int(pred_starts[i])
                pred_e = int(pred_ends[i])
                gold_s = int(gold_starts[i])
                gold_e = int(gold_ends[i])
                if pred_e < pred_s:
                    pred_e = pred_s

                exact = int(pred_s == gold_s and pred_e == gold_e)
                exact_matches += exact
                lang_exact[lang].append(exact)

                f1 = compute_overlap_f1(pred_s, pred_e, gold_s, gold_e)
                overlap_f1s.append(f1)
                lang_f1[lang].append(f1)

                total += 1

    # Classification metrics
    macro_f1    = f1_score(all_cls_labels, all_cls_preds, average='macro')
    report      = classification_report(
        all_cls_labels, all_cls_preds,
        target_names=['literal', 'idiomatic'], digits=4
    )

    # Span metrics
    exact_match = exact_matches / total if total > 0 else 0
    avg_overlap = np.mean(overlap_f1s) if overlap_f1s else 0

    print(f"\n── {split_name} results ──")
    print("Classification:")
    print(report)
    print(f"Span — Exact: {exact_match:.4f}  Overlap F1: {avg_overlap:.4f}")
    print(f"Total examples: {total}\n")

    print("Per-language breakdown:")
    for lang in sorted(lang_exact.keys()):
        lf1  = f1_score(lang_cls_labels[lang], lang_cls_preds[lang], average='macro')
        le   = np.mean(lang_exact[lang])
        lf   = np.mean(lang_f1[lang])
        n    = len(lang_exact[lang])
        print(f"  {lang:10s}  cls_macro_F1={lf1:.4f}  span_exact={le:.4f}  span_overlap_F1={lf:.4f}  ({n} examples)")

    return macro_f1, exact_match, avg_overlap, all_cls_preds, all_cls_labels


# ── Train ─────────────────────────────────────────────────────────────────────

def train(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device     = get_device(args.device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    config = vars(args)
    json.dump(config, open(output_dir / 'config.json', 'w'), indent=2)
    print(f"Languages: {args.langs}")
    print(f"Loss weights — cls: {args.cls_loss_weight}  span: {args.span_loss_weight}")

    # Build datasets
    train_examples = build_dataset_for_split('train', args.data_dir, args.langs, args.seed)
    dev_examples   = build_dataset_for_split('dev',   args.data_dir, args.langs, args.seed)
    test_langs = args.test_langs or args.langs
    test_examples  = build_dataset_for_split('test',  args.data_dir, test_langs, args.seed)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    print("Tokenizing train...")
    train_ds = JointDataset(train_examples, tokenizer, args.max_len)
    print("Tokenizing dev...")
    dev_ds   = JointDataset(dev_examples,   tokenizer, args.max_len)
    print("Tokenizing test...")
    test_ds  = JointDataset(test_examples,  tokenizer, args.max_len)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    dev_loader   = DataLoader(dev_ds,   batch_size=args.batch_size)
    test_loader  = DataLoader(test_ds,  batch_size=args.batch_size)

    print(f"Train: {len(train_ds)} | Dev: {len(dev_ds)} | Test: {len(test_ds)}")

    model = JointIdiomModel(args.model_name).to(device)

    cls_weights  = compute_cls_loss_weights(train_ds.valid_examples, device)
    cls_criterion  = torch.nn.CrossEntropyLoss(weight=cls_weights)
    span_criterion = torch.nn.CrossEntropyLoss()   # no weighting on span heads

    optimizer    = AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    # optimizer steps = ceil(batches_per_epoch / grad_accum_steps) * epochs
    total_steps  = ((len(train_loader) + args.grad_accum_steps - 1) // args.grad_accum_steps) * args.epochs
    warmup_steps = int(total_steps * args.warmup_ratio)
    scheduler    = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    if WANDB and args.use_wandb:
        wandb.init(project='idiom-joint', config=config,
                   name=Path(args.output_dir).name)

    best_dev_joint_f1 = 0.0   # optimise for joint F1 = geomean(cls_macro_f1, span_overlap_f1)
    best_epoch        = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}", unit='batch')
        for batch_idx, batch in enumerate(pbar):
            input_ids       = batch['input_ids'].to(device)
            attention_mask  = batch['attention_mask'].to(device)
            token_type_ids  = batch['token_type_ids'].to(device)
            cls_labels      = batch['cls_labels'].to(device)
            start_positions = batch['start_positions'].to(device)
            end_positions   = batch['end_positions'].to(device)

            cls_logits, start_logits, end_logits = model(
                input_ids, attention_mask, token_type_ids
            )

            cls_loss  = cls_criterion(cls_logits, cls_labels)
            span_loss = (
                span_criterion(start_logits, start_positions) +
                span_criterion(end_logits,   end_positions)
            ) / 2

            unscaled_loss = args.cls_loss_weight * cls_loss + args.span_loss_weight * span_loss

            if torch.isnan(unscaled_loss) or torch.isinf(unscaled_loss):
                # mDeBERTa-v3 disentangled attention can produce NaN/inf in the
                # span or cls head on the first few batches.  Skip the batch so
                # the optimizer state stays clean.
                print(f"  ⚠ NaN/inf loss (cls={cls_loss.item():.4f} "
                      f"span={span_loss.item():.4f}) — skipping batch")
                optimizer.zero_grad()
                continue

            # scale before backward so accumulated gradients equal one full-batch gradient
            (unscaled_loss / args.grad_accum_steps).backward()

            # mDeBERTa-v3 can emit NaN gradients via its position-bias
            # computation even when the forward loss is finite.  Clipping
            # a NaN is a no-op, so sanitise first.
            nan_params = 0
            for p in model.parameters():
                if p.grad is not None and (torch.isnan(p.grad).any() or
                                           torch.isinf(p.grad).any()):
                    p.grad = torch.nan_to_num(p.grad, nan=0.0, posinf=0.0, neginf=0.0)
                    nan_params += 1
            if nan_params:
                print(f"  ⚠ Sanitised NaN/inf grads in {nan_params} params")

            is_accum_step = (batch_idx + 1) % args.grad_accum_steps == 0
            is_last_batch = (batch_idx + 1) == len(train_loader)
            if is_accum_step or is_last_batch:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            total_loss += unscaled_loss.item()
            pbar.set_postfix({
                'loss': f'{unscaled_loss.item():.4f}',
                'cls':  f'{cls_loss.item():.4f}',
                'span': f'{span_loss.item():.4f}',
            })

        avg_loss = total_loss / len(train_loader)
        print(f"\nEpoch {epoch} avg loss: {avg_loss:.4f}")

        dev_cls_f1, dev_exact, dev_overlap, _, _ = evaluate(
            model, dev_loader, tokenizer, dev_ds.valid_examples,
            device, f'Dev (epoch {epoch})', args.max_len
        )

        # Joint F1 = geometric mean of cls macro F1 and span overlap F1
        # Geometric mean penalises models that sacrifice one task for the other
        import math
        dev_joint_f1 = math.sqrt(dev_cls_f1 * dev_overlap) if (dev_cls_f1 > 0 and dev_overlap > 0) else 0.0
        print(f"  Dev joint F1 (geomean): {dev_joint_f1:.4f}  (cls={dev_cls_f1:.4f}, span_overlap={dev_overlap:.4f})")

        if WANDB and args.use_wandb:
            wandb.log({
                'epoch':           epoch,
                'train_loss':      avg_loss,
                'dev_cls_f1':      dev_cls_f1,
                'dev_span_exact':  dev_exact,
                'dev_span_f1':     dev_overlap,
                'dev_joint_f1':    dev_joint_f1,
            })

        if dev_joint_f1 > best_dev_joint_f1:
            best_dev_joint_f1 = dev_joint_f1
            best_epoch        = epoch
            save_model(model, tokenizer, output_dir)
            print(f"  ✓ New best model saved (dev joint F1: {best_dev_joint_f1:.4f})")

    print(f"\nBest dev joint F1: {best_dev_joint_f1:.4f} at epoch {best_epoch}")

    # Final test eval
    print("\nLoading best model for test evaluation...")
    best_model = load_best_model(args.model_name, output_dir, device)

    test_cls_f1, test_exact, test_overlap, test_preds, test_labels = evaluate(
        best_model, test_loader, tokenizer, test_ds.valid_examples,
        device, 'Test (final)', args.max_len
    )

    # Save predictions
    best_model.eval()
    preds_out = []
    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            input_ids      = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            token_type_ids = batch['token_type_ids'].to(device)

            cls_logits, start_logits, end_logits = best_model(
                input_ids, attention_mask, token_type_ids
            )
            cls_preds   = torch.argmax(cls_logits,   dim=-1).cpu().numpy()
            pred_starts = torch.argmax(start_logits, dim=-1).cpu().numpy()
            pred_ends   = torch.argmax(end_logits,   dim=-1).cpu().numpy()
            gold_cls    = batch['cls_labels'].numpy()
            gold_starts = batch['start_positions'].numpy()
            gold_ends   = batch['end_positions'].numpy()

            batch_start = batch_idx * test_loader.batch_size
            for i in range(len(cls_preds)):
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
                pred_span = (
                    ex['sentence'][pred_char_s:pred_char_e]
                    if pred_char_s is not None else ''
                )

                preds_out.append({
                    **ex,
                    'pred_idiomaticity':  ID2LABEL[int(cls_preds[i])],
                    'cls_correct':        bool(int(cls_preds[i]) == int(gold_cls[i])),
                    'pred_span_start':    pred_char_s,
                    'pred_span_end':      pred_char_e,
                    'pred_matched_span':  pred_span,
                    'span_exact_match':   bool(pred_s == gold_s and pred_e == gold_e),
                    'span_overlap_f1':    round(compute_overlap_f1(pred_s, pred_e, gold_s, gold_e), 4),
                })

    preds_path = output_dir / 'test_predictions.jsonl'
    with open(preds_path, 'w', encoding='utf-8') as f:
        for p in preds_out:
            f.write(json.dumps(p, ensure_ascii=False) + '\n')
    print(f"Predictions saved → {preds_path}")

    metrics = {
        'best_dev_cls_f1':   dev_cls_f1,
        'best_dev_joint_f1': best_dev_joint_f1,
        'best_epoch':        best_epoch,
        'test_cls_macro_f1': test_cls_f1,
        'test_span_exact':   test_exact,
        'test_span_overlap': test_overlap,
        'model':             args.model_name,
        'langs':             args.langs,
        'test_langs':        test_langs,
        'cls_loss_weight':   args.cls_loss_weight,
        'span_loss_weight':  args.span_loss_weight,
        'train_size':        len(train_ds),
        'dev_size':          len(dev_ds),
        'test_size':         len(test_ds),
    }
    json.dump(metrics, open(output_dir / 'metrics.json', 'w'), indent=2)
    print(f"Metrics saved → {output_dir / 'metrics.json'}")

    if WANDB and args.use_wandb:
        wandb.log({
            'test_cls_f1':     test_cls_f1,
            'test_span_exact': test_exact,
            'test_span_f1':    test_overlap,
        })
        wandb.finish()


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == '__main__':
    args = parse_args()
    train(args)