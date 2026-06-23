"""
run_17_bio_cls_joint.py  —  REVISION_PLAN experiment E4

Hybrid BIO-span + CLS-classifier head, jointly trained on ONE mBERT encoder.

Why this exists
----------------
Systems D/E/F all pair a fine-tuned mBERT encoder with QA-style start/end
span heads. System G is BIO-only (no classifier at all, so its Joint F1 is
structurally penalized — see key_numbers.md). Nobody has tested whether a
classifier-equipped BIO tagger matches D/E on Joint F1. E4 builds that
missing cell: same encoder, same loss-weight convention (cls=0.3, span=1.9,
identical to Train_Join.py) as System E, but the span head is a 3-class
per-token BIO tagger (System G's exact decoder) instead of QA start/end
pointers. If E4 ≈ D/E: "simpler BIO architecture, no QA pointer needed, at
no Joint-F1 cost." If E4 < D/E: QA-style span heads earn their complexity.

Architecture : mBERT encoder → [CLS] token → Linear(H,2) cls head
                              → all tokens  → Linear(H,3) BIO head
Loss         : cls_loss_weight * cls_CE + bio_loss_weight * bio_CE
               (bio_CE: per-token CE, O downweighted via --o_weight, first-
               subtoken-only supervision — identical scheme to System G)
Decode       : decode_bio_to_char_span (verbatim from BiO_Task_mBERT_train.py —
               the bug-fixed multi-subtoken-endword + SP-leading-space walk)
Output       : test_predictions.jsonl in the EXACT Train_Join.py schema
               (pred_idiomaticity, cls_correct, pred_span_start/end,
               pred_matched_span, span_exact_match, span_overlap_f1) so it
               registers via Full_evaluation.py's --joint_preds flag and is
               directly comparable to System E/D's Joint F1 — no eval changes.

No reported metric is hardcoded (CLAUDE.md rule): all numbers computed live.

Usage
-----
    # local smoke test (CPU/MPS, tiny)
    .venv/bin/python3 experiments/rigor/run_17_bio_cls_joint.py \
        --output_dir /tmp/_e4_dryrun --langs English --epochs 1 --batch_size 8

    # full run (Colab GPU)
    python experiments/rigor/run_17_bio_cls_joint.py \
        --output_dir models/bio_cls_joint_mbert \
        --langs English Spanish Hindi Telugu \
        --test_langs English Spanish Hindi Telugu Indonesian \
        --seed 42

    # register, directly comparable to System D/E Joint F1
    python Evaluation/Full_evaluation.py \
        --joint_preds models/bio_cls_joint_mbert/test_predictions.jsonl \
        --output_dir  results/e4_bio_cls_joint
"""

import json
import argparse
from pathlib import Path
from collections import defaultdict, Counter

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup
from sklearn.metrics import classification_report, f1_score
from tqdm import tqdm

try:
    import wandb
    WANDB = True
except ImportError:
    WANDB = False


# ── Label schemes ────────────────────────────────────────────────────────────
CLS_LABEL2ID = {'literal': 0, 'idiomatic': 1}
CLS_ID2LABEL = {0: 'literal', 1: 'idiomatic'}
BIO_LABEL2ID = {'O': 0, 'B-IDIOM': 1, 'I-IDIOM': 2}
BIO_ID2LABEL = {0: 'O', 1: 'B-IDIOM', 2: 'I-IDIOM'}
NUM_BIO_LABELS = 3
IGNORE_IDX = -100


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--model_name',   default='bert-base-multilingual-cased',
                   help='Same default encoder as Systems D/E/F — head architecture is the only variable.')
    p.add_argument('--data_dir',     default='data/idioms_structured/Splits')
    p.add_argument('--output_dir',   default='models/bio_cls_joint_mbert')
    p.add_argument('--langs',        nargs='+', default=['English', 'Spanish', 'Hindi', 'Telugu'])
    p.add_argument('--test_langs',   nargs='+', default=None,
                   help='Eval languages (defaults to --langs). Add Indonesian for the held-out table.')
    p.add_argument('--epochs',       type=int,   default=7)
    p.add_argument('--batch_size',   type=int,   default=32)
    p.add_argument('--lr',           type=float, default=2e-5)
    p.add_argument('--max_len',      type=int,   default=128)
    p.add_argument('--warmup_ratio', type=float, default=0.05)
    p.add_argument('--dropout',      type=float, default=0.1)
    p.add_argument('--cls_loss_weight', type=float, default=0.3,
                   help='Weight on classification loss — identical default to Train_Join.py (System E).')
    p.add_argument('--bio_loss_weight', type=float, default=1.9,
                   help='Weight on BIO span loss — identical default to Train_Join.py span_loss_weight.')
    p.add_argument('--o_weight',     type=float, default=0.104,
                   help='Loss weight for O token class within the BIO term (B/I weighted 1.0) — '
                        'identical default to System G.')
    p.add_argument('--seed',         type=int,   default=42)
    p.add_argument('--use_wandb',    action='store_true')
    p.add_argument('--device',       default=None)
    return p.parse_args()


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


# ── Data ──────────────────────────────────────────────────────────────────────

def load_split(data_dir, split_name, langs):
    path = Path(data_dir) / f'{split_name}.jsonl'
    if not path.exists():
        raise FileNotFoundError(f"Split file not found: {path}")
    langs_set = set(langs)
    examples = [json.loads(l) for l in open(path, encoding='utf-8')]
    examples = [r for r in examples if r['language'] in langs_set]
    print(f"  {split_name}: {len(examples)} examples")
    return examples


def compute_cell_weights(train_examples):
    """Inverse-frequency per (language, idiomaticity) cell, min-normalised to 1.0.
    Identical scheme to Stage 1 / System E / System G — shared across cls and BIO
    loss terms so E4 isn't confounded by a different weighting convention."""
    cell_counts = Counter((ex['language'], ex['idiomaticity']) for ex in train_examples)
    total, n_cells = sum(cell_counts.values()), len(cell_counts)
    cell_weights = {c: total / (n_cells * n) for c, n in cell_counts.items()}
    min_w = min(cell_weights.values())
    return {c: w / min_w for c, w in cell_weights.items()}


def compute_cls_class_weights(train_examples, cell_weights, device):
    class_weighted = defaultdict(float)
    class_counts = defaultdict(int)
    for ex in train_examples:
        cls = ex['idiomaticity']
        class_weighted[cls] += cell_weights[(ex['language'], cls)]
        class_counts[cls] += 1
    literal_w   = class_weighted['literal']   / class_counts['literal']
    idiomatic_w = class_weighted['idiomatic'] / class_counts['idiomatic']
    min_w = min(literal_w, idiomatic_w)
    weights = torch.tensor([literal_w / min_w, idiomatic_w / min_w], dtype=torch.float).to(device)
    print(f"\nCls loss weights — literal: {weights[0]:.4f}, idiomatic: {weights[1]:.4f}")
    return weights


def align_bio_labels(offsets, word_ids, char_start, char_end, max_len):
    """Verbatim from BiO_Task_mBERT_train.py — first-subtoken-only supervision,
    overlap-based in_span check (handles SP leading-space offset, harmless no-op
    on WordPiece). Returns None if span has no anchor."""
    if char_start is None or char_end is None:
        return None
    labels = []
    prev_word_id = None
    for offset, word_id in zip(offsets, word_ids):
        if word_id is None:
            labels.append(IGNORE_IDX)
            prev_word_id = word_id
            continue
        if word_id == prev_word_id:
            labels.append(IGNORE_IDX)
            prev_word_id = word_id
            continue
        tok_char_s, tok_char_e = offset
        in_span = min(tok_char_e, char_end) > max(tok_char_s, char_start)
        if in_span:
            if BIO_LABEL2ID['B-IDIOM'] not in labels:
                labels.append(BIO_LABEL2ID['B-IDIOM'])
            else:
                labels.append(BIO_LABEL2ID['I-IDIOM'])
        else:
            labels.append(BIO_LABEL2ID['O'])
        prev_word_id = word_id
    labels += [IGNORE_IDX] * (max_len - len(labels))
    return labels[:max_len]


class BIOClsJointDataset(Dataset):
    """Provides both a cls label and a per-token BIO label sequence per example."""

    def __init__(self, examples, tokenizer, max_len, cell_weights=None):
        self.valid_examples, self.input_ids, self.attention_masks, self.token_type_ids = [], [], [], []
        self.cls_labels, self.bio_labels, self.example_weights = [], [], []
        skipped = 0

        for ex in examples:
            cs, ce = ex['span_start'], ex['span_end']
            enc = tokenizer(ex['sentence'], max_length=max_len, padding='max_length',
                            truncation=True, return_offsets_mapping=True, return_tensors='pt')
            offsets = enc['offset_mapping'][0].tolist()
            word_ids = enc.word_ids(batch_index=0)
            labels = align_bio_labels(offsets, word_ids, cs, ce, max_len)

            if labels is None or BIO_LABEL2ID['B-IDIOM'] not in labels:
                skipped += 1
                continue

            tid = enc.get('token_type_ids')
            self.valid_examples.append(ex)
            self.input_ids.append(enc['input_ids'].squeeze(0))
            self.attention_masks.append(enc['attention_mask'].squeeze(0))
            self.token_type_ids.append(tid.squeeze(0) if tid is not None
                                       else torch.zeros(max_len, dtype=torch.long))
            self.cls_labels.append(CLS_LABEL2ID[ex['idiomaticity']])
            self.bio_labels.append(torch.tensor(labels, dtype=torch.long))
            w = cell_weights.get((ex['language'], ex['idiomaticity']), 1.0) if cell_weights else 1.0
            self.example_weights.append(w)

        if skipped:
            print(f"  Skipped {skipped} examples (no span anchor / alignment failed)")

    def __len__(self):
        return len(self.valid_examples)

    def __getitem__(self, i):
        return {
            'input_ids':      self.input_ids[i],
            'attention_mask': self.attention_masks[i],
            'token_type_ids': self.token_type_ids[i],
            'cls_labels':     torch.tensor(self.cls_labels[i], dtype=torch.long),
            'bio_labels':     self.bio_labels[i],
            'example_weight': torch.tensor(self.example_weights[i], dtype=torch.float),
        }


# ── Model ─────────────────────────────────────────────────────────────────────

class BIOClsJointModel(torch.nn.Module):
    """One mBERT encoder, two heads:
      [CLS] token → cls_head  (literal / idiomatic)
      all tokens  → bio_head  (O / B-IDIOM / I-IDIOM)
    Both heads updated jointly every step — the multi-task architecture
    System E uses, with a BIO span head substituted for QA start/end heads."""

    def __init__(self, model_name, dropout=0.1):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        h = self.bert.config.hidden_size
        self.dropout  = torch.nn.Dropout(dropout)
        self.cls_head = torch.nn.Linear(h, 2)
        self.bio_head = torch.nn.Linear(h, NUM_BIO_LABELS)

    def forward(self, input_ids, attention_mask, token_type_ids):
        out = self.bert(input_ids=input_ids, attention_mask=attention_mask,
                        token_type_ids=token_type_ids)
        seq = out.last_hidden_state.float()           # [B, T, H]
        cls_repr = seq[:, 0, :]                        # [CLS]
        cls_logits = self.cls_head(cls_repr)            # [B, 2]
        bio_logits = self.bio_head(self.dropout(seq))   # [B, T, 3]
        return cls_logits, bio_logits


# ── Decode (verbatim from BiO_Task_mBERT_train.py — System G's bug-fixed decoder) ──

def decode_bio_to_char_span(bio_preds, encoding, sentence, max_len):
    offsets  = encoding['offset_mapping']
    word_ids = encoding.word_ids()

    first_subtokens = []
    seen_words = set()
    for i, (label, wid) in enumerate(zip(bio_preds, word_ids)):
        if wid is None or wid in seen_words:
            continue
        seen_words.add(wid)
        first_subtokens.append((i, label))

    span_tokens, in_span = [], False
    for tok_idx, label in first_subtokens:
        if label == BIO_LABEL2ID['B-IDIOM']:
            span_tokens = [tok_idx]
            in_span = True
        elif label == BIO_LABEL2ID['I-IDIOM'] and in_span:
            span_tokens.append(tok_idx)
        else:
            if in_span:
                break

    if not span_tokens:
        return None, None

    first_tok, last_tok = span_tokens[0], span_tokens[-1]
    last_word_id = word_ids[last_tok]
    j = last_tok
    while j + 1 < len(word_ids) and word_ids[j + 1] == last_word_id:
        j += 1
    last_tok = j

    if first_tok >= len(offsets) or last_tok >= len(offsets):
        return None, None

    char_start = offsets[first_tok][0]
    char_end   = offsets[last_tok][1]
    if char_start is None or char_end is None:
        return None, None
    char_end = min(char_end, len(sentence))

    while char_start < char_end and sentence[char_start].isspace():
        char_start += 1
    return int(char_start), int(char_end)


def compute_overlap_f1(ps, pe, gs, ge):
    if ps is None or pe is None:
        return 0.0
    pred, gold = set(range(ps, pe)), set(range(gs, ge))
    if not pred or not gold:
        return 0.0
    ov = len(pred & gold)
    if ov == 0:
        return 0.0
    p, r = ov / len(pred), ov / len(gold)
    return 2 * p * r / (p + r)


# ── Save / load ───────────────────────────────────────────────────────────────

def save_model(model, tokenizer, output_dir):
    best = Path(output_dir) / 'best_model'
    best.mkdir(parents=True, exist_ok=True)
    model.bert.save_pretrained(best, safe_serialization=True)
    weights = list(best.glob("*.safetensors")) + list(best.glob("pytorch_model.bin"))
    if not weights:
        raise RuntimeError(f"Encoder save wrote no weights to {best}: {[p.name for p in best.iterdir()]}")
    tokenizer.save_pretrained(best)
    torch.save({'cls_head': model.cls_head.state_dict(),
                'bio_head': model.bio_head.state_dict()}, best / 'task_heads.pt')
    print(f"  Saved encoder ({sum(p.stat().st_size for p in weights)/1e6:.1f} MB) + heads + tokenizer")


def load_best_model(model_name, output_dir, device, dropout):
    best = Path(output_dir) / 'best_model'
    if not best.exists():
        print(f"  ⚠ best_model/ not found. Loading base {model_name}.")
        return BIOClsJointModel(model_name, dropout=dropout).to(device)
    model = BIOClsJointModel(str(best), dropout=dropout)
    heads = torch.load(best / 'task_heads.pt', map_location='cpu', weights_only=True)
    model.cls_head.load_state_dict(heads['cls_head'])
    model.bio_head.load_state_dict(heads['bio_head'])
    return model.to(device)


# ── Evaluation ────────────────────────────────────────────────────────────────

def evaluate(model, loader, tokenizer, examples, device, split_name, max_len):
    model.eval()
    all_cls_preds, all_cls_labels = [], []
    lang_exact, lang_f1 = defaultdict(list), defaultdict(list)
    lang_cls_preds, lang_cls_labels = defaultdict(list), defaultdict(list)
    all_true_bio, all_pred_bio = [], []

    with torch.no_grad():
        for bi, batch in enumerate(tqdm(loader, desc=f'Eval {split_name}', leave=False)):
            cls_logits, bio_logits = model(
                batch['input_ids'].to(device), batch['attention_mask'].to(device),
                batch['token_type_ids'].to(device),
            )
            cls_preds = torch.argmax(cls_logits, dim=-1).cpu().numpy()
            bio_preds = torch.argmax(bio_logits, dim=-1).cpu()
            gold_cls  = batch['cls_labels'].numpy()
            gold_bio  = batch['bio_labels']

            for i in range(len(cls_preds)):
                ex_idx = bi * loader.batch_size + i
                if ex_idx >= len(examples):
                    break
                ex   = examples[ex_idx]
                lang = ex['language']

                all_cls_preds.append(int(cls_preds[i]))
                all_cls_labels.append(int(gold_cls[i]))
                lang_cls_preds[lang].append(int(cls_preds[i]))
                lang_cls_labels[lang].append(int(gold_cls[i]))

                enc = tokenizer(ex['sentence'], max_length=max_len, truncation=True,
                                return_offsets_mapping=True)
                pred_bio_list = bio_preds[i].tolist()
                ps, pe = decode_bio_to_char_span(pred_bio_list, enc, ex['sentence'], max_len)
                gs, ge = ex['span_start'], ex['span_end']

                exact = int(ps == gs and pe == ge) if ps is not None else 0
                lang_exact[lang].append(exact)
                f1 = compute_overlap_f1(ps, pe, gs, ge)
                lang_f1[lang].append(f1)

                true_bio_list = gold_bio[i].tolist()
                for t, p in zip(true_bio_list, pred_bio_list):
                    if t != IGNORE_IDX:
                        all_true_bio.append(t)
                        all_pred_bio.append(p)

    cls_macro_f1 = f1_score(all_cls_labels, all_cls_preds, average='macro')
    all_exact   = [v for vs in lang_exact.values() for v in vs]
    all_overlap = [v for vs in lang_f1.values()    for v in vs]
    span_exact   = float(np.mean(all_exact))   if all_exact   else 0.0
    span_overlap = float(np.mean(all_overlap)) if all_overlap else 0.0

    print(f"\n── {split_name} results ──")
    print("Classification:")
    print(classification_report(all_cls_labels, all_cls_preds,
          target_names=['literal', 'idiomatic'], digits=4))
    print(f"Span — Exact: {span_exact:.4f}  Overlap F1: {span_overlap:.4f}")
    print("\nPer-language breakdown:")
    for lang in sorted(lang_exact.keys()):
        lf1 = f1_score(lang_cls_labels[lang], lang_cls_preds[lang], average='macro')
        print(f"  {lang:10s}  cls_macro_F1={lf1:.4f}  span_exact={np.mean(lang_exact[lang]):.4f}  "
              f"span_overlap_F1={np.mean(lang_f1[lang]):.4f}  ({len(lang_exact[lang])} examples)")
    print("\n── BIO Token Report ──")
    print(classification_report(all_true_bio, all_pred_bio,
          target_names=[BIO_ID2LABEL[i] for i in range(NUM_BIO_LABELS)], digits=4, zero_division=0))

    return cls_macro_f1, span_exact, span_overlap, all_cls_preds, all_cls_labels


# ── Train ─────────────────────────────────────────────────────────────────────

def train(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = get_device(args.device)
    print(f"Device: {device}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    json.dump(vars(args), open(output_dir / 'config.json', 'w'), indent=2)
    print(f"Languages: {args.langs}")
    print(f"Loss weights — cls: {args.cls_loss_weight}  bio: {args.bio_loss_weight}  o_weight: {args.o_weight}")

    train_ex = load_split(args.data_dir, 'train', args.langs)
    dev_ex   = load_split(args.data_dir, 'dev', args.langs)
    test_langs = args.test_langs or args.langs
    test_ex  = load_split(args.data_dir, 'test', test_langs)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if not tokenizer.is_fast:
        raise SystemExit("E4 needs a fast tokenizer (word_ids / offset_mapping required).")

    cell_weights = compute_cell_weights(train_ex)
    print("\nTokenizing + building BIO + cls labels...")
    train_ds = BIOClsJointDataset(train_ex, tokenizer, args.max_len, cell_weights)
    dev_ds   = BIOClsJointDataset(dev_ex,   tokenizer, args.max_len)
    test_ds  = BIOClsJointDataset(test_ex,  tokenizer, args.max_len)
    print(f"Train: {len(train_ds)} | Dev: {len(dev_ds)} | Test: {len(test_ds)}")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    dev_loader   = DataLoader(dev_ds,   batch_size=args.batch_size)
    test_loader  = DataLoader(test_ds,  batch_size=args.batch_size)

    model = BIOClsJointModel(args.model_name, dropout=args.dropout).to(device)

    cls_weights = compute_cls_class_weights(train_ds.valid_examples, cell_weights, device)
    cls_criterion = torch.nn.CrossEntropyLoss(weight=cls_weights)
    bio_class_weights = torch.tensor([args.o_weight, 1.0, 1.0], dtype=torch.float).to(device)
    bio_criterion = torch.nn.CrossEntropyLoss(weight=bio_class_weights, ignore_index=IGNORE_IDX, reduction='none')

    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    total_steps = len(train_loader) * args.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, int(total_steps * args.warmup_ratio), total_steps)

    if WANDB and args.use_wandb:
        wandb.init(project='idiom-bio-cls-joint', config=vars(args), name=output_dir.name)

    best_dev_joint_f1, best_epoch = 0.0, 0
    import math

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}", unit='batch')
        for batch in pbar:
            cls_labels = batch['cls_labels'].to(device)
            bio_labels = batch['bio_labels'].to(device)
            ex_weights = batch['example_weight'].to(device)

            cls_logits, bio_logits = model(
                batch['input_ids'].to(device), batch['attention_mask'].to(device),
                batch['token_type_ids'].to(device),
            )

            cls_loss = cls_criterion(cls_logits, cls_labels)

            tok_loss = bio_criterion(bio_logits.view(-1, NUM_BIO_LABELS), bio_labels.view(-1))
            tok_loss = tok_loss.view(bio_labels.shape[0], -1)
            valid = (bio_labels != IGNORE_IDX).float()
            per_ex_bio = (tok_loss * valid).sum(dim=1) / valid.sum(dim=1).clamp(min=1)
            bio_loss = (per_ex_bio * ex_weights).mean()

            loss = args.cls_loss_weight * cls_loss + args.bio_loss_weight * bio_loss

            if torch.isnan(loss) or torch.isinf(loss):
                print(f"  ⚠ NaN/inf loss (cls={cls_loss.item():.4f} bio={bio_loss.item():.4f}) — skipping batch")
                optimizer.zero_grad()
                continue

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step(); scheduler.step(); optimizer.zero_grad()

            total_loss += loss.item()
            pbar.set_postfix({'loss': f'{loss.item():.4f}', 'cls': f'{cls_loss.item():.4f}', 'bio': f'{bio_loss.item():.4f}'})

        avg_loss = total_loss / len(train_loader)
        print(f"\nEpoch {epoch} avg loss: {avg_loss:.4f}")

        dev_cls_f1, dev_exact, dev_overlap, _, _ = evaluate(
            model, dev_loader, tokenizer, dev_ds.valid_examples, device, f'Dev (epoch {epoch})', args.max_len)

        dev_joint_f1 = math.sqrt(dev_cls_f1 * dev_overlap) if (dev_cls_f1 > 0 and dev_overlap > 0) else 0.0
        print(f"  Dev joint F1 (geomean): {dev_joint_f1:.4f}  (cls={dev_cls_f1:.4f}, span_overlap={dev_overlap:.4f})")

        if WANDB and args.use_wandb:
            wandb.log({'epoch': epoch, 'train_loss': avg_loss, 'dev_cls_f1': dev_cls_f1,
                       'dev_span_exact': dev_exact, 'dev_span_f1': dev_overlap, 'dev_joint_f1': dev_joint_f1})

        if dev_joint_f1 > best_dev_joint_f1:
            best_dev_joint_f1, best_epoch = dev_joint_f1, epoch
            save_model(model, tokenizer, output_dir)
            print(f"  ✓ New best model saved (dev joint F1: {best_dev_joint_f1:.4f})")

    print(f"\nBest dev joint F1: {best_dev_joint_f1:.4f} at epoch {best_epoch}")
    print("\nLoading best model for test evaluation...")
    best_model = load_best_model(args.model_name, output_dir, device, args.dropout)

    test_cls_f1, test_exact, test_overlap, _, _ = evaluate(
        best_model, test_loader, tokenizer, test_ds.valid_examples, device, 'Test (final)', args.max_len)

    # Predictions in Train_Join.py's exact schema → registers via --joint_preds.
    best_model.eval()
    preds_out = []
    with torch.no_grad():
        for bi, batch in enumerate(test_loader):
            cls_logits, bio_logits = best_model(
                batch['input_ids'].to(device), batch['attention_mask'].to(device),
                batch['token_type_ids'].to(device),
            )
            cls_preds = torch.argmax(cls_logits, dim=-1).cpu().numpy()
            bio_preds = torch.argmax(bio_logits, dim=-1).cpu()
            gold_cls  = batch['cls_labels'].numpy()

            for i in range(len(cls_preds)):
                ex_idx = bi * test_loader.batch_size + i
                if ex_idx >= len(test_ds.valid_examples):
                    break
                ex = test_ds.valid_examples[ex_idx]
                enc = tokenizer(ex['sentence'], max_length=args.max_len, truncation=True,
                                return_offsets_mapping=True)
                pred_bio_list = bio_preds[i].tolist()
                ps, pe = decode_bio_to_char_span(pred_bio_list, enc, ex['sentence'], args.max_len)
                if ps is None:
                    ps, pe = 0, 0
                gs, ge = ex['span_start'], ex['span_end']

                preds_out.append({
                    **ex,
                    'pred_idiomaticity':  CLS_ID2LABEL[int(cls_preds[i])],
                    'cls_correct':        bool(int(cls_preds[i]) == int(gold_cls[i])),
                    'pred_span_start':    ps,
                    'pred_span_end':      pe,
                    'pred_matched_span':  ex['sentence'][ps:pe],
                    'pred_bio_tags':      [BIO_ID2LABEL.get(l, 'O') for l in pred_bio_list if l != IGNORE_IDX],
                    'span_exact_match':   bool(ps == gs and pe == ge),
                    'span_overlap_f1':    round(compute_overlap_f1(ps, pe, gs, ge), 4),
                })

    preds_path = output_dir / 'test_predictions.jsonl'
    with open(preds_path, 'w', encoding='utf-8') as f:
        for p in preds_out:
            f.write(json.dumps(p, ensure_ascii=False) + '\n')
    print(f"Predictions saved → {preds_path}")

    json.dump({
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
        'bio_loss_weight':   args.bio_loss_weight,
        'o_weight':          args.o_weight,
        'train_size':        len(train_ds),
        'dev_size':          len(dev_ds),
        'test_size':         len(test_ds),
    }, open(output_dir / 'metrics.json', 'w'), indent=2)
    print(f"Metrics saved → {output_dir / 'metrics.json'}")

    if WANDB and args.use_wandb:
        wandb.log({'test_cls_f1': test_cls_f1, 'test_span_exact': test_exact, 'test_span_f1': test_overlap})
        wandb.finish()


if __name__ == '__main__':
    train(parse_args())
