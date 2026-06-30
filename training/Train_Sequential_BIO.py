"""
Train_Sequential_BIO.py

Sequential two-phase training where Phase 2 uses a BIO tagger head instead
of QA pointer heads. Completes the 2×2 matrix:

                    | Single-pass | Sequential/Pipeline |
  QA formulation   |  E (joint)  |  F/A/D              |
  BIO formulation  |  E4 ≈ E4bio |  THIS SCRIPT        |

Phase 1 — identical to Train_Sequential.py Phase 1 (cls-dominant, QA span head,
           saved on best dev cls macro F1).
           Output: models/sequential_bio/s{seed}/phase1/

Phase 2 — BIO span head initialized from Phase 1 encoder.
           Bottom layers + no QA heads (discarded). Top N encoder layers +
           BIO head unfrozen. Saved on best dev span overlap F1.
           Output: models/sequential_bio/s{seed}/phase2_bio/

At inference:
  - Classification  → Phase 1 model (cls head)
  - Span extraction → Phase 2 model (BIO head)

Prediction outputs are in the format expected by Full_evaluation.py:
  Phase 1 → test_predictions.jsonl  (pred_idiomaticity field)
  Phase 2 → test_predictions.jsonl  (pred_span_start, pred_span_end fields)

Joint evaluation:
  python Evaluation/Full_evaluation.py \\
      --seed 42 \\
      --seq_phase1 models/sequential_bio/s42/phase1/test_predictions.jsonl \\
      --seq_phase2 models/sequential_bio/s42/phase2_bio/test_predictions.jsonl

Usage:
    python training/Train_Sequential_BIO.py --seed 42
    python training/Train_Sequential_BIO.py --seed 42 --phase 1
    python training/Train_Sequential_BIO.py --seed 42 --phase 2
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

# ── Shared label schemes ───────────────────────────────────────────────────────

CLS_LABEL2ID = {'literal': 0, 'idiomatic': 1}
CLS_ID2LABEL = {0: 'literal', 1: 'idiomatic'}

BIO_LABEL2ID = {'O': 0, 'B-IDIOM': 1, 'I-IDIOM': 2}
BIO_ID2LABEL = {0: 'O', 1: 'B-IDIOM', 2: 'I-IDIOM'}
NUM_BIO_LABELS = 3
IGNORE_IDX = -100


# ── Args ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--model_name',          default='bert-base-multilingual-cased')
    p.add_argument('--data_dir',            default='data/idioms_structured/Splits')
    p.add_argument('--dropout',  type=float, default=0.1)
    p.add_argument('--output_dir',          default='models/sequential_bio')
    p.add_argument('--langs',               nargs='+', default=['English', 'Hindi', 'Telugu', 'Spanish'])
    p.add_argument('--test_langs',          nargs='+', default=None)
    p.add_argument('--phase',    type=int,  default=0,
                   help='1=phase1 only, 2=phase2 only, 0=both')
    # Phase 1 hyperparams (same as Train_Sequential.py)
    p.add_argument('--p1_epochs',       type=int,   default=7)
    p.add_argument('--p1_batch_size',   type=int,   default=32)
    p.add_argument('--p1_lr',           type=float, default=1e-5)
    p.add_argument('--p1_warmup_ratio', type=float, default=0.1)
    p.add_argument('--p1_cls_weight',   type=float, default=0.7)
    p.add_argument('--p1_span_weight',  type=float, default=0.3)
    # Phase 2 BIO hyperparams
    p.add_argument('--p2_epochs',       type=int,   default=6)
    p.add_argument('--p2_batch_size',   type=int,   default=32)
    p.add_argument('--p2_lr',           type=float, default=3.27e-5)
    p.add_argument('--p2_warmup_ratio', type=float, default=0.096)
    p.add_argument('--p2_o_weight',     type=float, default=0.104,
                   help='Loss weight for O class in BIO (B/I weighted 1.0)')
    p.add_argument('--unfreeze_top_layers', type=int, default=3)
    p.add_argument('--max_len',         type=int,   default=128)
    p.add_argument('--seed',            type=int,   default=42)
    p.add_argument('--use_wandb',       action='store_true')
    p.add_argument('--device',          default=None)
    return p.parse_args()


# ── Device ────────────────────────────────────────────────────────────────────

def get_device(forced=None):
    if forced:
        return torch.device(forced)
    if torch.cuda.is_available():
        return torch.device('cuda')
    if torch.backends.mps.is_available():
        print("MPS detected — precompiling...")
        d = torch.device('mps')
        _ = torch.zeros(1, device=d) + 1
        return d
    return torch.device('cpu')


# ── Data ──────────────────────────────────────────────────────────────────────

def load_split(data_dir, split_name, langs):
    langs_set = set(langs)
    path = Path(data_dir) / f'{split_name}.jsonl'
    return [
        json.loads(l) for l in open(path, encoding='utf-8')
        if json.loads(l)['language'] in langs_set
    ]


# ── Span helpers ──────────────────────────────────────────────────────────────

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


def decode_bio_to_char_span(bio_preds, encoding, sentence, max_len):
    """Convert predicted BIO label IDs → character offsets. Same logic as BiO_Task_mBERT_train.py."""
    offsets  = encoding['offset_mapping']
    word_ids = encoding.word_ids()

    first_subtokens = []
    seen_words = set()
    for i, (label, wid) in enumerate(zip(bio_preds, word_ids)):
        if wid is None:
            continue
        if wid in seen_words:
            continue
        seen_words.add(wid)
        first_subtokens.append((i, label))

    span_tokens = []
    in_span = False
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

    first_tok = span_tokens[0]
    last_tok  = span_tokens[-1]

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


# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 1 — Classification-dominant (identical to Train_Sequential.py Phase 1)
# ═══════════════════════════════════════════════════════════════════════════════

class QADataset(Dataset):
    """Joint cls + QA span dataset (used by Phase 1)."""

    def __init__(self, examples, tokenizer, max_len):
        self.valid_examples  = []
        self.input_ids       = []
        self.attention_masks = []
        self.token_type_ids  = []
        self.cls_labels      = []
        self.start_positions = []
        self.end_positions   = []

        skipped = 0
        for ex in examples:
            sentence   = ex['sentence']
            char_start = ex['span_start']
            char_end   = ex['span_end']

            encoding = tokenizer(sentence, max_length=max_len, padding='max_length',
                                 truncation=True, return_tensors='pt')
            enc_map  = tokenizer(sentence, max_length=max_len, truncation=True,
                                 return_offsets_mapping=True)

            token_start, token_end = char_to_token_span(enc_map, char_start, char_end, sentence)
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
            self.cls_labels.append(CLS_LABEL2ID[ex['idiomaticity']])
            self.start_positions.append(token_start)
            self.end_positions.append(token_end)

        if skipped:
            print(f"  Skipped {skipped} examples with failed span alignment")

    def __len__(self): return len(self.valid_examples)

    def __getitem__(self, idx):
        return {
            'input_ids':       self.input_ids[idx],
            'attention_mask':  self.attention_masks[idx],
            'token_type_ids':  self.token_type_ids[idx],
            'cls_labels':      torch.tensor(self.cls_labels[idx],      dtype=torch.long),
            'start_positions': torch.tensor(self.start_positions[idx], dtype=torch.long),
            'end_positions':   torch.tensor(self.end_positions[idx],   dtype=torch.long),
        }


class JointIdiomModel(torch.nn.Module):
    def __init__(self, model_name, dropout=0.1):
        super().__init__()
        self.bert       = AutoModel.from_pretrained(model_name)
        hidden_size     = self.bert.config.hidden_size
        self.drop       = torch.nn.Dropout(dropout)
        self.cls_head   = torch.nn.Linear(hidden_size, 2)
        self.start_head = torch.nn.Linear(hidden_size, 1)
        self.end_head   = torch.nn.Linear(hidden_size, 1)

    def forward(self, input_ids, attention_mask, token_type_ids):
        outputs      = self.bert(input_ids=input_ids, attention_mask=attention_mask,
                                 token_type_ids=token_type_ids)
        seq_output   = self.drop(outputs.last_hidden_state)
        cls_output   = seq_output[:, 0, :]
        cls_logits   = self.cls_head(cls_output)
        start_logits = self.start_head(seq_output).squeeze(-1)
        end_logits   = self.end_head(seq_output).squeeze(-1)
        mask         = attention_mask.bool()
        start_logits = start_logits.masked_fill(~mask, float('-inf'))
        end_logits   = end_logits.masked_fill(~mask,   float('-inf'))
        return cls_logits, start_logits, end_logits


def compute_cls_loss_weights(examples, device):
    cell_counts = Counter((ex['language'], ex['idiomaticity']) for ex in examples)
    total       = sum(cell_counts.values())
    n_cells     = len(cell_counts)
    cell_w      = {c: total / (n_cells * n) for c, n in cell_counts.items()}

    class_w = defaultdict(float)
    class_n = defaultdict(int)
    for ex in examples:
        cls = ex['idiomaticity']
        class_w[cls] += cell_w[(ex['language'], cls)]
        class_n[cls] += 1

    lit_w = class_w['literal']   / class_n['literal']
    idi_w = class_w['idiomatic'] / class_n['idiomatic']
    min_w = min(lit_w, idi_w)
    weights = torch.tensor([lit_w / min_w, idi_w / min_w], dtype=torch.float).to(device)
    print(f"  Cls loss weights — literal: {weights[0]:.4f}, idiomatic: {weights[1]:.4f}")
    return weights


def save_qa_model(model, tokenizer, output_dir):
    best = Path(output_dir) / 'best_model'
    best.mkdir(parents=True, exist_ok=True)
    model.bert.save_pretrained(best)
    tokenizer.save_pretrained(best)
    torch.save({
        'cls_head':   model.cls_head.state_dict(),
        'start_head': model.start_head.state_dict(),
        'end_head':   model.end_head.state_dict(),
    }, best / 'task_heads.pt')


def load_qa_model(model_name, output_dir, device, dropout=0.1):
    best  = Path(output_dir) / 'best_model'
    model = JointIdiomModel(model_name, dropout=dropout)
    model.bert = AutoModel.from_pretrained(best)
    heads = torch.load(best / 'task_heads.pt', map_location='cpu', weights_only=True)
    model.cls_head.load_state_dict(heads['cls_head'])
    model.start_head.load_state_dict(heads['start_head'])
    model.end_head.load_state_dict(heads['end_head'])
    return model.to(device)


def token_to_char_span(tokenizer, sentence, token_start, token_end, max_len):
    enc     = tokenizer(sentence, max_length=max_len, truncation=True,
                        return_offsets_mapping=True)
    offsets = enc['offset_mapping']
    if token_start >= len(offsets) or token_end >= len(offsets):
        return None, None
    return offsets[token_start][0], offsets[token_end][1]


def evaluate_phase1(model, loader, tokenizer, examples, device, label, max_len):
    model.eval()
    all_cls_preds, all_cls_labels = [], []
    exact_matches = 0
    overlap_f1s   = []
    lang_exact    = defaultdict(list)
    lang_f1       = defaultdict(list)
    lang_cls_p    = defaultdict(list)
    lang_cls_l    = defaultdict(list)
    total = 0

    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(loader, desc=f'Eval {label}', leave=False)):
            cls_logits, start_logits, end_logits = model(
                batch['input_ids'].to(device),
                batch['attention_mask'].to(device),
                batch['token_type_ids'].to(device),
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
                if ex_idx >= len(examples): break
                lang = examples[ex_idx]['language']

                all_cls_preds.append(int(cls_preds[i]))
                all_cls_labels.append(int(gold_cls[i]))
                lang_cls_p[lang].append(int(cls_preds[i]))
                lang_cls_l[lang].append(int(gold_cls[i]))

                pred_s = int(pred_starts[i])
                pred_e = int(pred_ends[i])
                gold_s = int(gold_starts[i])
                gold_e = int(gold_ends[i])
                if pred_e < pred_s: pred_e = pred_s

                exact = int(pred_s == gold_s and pred_e == gold_e)
                exact_matches += exact
                lang_exact[lang].append(exact)
                f1 = compute_overlap_f1(pred_s, pred_e, gold_s, gold_e)
                overlap_f1s.append(f1)
                lang_f1[lang].append(f1)
                total += 1

    macro_f1    = f1_score(all_cls_labels, all_cls_preds, average='macro')
    exact_match = exact_matches / total if total > 0 else 0
    avg_overlap = np.mean(overlap_f1s) if overlap_f1s else 0

    print(f"\n── {label} ──")
    print(classification_report(all_cls_labels, all_cls_preds,
                                  target_names=['literal', 'idiomatic'], digits=4))
    print(f"Span — Exact: {exact_match:.4f}  Overlap F1: {avg_overlap:.4f}  N={total}")
    for lang in sorted(lang_exact):
        lf1 = f1_score(lang_cls_l[lang], lang_cls_p[lang], average='macro')
        le  = np.mean(lang_exact[lang])
        lf  = np.mean(lang_f1[lang])
        print(f"  {lang:10s}  cls={lf1:.4f}  exact={le:.4f}  overlap={lf:.4f}")

    return macro_f1, exact_match, avg_overlap


def write_phase1_predictions(model, loader, tokenizer, examples, device, max_len, out_path):
    model.eval()
    preds_out = []
    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            cls_logits, start_logits, end_logits = model(
                batch['input_ids'].to(device),
                batch['attention_mask'].to(device),
                batch['token_type_ids'].to(device),
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
                if ex_idx >= len(examples): break
                ex = examples[ex_idx]

                pred_s = int(pred_starts[i])
                pred_e = int(pred_ends[i])
                if pred_e < pred_s: pred_e = pred_s

                pred_char_s, pred_char_e = token_to_char_span(
                    tokenizer, ex['sentence'], pred_s, pred_e, max_len
                )
                preds_out.append({
                    **ex,
                    'pred_idiomaticity': CLS_ID2LABEL[int(cls_preds[i])],
                    'cls_correct':       bool(int(cls_preds[i]) == int(gold_cls[i])),
                    'pred_span_start':   pred_char_s,
                    'pred_span_end':     pred_char_e,
                })

    with open(out_path, 'w', encoding='utf-8') as f:
        for p in preds_out:
            f.write(json.dumps(p, ensure_ascii=False) + '\n')
    print(f"Phase 1 predictions → {out_path}")


def phase1(args, device, tokenizer, train_ds, dev_ds, test_ds, output_dir):
    print(f"\n{'='*70}")
    print(f"PHASE 1 — Classification-dominant (same as System F)")
    print(f"  cls={args.p1_cls_weight}  span={args.p1_span_weight}  "
          f"epochs={args.p1_epochs}  lr={args.p1_lr}  batch={args.p1_batch_size}")
    print(f"{'='*70}")

    output_dir = Path(output_dir) / 'phase1'
    output_dir.mkdir(parents=True, exist_ok=True)

    train_loader = DataLoader(train_ds, batch_size=args.p1_batch_size, shuffle=True)
    dev_loader   = DataLoader(dev_ds,   batch_size=args.p1_batch_size)
    test_loader  = DataLoader(test_ds,  batch_size=args.p1_batch_size)

    model       = JointIdiomModel(args.model_name, dropout=args.dropout).to(device)
    cls_weights = compute_cls_loss_weights(train_ds.valid_examples, device)
    cls_crit    = torch.nn.CrossEntropyLoss(weight=cls_weights)
    span_crit   = torch.nn.CrossEntropyLoss()

    optimizer    = AdamW(model.parameters(), lr=args.p1_lr, weight_decay=0.01)
    total_steps  = len(train_loader) * args.p1_epochs
    warmup_steps = int(total_steps * args.p1_warmup_ratio)
    scheduler    = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    best_score, best_epoch = 0.0, 0
    for epoch in range(1, args.p1_epochs + 1):
        model.train()
        total_loss = 0.0
        pbar = tqdm(train_loader, desc=f"[Phase1] Epoch {epoch}/{args.p1_epochs}", unit='batch')
        for batch in pbar:
            cls_logits, start_logits, end_logits = model(
                batch['input_ids'].to(device),
                batch['attention_mask'].to(device),
                batch['token_type_ids'].to(device),
            )
            cls_loss  = cls_crit(cls_logits, batch['cls_labels'].to(device))
            span_loss = (span_crit(start_logits, batch['start_positions'].to(device)) +
                         span_crit(end_logits,   batch['end_positions'].to(device))) / 2
            loss = args.p1_cls_weight * cls_loss + args.p1_span_weight * span_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step(); scheduler.step(); optimizer.zero_grad()
            total_loss += loss.item()
            pbar.set_postfix({'loss': f'{loss.item():.4f}', 'cls': f'{cls_loss.item():.4f}'})

        print(f"\n[Phase1] Epoch {epoch} avg loss: {total_loss/len(train_loader):.4f}")
        dev_cls_f1, _, _ = evaluate_phase1(
            model, dev_loader, tokenizer, dev_ds.valid_examples,
            device, f'Dev epoch {epoch}', args.max_len
        )
        if dev_cls_f1 > best_score:
            best_score = dev_cls_f1
            best_epoch = epoch
            save_qa_model(model, tokenizer, output_dir)
            print(f"  ✓ [Phase1] New best cls_f1={best_score:.4f} at epoch {epoch}")

    print(f"\n[Phase1] Best dev cls_f1: {best_score:.4f} at epoch {best_epoch}")

    print("\n[Phase1] Loading best model for test evaluation...")
    best_model = load_qa_model(args.model_name, output_dir, device, dropout=args.dropout)
    test_cls_f1, test_exact, test_overlap = evaluate_phase1(
        best_model, test_loader, tokenizer, test_ds.valid_examples,
        device, 'Test Phase1', args.max_len
    )
    write_phase1_predictions(
        best_model, test_loader, tokenizer, test_ds.valid_examples,
        device, args.max_len, output_dir / 'test_predictions.jsonl'
    )
    json.dump({
        'phase': 1, 'save_criterion': 'cls_f1',
        'best_dev_cls_f1': best_score, 'best_epoch': best_epoch,
        'test_cls_macro_f1': test_cls_f1,
        'test_span_exact': test_exact, 'test_span_overlap': test_overlap,
        'model': args.model_name, 'langs': args.langs,
    }, open(output_dir / 'metrics.json', 'w'), indent=2)
    print(f"[Phase1] Metrics → {output_dir / 'metrics.json'}")

    return output_dir


# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 2 — BIO span head initialized from Phase 1 encoder
# ═══════════════════════════════════════════════════════════════════════════════

class BIODataset(Dataset):
    """Per-token BIO labels derived from character span annotations."""

    def __init__(self, examples, tokenizer, max_len):
        self.valid_examples  = []
        self.input_ids       = []
        self.attention_masks = []
        self.token_type_ids  = []
        self.bio_labels      = []

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
                return_tensors='pt',
                return_offsets_mapping=True,
            )
            word_ids = encoding.word_ids(batch_index=0)
            offsets  = encoding['offset_mapping'][0].tolist()

            labels   = []
            prev_wid = None
            for wid, (cs, ce) in zip(word_ids, offsets):
                if wid is None:
                    labels.append(IGNORE_IDX)  # [CLS]/[SEP]/padding
                    prev_wid = wid
                    continue
                # Subtoken of same word as previous → inherit label, mark as I or O
                if wid == prev_wid:
                    labels.append(IGNORE_IDX)  # only first subtoken gets a label
                    continue
                # First subtoken of this word
                if cs >= char_start and ce <= char_end:
                    # Within gold span
                    if cs == char_start:
                        labels.append(BIO_LABEL2ID['B-IDIOM'])
                    else:
                        labels.append(BIO_LABEL2ID['I-IDIOM'])
                elif cs < char_end and ce > char_start:
                    # Partial overlap (subword boundary crossed idiom boundary)
                    labels.append(BIO_LABEL2ID['I-IDIOM'])
                else:
                    labels.append(BIO_LABEL2ID['O'])
                prev_wid = wid

            # Pad remaining
            while len(labels) < max_len:
                labels.append(IGNORE_IDX)
            labels = labels[:max_len]

            # Verify at least one B-IDIOM assigned
            if BIO_LABEL2ID['B-IDIOM'] not in labels:
                skipped += 1
                continue

            self.valid_examples.append(ex)
            ids  = encoding['input_ids'].squeeze(0)
            mask = encoding['attention_mask'].squeeze(0)
            self.input_ids.append(ids)
            self.attention_masks.append(mask)
            tid = encoding.get('token_type_ids')
            self.token_type_ids.append(
                tid.squeeze(0) if tid is not None
                else torch.zeros(max_len, dtype=torch.long)
            )
            self.bio_labels.append(torch.tensor(labels, dtype=torch.long))

        if skipped:
            print(f"  BIODataset: skipped {skipped} examples with no B-IDIOM alignment")

    def __len__(self): return len(self.valid_examples)

    def __getitem__(self, idx):
        return {
            'input_ids':      self.input_ids[idx],
            'attention_mask': self.attention_masks[idx],
            'token_type_ids': self.token_type_ids[idx],
            'bio_labels':     self.bio_labels[idx],
        }


class SeqBIOModel(torch.nn.Module):
    """Phase 1 encoder + BIO tagger head. Initialized from Phase 1 bert weights."""

    def __init__(self, encoder, dropout=0.1):
        super().__init__()
        self.bert     = encoder
        hidden        = self.bert.config.hidden_size
        self.drop     = torch.nn.Dropout(dropout)
        self.bio_head = torch.nn.Linear(hidden, NUM_BIO_LABELS)

    def forward(self, input_ids, attention_mask, token_type_ids):
        out    = self.bert(input_ids=input_ids, attention_mask=attention_mask,
                           token_type_ids=token_type_ids)
        hidden = self.drop(out.last_hidden_state)
        return self.bio_head(hidden)  # [B, T, 3]


def freeze_for_phase2_bio(model, unfreeze_top_n):
    for param in model.parameters():
        param.requires_grad = False
    for param in model.bio_head.parameters():
        param.requires_grad = True
    layers      = model.bert.encoder.layer
    n_total     = len(layers)
    for i in range(n_total - unfreeze_top_n, n_total):
        for param in layers[i].parameters():
            param.requires_grad = True
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    print(f"  Phase2 BIO frozen: {total - trainable:,} | trainable: {trainable:,} "
          f"({100*trainable/total:.1f}%) — bio_head + top {unfreeze_top_n} encoder layers")


def save_bio_model(model, tokenizer, output_dir):
    best = Path(output_dir) / 'best_model'
    best.mkdir(parents=True, exist_ok=True)
    model.bert.save_pretrained(best)
    tokenizer.save_pretrained(best)
    torch.save(model.bio_head.state_dict(), best / 'bio_head.pt')


def load_bio_model(output_dir, device, dropout=0.1):
    best    = Path(output_dir) / 'best_model'
    encoder = AutoModel.from_pretrained(best)
    model   = SeqBIOModel(encoder, dropout=dropout)
    model.bio_head.load_state_dict(
        torch.load(best / 'bio_head.pt', map_location='cpu', weights_only=True)
    )
    return model.to(device)


def evaluate_phase2_bio(model, loader, tokenizer, examples, device, label, max_len):
    model.eval()
    lang_exact   = defaultdict(list)
    lang_overlap = defaultdict(list)

    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(loader, desc=f'Eval {label}', leave=False)):
            logits = model(
                batch['input_ids'].to(device),
                batch['attention_mask'].to(device),
                batch['token_type_ids'].to(device),
            )
            preds = torch.argmax(logits, dim=-1).cpu().numpy()

            batch_start = batch_idx * loader.batch_size
            for i in range(len(preds)):
                ex_idx = batch_start + i
                if ex_idx >= len(examples): break
                ex       = examples[ex_idx]
                sentence = ex['sentence']
                lang     = ex['language']
                gold_s   = ex['span_start']
                gold_e   = ex['span_end']

                pred_bio = preds[i].tolist()
                enc = tokenizer(sentence, max_length=max_len, truncation=True,
                                return_offsets_mapping=True)
                pred_char_s, pred_char_e = decode_bio_to_char_span(pred_bio, enc, sentence, max_len)

                exact   = int(pred_char_s == gold_s and pred_char_e == gold_e) \
                          if pred_char_s is not None else 0
                overlap = compute_overlap_f1(pred_char_s, pred_char_e, gold_s, gold_e)
                lang_exact[lang].append(exact)
                lang_overlap[lang].append(overlap)

    all_exact   = [v for vals in lang_exact.values()   for v in vals]
    all_overlap = [v for vals in lang_overlap.values() for v in vals]

    print(f"\n── {label} BIO Span ──")
    print(f"  {'Language':<12} {'Exact':<10} {'Overlap F1'}")
    for lang in sorted(lang_exact):
        print(f"  {lang:<12} {np.mean(lang_exact[lang]):<10.4f} {np.mean(lang_overlap[lang]):.4f}"
              f"  (n={len(lang_exact[lang])})")
    print(f"  {'Overall':<12} {np.mean(all_exact):<10.4f} {np.mean(all_overlap):.4f}")

    return np.mean(all_exact), np.mean(all_overlap)


def write_phase2_bio_predictions(model, loader, tokenizer, examples, device, max_len, out_path):
    model.eval()
    preds_out = []
    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            logits = model(
                batch['input_ids'].to(device),
                batch['attention_mask'].to(device),
                batch['token_type_ids'].to(device),
            )
            preds = torch.argmax(logits, dim=-1).cpu().numpy()

            batch_start = batch_idx * loader.batch_size
            for i in range(len(preds)):
                ex_idx = batch_start + i
                if ex_idx >= len(examples): break
                ex       = examples[ex_idx]
                sentence = ex['sentence']

                pred_bio = preds[i].tolist()
                enc = tokenizer(sentence, max_length=max_len, truncation=True,
                                return_offsets_mapping=True)
                pred_char_s, pred_char_e = decode_bio_to_char_span(pred_bio, enc, sentence, max_len)

                exact   = int(pred_char_s == ex['span_start'] and pred_char_e == ex['span_end']) \
                          if pred_char_s is not None else 0
                overlap = compute_overlap_f1(pred_char_s, pred_char_e, ex['span_start'], ex['span_end'])

                preds_out.append({
                    **ex,
                    'pred_span_start':   pred_char_s,
                    'pred_span_end':     pred_char_e,
                    'span_exact_match':  bool(exact),
                    'span_overlap_f1':   round(overlap, 4),
                    'pred_bio_tags':     [BIO_ID2LABEL.get(l, 'O') for l in pred_bio
                                         if l != IGNORE_IDX],
                })

    with open(out_path, 'w', encoding='utf-8') as f:
        for p in preds_out:
            f.write(json.dumps(p, ensure_ascii=False) + '\n')
    print(f"Phase2 BIO predictions → {out_path}")


def phase2_bio(args, device, tokenizer, train_ds, dev_ds, test_ds,
               phase1_dir, output_dir):
    print(f"\n{'='*70}")
    print(f"PHASE 2 BIO — BIO tagger initialized from Phase 1 encoder")
    print(f"  lr={args.p2_lr}  epochs={args.p2_epochs}  "
          f"batch={args.p2_batch_size}  o_weight={args.p2_o_weight}")
    print(f"  Unfreezing top {args.unfreeze_top_layers} encoder layers + bio_head")
    print(f"{'='*70}")

    output_dir = Path(output_dir) / 'phase2_bio'
    output_dir.mkdir(parents=True, exist_ok=True)

    train_loader = DataLoader(train_ds, batch_size=args.p2_batch_size, shuffle=True)
    dev_loader   = DataLoader(dev_ds,   batch_size=args.p2_batch_size)
    test_loader  = DataLoader(test_ds,  batch_size=args.p2_batch_size)

    # Load Phase 1 encoder, discard QA heads, attach BIO head
    print(f"\n[Phase2 BIO] Loading Phase 1 encoder from {phase1_dir}...")
    p1_model = load_qa_model(args.model_name, phase1_dir, device, dropout=args.dropout)
    model    = SeqBIOModel(p1_model.bert, dropout=args.dropout).to(device)
    del p1_model

    freeze_for_phase2_bio(model, args.unfreeze_top_layers)

    # BIO loss: downweight O class to focus on span tokens
    bio_weights = torch.tensor(
        [args.p2_o_weight, 1.0, 1.0], dtype=torch.float
    ).to(device)
    bio_crit    = torch.nn.CrossEntropyLoss(weight=bio_weights, ignore_index=IGNORE_IDX)

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer    = AdamW(trainable_params, lr=args.p2_lr, weight_decay=0.01)
    total_steps  = len(train_loader) * args.p2_epochs
    warmup_steps = int(total_steps * args.p2_warmup_ratio)
    scheduler    = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    best_score, best_epoch = 0.0, 0
    for epoch in range(1, args.p2_epochs + 1):
        model.train()
        total_loss = 0.0
        pbar = tqdm(train_loader, desc=f"[Phase2 BIO] Epoch {epoch}/{args.p2_epochs}", unit='batch')
        for batch in pbar:
            logits = model(
                batch['input_ids'].to(device),
                batch['attention_mask'].to(device),
                batch['token_type_ids'].to(device),
            )  # [B, T, 3]
            loss = bio_crit(
                logits.view(-1, NUM_BIO_LABELS),
                batch['bio_labels'].to(device).view(-1),
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], 1.0
            )
            optimizer.step(); scheduler.step(); optimizer.zero_grad()
            total_loss += loss.item()
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})

        print(f"\n[Phase2 BIO] Epoch {epoch} avg loss: {total_loss/len(train_loader):.4f}")
        _, dev_overlap = evaluate_phase2_bio(
            model, dev_loader, tokenizer, dev_ds.valid_examples,
            device, f'Dev epoch {epoch}', args.max_len
        )
        if dev_overlap > best_score:
            best_score = dev_overlap
            best_epoch = epoch
            save_bio_model(model, tokenizer, output_dir)
            print(f"  ✓ [Phase2 BIO] New best overlap_f1={best_score:.4f} at epoch {epoch}")

    print(f"\n[Phase2 BIO] Best dev overlap_f1: {best_score:.4f} at epoch {best_epoch}")

    print("\n[Phase2 BIO] Loading best model for test evaluation...")
    best_model = load_bio_model(output_dir, device, dropout=args.dropout)
    test_exact, test_overlap = evaluate_phase2_bio(
        best_model, test_loader, tokenizer, test_ds.valid_examples,
        device, 'Test Phase2 BIO', args.max_len
    )
    write_phase2_bio_predictions(
        best_model, test_loader, tokenizer, test_ds.valid_examples,
        device, args.max_len, output_dir / 'test_predictions.jsonl'
    )
    json.dump({
        'phase': '2_bio', 'save_criterion': 'overlap_f1',
        'p2_o_weight': args.p2_o_weight,
        'unfreeze_top_layers': args.unfreeze_top_layers,
        'initialized_from': str(phase1_dir),
        'best_dev_overlap_f1': best_score, 'best_epoch': best_epoch,
        'test_span_exact': test_exact, 'test_span_overlap': test_overlap,
        'model': args.model_name, 'langs': args.langs,
    }, open(output_dir / 'metrics.json', 'w'), indent=2)
    print(f"[Phase2 BIO] Metrics → {output_dir / 'metrics.json'}")

    return output_dir


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = get_device(args.device)
    print(f"Device: {device}  |  Seed: {args.seed}  |  Langs: {args.langs}")

    train_examples = load_split(args.data_dir, 'train', args.langs)
    dev_examples   = load_split(args.data_dir, 'dev',   args.langs)
    test_langs     = args.test_langs or args.langs
    test_examples  = load_split(args.data_dir, 'test',  test_langs)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    seed_output_dir = Path(args.output_dir) / f's{args.seed}'
    seed_output_dir.mkdir(parents=True, exist_ok=True)

    phase1_dir = seed_output_dir / 'phase1'

    if args.phase in (0, 1):
        print("\nTokenizing QA datasets (Phase 1)...")
        train_qa = QADataset(train_examples, tokenizer, args.max_len)
        dev_qa   = QADataset(dev_examples,   tokenizer, args.max_len)
        test_qa  = QADataset(test_examples,  tokenizer, args.max_len)
        print(f"  Train: {len(train_qa)} | Dev: {len(dev_qa)} | Test: {len(test_qa)}")
        phase1_dir = phase1(args, device, tokenizer, train_qa, dev_qa, test_qa, seed_output_dir)

    if args.phase in (0, 2):
        if not (phase1_dir / 'best_model').exists():
            raise FileNotFoundError(
                f"Phase 1 model not found at {phase1_dir / 'best_model'}. "
                f"Run phase 1 first: --phase 1"
            )
        print("\nTokenizing BIO datasets (Phase 2)...")
        train_bio = BIODataset(train_examples, tokenizer, args.max_len)
        dev_bio   = BIODataset(dev_examples,   tokenizer, args.max_len)
        test_bio  = BIODataset(test_examples,  tokenizer, args.max_len)
        print(f"  Train: {len(train_bio)} | Dev: {len(dev_bio)} | Test: {len(test_bio)}")
        phase2_bio(args, device, tokenizer, train_bio, dev_bio, test_bio,
                   phase1_dir, seed_output_dir)

    print("\n" + "="*70)
    print("Done. Evaluate with:")
    print(f"  python Evaluation/Full_evaluation.py \\")
    print(f"      --seed {args.seed} \\")
    print(f"      --seq_phase1  {seed_output_dir}/phase1/test_predictions.jsonl \\")
    print(f"      --seq_phase2  {seed_output_dir}/phase2_bio/test_predictions.jsonl")
    print("="*70)


if __name__ == '__main__':
    main()
