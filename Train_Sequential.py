"""
train_sequential.py

Sequential two-phase mBERT training:

  Phase 1 — Classification-dominant (cls=0.7, span=0.3)
             Full model trained, saved on best dev cls macro F1.
             Output: models/sequential/phase1/

  Phase 2 — Span-dominant (cls=0.3, span=0.7)
             Initialised from Phase 1 weights.
             Bottom encoder layers + cls head FROZEN.
             Only top N transformer layers + span heads updated.
             Saved on best dev span overlap F1.
             Output: models/sequential/phase2/

At inference time:
  - Classification  → Phase 1 model (cls head)
  - Span extraction → Phase 2 model (span heads)

Final joint accuracy + overlap F1 computed in Joint_Evaluation.py by
pointing --stage1_mbert at phase1/test_predictions.jsonl and
--stage2_mbert at phase2/test_predictions.jsonl.

Usage:
    python train_sequential.py

    # Phase 1 only
    python train_sequential.py --phase 1

    # Phase 2 only (requires phase1 already trained)
    python train_sequential.py --phase 2

    # Unfreeze top 4 encoder layers in phase 2 (default: 3)
    python train_sequential.py --unfreeze_top_layers 4
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

LABEL2ID = {'literal': 0, 'idiomatic': 1}
ID2LABEL  = {0: 'literal', 1: 'idiomatic'}


# ── Args ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--model_name',         default='bert-base-multilingual-cased')
    p.add_argument('--data_dir',           default='idioms_structured/Splits')
    p.add_argument('--output_dir',         default='models/sequential')
    p.add_argument('--langs',              nargs='+', default=['English', 'Hindi', 'Telugu'])
    p.add_argument('--phase',              type=int, default=0,
                   help='1=phase1 only, 2=phase2 only, 0=both (default)')
    # Phase 1 hyperparams
    p.add_argument('--p1_epochs',          type=int,   default=7)
    p.add_argument('--p1_batch_size',      type=int,   default=32)
    p.add_argument('--p1_lr',              type=float, default=1e-5)
    p.add_argument('--p1_warmup_ratio',    type=float, default=0.1)
    p.add_argument('--p1_cls_weight',      type=float, default=0.7)
    p.add_argument('--p1_span_weight',     type=float, default=0.3)
    # Phase 2 hyperparams
    p.add_argument('--p2_epochs',          type=int,   default=5)
    p.add_argument('--p2_batch_size',      type=int,   default=16)
    p.add_argument('--p2_lr',              type=float, default=3e-5)
    p.add_argument('--p2_warmup_ratio',    type=float, default=0.06)
    p.add_argument('--p2_cls_weight',      type=float, default=0.3)
    p.add_argument('--p2_span_weight',     type=float, default=0.7)
    p.add_argument('--unfreeze_top_layers',type=int,   default=3,
                   help='Number of top encoder layers to unfreeze in phase 2')
    p.add_argument('--max_len',            type=int,   default=128)
    p.add_argument('--seed',               type=int,   default=42)
    p.add_argument('--use_wandb',          action='store_true')
    p.add_argument('--device',             default=None)
    return p.parse_args()


# ── Device ────────────────────────────────────────────────────────────────────

def get_device(forced=None):
    if forced:
        return torch.device(forced)
    if torch.cuda.is_available():
        return torch.device('cuda')
    if torch.backends.mps.is_available():
        print("MPS detected — precompiling shaders...")
        d = torch.device('mps')
        _ = torch.zeros(1, device=d) + 1
        print("MPS ready.")
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


# ── Loss weights ──────────────────────────────────────────────────────────────

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
    print(f"Cls loss weights — literal: {weights[0]:.4f}, idiomatic: {weights[1]:.4f}")
    return weights


# ── Span helpers ──────────────────────────────────────────────────────────────

def char_to_token_span(encoding, char_start, char_end, sentence):
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
    enc     = tokenizer(sentence, max_length=max_len, truncation=True,
                        return_offsets_mapping=True)
    offsets = enc['offset_mapping']
    if token_start >= len(offsets) or token_end >= len(offsets):
        return None, None
    return offsets[token_start][0], offsets[token_end][1]


def compute_overlap_f1(pred_start, pred_end, gold_start, gold_end):
    pred_set = set(range(pred_start, pred_end + 1))
    gold_set = set(range(gold_start, gold_end + 1))
    if not pred_set or not gold_set:
        return 0.0
    overlap = len(pred_set & gold_set)
    if overlap == 0:
        return 0.0
    p = overlap / len(pred_set)
    r = overlap / len(gold_set)
    return 2 * p * r / (p + r)


# ── Dataset ───────────────────────────────────────────────────────────────────

class JointDataset(Dataset):
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
            self.cls_labels.append(LABEL2ID[ex['idiomaticity']])
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


# ── Model ─────────────────────────────────────────────────────────────────────

class JointIdiomModel(torch.nn.Module):
    def __init__(self, model_name):
        super().__init__()
        self.bert       = AutoModel.from_pretrained(model_name)
        hidden_size     = self.bert.config.hidden_size
        self.cls_head   = torch.nn.Linear(hidden_size, 2)
        self.start_head = torch.nn.Linear(hidden_size, 1)
        self.end_head   = torch.nn.Linear(hidden_size, 1)

    def forward(self, input_ids, attention_mask, token_type_ids):
        outputs      = self.bert(input_ids=input_ids, attention_mask=attention_mask,
                                 token_type_ids=token_type_ids)
        seq_output   = outputs.last_hidden_state
        cls_output   = seq_output[:, 0, :]
        cls_logits   = self.cls_head(cls_output)
        start_logits = self.start_head(seq_output).squeeze(-1)
        end_logits   = self.end_head(seq_output).squeeze(-1)
        mask         = attention_mask.bool()
        start_logits = start_logits.masked_fill(~mask, float('-inf'))
        end_logits   = end_logits.masked_fill(~mask,   float('-inf'))
        return cls_logits, start_logits, end_logits


# ── Freeze helpers ────────────────────────────────────────────────────────────

def freeze_for_phase2(model, unfreeze_top_n):
    """
    Freeze everything, then selectively unfreeze:
      - Top N transformer encoder layers
      - Span heads (start_head, end_head)
    Keep frozen:
      - Embeddings
      - Bottom encoder layers
      - Pooler
      - cls_head  ← don't degrade phase 1 classification
    """
    # Freeze everything first
    for param in model.parameters():
        param.requires_grad = False

    # Unfreeze span heads
    for param in model.start_head.parameters():
        param.requires_grad = True
    for param in model.end_head.parameters():
        param.requires_grad = True

    # Unfreeze top N encoder layers
    encoder_layers = model.bert.encoder.layer
    total_layers   = len(encoder_layers)
    for i in range(total_layers - unfreeze_top_n, total_layers):
        for param in encoder_layers[i].parameters():
            param.requires_grad = True

    # Report
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    print(f"Phase 2 frozen: {total - trainable:,} params | "
          f"trainable: {trainable:,} params ({100*trainable/total:.1f}%)")
    print(f"Unfrozen: span heads + top {unfreeze_top_n} encoder layers "
          f"(layers {total_layers - unfreeze_top_n}–{total_layers - 1})")


# ── Save / load ───────────────────────────────────────────────────────────────

def save_model(model, tokenizer, output_dir):
    best = Path(output_dir) / 'best_model'
    best.mkdir(parents=True, exist_ok=True)
    model.bert.save_pretrained(best)
    tokenizer.save_pretrained(best)
    torch.save({
        'cls_head':   model.cls_head.state_dict(),
        'start_head': model.start_head.state_dict(),
        'end_head':   model.end_head.state_dict(),
    }, best / 'task_heads.pt')


def load_model(model_name, output_dir, device):
    best  = Path(output_dir) / 'best_model'
    model = JointIdiomModel(model_name)
    model.bert = AutoModel.from_pretrained(best)
    heads = torch.load(best / 'task_heads.pt', map_location='cpu')
    model.cls_head.load_state_dict(heads['cls_head'])
    model.start_head.load_state_dict(heads['start_head'])
    model.end_head.load_state_dict(heads['end_head'])
    return model.to(device)


# ── Evaluation ────────────────────────────────────────────────────────────────

def evaluate(model, loader, tokenizer, examples, device, label, max_len):
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
        print(f"  {lang:10s}  cls={lf1:.4f}  exact={le:.4f}  overlap={lf:.4f}  ({len(lang_exact[lang])})")

    return macro_f1, exact_match, avg_overlap


# ── Prediction writer ─────────────────────────────────────────────────────────

def write_predictions(model, loader, tokenizer, examples, device, max_len, out_path):
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
                    'pred_idiomaticity': ID2LABEL[int(cls_preds[i])],
                    'cls_correct':       bool(int(cls_preds[i]) == int(gold_cls[i])),
                    'pred_span_start':   pred_char_s,
                    'pred_span_end':     pred_char_e,
                    'pred_matched_span': ex['sentence'][pred_char_s:pred_char_e] if pred_char_s is not None else '',
                    'span_exact_match':  bool(pred_s == int(gold_starts[i]) and pred_e == int(gold_ends[i])),
                    'span_overlap_f1':   round(compute_overlap_f1(
                                             pred_s, pred_e,
                                             int(gold_starts[i]), int(gold_ends[i])), 4),
                })

    with open(out_path, 'w', encoding='utf-8') as f:
        for p in preds_out:
            f.write(json.dumps(p, ensure_ascii=False) + '\n')
    print(f"Predictions saved → {out_path}")


# ── Training loop (shared by both phases) ─────────────────────────────────────

def run_training_loop(
    model, optimizer, scheduler,
    cls_criterion, span_criterion,
    cls_loss_weight, span_loss_weight,
    train_loader, dev_loader,
    tokenizer, dev_ds,
    device, args_epochs, max_len,
    save_criterion,   # 'cls_f1' or 'overlap_f1'
    output_dir,
    tokenizer_ref,
    phase_label,
    use_wandb,
):
    best_score = 0.0
    best_epoch = 0

    for epoch in range(1, args_epochs + 1):
        model.train()
        total_loss = 0.0

        pbar = tqdm(train_loader, desc=f"[{phase_label}] Epoch {epoch}/{args_epochs}", unit='batch')
        for batch in pbar:
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
            span_loss = (span_criterion(start_logits, start_positions) +
                         span_criterion(end_logits,   end_positions)) / 2
            loss      = cls_loss_weight * cls_loss + span_loss_weight * span_loss

            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], 1.0
            )
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            total_loss += loss.item()
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'cls':  f'{cls_loss.item():.4f}',
                'span': f'{span_loss.item():.4f}',
            })

        avg_loss = total_loss / len(train_loader)
        print(f"\n[{phase_label}] Epoch {epoch} avg loss: {avg_loss:.4f}")

        dev_cls_f1, dev_exact, dev_overlap = evaluate(
            model, dev_loader, tokenizer_ref, dev_ds.valid_examples,
            device, f'Dev epoch {epoch}', max_len
        )

        score = dev_cls_f1 if save_criterion == 'cls_f1' else dev_overlap

        if use_wandb and WANDB:
            wandb.log({
                'epoch': epoch, 'train_loss': avg_loss,
                'dev_cls_f1': dev_cls_f1, 'dev_span_exact': dev_exact,
                'dev_span_overlap': dev_overlap,
            })

        if score > best_score:
            best_score = score
            best_epoch = epoch
            save_model(model, tokenizer_ref, output_dir)
            print(f"  ✓ [{phase_label}] New best [{save_criterion}={best_score:.4f}] at epoch {epoch}")

    print(f"\n[{phase_label}] Best dev {save_criterion}: {best_score:.4f} at epoch {best_epoch}")
    return best_score, best_epoch


# ── Phase 1 ───────────────────────────────────────────────────────────────────

def phase1(args, device, tokenizer, train_ds, dev_ds, test_ds):
    print(f"\n{'='*70}")
    print(f"PHASE 1 — Classification-dominant")
    print(f"  cls_weight={args.p1_cls_weight}  span_weight={args.p1_span_weight}")
    print(f"  epochs={args.p1_epochs}  lr={args.p1_lr}  batch={args.p1_batch_size}")
    print(f"  Save criterion: cls macro F1")
    print(f"{'='*70}")

    output_dir = Path(args.output_dir) / 'phase1'
    output_dir.mkdir(parents=True, exist_ok=True)

    train_loader = DataLoader(train_ds, batch_size=args.p1_batch_size, shuffle=True)
    dev_loader   = DataLoader(dev_ds,   batch_size=args.p1_batch_size)
    test_loader  = DataLoader(test_ds,  batch_size=args.p1_batch_size)

    model       = JointIdiomModel(args.model_name).to(device)
    cls_weights = compute_cls_loss_weights(train_ds.valid_examples, device)
    cls_crit    = torch.nn.CrossEntropyLoss(weight=cls_weights)
    span_crit   = torch.nn.CrossEntropyLoss()

    optimizer    = AdamW(model.parameters(), lr=args.p1_lr, weight_decay=0.01)
    total_steps  = len(train_loader) * args.p1_epochs
    warmup_steps = int(total_steps * args.p1_warmup_ratio)
    scheduler    = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    if WANDB and args.use_wandb:
        wandb.init(project='idiom-sequential', config=vars(args),
                   name='phase1', reinit=True)

    best_score, best_epoch = run_training_loop(
        model=model, optimizer=optimizer, scheduler=scheduler,
        cls_criterion=cls_crit, span_criterion=span_crit,
        cls_loss_weight=args.p1_cls_weight, span_loss_weight=args.p1_span_weight,
        train_loader=train_loader, dev_loader=dev_loader,
        tokenizer=tokenizer, dev_ds=dev_ds,
        device=device, args_epochs=args.p1_epochs, max_len=args.max_len,
        save_criterion='cls_f1',
        output_dir=output_dir,
        tokenizer_ref=tokenizer,
        phase_label='Phase 1',
        use_wandb=args.use_wandb,
    )

    # Test evaluation
    print("\n[Phase 1] Loading best model for test evaluation...")
    best_model = load_model(args.model_name, output_dir, device)
    test_cls_f1, test_exact, test_overlap = evaluate(
        best_model, test_loader, tokenizer, test_ds.valid_examples,
        device, 'Test Phase 1', args.max_len
    )
    write_predictions(
        best_model, test_loader, tokenizer, test_ds.valid_examples,
        device, args.max_len, output_dir / 'test_predictions.jsonl'
    )

    json.dump({
        'phase': 1, 'save_criterion': 'cls_f1',
        'p1_cls_weight': args.p1_cls_weight, 'p1_span_weight': args.p1_span_weight,
        'best_dev_cls_f1': best_score, 'best_epoch': best_epoch,
        'test_cls_macro_f1': test_cls_f1,
        'test_span_exact': test_exact, 'test_span_overlap': test_overlap,
        'model': args.model_name, 'langs': args.langs,
    }, open(output_dir / 'metrics.json', 'w'), indent=2)
    print(f"[Phase 1] Metrics saved → {output_dir / 'metrics.json'}")

    if WANDB and args.use_wandb:
        wandb.finish()

    return output_dir


# ── Phase 2 ───────────────────────────────────────────────────────────────────

def phase2(args, device, tokenizer, train_ds, dev_ds, test_ds, phase1_dir):
    print(f"\n{'='*70}")
    print(f"PHASE 2 — Span-dominant (initialised from Phase 1)")
    print(f"  cls_weight={args.p2_cls_weight}  span_weight={args.p2_span_weight}")
    print(f"  epochs={args.p2_epochs}  lr={args.p2_lr}  batch={args.p2_batch_size}")
    print(f"  Unfreezing top {args.unfreeze_top_layers} encoder layers + span heads")
    print(f"  Save criterion: span overlap F1")
    print(f"{'='*70}")

    output_dir = Path(args.output_dir) / 'phase2'
    output_dir.mkdir(parents=True, exist_ok=True)

    train_loader = DataLoader(train_ds, batch_size=args.p2_batch_size, shuffle=True)
    dev_loader   = DataLoader(dev_ds,   batch_size=args.p2_batch_size)
    test_loader  = DataLoader(test_ds,  batch_size=args.p2_batch_size)

    # Load Phase 1 weights as starting point
    print(f"\n[Phase 2] Loading Phase 1 weights from {phase1_dir}...")
    model = load_model(args.model_name, phase1_dir, device)

    # Freeze bottom layers and cls head; unfreeze top N layers and span heads
    freeze_for_phase2(model, args.unfreeze_top_layers)

    cls_weights = compute_cls_loss_weights(train_ds.valid_examples, device)
    cls_crit    = torch.nn.CrossEntropyLoss(weight=cls_weights)
    span_crit   = torch.nn.CrossEntropyLoss()

    # Only pass trainable params to optimizer
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer    = AdamW(trainable_params, lr=args.p2_lr, weight_decay=0.01)
    total_steps  = len(train_loader) * args.p2_epochs
    warmup_steps = int(total_steps * args.p2_warmup_ratio)
    scheduler    = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    if WANDB and args.use_wandb:
        wandb.init(project='idiom-sequential', config=vars(args),
                   name='phase2', reinit=True)

    best_score, best_epoch = run_training_loop(
        model=model, optimizer=optimizer, scheduler=scheduler,
        cls_criterion=cls_crit, span_criterion=span_crit,
        cls_loss_weight=args.p2_cls_weight, span_loss_weight=args.p2_span_weight,
        train_loader=train_loader, dev_loader=dev_loader,
        tokenizer=tokenizer, dev_ds=dev_ds,
        device=device, args_epochs=args.p2_epochs, max_len=args.max_len,
        save_criterion='overlap_f1',
        output_dir=output_dir,
        tokenizer_ref=tokenizer,
        phase_label='Phase 2',
        use_wandb=args.use_wandb,
    )

    # Test evaluation
    print("\n[Phase 2] Loading best model for test evaluation...")
    best_model = load_model(args.model_name, output_dir, device)
    test_cls_f1, test_exact, test_overlap = evaluate(
        best_model, test_loader, tokenizer, test_ds.valid_examples,
        device, 'Test Phase 2', args.max_len
    )
    write_predictions(
        best_model, test_loader, tokenizer, test_ds.valid_examples,
        device, args.max_len, output_dir / 'test_predictions.jsonl'
    )

    json.dump({
        'phase': 2, 'save_criterion': 'overlap_f1',
        'p2_cls_weight': args.p2_cls_weight, 'p2_span_weight': args.p2_span_weight,
        'unfreeze_top_layers': args.unfreeze_top_layers,
        'initialized_from': str(phase1_dir),
        'best_dev_overlap_f1': best_score, 'best_epoch': best_epoch,
        'test_cls_macro_f1': test_cls_f1,
        'test_span_exact': test_exact, 'test_span_overlap': test_overlap,
        'model': args.model_name, 'langs': args.langs,
    }, open(output_dir / 'metrics.json', 'w'), indent=2)
    print(f"[Phase 2] Metrics saved → {output_dir / 'metrics.json'}")

    if WANDB and args.use_wandb:
        wandb.finish()


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = get_device(args.device)
    print(f"Languages: {args.langs}")

    train_examples = load_split(args.data_dir, 'train', args.langs)
    dev_examples   = load_split(args.data_dir, 'dev',   args.langs)
    test_examples  = load_split(args.data_dir, 'test',  args.langs)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    print("Tokenizing...")
    train_ds = JointDataset(train_examples, tokenizer, args.max_len)
    dev_ds   = JointDataset(dev_examples,   tokenizer, args.max_len)
    test_ds  = JointDataset(test_examples,  tokenizer, args.max_len)
    print(f"Train: {len(train_ds)} | Dev: {len(dev_ds)} | Test: {len(test_ds)}")

    phase1_dir = Path(args.output_dir) / 'phase1'

    if args.phase in (0, 1):
        phase1_dir = phase1(args, device, tokenizer, train_ds, dev_ds, test_ds)

    if args.phase in (0, 2):
        if not (phase1_dir / 'best_model').exists():
            raise FileNotFoundError(
                f"Phase 1 model not found at {phase1_dir / 'best_model'}. "
                f"Run phase 1 first: --phase 1"
            )
        phase2(args, device, tokenizer, train_ds, dev_ds, test_ds, phase1_dir)

    print("\n" + "="*70)
    print("Done. Update Joint_Evaluation.py:")
    print(f"  --stage1_mbert  {args.output_dir}/phase1/test_predictions.jsonl")
    print(f"  --stage2_mbert  {args.output_dir}/phase2/test_predictions.jsonl")
    print("="*70)


if __name__ == '__main__':
    main()