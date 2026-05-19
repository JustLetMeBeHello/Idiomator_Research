"""
train_stage1_classifier.py

Stage 1: Fine-tune mBERT (or monolingual BERT) for idiomaticity classification.
Input  : raw sentence
Output : idiomatic (1) or literal (0)

Key design decisions:
  - Hindi and Telugu use ALL available examples (not sampled)
  - English uses pre-built sampled splits (2x Hindi+Telugu)
  - Loss weighted by language frequency AND per-class idiomaticity imbalance
  - MPS shader precompilation enabled for Apple Silicon

Usage examples:
    # mBERT — all three languages
    python train_stage1_classifier.py \
        --output_dir models/stage1_mbert_en_hi_te \
        --langs English Hindi Telugu

    # mBERT — Hindi + Telugu only (all examples)
    python train_stage1_classifier.py \
        --output_dir models/stage1_mbert_hi_te \
        --langs Hindi Telugu

    # mBERT — English + Telugu
    python train_stage1_classifier.py \
        --output_dir models/stage1_mbert_en_te \
        --langs English Telugu

    # Monolingual BERT — English only
    python train_stage1_classifier.py \
        --model_name bert-base-uncased \
        --output_dir models/stage1_bert_en \
        --langs English
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
    AutoModelForSequenceClassification,
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
    print("wandb not installed — skipping experiment tracking. pip install wandb to enable.")


# ── Config ────────────────────────────────────────────────────────────────────

LABEL2ID = {'literal': 0, 'idiomatic': 1}
ID2LABEL  = {0: 'literal', 1: 'idiomatic'}

# Languages that use ALL examples (no sampling cap)
LOW_RESOURCE_LANGS = {'Hindi', 'Telugu'}
#Research_And_Training/
# Raw data paths (used to load full Hi/Te data bypassing sampled splits)



# ── Args ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--model_name',   default='bert-base-multilingual-cased')
    p.add_argument('--data_dir',     default='idioms_structured/Splits')
    p.add_argument('--output_dir',   default='models/stage1_mbert_en_hi_te')
    p.add_argument('--langs',        nargs='+', default=['English', 'Hindi', 'Telugu','Spanish'],
                   help='Languages to include e.g. --langs English Hindi Telugu')
    p.add_argument('--test_langs',   nargs='+', default=None,
                   help='Languages to evaluate on. Defaults to --langs. Use all target languages for cross-lingual ablations.')
    p.add_argument('--epochs',       type=int,   default=7)
    p.add_argument('--batch_size',   type=int,   default=32)
    p.add_argument('--lr',           type=float, default=3e-5)
    p.add_argument('--max_len',      type=int,   default=128)
    p.add_argument('--warmup_ratio', type=float, default=0.1)
    p.add_argument('--seed',         type=int,   default=42)
    p.add_argument('--use_wandb',    action='store_true')
    p.add_argument('--device',       default=None, help='Force device: cuda, mps, cpu')
    return p.parse_args()


# ── Device ────────────────────────────────────────────────────────────────────

def get_device(forced=None):
    if forced:
        device = torch.device(forced)
    elif torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')

    # Warm up MPS shader cache on Apple Silicon
    if device.type == 'mps':
        print("MPS device detected — precompiling shaders...")
        _x = torch.zeros(1, device=device)
        _y = _x + 1
        del _x, _y
        print("MPS shaders ready.")

    print(f"Device: {device}")
    return device


# ── Data loading ──────────────────────────────────────────────────────────────

# ── Data loading ──────────────────────────────────────────────────────────────

def load_split_examples(split_path, langs):
    """
    Load examples directly from precomputed split JSONL files.
    """
    langs_set = set(langs)
    examples = []

    with open(split_path, encoding='utf-8') as f:
        for line in f:
            r = json.loads(line)
            if r['language'] in langs_set:
                examples.append(r)

    return examples


def build_dataset_for_split(split_name, data_dir, langs):
    """
    Load train/dev/test directly from:
      idioms_structured/Splits/{split_name}.jsonl
    """
    split_path = Path(data_dir) / f'{split_name}.jsonl'
    return load_split_examples(split_path, langs)


# ── Loss weights ──────────────────────────────────────────────────────────────

def compute_loss_weights(train_examples, langs, device):
    """
    Compute per-class loss weights accounting for:
      1. Language imbalance (English >> Hindi/Telugu)
      2. Idiomaticity imbalance within each language

    Strategy: inverse frequency over (language, idiomaticity) cells,
    marginalized to per-class weights for CrossEntropyLoss.
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

    # Scale so min weight = 1.0
    min_w       = min(literal_w, idiomatic_w)
    literal_w  /= min_w
    idiomatic_w /= min_w

    weights = torch.tensor([literal_w, idiomatic_w], dtype=torch.float).to(device)
    print(f"\nLoss weights — literal: {weights[0]:.4f}, idiomatic: {weights[1]:.4f}")
    print("Per (language, class) training counts:")
    for (lang, cls), count in sorted(cell_counts.items()):
        print(f"  {lang:10s} {cls:10s}: {count:5d}  cell_weight={cell_weights[(lang,cls)]:.4f}")

    return weights


# ── Dataset ───────────────────────────────────────────────────────────────────

class IdiomDataset(Dataset):
    def __init__(self, examples, tokenizer, max_len):
        self.examples = examples
        self.labels   = [LABEL2ID[ex['idiomaticity']] for ex in examples]

        self.encodings = tokenizer(
            [ex['sentence'] for ex in examples],
            max_length = max_len,
            padding ='max_length',
            truncation = True,
            return_tensors ='pt',
        )

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            'input_ids':      self.encodings['input_ids'][idx],
            'attention_mask': self.encodings['attention_mask'][idx],
            'token_type_ids': self.encodings.get(
                                'token_type_ids',
                                torch.zeros_like(self.encodings['input_ids']))[idx],
            'labels': torch.tensor(self.labels[idx], dtype=torch.long),
        }


# ── Eval ──────────────────────────────────────────────────────────────────────

def evaluate(model, loader, device, examples, split_name='dev'):
    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for batch in loader:
            input_ids      = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            token_type_ids = batch['token_type_ids'].to(device)
            labels         = batch['labels'].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
            )
            preds = torch.argmax(outputs.logits, dim=-1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    report   = classification_report(all_labels, all_preds,
                                      target_names=['literal', 'idiomatic'], digits=4)
    macro_f1 = f1_score(all_labels, all_preds, average='macro')

    print(f"\n── {split_name} results ──")
    print(report)

    # Per-language breakdown
    for lang in sorted(set(ex['language'] for ex in examples)):
        idxs     = [i for i, ex in enumerate(examples) if ex['language'] == lang]
        l_preds  = [all_preds[i]  for i in idxs]
        l_labels = [all_labels[i] for i in idxs]
        lang_f1  = f1_score(l_labels, l_preds, average='macro')
        print(f"  {lang}: macro F1 = {lang_f1:.4f} ({len(idxs)} examples)")

    return macro_f1, all_preds, all_labels


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

    # Build datasets
    train_examples = build_dataset_for_split('train', args.data_dir, args.langs)
    dev_examples   = build_dataset_for_split('dev',   args.data_dir, args.langs)
    test_langs = args.test_langs or args.langs
    test_examples  = build_dataset_for_split('test',  args.data_dir, test_langs)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    train_ds  = IdiomDataset(train_examples, tokenizer, args.max_len)
    dev_ds    = IdiomDataset(dev_examples,   tokenizer, args.max_len)
    test_ds   = IdiomDataset(test_examples,  tokenizer, args.max_len)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    dev_loader   = DataLoader(dev_ds,   batch_size=args.batch_size)
    test_loader  = DataLoader(test_ds,  batch_size=args.batch_size)

    print(f"Train: {len(train_ds)} | Dev: {len(dev_ds)} | Test: {len(test_ds)}")

    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=2,
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    ).to(device)

    loss_weights = compute_loss_weights(train_examples, args.langs, device)
    criterion    = torch.nn.CrossEntropyLoss(weight=loss_weights)

    optimizer    = AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    total_steps  = len(train_loader) * args.epochs
    warmup_steps = int(total_steps * args.warmup_ratio)
    scheduler    = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    if WANDB and args.use_wandb:
        wandb.init(project='idiom-classification', config=config,
                   name=Path(args.output_dir).name)

    best_f1    = 0.0
    best_epoch = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}", unit='batch')
        for batch in pbar:
            input_ids      = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            token_type_ids = batch['token_type_ids'].to(device)
            labels         = batch['labels'].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
            )
            loss = criterion(outputs.logits, labels)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            total_loss += loss.item()
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})

        avg_loss = total_loss / len(train_loader)
        print(f"\nEpoch {epoch} avg loss: {avg_loss:.4f}")

        dev_f1, _, _ = evaluate(model, dev_loader, device, dev_examples,
                                f'Dev (epoch {epoch})')

        if WANDB and args.use_wandb:
            wandb.log({'epoch': epoch, 'train_loss': avg_loss, 'dev_macro_f1': dev_f1})

        if dev_f1 > best_f1:
            best_f1    = dev_f1
            best_epoch = epoch
            model.save_pretrained(output_dir / 'best_model')
            tokenizer.save_pretrained(output_dir / 'best_model')
            print(f"  ✓ New best model saved (dev F1: {best_f1:.4f})")

    print(f"\nBest dev F1: {best_f1:.4f} at epoch {best_epoch}")

    # Final test eval with best model
    print("\nLoading best model for test evaluation...")
    model = AutoModelForSequenceClassification.from_pretrained(
        output_dir / 'best_model'
    ).to(device)

    test_f1, test_preds, test_labels = evaluate(
        model, test_loader, device, test_examples, 'Test (final)'
    )

    # Save predictions
    preds_out = []
    for ex, pred, label in zip(test_examples, test_preds, test_labels):
        preds_out.append({
            **ex,
            'pred_idiomaticity': ID2LABEL[int(pred)],
            'correct':           bool(pred == label),
        })

    preds_path = output_dir / 'test_predictions.jsonl'
    with open(preds_path, 'w', encoding='utf-8') as f:
        for p in preds_out:
            f.write(json.dumps(p, ensure_ascii=False) + '\n')
    print(f"Predictions saved → {preds_path}")

    # Save metrics
    metrics = {
        'best_dev_f1':   best_f1,
        'best_epoch':    best_epoch,
        'test_macro_f1': test_f1,
        'model':         args.model_name,
        'langs':         args.langs,
        'test_langs':    test_langs,
        'train_size':    len(train_examples),
        'dev_size':      len(dev_examples),
        'test_size':     len(test_examples),
    }
    json.dump(metrics, open(output_dir / 'metrics.json', 'w'), indent=2)
    print(f"Metrics saved → {output_dir / 'metrics.json'}")

    if WANDB and args.use_wandb:
        wandb.log({'test_macro_f1': test_f1})
        wandb.finish()


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == '__main__':
    args = parse_args()
    train(args)
