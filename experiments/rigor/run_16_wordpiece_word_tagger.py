"""
run_16_wordpiece_word_tagger.py  —  REVISION_PLAN experiment E3

A WordPiece WORD-LEVEL POS-style tagger for idiom-span extraction.

Why this exists
---------------
C1 concludes that "a simpler word-level tagger suffices" — i.e. once the
SentencePiece trailing-punctuation artifact is removed, the QA span-pointer
buys nothing over a plain sequence labeller working at WORD granularity on a
WordPiece encoder. That is currently an INFERENCE in the paper, never built.
E3 turns the recommendation into a measured System: train it, score it with the
same eval as Systems A–G, and report whether it matches the QA/Joint span EM.

How it differs from System G (BiO_Task_mBERT_train.py)
------------------------------------------------------
System G is a SUBWORD BIO tagger: it classifies every subword token, supervises
the first subword of each word, and decodes by walking subword B/I runs. E3 is
genuinely WORD-LEVEL, POS-tagger style:
  * subword hidden states are MEAN-POOLED into one vector per word
    (a real word representation, not "first subword only"),
  * exactly ONE B/I/O decision is made per word,
  * decoding uses WORD char boundaries → no mid-word truncation, and because
    the encoder is WordPiece (mBERT) there is no ▁ leading-space offset to
    correct, which is the whole point of C1's recommendation.

WordPiece is intended (default mBERT). A SentencePiece encoder would reintroduce
the very offset confound C1 isolates; the script warns if given one.

Architecture : mBERT encoder → mean-pool subwords→words → dropout → Linear(H,3)
Loss         : weighted CrossEntropy over word labels (O downweighted),
               per-example (language, idiomaticity) inverse-freq weighting
               (identical scheme to Stage 1 / System G).
Output       : per-word B/I/O → decoded to char offsets, saved in the SAME
               test_predictions.jsonl schema Full_evaluation.py consumes, so
               this plugs in as a new system row with no eval changes.

No reported metric is hardcoded (CLAUDE.md rule): all numbers computed live.

Usage
-----
    # local smoke test (CPU/MPS, tiny)
    .venv/bin/python3 experiments/rigor/run_16_wordpiece_word_tagger.py \
        --output_dir /tmp/_e3_dryrun --langs English --epochs 1 --batch_size 8

    # full run (Colab GPU; ~5–8 GPU-hr)
    python experiments/rigor/run_16_wordpiece_word_tagger.py \
        --output_dir models/word_tagger_mbert \
        --langs English Spanish Hindi Telugu \
        --test_langs English Spanish Hindi Telugu Indonesian \
        --seed 42

    # register as a system row
    python Evaluation/Full_evaluation.py --bio_preds models/word_tagger_mbert/test_predictions.jsonl ...
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
from sklearn.metrics import classification_report
from tqdm import tqdm

try:
    import wandb
    WANDB = True
except ImportError:
    WANDB = False


# ── Label scheme (word-level) ───────────────────────────────────────────────
LABEL2ID = {'O': 0, 'B-IDIOM': 1, 'I-IDIOM': 2}
ID2LABEL = {0: 'O', 1: 'B-IDIOM', 2: 'I-IDIOM'}
NUM_LABELS = 3
IGNORE_IDX = -100


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--model_name',   default='bert-base-multilingual-cased',
                   help='WordPiece encoder (mBERT). SentencePiece encoders are warned against.')
    p.add_argument('--data_dir',     default='data/idioms_structured/Splits')
    p.add_argument('--output_dir',   default='models/word_tagger_mbert')
    p.add_argument('--langs',        nargs='+', default=['English', 'Spanish', 'Hindi', 'Telugu'])
    p.add_argument('--test_langs',   nargs='+', default=None,
                   help='Eval languages (defaults to --langs). Add Indonesian for the held-out table.')
    p.add_argument('--epochs',       type=int,   default=6)
    p.add_argument('--batch_size',   type=int,   default=32)
    p.add_argument('--lr',           type=float, default=3.27e-5)
    p.add_argument('--max_len',      type=int,   default=128)
    p.add_argument('--max_words',    type=int,   default=64,
                   help='Max words per sentence (word slots). Sentences with more are truncated.')
    p.add_argument('--warmup_ratio', type=float, default=0.096)
    p.add_argument('--dropout',      type=float, default=0.1)
    p.add_argument('--o_weight',     type=float, default=0.104,
                   help='Loss weight for O word class (B/I weighted 1.0).')
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
        return torch.device('mps')
    return torch.device('cpu')


# ── Data ────────────────────────────────────────────────────────────────────

def load_split(data_dir, split_name, langs):
    path = Path(data_dir) / f'{split_name}.jsonl'
    if not path.exists():
        raise FileNotFoundError(f"Split file not found: {path}")
    langs_set = set(langs)
    examples = [json.loads(l) for l in open(path, encoding='utf-8')]
    examples = [r for r in examples if r['language'] in langs_set]
    print(f"  {split_name}: {len(examples)} examples")
    return examples


def compute_lang_loss_weights(train_examples):
    """Inverse-frequency per (language, idiomaticity) cell, min-normalised to 1.0.
    Identical to Stage 1 / System G so E3 is comparable on the same footing."""
    cell_counts = Counter((ex['language'], ex['idiomaticity']) for ex in train_examples)
    total, n_cells = sum(cell_counts.values()), len(cell_counts)
    cell_weights = {c: total / (n_cells * n) for c, n in cell_counts.items()}
    min_w = min(cell_weights.values())
    return {c: w / min_w for c, w in cell_weights.items()}


def word_spans(offsets, word_ids, max_words):
    """Group subtoken offsets by word_id → per-word (char_start, char_end) and
    the list of subtoken positions feeding each word (for mean-pooling).

    Returns (word_offsets, word_subtokens) where index = word slot < max_words.
    Words beyond max_words are dropped (sentence truncated)."""
    word_offsets = {}
    word_subtokens = defaultdict(list)
    for tok_idx, (off, wid) in enumerate(zip(offsets, word_ids)):
        if wid is None or wid >= max_words:
            continue
        cs, ce = off
        if wid not in word_offsets:
            word_offsets[wid] = [cs, ce]
        else:
            word_offsets[wid][0] = min(word_offsets[wid][0], cs)
            word_offsets[wid][1] = max(word_offsets[wid][1], ce)
        word_subtokens[wid].append(tok_idx)
    n_words = len(word_offsets)
    offs = [tuple(word_offsets[w]) for w in range(n_words)]
    subs = [word_subtokens[w] for w in range(n_words)]
    return offs, subs


def make_word_labels(word_offsets, char_start, char_end):
    """One B/I/O per word from char-span overlap. First overlapping word = B."""
    labels = []
    seen_b = False
    for (cs, ce) in word_offsets:
        in_span = min(ce, char_end) > max(cs, char_start)
        if in_span:
            labels.append(LABEL2ID['B-IDIOM'] if not seen_b else LABEL2ID['I-IDIOM'])
            seen_b = True
        else:
            labels.append(LABEL2ID['O'])
    return labels


class WordTaggerDataset(Dataset):
    """Each item carries a [max_words, max_len] mean-pool matrix that maps the
    encoder's subtoken outputs onto word slots, plus word-level labels."""

    def __init__(self, examples, tokenizer, max_len, max_words, cell_weights=None):
        self.valid_examples = []
        self.input_ids, self.attention_masks, self.token_type_ids = [], [], []
        self.pool_matrices, self.word_labels, self.example_weights = [], [], []
        skipped = 0

        for ex in examples:
            cs, ce = ex['span_start'], ex['span_end']
            if cs is None or ce is None:
                skipped += 1
                continue
            enc = tokenizer(ex['sentence'], max_length=max_len, padding='max_length',
                            truncation=True, return_offsets_mapping=True, return_tensors='pt')
            offsets = enc['offset_mapping'][0].tolist()
            word_ids = enc.word_ids(batch_index=0)
            w_offs, w_subs = word_spans(offsets, word_ids, max_words)
            labels = make_word_labels(w_offs, cs, ce)

            # Need at least one B word in the kept window, else span is unreachable.
            if not labels or LABEL2ID['B-IDIOM'] not in labels:
                skipped += 1
                continue

            pool = torch.zeros(max_words, max_len, dtype=torch.float)
            for w, sub_idx in enumerate(w_subs):
                if sub_idx:
                    pool[w, sub_idx] = 1.0 / len(sub_idx)   # mean-pool weights
            lab = torch.full((max_words,), IGNORE_IDX, dtype=torch.long)
            for w, l in enumerate(labels):
                lab[w] = l

            tid = enc.get('token_type_ids')
            self.valid_examples.append(ex)
            self.input_ids.append(enc['input_ids'].squeeze(0))
            self.attention_masks.append(enc['attention_mask'].squeeze(0))
            self.token_type_ids.append(tid.squeeze(0) if tid is not None
                                       else torch.zeros(max_len, dtype=torch.long))
            self.pool_matrices.append(pool)
            self.word_labels.append(lab)
            w = cell_weights.get((ex['language'], ex['idiomaticity']), 1.0) if cell_weights else 1.0
            self.example_weights.append(w)

        if skipped:
            print(f"  Skipped {skipped} examples (no span / span outside window)")

    def __len__(self):
        return len(self.valid_examples)

    def __getitem__(self, i):
        return {
            'input_ids':      self.input_ids[i],
            'attention_mask': self.attention_masks[i],
            'token_type_ids': self.token_type_ids[i],
            'pool_matrix':    self.pool_matrices[i],
            'word_labels':    self.word_labels[i],
            'example_weight': torch.tensor(self.example_weights[i], dtype=torch.float),
        }


# ── Model ───────────────────────────────────────────────────────────────────

class WordTagger(torch.nn.Module):
    """mBERT → mean-pool subwords to words → linear B/I/O head (one tag per word)."""

    def __init__(self, model_name, num_labels=NUM_LABELS, dropout=0.1):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        h = self.bert.config.hidden_size
        self.dropout = torch.nn.Dropout(dropout)
        self.head = torch.nn.Linear(h, num_labels)

    def forward(self, input_ids, attention_mask, token_type_ids, pool_matrix):
        out = self.bert(input_ids=input_ids, attention_mask=attention_mask,
                        token_type_ids=token_type_ids)
        seq = out.last_hidden_state.float()              # [B, T, H]
        word_repr = torch.bmm(pool_matrix, seq)          # [B, W, H] mean-pooled
        return self.head(self.dropout(word_repr))        # [B, W, 3]


# ── Decode + metrics ────────────────────────────────────────────────────────

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


def decode_words_to_span(word_preds, word_offsets, sentence):
    """First B word, extend through consecutive I words → char span."""
    span = []
    in_span = False
    for w, label in enumerate(word_preds):
        if w >= len(word_offsets):
            break
        if label == LABEL2ID['B-IDIOM']:
            span = [w]
            in_span = True
        elif label == LABEL2ID['I-IDIOM'] and in_span:
            span.append(w)
        elif in_span:
            break
    if not span:
        return None, None
    cs = word_offsets[span[0]][0]
    ce = min(word_offsets[span[-1]][1], len(sentence))
    return int(cs), int(ce)


def predict_batch(model, batch, device):
    logits = model(batch['input_ids'].to(device), batch['attention_mask'].to(device),
                   batch['token_type_ids'].to(device), batch['pool_matrix'].to(device))
    return torch.argmax(logits, dim=-1).cpu()   # [B, W]


def word_offsets_for(tokenizer, sentence, max_len, max_words):
    enc = tokenizer(sentence, max_length=max_len, truncation=True, return_offsets_mapping=True)
    offs, _subs = word_spans(enc['offset_mapping'], enc.word_ids(), max_words)
    return offs


def evaluate(model, loader, tokenizer, examples, device, split_name, max_len, max_words):
    model.eval()
    lang_exact, lang_f1 = defaultdict(list), defaultdict(list)
    all_true, all_pred = [], []
    with torch.no_grad():
        for bi, batch in enumerate(tqdm(loader, desc=f'Eval {split_name}', leave=False)):
            preds = predict_batch(model, batch, device)
            true = batch['word_labels']
            for i in range(len(preds)):
                ex_idx = bi * loader.batch_size + i
                if ex_idx >= len(examples):
                    break
                ex = examples[ex_idx]
                offs = word_offsets_for(tokenizer, ex['sentence'], max_len, max_words)
                ps, pe = decode_words_to_span(preds[i].tolist(), offs, ex['sentence'])
                gs, ge = ex['span_start'], ex['span_end']
                lang_exact[ex['language']].append(int(ps == gs and pe == ge) if ps is not None else 0)
                lang_f1[ex['language']].append(compute_overlap_f1(ps, pe, gs, ge))
                for t, p in zip(true[i].tolist(), preds[i].tolist()):
                    if t != IGNORE_IDX:
                        all_true.append(t); all_pred.append(p)

    all_exact = [v for vs in lang_exact.values() for v in vs]
    all_overlap = [v for vs in lang_f1.values() for v in vs]
    print(f"\n── {split_name} Span Results ──")
    print(f"  {'Language':<12} {'Exact':<10} {'Overlap F1':<12} {'N':<6}")
    for lang in sorted(lang_exact.keys()):
        print(f"  {lang:<12} {np.mean(lang_exact[lang]):<10.4f} "
              f"{np.mean(lang_f1[lang]):<12.4f} {len(lang_exact[lang]):<6}")
    print(f"  {'Overall':<12} {np.mean(all_exact):<10.4f} {np.mean(all_overlap):<12.4f} {len(all_exact):<6}")
    print(f"\n── {split_name} Word BIO Report ──")
    print(classification_report(all_true, all_pred,
          target_names=[ID2LABEL[i] for i in range(NUM_LABELS)], digits=4, zero_division=0))
    return float(np.mean(all_exact)), float(np.mean(all_overlap))


# ── Save ────────────────────────────────────────────────────────────────────

def save_model(model, tokenizer, output_dir):
    best = Path(output_dir) / 'best_model'
    best.mkdir(parents=True, exist_ok=True)
    model.bert.save_pretrained(best, safe_serialization=True)
    weights = list(best.glob("*.safetensors")) + list(best.glob("pytorch_model.bin"))
    if not weights:
        raise RuntimeError(f"Encoder save wrote no weights to {best}: {[p.name for p in best.iterdir()]}")
    tokenizer.save_pretrained(best)
    torch.save(model.head.state_dict(), best / 'word_head.pt')
    print(f"  Saved encoder ({sum(p.stat().st_size for p in weights)/1e6:.1f} MB) + head + tokenizer")


def load_best_model(model_name, output_dir, device, dropout):
    best = Path(output_dir) / 'best_model'
    if not best.exists():
        print(f"  ⚠ best_model/ not found. Loading base {model_name}.")
        model = WordTagger(model_name, dropout=dropout)
    else:
        model = WordTagger(str(best), dropout=dropout)
        model.head.load_state_dict(torch.load(best / 'word_head.pt', map_location=device, weights_only=True))
    return model.to(device)


# ── Train ───────────────────────────────────────────────────────────────────

def train(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = get_device(args.device)
    print(f"Device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if not tokenizer.is_fast:
        raise SystemExit("E3 needs a fast tokenizer (word_ids / offset_mapping required).")
    # Warn if the encoder is SentencePiece — E3 is a WordPiece recommendation by design.
    sp_markers = ('xlm-roberta', 'rembert', 'mdeberta', 'sentencepiece', 'albert', 'camembert')
    if any(m in args.model_name.lower() for m in sp_markers):
        print(f"⚠ {args.model_name} looks SentencePiece. E3 is C1's *WordPiece* word-level "
              f"recommendation; an SP encoder reintroduces the ▁ offset confound. Proceeding anyway.")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    json.dump(vars(args), open(output_dir / 'config.json', 'w'), indent=2)

    print("\nLoading splits...")
    train_ex = load_split(args.data_dir, 'train', args.langs)
    dev_ex   = load_split(args.data_dir, 'dev', args.langs)
    test_langs = args.test_langs or args.langs
    test_ex  = load_split(args.data_dir, 'test', test_langs)

    cell_weights = compute_lang_loss_weights(train_ex)
    print("\nTokenizing + building word-level labels...")
    train_ds = WordTaggerDataset(train_ex, tokenizer, args.max_len, args.max_words, cell_weights)
    dev_ds   = WordTaggerDataset(dev_ex, tokenizer, args.max_len, args.max_words)
    test_ds  = WordTaggerDataset(test_ex, tokenizer, args.max_len, args.max_words)
    print(f"Train: {len(train_ds)} | Dev: {len(dev_ds)} | Test: {len(test_ds)}")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    dev_loader   = DataLoader(dev_ds, batch_size=args.batch_size)
    test_loader  = DataLoader(test_ds, batch_size=args.batch_size)

    model = WordTagger(args.model_name, dropout=args.dropout).to(device)
    class_weights = torch.tensor([args.o_weight, 1.0, 1.0], dtype=torch.float).to(device)
    criterion = torch.nn.CrossEntropyLoss(weight=class_weights, ignore_index=IGNORE_IDX, reduction='none')

    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    total_steps = len(train_loader) * args.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, int(total_steps * args.warmup_ratio), total_steps)

    if WANDB and args.use_wandb:
        wandb.init(project='idiom-word-tagger', config=vars(args), name=output_dir.name)

    best_dev_overlap, best_epoch = 0.0, 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}", unit='batch')
        for batch in pbar:
            labels = batch['word_labels'].to(device)            # [B, W]
            ex_w = batch['example_weight'].to(device)           # [B]
            logits = model(batch['input_ids'].to(device), batch['attention_mask'].to(device),
                           batch['token_type_ids'].to(device), batch['pool_matrix'].to(device))
            tok_loss = criterion(logits.view(-1, NUM_LABELS), labels.view(-1)).view(labels.shape[0], -1)
            valid = (labels != IGNORE_IDX).float()
            per_ex = (tok_loss * valid).sum(dim=1) / valid.sum(dim=1).clamp(min=1)
            loss = (per_ex * ex_w).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step(); scheduler.step(); optimizer.zero_grad()
            total_loss += loss.item()
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})

        print(f"\nEpoch {epoch} avg loss: {total_loss/len(train_loader):.4f}")
        dev_exact, dev_overlap = evaluate(model, dev_loader, tokenizer, dev_ds.valid_examples,
                                          device, f'Dev (epoch {epoch})', args.max_len, args.max_words)
        if WANDB and args.use_wandb:
            wandb.log({'epoch': epoch, 'train_loss': total_loss/len(train_loader),
                       'dev_exact': dev_exact, 'dev_overlap': dev_overlap})
        if dev_overlap > best_dev_overlap:
            best_dev_overlap, best_epoch = dev_overlap, epoch
            save_model(model, tokenizer, output_dir)
            print(f"  ✓ New best (dev overlap F1: {best_dev_overlap:.4f})")

    print(f"\nBest dev overlap F1: {best_dev_overlap:.4f} at epoch {best_epoch}")
    print("\nLoading best model for test...")
    best_model = load_best_model(args.model_name, output_dir, device, args.dropout)
    test_exact, test_overlap = evaluate(best_model, test_loader, tokenizer, test_ds.valid_examples,
                                        device, 'Test (final)', args.max_len, args.max_words)

    # Predictions in Full_evaluation schema (same as System G).
    best_model.eval()
    preds_out = []
    with torch.no_grad():
        for bi, batch in enumerate(test_loader):
            preds = predict_batch(best_model, batch, device)
            for i in range(len(preds)):
                ex_idx = bi * test_loader.batch_size + i
                if ex_idx >= len(test_ds.valid_examples):
                    break
                ex = test_ds.valid_examples[ex_idx]
                offs = word_offsets_for(tokenizer, ex['sentence'], args.max_len, args.max_words)
                ps, pe = decode_words_to_span(preds[i].tolist(), offs, ex['sentence'])
                if ps is None:
                    ps, pe = 0, 0
                gs, ge = ex['span_start'], ex['span_end']
                preds_out.append({
                    **ex,
                    'pred_span_start': ps, 'pred_span_end': pe,
                    'pred_matched_span': ex['sentence'][ps:pe],
                    'span_exact_match': bool(ps == gs and pe == ge),
                    'span_overlap_f1': round(compute_overlap_f1(ps, pe, gs, ge), 4),
                })
    preds_path = output_dir / 'test_predictions.jsonl'
    with open(preds_path, 'w', encoding='utf-8') as f:
        for p in preds_out:
            f.write(json.dumps(p, ensure_ascii=False) + '\n')
    print(f"Predictions saved → {preds_path}")

    json.dump({
        'model': args.model_name, 'langs': args.langs, 'test_langs': test_langs,
        'best_epoch': best_epoch, 'best_dev_overlap': round(best_dev_overlap, 4),
        'test_exact_match': round(test_exact, 4), 'test_overlap_f1': round(test_overlap, 4),
        'train_size': len(train_ds), 'dev_size': len(dev_ds), 'test_size': len(test_ds),
        'o_weight': args.o_weight, 'max_words': args.max_words,
    }, open(output_dir / 'metrics.json', 'w'), indent=2)
    print(f"Metrics saved → {output_dir / 'metrics.json'}")
    if WANDB and args.use_wandb:
        wandb.finish()


if __name__ == '__main__':
    train(parse_args())
