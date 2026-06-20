"""
bio_tagger_hparam_search.py

Optuna TPE hyperparameter search for the BIO tagger.
Objective: maximise Hindi + Telugu average overlap F1 on dev set.

── Colab / session-timeout safe ──────────────────────────────────────────────
The Optuna study is backed by a SQLite database so it SURVIVES session crashes.
Re-running the script continues exactly where it left off — completed trials
are never repeated. Per-trial results are also saved as individual JSON files
so you can inspect progress even mid-run.

Recommended Colab setup at the top of your notebook:
    from google.colab import drive
    drive.mount('/content/drive')

Then point --output_dir at Drive:
    --output_dir /content/drive/MyDrive/idiom_research/bio_hparam

The SQLite file and per-trial checkpoints are written there. A crashed session
picks up automatically on re-run with no flags to change.
──────────────────────────────────────────────────────────────────────────────

Usage:
    # First run or resume after crash — same command either way:
    python bio_tagger_hparam_search.py \\
        --data_dir  data/idioms_structured/Splits \\
        --output_dir /content/drive/MyDrive/idiom_research/bio_hparam \\
        --n_trials  30

    # Smoke test:
    python bio_tagger_hparam_search.py --n_trials 5 --max_epochs 3
"""

import json
import argparse
import numpy as np
from pathlib import Path
from collections import defaultdict, Counter

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup
from torch.optim import AdamW
from tqdm import tqdm

try:
    import optuna
    from optuna.samplers import TPESampler
    from optuna.pruners import MedianPruner
except ImportError:
    raise ImportError("pip install optuna")

# ── BIO constants ──────────────────────────────────────────────────────────────

LABEL2ID   = {'O': 0, 'B-IDIOM': 1, 'I-IDIOM': 2}
ID2LABEL   = {0: 'O', 1: 'B-IDIOM', 2: 'I-IDIOM'}
NUM_LABELS = 3
IGNORE_IDX = -100


# ── Colab helpers ──────────────────────────────────────────────────────────────

def is_colab():
    try:
        import google.colab  # noqa
        return True
    except ImportError:
        return False


def ensure_drive_mounted(mount_point='/content/drive'):
    if not is_colab():
        return
    if Path(mount_point).exists() and any(Path(mount_point).iterdir()):
        print(f"Drive already mounted at {mount_point}")
        return
    try:
        from google.colab import drive
        drive.mount(mount_point)
        print(f"Drive mounted at {mount_point}")
    except Exception as e:
        print(f"  Could not mount Drive: {e} — checkpoints will be local only")


def save_trial_checkpoint(output_dir, trial_number, params, score, per_lang):
    """Write one JSON per trial immediately after it finishes."""
    ckpt_dir = Path(output_dir) / 'trial_checkpoints'
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt = {'trial': trial_number, 'score': score,
            'per_lang': per_lang, 'params': params}
    with open(ckpt_dir / f'trial_{trial_number:03d}.json', 'w') as f:
        json.dump(ckpt, f, indent=2)


def load_existing_checkpoints(output_dir):
    """Print a summary of already-completed trials (useful after resume)."""
    ckpt_dir = Path(output_dir) / 'trial_checkpoints'
    if not ckpt_dir.exists():
        return
    files = sorted(ckpt_dir.glob('trial_*.json'))
    if not files:
        return
    print(f"\nFound {len(files)} existing checkpoint(s):")
    best_score, best_trial = 0.0, -1
    for f in files:
        t = json.load(open(f))
        score = t.get('score', 0.0)
        print(f"  Trial {t['trial']:03d}  HI+TE={score:.4f}  {t['params']}")
        if score > best_score:
            best_score, best_trial = score, t['trial']
    print(f"  → Best so far: trial {best_trial:03d}  HI+TE={best_score:.4f}\n")


# ── Args ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--model_name',  default='bert-base-multilingual-cased')
    p.add_argument('--data_dir',    default='data/idioms_structured/Splits')
    p.add_argument('--output_dir',  default='models/bio_tagger_hparam')
    p.add_argument('--langs',       nargs='+', default=['English', 'Hindi', 'Telugu'])
    p.add_argument('--n_trials',    type=int, default=30,
                   help='Total trials including already-completed ones')
    p.add_argument('--max_epochs',  type=int, default=7)
    p.add_argument('--max_len',     type=int, default=128)
    p.add_argument('--seed',        type=int, default=42)
    p.add_argument('--device',      default=None)
    p.add_argument('--mount_drive', action='store_true',
                   help='Auto-mount Google Drive in Colab before starting')
    return p.parse_args()


# ── Device ─────────────────────────────────────────────────────────────────────

def get_device(forced=None):
    if forced:
        return torch.device(forced)
    if torch.cuda.is_available():
        return torch.device('cuda')
    if torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


# ── Data ───────────────────────────────────────────────────────────────────────

def load_split(data_dir, split_name, langs):
    path = Path(data_dir) / f'{split_name}.jsonl'
    if not path.exists():
        raise FileNotFoundError(path)
    langs_set = set(langs)
    examples = []
    for line in open(path, encoding='utf-8'):
        r = json.loads(line)
        if r['language'] in langs_set:
            examples.append(r)
    return examples


def compute_lang_loss_weights(train_examples):
    cell_counts = Counter(
        (ex['language'], ex['idiomaticity']) for ex in train_examples
    )
    total   = sum(cell_counts.values())
    n_cells = len(cell_counts)
    cell_weights = {c: total / (n_cells * n) for c, n in cell_counts.items()}
    min_w = min(cell_weights.values())
    return {c: w / min_w for c, w in cell_weights.items()}


def align_bio_labels(offsets, word_ids, char_start, char_end, max_len):
    labels, prev_word_id = [], None
    for offset, word_id in zip(offsets, word_ids):
        if word_id is None or word_id == prev_word_id:
            labels.append(IGNORE_IDX)
            prev_word_id = word_id
            continue
        tok_s, tok_e = offset
        in_span = (tok_s >= char_start) and (tok_e <= char_end)
        if in_span:
            has_b = any(l == LABEL2ID['B-IDIOM'] for l in labels)
            labels.append(LABEL2ID['B-IDIOM'] if not has_b else LABEL2ID['I-IDIOM'])
        else:
            labels.append(LABEL2ID['O'])
        prev_word_id = word_id
    labels += [IGNORE_IDX] * (max_len - len(labels))
    return labels[:max_len]


class BIODataset(Dataset):
    def __init__(self, examples, tokenizer, max_len, cell_weights=None):
        self.valid_examples  = []
        self.input_ids       = []
        self.attention_masks = []
        self.token_type_ids  = []
        self.bio_labels      = []
        self.example_weights = []

        for ex in examples:
            enc = tokenizer(
                ex['sentence'], max_length=max_len, padding='max_length',
                truncation=True, return_offsets_mapping=True, return_tensors='pt',
            )
            labels = align_bio_labels(
                enc['offset_mapping'][0].tolist(),
                enc.word_ids(batch_index=0),
                ex['span_start'], ex['span_end'], max_len
            )
            if LABEL2ID['B-IDIOM'] not in labels:
                continue

            tid = enc.get('token_type_ids')
            seq_len = enc['input_ids'].shape[1]
            self.valid_examples.append(ex)
            self.input_ids.append(enc['input_ids'].squeeze(0))
            self.attention_masks.append(enc['attention_mask'].squeeze(0))
            self.token_type_ids.append(
                tid.squeeze(0) if tid is not None
                else torch.zeros(seq_len, dtype=torch.long)
            )
            self.bio_labels.append(torch.tensor(labels, dtype=torch.long))
            w = cell_weights.get((ex['language'], ex['idiomaticity']), 1.0) \
                if cell_weights else 1.0
            self.example_weights.append(w)

    def __len__(self):
        return len(self.valid_examples)

    def __getitem__(self, idx):
        return {
            'input_ids':      self.input_ids[idx],
            'attention_mask': self.attention_masks[idx],
            'token_type_ids': self.token_type_ids[idx],
            'bio_labels':     self.bio_labels[idx],
            'example_weight': torch.tensor(self.example_weights[idx], dtype=torch.float),
        }


# ── Model ──────────────────────────────────────────────────────────────────────

class BIOTagger(torch.nn.Module):
    def __init__(self, model_name, dropout=0.1):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        self.drop = torch.nn.Dropout(dropout)
        self.head = torch.nn.Linear(self.bert.config.hidden_size, NUM_LABELS)

    def forward(self, input_ids, attention_mask, token_type_ids):
        out = self.bert(input_ids=input_ids, attention_mask=attention_mask,
                        token_type_ids=token_type_ids)
        return self.head(self.drop(out.last_hidden_state))


# ── Span decoding ──────────────────────────────────────────────────────────────

def decode_bio_to_char_span(bio_preds, encoding, sentence, max_len):
    offsets, word_ids = encoding['offset_mapping'], encoding.word_ids()
    seen, first_subtokens = set(), []
    for i, (label, wid) in enumerate(zip(bio_preds, word_ids)):
        if wid is None or wid in seen:
            continue
        seen.add(wid)
        first_subtokens.append((i, label))

    span_tokens, in_span = [], False
    for tok_idx, label in first_subtokens:
        if label == LABEL2ID['B-IDIOM']:
            span_tokens, in_span = [tok_idx], True
        elif label == LABEL2ID['I-IDIOM'] and in_span:
            span_tokens.append(tok_idx)
        elif in_span:
            break

    if not span_tokens:
        return None, None
    ft, lt = span_tokens[0], span_tokens[-1]
    # lt is the first subtoken of the last span word; extend to the last
    # subtoken of that word so char_end isn't truncated mid-word.
    last_wid = word_ids[lt]
    while lt + 1 < len(word_ids) and word_ids[lt + 1] == last_wid:
        lt += 1
    if ft >= len(offsets) or lt >= len(offsets):
        return None, None
    return int(offsets[ft][0]), int(min(offsets[lt][1], len(sentence)))


def compute_overlap_f1(pred_s, pred_e, gold_s, gold_e):
    if pred_s is None or pred_e is None:
        return 0.0
    ps, gs = set(range(pred_s, pred_e)), set(range(gold_s, gold_e))
    if not ps or not gs:
        return 0.0
    ov = len(ps & gs)
    if ov == 0:
        return 0.0
    return 2 * (ov / len(ps)) * (ov / len(gs)) / ((ov / len(ps)) + (ov / len(gs)))


# ── Eval ───────────────────────────────────────────────────────────────────────

def evaluate(model, loader, tokenizer, examples, device, max_len):
    model.eval()
    lang_f1 = defaultdict(list)
    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            logits = model(batch['input_ids'].to(device),
                           batch['attention_mask'].to(device),
                           batch['token_type_ids'].to(device))
            preds  = torch.argmax(logits, dim=-1).cpu()
            for i in range(len(preds)):
                ex_idx = batch_idx * loader.batch_size + i
                if ex_idx >= len(examples):
                    break
                ex  = examples[ex_idx]
                enc = tokenizer(ex['sentence'], max_length=max_len,
                                truncation=True, return_offsets_mapping=True)
                ps, pe = decode_bio_to_char_span(preds[i].tolist(), enc,
                                                 ex['sentence'], max_len)
                lang_f1[ex['language']].append(
                    compute_overlap_f1(ps, pe, ex['span_start'], ex['span_end'])
                )
    per_lang = {l: float(np.mean(v)) for l, v in lang_f1.items()}
    objective = float(np.mean([per_lang.get('Hindi', 0.0),
                                per_lang.get('Telugu', 0.0)]))
    return per_lang, objective


# ── Single trial ───────────────────────────────────────────────────────────────

def run_trial(trial, args, train_ds, dev_ds, tokenizer, device, output_dir):
    lr           = trial.suggest_float('lr',           1e-5, 5e-5, log=True)
    batch_size   = trial.suggest_categorical('batch_size', [16, 32])
    epochs       = trial.suggest_int('epochs',         3, args.max_epochs)
    warmup_ratio = trial.suggest_float('warmup_ratio', 0.03, 0.15)
    o_weight     = trial.suggest_float('o_weight',     0.05, 0.5,  log=True)
    dropout      = trial.suggest_float('dropout',      0.1,  0.3)

    print(f"\nTrial {trial.number}: lr={lr:.2e} bs={batch_size} ep={epochs} "
          f"warmup={warmup_ratio:.3f} o_w={o_weight:.3f} drop={dropout:.2f}")

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    dev_loader   = DataLoader(dev_ds,   batch_size=batch_size)

    model     = BIOTagger(args.model_name, dropout=dropout).to(device)
    criterion = torch.nn.CrossEntropyLoss(
        weight=torch.tensor([o_weight, 1.0, 1.0], dtype=torch.float).to(device),
        ignore_index=IGNORE_IDX, reduction='none'
    )
    optimizer   = AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    total_steps = len(train_loader) * epochs
    scheduler   = get_linear_schedule_with_warmup(
        optimizer, int(total_steps * warmup_ratio), total_steps
    )

    best_score, best_per_lang = 0.0, {}

    for epoch in range(1, epochs + 1):
        model.train()
        for batch in tqdm(train_loader, desc=f"  T{trial.number} E{epoch}", leave=False):
            logits     = model(batch['input_ids'].to(device),
                               batch['attention_mask'].to(device),
                               batch['token_type_ids'].to(device))
            bio_labels = batch['bio_labels'].to(device)
            ex_weights = batch['example_weight'].to(device)

            token_loss = criterion(logits.view(-1, NUM_LABELS), bio_labels.view(-1))
            token_loss = token_loss.view(bio_labels.shape[0], -1)
            valid_mask = (bio_labels != IGNORE_IDX).float()
            per_ex     = (token_loss * valid_mask).sum(1) / valid_mask.sum(1).clamp(min=1)
            (per_ex * ex_weights).mean().backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step(); scheduler.step(); optimizer.zero_grad()

        per_lang, score = evaluate(model, dev_loader, tokenizer,
                                   dev_ds.valid_examples, device, args.max_len)
        lang_str = '  '.join(f"{l[:2]}={v:.4f}" for l, v in sorted(per_lang.items()))
        print(f"  Epoch {epoch}: {lang_str}  →  obj={score:.4f}")

        if score > best_score:
            best_score, best_per_lang = score, per_lang

        trial.report(score, epoch)
        if trial.should_prune():
            print(f"  Pruned at epoch {epoch}")
            save_trial_checkpoint(output_dir, trial.number,
                                  trial.params, best_score, best_per_lang)
            raise optuna.exceptions.TrialPruned()

    # Save immediately after completion, before session can die
    save_trial_checkpoint(output_dir, trial.number,
                          trial.params, best_score, best_per_lang)
    return best_score


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    args   = parse_args()
    device = get_device(args.device)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.mount_drive:
        ensure_drive_mounted()

    print(f"Device     : {device}")
    print(f"Output dir : {output_dir}")

    load_existing_checkpoints(output_dir)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    print("Loading data...")
    train_examples = load_split(args.data_dir, 'train', args.langs)
    dev_examples   = load_split(args.data_dir, 'dev',   args.langs)
    tokenizer      = AutoTokenizer.from_pretrained(args.model_name)
    cell_weights   = compute_lang_loss_weights(train_examples)

    print("Tokenising...")
    train_ds = BIODataset(train_examples, tokenizer, args.max_len, cell_weights)
    dev_ds   = BIODataset(dev_examples,   tokenizer, args.max_len)
    print(f"Train: {len(train_ds)}  Dev: {len(dev_ds)}")

    # ── SQLite-backed study — survives session crashes ─────────────────────────
    db_path    = output_dir / 'study.db'
    storage    = f'sqlite:///{db_path}'
    study_name = 'bio_tagger_hparam'

    print(f"\nOptuna DB: {db_path}")
    print("Resuming existing study." if db_path.exists() else "Starting new study.")

    study = optuna.create_study(
        study_name=study_name,
        storage=storage,
        load_if_exists=True,       # ← resumes automatically if DB exists
        direction='maximize',
        sampler=TPESampler(seed=args.seed),
        pruner=MedianPruner(n_startup_trials=5, n_warmup_steps=2),
    )

    completed = len([t for t in study.trials
                     if t.state == optuna.trial.TrialState.COMPLETE])
    remaining = max(0, args.n_trials - completed)
    print(f"Completed: {completed}  |  Remaining: {remaining}")

    if remaining > 0:
        def objective(trial):
            return run_trial(trial, args, train_ds, dev_ds,
                             tokenizer, device, output_dir)
        study.optimize(objective, n_trials=remaining, show_progress_bar=False)

    best = study.best_trial
    print("\n" + "=" * 60)
    print("BEST TRIAL")
    print("=" * 60)
    print(f"  Trial    : {best.number}")
    print(f"  HI+TE F1 : {best.value:.4f}")
    for k, v in best.params.items():
        print(f"  {k:<20} = {v}")

    results = {
        'best_trial':  best.number,
        'best_score':  best.value,
        'best_params': best.params,
        'all_trials': [
            {'number': t.number, 'value': t.value,
             'params': t.params, 'state': str(t.state)}
            for t in study.trials
        ]
    }
    out_path = output_dir / 'hparam_results.json'
    json.dump(results, open(out_path, 'w'), indent=2)
    print(f"\nFull results → {out_path}")

    p = best.params
    print("\nBest config as CLI args:")
    print(f"  --lr {p['lr']:.2e} \\")
    print(f"  --batch_size {p['batch_size']} \\")
    print(f"  --epochs {p['epochs']} \\")
    print(f"  --warmup_ratio {p['warmup_ratio']:.3f} \\")
    print(f"  --o_weight {p['o_weight']:.3f}")


if __name__ == '__main__':
    main()