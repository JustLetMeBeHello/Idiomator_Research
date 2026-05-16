"""
bio_tagger_hparam_search.py

Bayesian hyperparameter optimisation for the BIO tagger using Optuna (TPE sampler).
Optimises for Hindi + Telugu average overlap F1 on dev set.

Why Optuna over grid/random search:
  - TPE builds a probabilistic model of which configs are promising
  - Each trial informs the next — it learns the landscape as it goes
  - Finds better configs in ~half the trials of random search

Usage:
    # 30 trials, Bayesian TPE (recommended)
    python bio_tagger_hparam_search.py --n_trials 30

    # Resume a previous study (adds more trials to existing DB)
    python bio_tagger_hparam_search.py --n_trials 30 --study_name bio_sweep_v1

    # Quick smoke test (5 trials, max 3 epochs each)
    python bio_tagger_hparam_search.py --n_trials 5 --max_epochs 3
"""

import json
import argparse
import time
import numpy as np
from pathlib import Path
from collections import defaultdict, Counter
from datetime import datetime

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup
from torch.optim import AdamW
# Change this import at the top of your script:
from tqdm.notebook import tqdm

import optuna
from optuna.samplers import TPESampler


# ── Fixed config — edit to match your paths ───────────────────────────────────

FIXED = {
    'model_name': 'bert-base-multilingual-cased',
    'data_dir':   'idioms_structured/Splits',
    'langs':      ['English', 'Hindi', 'Telugu'],
    'max_len':    128,
    'seed':       42,
}

# Output dirs
SWEEP_DIR    = Path('bio_sweep_results')
STUDY_DB     = SWEEP_DIR / 'optuna_study.db'
SUMMARY_PATH = SWEEP_DIR / 'summary.jsonl'


# ── BIO constants ─────────────────────────────────────────────────────────────

LABEL2ID  = {'O': 0, 'B-IDIOM': 1, 'I-IDIOM': 2}
ID2LABEL  = {0: 'O', 1: 'B-IDIOM', 2: 'I-IDIOM'}
NUM_LABELS = 3
IGNORE_IDX = -100


# ── Args ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--n_trials',     type=int,   default=30,
                   help='Number of Optuna trials to run')
    p.add_argument('--study_name',   default='bio_sweep_v1',
                   help='Study name — reuse to resume a previous sweep')
    p.add_argument('--max_epochs',  type=int,   default=7,
                   help='Max epochs per trial')
    p.add_argument('--device',      default=None)
    p.add_argument('--timeout',      type=int,   default=None,
                   help='Stop after this many seconds regardless of n_trials')
    return p.parse_args()


# ── Device ────────────────────────────────────────────────────────────────────

def get_device(forced=None):
    if forced:
        return torch.device(forced)
    if torch.cuda.is_available():
        return torch.device('cuda')
    if torch.backends.mps.is_available():
        d = torch.device('mps')
        _ = torch.zeros(1, device=d) + 1
        return d
    return torch.device('cpu')


# ── Data ──────────────────────────────────────────────────────────────────────

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
            encoding = tokenizer(
                ex['sentence'],
                max_length=max_len,
                padding='max_length',
                truncation=True,
                return_offsets_mapping=True,
                return_tensors='pt',
            )
            labels = align_bio_labels(
                encoding['offset_mapping'][0].tolist(),
                encoding.word_ids(batch_index=0),
                ex['span_start'], ex['span_end'], max_len
            )
            if LABEL2ID['B-IDIOM'] not in labels:
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
            w = 1.0
            if cell_weights is not None:
                w = cell_weights.get((ex['language'], ex['idiomaticity']), 1.0)
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


# ── Model ─────────────────────────────────────────────────────────────────────

class BIOTagger(torch.nn.Module):
    def __init__(self, model_name, dropout=0.1):
        super().__init__()
        self.bert   = AutoModel.from_pretrained(model_name)
        hidden_size = self.bert.config.hidden_size
        self.drop   = torch.nn.Dropout(dropout)
        self.head   = torch.nn.Linear(hidden_size, NUM_LABELS)

    def forward(self, input_ids, attention_mask, token_type_ids):
        out    = self.bert(input_ids=input_ids, attention_mask=attention_mask,
                           token_type_ids=token_type_ids)
        seq    = self.drop(out.last_hidden_state)
        return self.head(seq)


# ── Span decoding ─────────────────────────────────────────────────────────────

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

    span_tokens = []
    in_span = False
    for tok_idx, label in first_subtokens:
        if label == LABEL2ID['B-IDIOM']:
            span_tokens = [tok_idx]
            in_span = True
        elif label == LABEL2ID['I-IDIOM'] and in_span:
            span_tokens.append(tok_idx)
        elif in_span:
            break

    if not span_tokens:
        return None, None

    first_tok, last_tok = span_tokens[0], span_tokens[-1]
    if first_tok >= len(offsets) or last_tok >= len(offsets):
        return None, None

    char_start = offsets[first_tok][0]
    char_end   = min(offsets[last_tok][1], len(sentence))
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


# ── Eval ──────────────────────────────────────────────────────────────────────

def evaluate_overlap(model, loader, tokenizer, examples, device, max_len):
    """Returns per-language overlap F1 dict and overall mean."""
    model.eval()
    lang_f1 = defaultdict(list)

    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            input_ids      = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            token_type_ids = batch['token_type_ids'].to(device)

            logits = model(input_ids, attention_mask, token_type_ids)
            preds  = torch.argmax(logits, dim=-1).cpu()

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

                enc = tokenizer(sentence, max_length=max_len, truncation=True,
                                return_offsets_mapping=True)
                pred_bio = preds[i].tolist()
                pred_s, pred_e = decode_bio_to_char_span(pred_bio, enc, sentence, max_len)
                lang_f1[lang].append(compute_overlap_f1(pred_s, pred_e, gold_s, gold_e))

    per_lang = {lang: float(np.mean(scores)) for lang, scores in lang_f1.items()}

    # Objective: HI + TE average
    low_resource = [per_lang.get('Hindi', 0.0), per_lang.get('Telugu', 0.0)]
    objective_score = float(np.mean(low_resource))
    return per_lang, objective_score


# ── Search space ──────────────────────────────────────────────────────────────

def sample_config(trial: optuna.Trial, max_epochs: int) -> dict:
    """Define the hyperparameter search space."""
    lr           = trial.suggest_float('lr',           1e-5, 5e-5, log=True)
    batch_size   = trial.suggest_categorical('batch_size',   [16, 32])
    epochs       = trial.suggest_int('epochs',         3, max_epochs)
    warmup_ratio = trial.suggest_float('warmup_ratio', 0.03, 0.15)
    o_weight     = trial.suggest_float('o_weight',     0.05, 0.5, log=True)
    dropout      = trial.suggest_float('dropout',      0.1, 0.3)

    return {
        'lr':           lr,
        'batch_size':   batch_size,
        'epochs':       epochs,
        'warmup_ratio': warmup_ratio,
        'o_weight':     o_weight,
        'dropout':      dropout,
    }


# ── Trial runner ──────────────────────────────────────────────────────────────

def run_trial(config: dict, trial_num: int, args, train_ds, dev_ds, tokenizer, device) -> tuple[float, dict]:
    """Runs a single in-process trial using the specified configuration parameters."""
    lr           = config['lr']
    batch_size   = config['batch_size']
    epochs       = config['epochs']
    warmup_ratio = config['warmup_ratio']
    o_weight     = config['o_weight']
    dropout      = config['dropout']

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    dev_loader   = DataLoader(dev_ds,   batch_size=batch_size)

    model     = BIOTagger(FIXED['model_name'], dropout=dropout).to(device)
    class_wts = torch.tensor([o_weight, 1.0, 1.0], dtype=torch.float).to(device)
    criterion = torch.nn.CrossEntropyLoss(
        weight=class_wts, ignore_index=IGNORE_IDX, reduction='none'
    )

    optimizer    = AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    total_steps  = len(train_loader) * epochs
    warmup_steps = int(total_steps * warmup_ratio)
    scheduler    = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    best_score = 0.0
    best_per_lang = {}

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0

        for batch in tqdm(train_loader, desc=f"  T{trial_num:03d} E{epoch}", leave=False):
            input_ids       = batch['input_ids'].to(device)
            attention_mask  = batch['attention_mask'].to(device)
            token_type_ids  = batch['token_type_ids'].to(device)
            bio_labels      = batch['bio_labels'].to(device)
            example_weights = batch['example_weight'].to(device)

            logits     = model(input_ids, attention_mask, token_type_ids)
            token_loss = criterion(logits.view(-1, NUM_LABELS), bio_labels.view(-1))
            token_loss = token_loss.view(bio_labels.shape[0], -1)
            valid_mask = (bio_labels != IGNORE_IDX).float()
            denom      = valid_mask.sum(dim=1).clamp(min=1)
            per_ex     = (token_loss * valid_mask).sum(dim=1) / denom
            loss       = (per_ex * example_weights).mean()

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            total_loss += loss.item()

        per_lang, score = evaluate_overlap(
            model, dev_loader, tokenizer, dev_ds.valid_examples, device, FIXED['max_len']
        )
        lang_str = '  '.join(f"{l[:2]}={v:.4f}" for l, v in sorted(per_lang.items()))
        print(f"  Epoch {epoch}: {lang_str}  →  HI+TE avg={score:.4f}")

        if score > best_score:
            best_score = score
            best_per_lang = per_lang

    return best_score, best_per_lang


# ── Optuna objective ──────────────────────────────────────────────────────────

def make_objective(args, train_ds, dev_ds, tokenizer, device):
    def objective(trial: optuna.Trial) -> float:
        config    = sample_config(trial, args.max_epochs)
        trial_num = trial.number

        print(f"\n{'='*70}")
        print(f"TRIAL {trial_num}  |  {datetime.now().strftime('%H:%M:%S')}")
        print(f"Config: {config}")
        print(f"{'='*70}")

        t0      = time.time()
        best_score, best_per_lang = run_trial(config, trial_num, args, train_ds, dev_ds, tokenizer, device)
        elapsed = time.time() - t0

        print(f"\n  ✓ HI+TE avg F1={best_score:.4f}  ({elapsed/60:.1f} min)")

        # Log all metrics as Optuna user attributes for inspection later
        trial.set_user_attr('hi_te_avg', best_score)
        for lang, val in best_per_lang.items():
            trial.set_user_attr(f'{lang.lower()}_f1', val)
        trial.set_user_attr('elapsed_min',  round(elapsed / 60, 1))

        # Save to summary log
        SWEEP_DIR.mkdir(parents=True, exist_ok=True)
        row = {
            'trial':        trial_num,
            'config':       config,
            'hi_te_avg':    round(best_score, 4),
            'elapsed_min':  round(elapsed / 60, 1),
            'timestamp':    datetime.now().isoformat(),
        }
        for lang, val in best_per_lang.items():
            row[f'{lang.lower()}_f1'] = round(val, 4)

# ... (keep the existing dictionary logging code here) ...

        with open(SUMMARY_PATH, 'a') as f:
            f.write(json.dumps(row) + '\n')

        # NEW: Clear Colab output and print an updated leaderboard immediately
        try:
            from google.colab import output
            output.clear()  # Wipes the old wall of text clean
        except ImportError:
            pass

        # Print the immediate Top Leaderboard right in place
        if SUMMARY_PATH.exists():
            rows = [json.loads(l) for l in open(SUMMARY_PATH)]
            rows.sort(key=lambda r: r['hi_te_avg'], reverse=True)
            print(f"\n── CURRENT TOP TRIALS (Updated {datetime.now().strftime('%H:%M:%S')}) ──")
            print(f"{'#':<5} {'HI+TE avg':>10} {'English':>9} {'Hindi':>9} {'Telugu':>10} {'time(min)':>10}")
            print('─' * 60)
            for r in rows[:10]:
                en = r.get('english_f1', 0.0)
                hi = r.get('hindi_f1', 0.0)
                te = r.get('telugu_f1', 0.0)
                print(f"{r['trial']:<5} {r['hi_te_avg']:>10.4f} {en:>9.4f} {hi:>9.4f} {te:>10.4f} {r['elapsed_min']:>10.1f}")

        return best_score

        return best_score  # ← this is what Optuna maximises

    return objective


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args   = parse_args()
    device = get_device(args.device)
    print(f"Device: {device}")

    SWEEP_DIR.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(FIXED['seed'])
    np.random.seed(FIXED['seed'])

    print("Loading data...")
    train_examples = load_split(FIXED['data_dir'], 'train', FIXED['langs'])
    dev_examples   = load_split(FIXED['data_dir'], 'dev',   FIXED['langs'])

    tokenizer    = AutoTokenizer.from_pretrained(FIXED['model_name'])
    cell_weights = compute_lang_loss_weights(train_examples)

    print("Tokenising...")
    train_ds = BIODataset(train_examples, tokenizer, FIXED['max_len'], cell_weights)
    dev_ds   = BIODataset(dev_examples,   tokenizer, FIXED['max_len'])
    print(f"Train: {len(train_ds)}  Dev: {len(dev_ds)}")

    sampler = TPESampler(seed=FIXED['seed'])

    # load_if_exists=True means you can resume a previous study by reusing the same study_name
    study = optuna.create_study(
        study_name    = args.study_name,
        direction     = 'maximize',
        sampler       = sampler,
        pruner        = optuna.pruners.NopPruner(),  # Pruning disabled completely
        storage       = f'sqlite:///{STUDY_DB}',
        load_if_exists= True,
    )

    print(f"Study: {args.study_name}")
    print(f"DB:    {STUDY_DB}")
    print(f"Optimising: HI+TE average overlap F1 on dev set")
    print(f"Running {args.n_trials} trials...\n")

    study.optimize(
        make_objective(args, train_ds, dev_ds, tokenizer, device),
        n_trials  = args.n_trials,
        timeout   = args.timeout,
        show_progress_bar = True,
    )

    # ── Results ───────────────────────────────────────────────────────────────
    best = study.best_trial
    print(f"\n{'='*70}")
    print(f"SWEEP COMPLETE — {len(study.trials)} total trials")
    print(f"Best HI+TE avg F1 : {best.value:.4f}")
    print(f"Best trial        : #{best.number}")
    print(f"Best config       :")
    for k, v in best.params.items():
        print(f"  {k:<22} {v}")
    print(f"{'='*70}")

    best_out = {
        'study_name':       args.study_name,
        'best_trial':       best.number,
        'best_hi_te_avg':   best.value,
        'best_english_f1':  best.user_attrs.get('english_f1'),
        'best_hindi_f1':    best.user_attrs.get('hindi_f1'),
        'best_telugu_f1':   best.user_attrs.get('telugu_f1'),
        'config':           best.params,
    }
    out_path = SWEEP_DIR / 'best_config.json'
    json.dump(best_out, open(out_path, 'w'), indent=2)
    print(f"\nBest config saved → {out_path}")

    # Leaderboard
    if SUMMARY_PATH.exists():
        rows = [json.loads(l) for l in open(SUMMARY_PATH)]
        rows.sort(key=lambda r: r['hi_te_avg'], reverse=True)
        print(f"\n── Top 10 trials by HI+TE avg F1 ──")
        print(f"{'#':<5} {'HI+TE avg':>10} {'English':>9} {'Hindi':>9} {'Telugu':>10} {'time(min)':>10}  config")
        print('─' * 95)
        for r in rows[:10]:
            cfg_str = f"lr={r['config']['lr']:.2e}  bs={r['config']['batch_size']}  ep={r['config']['epochs']}  warm={r['config']['warmup_ratio']:.3f}  o_wt={r['config']['o_weight']:.3f}  drop={r['config']['dropout']:.2f}"
            en = r.get('english_f1', 0.0)
            hi = r.get('hindi_f1', 0.0)
            te = r.get('telugu_f1', 0.0)
            print(f"{r['trial']:<5} {r['hi_te_avg']:>10.4f} {en:>9.4f} {hi:>9.4f} {te:>10.4f} {r['elapsed_min']:>10.1f}  {cfg_str}")

    # Importance analysis — which hyperparameter mattered most
    try:
        importance = optuna.importance.get_param_importances(study)
        print(f"\n── Hyperparameter importance ──")
        for param, imp in sorted(importance.items(), key=lambda x: -x[1]):
            bar = '█' * int(imp * 40)
            print(f"  {param:<22} {imp:.3f}  {bar}")
    except Exception:
        pass  # needs >= 5 completed trials


if __name__ == '__main__':
    main()