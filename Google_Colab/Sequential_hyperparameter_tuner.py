"""
sequential_hparam_search.py

Optuna TPE hyperparameter search for the Sequential two-phase mBERT model.
Tunes Phase 1 and Phase 2 jointly — Phase 2 always initialises from the
Phase 1 checkpoint produced in the same trial.

Objective: macro-avg of Hindi + Telugu joint F1 on dev set
(consistent with joint model and BIO tagger searches).

── Colab / session-timeout safe ──────────────────────────────────────────────
Optuna study is backed by SQLite. Re-running resumes automatically.
Per-trial checkpoints written immediately after each trial completes.

Recommended Colab setup:
    from google.colab import drive
    drive.mount('/content/drive')

    !python sequential_hparam_search.py \\
        --data_dir  idioms_structured/Splits \\
        --output_dir /content/drive/MyDrive/idiom_research/sequential_hparam \\
        --n_trials 30
──────────────────────────────────────────────────────────────────────────────
"""

import json
import shutil
import argparse
import numpy as np
from pathlib import Path
from collections import defaultdict, Counter

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup
from torch.optim import AdamW
from sklearn.metrics import f1_score
from tqdm import tqdm

try:
    import optuna
    from optuna.samplers import TPESampler
    from optuna.pruners import MedianPruner
except ImportError:
    raise ImportError("pip install optuna")

LABEL2ID = {'literal': 0, 'idiomatic': 1}
ID2LABEL = {0: 'literal', 1: 'idiomatic'}


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
    except Exception as e:
        print(f"Could not mount Drive: {e}")


def save_trial_checkpoint(output_dir, trial_number, params, score, per_lang):
    ckpt_dir = Path(output_dir) / 'trial_checkpoints'
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt = {'trial': trial_number, 'score': score,
            'per_lang': per_lang, 'params': params}
    with open(ckpt_dir / f'trial_{trial_number:03d}.json', 'w') as f:
        json.dump(ckpt, f, indent=2)


def load_existing_checkpoints(output_dir):
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
    p.add_argument('--data_dir',    default='idioms_structured/Splits')
    p.add_argument('--output_dir',  default='models/sequential_hparam')
    p.add_argument('--langs',       nargs='+', default=['English', 'Hindi', 'Telugu'])
    p.add_argument('--n_trials',    type=int, default=30)
    p.add_argument('--max_len',     type=int, default=128)
    p.add_argument('--seed',        type=int, default=42)
    p.add_argument('--device',      default=None)
    p.add_argument('--mount_drive', action='store_true')
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
    return [json.loads(l) for l in open(path, encoding='utf-8')
            if json.loads(l)['language'] in langs_set]


def compute_cls_loss_weights(examples, device):
    cell_counts = Counter((ex['language'], ex['idiomaticity']) for ex in examples)
    total       = sum(cell_counts.values())
    n_cells     = len(cell_counts)
    cell_w      = {c: total / (n_cells * n) for c, n in cell_counts.items()}
    class_w     = defaultdict(float)
    class_n     = defaultdict(int)
    for ex in examples:
        cls = ex['idiomaticity']
        class_w[cls] += cell_w[(ex['language'], cls)]
        class_n[cls] += 1
    lit_w = class_w['literal']   / class_n['literal']
    idi_w = class_w['idiomatic'] / class_n['idiomatic']
    min_w = min(lit_w, idi_w)
    return torch.tensor([lit_w / min_w, idi_w / min_w], dtype=torch.float).to(device)


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
            encoding = tokenizer(
                ex['sentence'], max_length=max_len, padding='max_length',
                truncation=True, return_tensors='pt',
            )
            cls_label = LABEL2ID[ex['idiomaticity']]
            token_start, token_end = char_to_token_span(
                tokenizer(
                    ex['sentence'], max_length=max_len, truncation=True,
                    return_offsets_mapping=True,
                ),
                ex['span_start'], ex['span_end'], ex['sentence']
            )
            if token_start is None:
                skipped += 1
                continue

            token_start = min(token_start, max_len - 1)
            token_end   = min(token_end,   max_len - 1)

            tid = encoding.get('token_type_ids')
            seq_len = encoding['input_ids'].shape[1]
            self.valid_examples.append(ex)
            self.input_ids.append(encoding['input_ids'].squeeze(0))
            self.attention_masks.append(encoding['attention_mask'].squeeze(0))
            self.token_type_ids.append(
                tid.squeeze(0) if tid is not None
                else torch.zeros(seq_len, dtype=torch.long)
            )
            self.cls_labels.append(cls_label)
            self.start_positions.append(token_start)
            self.end_positions.append(token_end)

        if skipped:
            print(f"  Skipped {skipped} examples with unresolvable spans")

    def __len__(self):
        return len(self.valid_examples)

    def __getitem__(self, idx):
        return {
            'input_ids':       self.input_ids[idx],
            'attention_mask':  self.attention_masks[idx],
            'token_type_ids':  self.token_type_ids[idx],
            'cls_label':       torch.tensor(self.cls_labels[idx],      dtype=torch.long),
            'start_position':  torch.tensor(self.start_positions[idx], dtype=torch.long),
            'end_position':    torch.tensor(self.end_positions[idx],   dtype=torch.long),
        }


# ── Model ──────────────────────────────────────────────────────────────────────

class JointIdiomModel(torch.nn.Module):
    def __init__(self, model_name, dropout=0.1):
        super().__init__()
        self.bert      = AutoModel.from_pretrained(model_name)
        hidden_size    = self.bert.config.hidden_size
        self.drop      = torch.nn.Dropout(dropout)
        self.cls_head  = torch.nn.Linear(hidden_size, 2)
        self.start_head = torch.nn.Linear(hidden_size, 1)
        self.end_head   = torch.nn.Linear(hidden_size, 1)

    def forward(self, input_ids, attention_mask, token_type_ids):
        out     = self.bert(input_ids=input_ids, attention_mask=attention_mask,
                            token_type_ids=token_type_ids)
        pooled  = self.drop(out.last_hidden_state[:, 0, :])
        seq     = self.drop(out.last_hidden_state)
        cls_logits  = self.cls_head(pooled)
        start_logits = self.start_head(seq).squeeze(-1)
        end_logits   = self.end_head(seq).squeeze(-1)
        mask = attention_mask.bool()
        start_logits = start_logits.masked_fill(~mask, float('-inf'))
        end_logits   = end_logits.masked_fill(~mask, float('-inf'))
        return cls_logits, start_logits, end_logits


def freeze_for_phase2(model, unfreeze_top_layers):
    """Freeze all encoder layers and cls head; unfreeze top N layers + span heads."""
    for param in model.parameters():
        param.requires_grad = False
    # Unfreeze top N transformer layers
    encoder_layers = model.bert.encoder.layer
    n_layers = len(encoder_layers)
    for layer_idx in range(n_layers - unfreeze_top_layers, n_layers):
        for param in encoder_layers[layer_idx].parameters():
            param.requires_grad = True
    # Always unfreeze span heads
    for param in model.start_head.parameters():
        param.requires_grad = True
    for param in model.end_head.parameters():
        param.requires_grad = True
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    print(f"  Trainable params: {trainable:,} / {total:,} ({100*trainable/total:.1f}%)")


def save_model(model, tokenizer, output_dir):
    path = Path(output_dir) / 'best_model'
    path.mkdir(parents=True, exist_ok=True)
    model.bert.save_pretrained(path)
    tokenizer.save_pretrained(path)
    torch.save({
        'cls_head':   model.cls_head.state_dict(),
        'start_head': model.start_head.state_dict(),
        'end_head':   model.end_head.state_dict(),
    }, path / 'task_heads.pt')


def load_model(model_name, output_dir, device, dropout=0.1):
    best = Path(output_dir) / 'best_model'
    model = JointIdiomModel(model_name, dropout=dropout)
    model.bert = AutoModel.from_pretrained(best)
    heads = torch.load(best / 'task_heads.pt', map_location='cpu')
    model.cls_head.load_state_dict(heads['cls_head'])
    model.start_head.load_state_dict(heads['start_head'])
    model.end_head.load_state_dict(heads['end_head'])
    return model.to(device)


# ── Eval ───────────────────────────────────────────────────────────────────────

def evaluate_dev(model, loader, tokenizer, examples, device, max_len):
    """Returns cls macro F1 and per-language joint macro F1."""
    model.eval()
    all_cls_gold, all_cls_pred = [], []
    lang_joint_gold = defaultdict(list)
    lang_joint_pred = defaultdict(list)

    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            cls_logits, start_logits, end_logits = model(
                batch['input_ids'].to(device),
                batch['attention_mask'].to(device),
                batch['token_type_ids'].to(device),
            )
            cls_preds   = torch.argmax(cls_logits, dim=-1).cpu().tolist()
            start_preds = torch.argmax(start_logits, dim=-1).cpu().tolist()
            end_preds   = torch.argmax(end_logits, dim=-1).cpu().tolist()

            for i in range(len(cls_preds)):
                ex_idx = batch_idx * loader.batch_size + i
                if ex_idx >= len(examples):
                    break
                ex        = examples[ex_idx]
                gold_cls  = LABEL2ID[ex['idiomaticity']]
                pred_cls  = cls_preds[i]
                all_cls_gold.append(gold_cls)
                all_cls_pred.append(pred_cls)

                # Joint macro-F1 logic matching Evaluation/Full_evaluation.py:
                # literals are class 0; idiomatic examples require correct cls
                # and any span overlap to count as class 1.
                gold_s, gold_e = ex['span_start'], ex['span_end']
                if gold_cls == LABEL2ID['literal']:
                    gold_joint = 0
                    pred_joint = 0 if pred_cls == LABEL2ID['literal'] else 1
                else:
                    gold_joint = 1
                    if pred_cls != LABEL2ID['idiomatic']:
                        pred_joint = 0
                    else:
                        pred_start = start_preds[i]
                        pred_end = end_preds[i]
                        if pred_end < pred_start:
                            pred_end = pred_start
                        ps, pe = token_to_char_span(
                            tokenizer, ex['sentence'],
                            pred_start, pred_end, max_len
                        )
                        if ps is None:
                            pred_joint = 0
                        else:
                            f1 = compute_overlap_f1(ps, pe, gold_s, gold_e)
                            pred_joint = 1 if f1 > 0.0 else 0
                lang_joint_gold[ex['language']].append(gold_joint)
                lang_joint_pred[ex['language']].append(pred_joint)

    cls_f1   = f1_score(all_cls_gold, all_cls_pred, average='macro', zero_division=0)
    per_lang_joint = {
        l: float(f1_score(lang_joint_gold[l], lang_joint_pred[l],
                          average='macro', zero_division=0))
        for l in lang_joint_gold
    }
    hi_te_joint = float(np.mean([
        per_lang_joint.get('Hindi', 0.0),
        per_lang_joint.get('Telugu', 0.0),
    ]))
    return cls_f1, per_lang_joint, hi_te_joint


# ── Training loop ──────────────────────────────────────────────────────────────

def train_phase(
    model, optimizer, scheduler, cls_criterion, span_criterion,
    cls_loss_weight, span_loss_weight,
    train_loader, dev_loader, tokenizer,
    dev_examples, device, n_epochs, max_len,
    save_criterion, output_dir, phase_label,
    trial=None, pruning_offset=0,
):
    """
    save_criterion: 'cls_f1' (phase 1) or 'hi_te_joint' (phase 2)
    pruning_offset: epoch number to offset for Optuna reporting (phase2 continues from phase1)
    Returns best_score, best_per_lang_joint
    """
    best_score, best_per_lang = 0.0, {}

    for epoch in range(1, n_epochs + 1):
        model.train()
        for batch in tqdm(train_loader, desc=f"  {phase_label} E{epoch}", leave=False):
            cls_logits, start_logits, end_logits = model(
                batch['input_ids'].to(device),
                batch['attention_mask'].to(device),
                batch['token_type_ids'].to(device),
            )
            cls_loss  = cls_criterion(cls_logits, batch['cls_label'].to(device))
            start_loss = span_criterion(
                start_logits, batch['start_position'].to(device)
            )
            end_loss = span_criterion(
                end_logits, batch['end_position'].to(device)
            )
            span_loss = (start_loss + end_loss) / 2
            loss = cls_loss_weight * cls_loss + span_loss_weight * span_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step(); scheduler.step(); optimizer.zero_grad()

        cls_f1, per_lang, hi_te = evaluate_dev(
            model, dev_loader, tokenizer, dev_examples, device, max_len
        )
        lang_str = '  '.join(f"{l[:2]}={v:.4f}" for l, v in sorted(per_lang.items()))
        print(f"  {phase_label} Epoch {epoch}: cls={cls_f1:.4f}  {lang_str}  HI+TE={hi_te:.4f}")

        score = cls_f1 if save_criterion == 'cls_f1' else hi_te
        if score > best_score:
            best_score  = score
            best_per_lang = per_lang
            save_model(model, tokenizer, output_dir)

        if trial is not None:
            trial.report(hi_te, pruning_offset + epoch)
            if trial.should_prune():
                print(f"  Pruned at {phase_label} epoch {epoch}")
                raise optuna.exceptions.TrialPruned()

    return best_score, best_per_lang


# ── Single trial ───────────────────────────────────────────────────────────────

def run_trial(trial, args, train_ds, dev_ds, tokenizer, device, output_dir):
    # ── Search space ──────────────────────────────────────────────────────────
    # Phase 1
    p1_lr           = trial.suggest_float('p1_lr',           5e-6, 3e-5, log=True)
    p1_batch_size   = trial.suggest_categorical('p1_batch_size', [16, 32])
    p1_epochs       = trial.suggest_int('p1_epochs',         4, 8)
    p1_warmup_ratio = trial.suggest_float('p1_warmup_ratio', 0.05, 0.15)
    p1_cls_weight   = trial.suggest_float('p1_cls_weight',   0.5, 0.8)

    # Phase 2
    p2_lr             = trial.suggest_float('p2_lr',             1e-5, 5e-5, log=True)
    p2_batch_size     = trial.suggest_categorical('p2_batch_size', [16, 32])
    p2_epochs         = trial.suggest_int('p2_epochs',            3, 6)
    p2_warmup_ratio   = trial.suggest_float('p2_warmup_ratio',    0.03, 0.12)
    p2_span_weight    = trial.suggest_float('p2_span_weight',     0.5, 0.9)
    unfreeze_top      = trial.suggest_int('unfreeze_top_layers',  2, 5)
    dropout           = trial.suggest_float('dropout',            0.1, 0.3)

    print(f"\nTrial {trial.number}:")
    print(f"  P1: lr={p1_lr:.2e} bs={p1_batch_size} ep={p1_epochs} "
          f"warmup={p1_warmup_ratio:.3f} cls_w={p1_cls_weight:.2f}")
    print(f"  P2: lr={p2_lr:.2e} bs={p2_batch_size} ep={p2_epochs} "
          f"warmup={p2_warmup_ratio:.3f} span_w={p2_span_weight:.2f} "
          f"unfreeze={unfreeze_top} drop={dropout:.2f}")

    trial_dir = Path(output_dir) / f'trial_{trial.number:03d}'
    p1_dir    = trial_dir / 'phase1'
    p1_dir.mkdir(parents=True, exist_ok=True)

    cls_weights = compute_cls_loss_weights(train_ds.valid_examples, device)
    cls_crit    = torch.nn.CrossEntropyLoss(weight=cls_weights)
    span_crit   = torch.nn.CrossEntropyLoss()

    # ── Phase 1 ───────────────────────────────────────────────────────────────
    print(f"\n  [P1] Classification-dominant (cls_w={p1_cls_weight:.2f})")
    model = JointIdiomModel(args.model_name, dropout=dropout).to(device)
    train_loader = DataLoader(train_ds, batch_size=p1_batch_size, shuffle=True)
    dev_loader   = DataLoader(dev_ds,   batch_size=p1_batch_size)
    total_steps  = len(train_loader) * p1_epochs
    optimizer    = AdamW(model.parameters(), lr=p1_lr, weight_decay=0.01)
    scheduler    = get_linear_schedule_with_warmup(
        optimizer, int(total_steps * p1_warmup_ratio), total_steps
    )

    try:
        train_phase(
            model, optimizer, scheduler, cls_crit, span_crit,
            cls_loss_weight=p1_cls_weight, span_loss_weight=1 - p1_cls_weight,
            train_loader=train_loader, dev_loader=dev_loader,
            tokenizer=tokenizer, dev_examples=dev_ds.valid_examples,
            device=device, n_epochs=p1_epochs, max_len=args.max_len,
            save_criterion='cls_f1', output_dir=p1_dir,
            phase_label='P1', trial=trial, pruning_offset=0,
        )
    except optuna.exceptions.TrialPruned:
        save_trial_checkpoint(output_dir, trial.number, trial.params, 0.0, {})
        raise

    # ── Phase 2 ───────────────────────────────────────────────────────────────
    print(f"\n  [P2] Span-dominant, initialised from P1 (span_w={p2_span_weight:.2f})")
    model = load_model(args.model_name, p1_dir, device, dropout=dropout)
    freeze_for_phase2(model, unfreeze_top)

    train_loader = DataLoader(train_ds, batch_size=p2_batch_size, shuffle=True)
    dev_loader   = DataLoader(dev_ds,   batch_size=p2_batch_size)
    total_steps  = len(train_loader) * p2_epochs
    trainable    = [p for p in model.parameters() if p.requires_grad]
    optimizer    = AdamW(trainable, lr=p2_lr, weight_decay=0.01)
    scheduler    = get_linear_schedule_with_warmup(
        optimizer, int(total_steps * p2_warmup_ratio), total_steps
    )

    p2_dir = trial_dir / 'phase2'
    p2_dir.mkdir(parents=True, exist_ok=True)

    try:
        best_score, best_per_lang = train_phase(
            model, optimizer, scheduler, cls_crit, span_crit,
            cls_loss_weight=1 - p2_span_weight, span_loss_weight=p2_span_weight,
            train_loader=train_loader, dev_loader=dev_loader,
            tokenizer=tokenizer, dev_examples=dev_ds.valid_examples,
            device=device, n_epochs=p2_epochs, max_len=args.max_len,
            save_criterion='hi_te_joint', output_dir=p2_dir,
            phase_label='P2', trial=trial, pruning_offset=p1_epochs,
        )
    except optuna.exceptions.TrialPruned:
        save_trial_checkpoint(output_dir, trial.number, trial.params, 0.0, {})
        raise

    # Save checkpoint then clean up large trial dir to save Drive space
    save_trial_checkpoint(output_dir, trial.number,
                          trial.params, best_score, best_per_lang)
    shutil.rmtree(trial_dir, ignore_errors=True)

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

    print("Tokenising...")
    train_ds = JointDataset(train_examples, tokenizer, args.max_len)
    dev_ds   = JointDataset(dev_examples,   tokenizer, args.max_len)
    print(f"Train: {len(train_ds)}  Dev: {len(dev_ds)}")

    # ── SQLite-backed study ────────────────────────────────────────────────────
    db_path    = output_dir / 'study.db'
    storage    = f'sqlite:///{db_path}'
    # v2 avoids mixing older trials that used a different model-head layout and
    # optimized joint correctness rate rather than joint macro F1.
    study_name = 'sequential_hparam_v2_joint_macro_f1'

    print(f"\nOptuna DB: {db_path}")
    print("Resuming." if db_path.exists() else "Starting new study.")

    study = optuna.create_study(
        study_name=study_name,
        storage=storage,
        load_if_exists=True,
        direction='maximize',
        sampler=TPESampler(seed=args.seed),
        pruner=MedianPruner(n_startup_trials=5, n_warmup_steps=3),
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
    print(f"  Trial      : {best.number}")
    print(f"  HI+TE F1   : {best.value:.4f}")
    for k, v in best.params.items():
        print(f"  {k:<25} = {v}")

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
    print("\nBest config as CLI args for train_sequential.py:")
    print(f"  --p1_lr {p['p1_lr']:.2e} --p1_batch_size {p['p1_batch_size']} "
          f"--p1_epochs {p['p1_epochs']} --p1_warmup_ratio {p['p1_warmup_ratio']:.3f} "
          f"--p1_cls_weight {p['p1_cls_weight']:.2f} "
          f"--p1_span_weight {1 - p['p1_cls_weight']:.2f} \\")
    print(f"  --p2_lr {p['p2_lr']:.2e} --p2_batch_size {p['p2_batch_size']} "
          f"--p2_epochs {p['p2_epochs']} --p2_warmup_ratio {p['p2_warmup_ratio']:.3f} "
          f"--p2_cls_weight {1 - p['p2_span_weight']:.2f} "
          f"--p2_span_weight {p['p2_span_weight']:.2f} "
          f"--unfreeze_top_layers {p['unfreeze_top_layers']} "
          f"--dropout {p['dropout']:.3f}")


if __name__ == '__main__':
    main()
