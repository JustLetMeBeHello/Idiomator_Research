"""
run_06_idiombert_v2.py
======================

IdiomBERT-v2: a unified training script that toggles three MWE-specific
training innovations on top of the QA-style multi-task baseline, for a
benchmark study on multilingual / non-Latin MWE detection.

Components (each flag-toggleable):

  --use_scl          Span Contrastive Loss (SupCon-style, in-batch).
                     For each anchor (figurative use of idiom X), positives
                     are other figurative uses of X in the batch; hard
                     negatives are literal uses of X in the batch.
                     Matheny et al. 2026 (arXiv 2603.22799) introduced
                     SCL+HNR for English idiomaticity; we extend to 5-lang.

  --use_hnr          Hard-Negative Reweighting on the classification +
                     span losses: literal examples (the natural hard
                     negatives) are upweighted by --hnr_lit_weight.

  --use_li           Lateral Inhibition at inference: enumerate top-k
                     candidate (start, end) spans, suppress overlapping
                     candidates with lower joint score (learned-NMS-style,
                     following Avram et al. 2023 arXiv 2306.10419 for MWE).
                     Output-side only — encoder-side variant left as TODO.

Composability: any subset of the three flags can be set. The all-three
config is "IdiomBERT-v2-full". Recommended ablation matrix is 8 cells:
  baseline / +SCL / +HNR / +LI / +SCL+HNR / +SCL+LI / +HNR+LI / all-three

Usage (single full-config run on EN+ES+HI+TE, evaluated on all five):
    python experiments/rigor/run_06_idiombert_v2.py \
        --output_dir models/idiombert_v2/full \
        --langs English Spanish Hindi Telugu \
        --test_langs English Spanish Hindi Telugu Indonesian \
        --use_scl --use_hnr --use_li \
        --scl_weight 0.1 --hnr_lit_weight 3.0

Notes
-----
- The existing training/Train_Join.py architecture (mBERT + cls_head + start_head +
  end_head) is the baseline. This script reuses its data pipeline and
  evaluation routines to keep results comparable across the matrix.
- SCL requires batches that contain multiple uses of the same idiom. The
  IdiomBalancedBatchSampler below groups by idiom_id and emits batches
  with at least --scl_min_idioms_per_batch distinct idioms appearing
  ≥2 times. Falls back to random batching when group sizes are too small.
- Saved per-example test predictions remain in the same JSONL format as
  the other systems so Evaluation/Full_evaluation.py works unchanged.
"""

import json
import math
import random
import argparse
from pathlib import Path
from collections import defaultdict, Counter

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Sampler
from torch.optim import AdamW
from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup
from sklearn.metrics import f1_score, classification_report
from tqdm import tqdm

import sys
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
# Reuse data pipeline + alignment helpers from the existing trainer.
from Train_Join import (
    LABEL2ID,
    ID2LABEL,
    char_to_token_span,
    compute_overlap_f1,
    JointDataset,
    JointIdiomModel,
    compute_cls_loss_weights,
)


# ── Args ─────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    # ── Base trainer args (same defaults as training/Train_Join.py) ──
    p.add_argument('--model_name', default='bert-base-multilingual-cased')
    p.add_argument('--data_dir',   default='data/idioms_structured/Splits')
    p.add_argument('--output_dir', required=True)
    p.add_argument('--langs', nargs='+',
                   default=['English', 'Spanish', 'Hindi', 'Telugu'])
    p.add_argument('--test_langs', nargs='+', default=None,
                   help='Defaults to --langs')
    p.add_argument('--epochs',     type=int,   default=7)
    p.add_argument('--batch_size', type=int,   default=32)
    p.add_argument('--lr',         type=float, default=2e-5)
    p.add_argument('--max_len',    type=int,   default=128)
    p.add_argument('--warmup_ratio', type=float, default=0.05)
    p.add_argument('--seed',       type=int,   default=42)
    p.add_argument('--cls_loss_weight',  type=float, default=0.3)
    p.add_argument('--span_loss_weight', type=float, default=1.9)
    p.add_argument('--device', default=None)
    # ── SCL ──
    p.add_argument('--use_scl', action='store_true')
    p.add_argument('--scl_weight', type=float, default=0.1,
                   help='λ_span in Matheny et al.; total loss += scl_weight * L_scl')
    p.add_argument('--scl_temperature', type=float, default=0.07)
    p.add_argument('--scl_proj_dim', type=int, default=128)
    p.add_argument('--scl_min_idioms_per_batch', type=int, default=4,
                   help='Min idioms appearing ≥2× in a batch (else random fallback)')
    # ── HNR ──
    p.add_argument('--use_hnr', action='store_true')
    p.add_argument('--hnr_lit_weight', type=float, default=3.0,
                   help='Upweight factor for literal examples in cls + span losses')
    # ── Lateral inhibition (output-side) ──
    p.add_argument('--use_li', action='store_true')
    p.add_argument('--li_top_k', type=int, default=5,
                   help='Top-k candidate spans to enumerate per example')
    p.add_argument('--li_iou_threshold', type=float, default=0.5,
                   help='Suppress overlapping candidates above this IoU')
    return p.parse_args()


# ── Idiom-balanced batch sampler (for SCL) ───────────────────────────────────

class IdiomBalancedBatchSampler(Sampler):
    """
    Emits batches where ≥ min_idioms_per_batch idioms each appear ≥2 times.
    Falls back to random sampling when group sizes can't satisfy that.

    Strategy: shuffle idiom groups; greedy-pack each batch by picking an
    idiom group at random, taking ≥2 examples from it, then filling the
    remainder with random singletons. Yields one epoch worth of batches.
    """
    def __init__(self, examples, batch_size, min_idioms_per_batch=4, seed=42):
        self.batch_size  = batch_size
        self.min_idioms  = min_idioms_per_batch
        self.rng         = random.Random(seed)
        # Group indices by idiom_id
        self.groups = defaultdict(list)
        for i, ex in enumerate(examples):
            self.groups[ex.get('idiom_id', f'__solo_{i}')].append(i)
        self.n_examples = len(examples)

    def __iter__(self):
        all_indices = list(range(self.n_examples))
        self.rng.shuffle(all_indices)
        used = set()
        # Live pool of group-ids that still have ≥2 unused indices.
        # Maintained explicitly so we never pick an exhausted group twice.
        scl_groups = [gid for gid, idxs in self.groups.items() if len(idxs) >= 2]

        while len(used) < self.n_examples:
            batch = []
            tried_idioms = 0
            # Refresh: drop groups that no longer have ≥2 unused
            scl_groups = [g for g in scl_groups
                          if sum(1 for i in self.groups[g] if i not in used) >= 2]
            while tried_idioms < self.min_idioms and len(batch) < self.batch_size and scl_groups:
                gid = self.rng.choice(scl_groups)
                scl_groups.remove(gid)  # consume — won't be picked again this batch
                pool = [i for i in self.groups[gid] if i not in used]
                self.rng.shuffle(pool)
                take = min(2, self.batch_size - len(batch))
                for i in pool[:take]:
                    batch.append(i); used.add(i)
                tried_idioms += 1
            # Fill remainder with random unused indices
            remaining = [i for i in all_indices if i not in used]
            self.rng.shuffle(remaining)
            for i in remaining:
                if len(batch) >= self.batch_size:
                    break
                batch.append(i); used.add(i)
            if not batch:
                break
            yield batch

    def __len__(self):
        return math.ceil(self.n_examples / self.batch_size)


# ── Augmented dataset: also expose idiom_id and idiomaticity in each item ────

class JointDatasetV2(JointDataset):
    """Extends JointDataset to expose idiom_id + idiomaticity for SCL/HNR."""
    def __getitem__(self, idx):
        item = super().__getitem__(idx)
        ex   = self.valid_examples[idx]
        item['idiom_id_str']    = ex.get('idiom_id', '')
        item['is_literal']      = 1 if ex['idiomaticity'] == 'literal' else 0
        return item


def collate_v2(batch):
    out = {k: torch.stack([b[k] for b in batch]) if torch.is_tensor(batch[0][k])
           else [b[k] for b in batch] for k in batch[0]}
    return out


# ── Model with optional SCL projection head ──────────────────────────────────

class IdiomBERTv2(JointIdiomModel):
    """Adds a projection head over span representations for SCL."""
    def __init__(self, model_name, scl_proj_dim=128, use_scl=False):
        super().__init__(model_name)
        self.use_scl = use_scl
        if use_scl:
            h = self.bert.config.hidden_size
            self.scl_proj = torch.nn.Sequential(
                torch.nn.Linear(h * 2, h),
                torch.nn.ReLU(),
                torch.nn.Linear(h, scl_proj_dim),
            )

    def forward_with_repr(self, input_ids, attention_mask, token_type_ids):
        """Returns (cls_logits, start_logits, end_logits, seq_output)."""
        outputs    = self.bert(input_ids=input_ids,
                               attention_mask=attention_mask,
                               token_type_ids=token_type_ids)
        seq_output = outputs.last_hidden_state
        cls_output = seq_output[:, 0, :]
        cls_logits   = self.cls_head(cls_output)
        start_logits = self.start_head(seq_output).squeeze(-1)
        end_logits   = self.end_head(seq_output).squeeze(-1)
        mask = attention_mask.bool()
        start_logits = start_logits.masked_fill(~mask, float('-inf'))
        end_logits   = end_logits.masked_fill(~mask,   float('-inf'))
        return cls_logits, start_logits, end_logits, seq_output

    def span_repr(self, seq_output, start_idx, end_idx):
        """[start;end] concatenation → projection. Shape [B, scl_proj_dim]."""
        b = seq_output.size(0)
        starts = seq_output[torch.arange(b), start_idx]    # [B, H]
        ends   = seq_output[torch.arange(b), end_idx]      # [B, H]
        cat    = torch.cat([starts, ends], dim=-1)         # [B, 2H]
        z      = self.scl_proj(cat)                        # [B, D]
        return F.normalize(z, dim=-1)


# ── Loss functions ───────────────────────────────────────────────────────────

def span_contrastive_loss(z, idiom_ids, is_literal, temperature=0.07):
    """
    SupCon-style contrastive over span embeddings z [B, D] (L2-normalized).
    For each anchor i:
      positives  P(i) = {j : idiom_ids[j] == idiom_ids[i], j != i, both figurative}
      negatives  N(i) = all other j (literal-of-same-idiom included → hard)
    Loss: -log( sum_p exp(sim_ip/τ) / sum_{j != i} exp(sim_ij/τ) )
    Numerically stable: subtract per-row max, multiply self out of the
    denominator (rather than using -inf, which would NaN through 0 × -inf
    when later combined with the boolean positive mask).
    Anchors with no positive are skipped (returns 0 contribution).
    """
    B = z.size(0)
    if B < 2:
        return z.new_zeros(())

    sim = (z @ z.T) / temperature                            # [B, B]
    # Per-row max subtraction for numerical stability
    sim_max, _ = sim.max(dim=1, keepdim=True)
    sim = sim - sim_max.detach()

    self_mask = torch.eye(B, device=z.device, dtype=torch.bool)
    not_self  = (~self_mask).float()                         # [B, B]

    # Positive mask: same idiom_id (non-empty) AND both figurative
    same_idiom = torch.tensor(
        [[idiom_ids[i] == idiom_ids[j] and idiom_ids[i] != '' for j in range(B)]
         for i in range(B)], device=z.device)
    fig = (torch.tensor(is_literal, device=z.device) == 0)
    both_fig = fig.unsqueeze(0) & fig.unsqueeze(1)
    pos_mask = (same_idiom & both_fig & (~self_mask)).float()  # [B, B]

    has_pos = pos_mask.sum(dim=1) > 0
    if not has_pos.any():
        return z.new_zeros(())

    # Stable log-softmax: log(exp(sim) / sum_{j != i} exp(sim)).
    # exp_sim is finite (no -inf), self is zeroed via not_self.
    exp_sim = torch.exp(sim) * not_self
    log_denom = torch.log(exp_sim.sum(dim=1, keepdim=True).clamp(min=1e-12))
    log_prob = sim - log_denom                               # [B, B]

    pos_count = pos_mask.sum(dim=1).clamp(min=1)
    pos_log_prob = (log_prob * pos_mask).sum(dim=1) / pos_count
    loss = -pos_log_prob[has_pos].mean()
    return loss


def hnr_weighted_ce(logits, targets, is_literal, base_weight, lit_weight):
    """Per-example reweighted cross-entropy. is_literal: [B] in {0,1}."""
    ce = F.cross_entropy(logits, targets, reduction='none')
    w  = torch.where(is_literal.bool(), lit_weight, base_weight)
    return (ce * w).mean()


# ── Lateral inhibition (output-side NMS over candidate spans) ────────────────

def lateral_inhibition_decode(start_logits, end_logits, attention_mask,
                              top_k=5, iou_threshold=0.5, max_span_len=15):
    """
    For each example: enumerate top-k candidate (s, e) spans by joint score
    start_logits[s] + end_logits[e] subject to s ≤ e ≤ s+max_span_len.
    Sort by score; greedily accept; suppress any later candidate with
    IoU > iou_threshold vs an accepted one. Return the surviving top-1.

    Args:
        start_logits, end_logits: [B, T]
        attention_mask:           [B, T]
    Returns:
        pred_starts, pred_ends:   [B] long tensors
    """
    B, T = start_logits.shape
    pred_s = torch.zeros(B, dtype=torch.long, device=start_logits.device)
    pred_e = torch.zeros(B, dtype=torch.long, device=start_logits.device)

    for b in range(B):
        valid_len = int(attention_mask[b].sum().item())
        s_logits  = start_logits[b, :valid_len]
        e_logits  = end_logits[b, :valid_len]

        # Top-k starts and ends
        k = min(top_k, valid_len)
        s_top = torch.topk(s_logits, k).indices.tolist()
        e_top = torch.topk(e_logits, k).indices.tolist()

        # Enumerate candidate (s, e) pairs with valid ordering
        cands = []
        for s in s_top:
            for e in e_top:
                if e < s or (e - s) > max_span_len:
                    continue
                score = float(s_logits[s] + e_logits[e])
                cands.append((score, s, e))
        if not cands:
            # Fallback: argmax of each
            pred_s[b] = int(torch.argmax(s_logits))
            pred_e[b] = max(int(torch.argmax(e_logits)), int(pred_s[b]))
            continue
        cands.sort(reverse=True)
        accepted = []
        for score, s, e in cands:
            if all(_iou(s, e, s2, e2) <= iou_threshold for _, s2, e2 in accepted):
                accepted.append((score, s, e))
            if len(accepted) >= 1:
                break  # we only need top-1 after inhibition
        _, s, e = accepted[0]
        pred_s[b], pred_e[b] = s, e
    return pred_s, pred_e


def _iou(s1, e1, s2, e2):
    inter = max(0, min(e1, e2) - max(s1, s2) + 1)
    union = (e1 - s1 + 1) + (e2 - s2 + 1) - inter
    return inter / union if union > 0 else 0.0


# ── Training loop ────────────────────────────────────────────────────────────

def train_one_epoch(model, loader, optim, sched, device, args,
                    cls_criterion_base, epoch):
    model.train()
    running = defaultdict(float); n = 0
    for batch in tqdm(loader, desc=f'epoch {epoch}', leave=False):
        input_ids      = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        token_type_ids = batch['token_type_ids'].to(device)
        cls_labels     = batch['cls_labels'].to(device)
        start_pos      = batch['start_positions'].to(device)
        end_pos        = batch['end_positions'].to(device)
        is_literal     = torch.tensor(batch['is_literal']).to(device)

        cls_logits, start_logits, end_logits, seq_out = model.forward_with_repr(
            input_ids, attention_mask, token_type_ids)

        # ── Classification + span losses, with optional HNR ──
        if args.use_hnr:
            cls_loss   = hnr_weighted_ce(cls_logits, cls_labels, is_literal,
                                         base_weight=1.0, lit_weight=args.hnr_lit_weight)
            start_loss = hnr_weighted_ce(start_logits, start_pos, is_literal,
                                         base_weight=1.0, lit_weight=args.hnr_lit_weight)
            end_loss   = hnr_weighted_ce(end_logits,   end_pos,   is_literal,
                                         base_weight=1.0, lit_weight=args.hnr_lit_weight)
        else:
            cls_loss   = cls_criterion_base(cls_logits, cls_labels)
            start_loss = F.cross_entropy(start_logits, start_pos)
            end_loss   = F.cross_entropy(end_logits,   end_pos)
        span_loss = (start_loss + end_loss) / 2

        total = args.cls_loss_weight  * cls_loss + \
                args.span_loss_weight * span_loss

        # ── SCL ──
        scl_loss = torch.tensor(0.0, device=device)
        if args.use_scl:
            z = model.span_repr(seq_out, start_pos, end_pos)
            scl_loss = span_contrastive_loss(
                z, batch['idiom_id_str'], batch['is_literal'],
                temperature=args.scl_temperature)
            total = total + args.scl_weight * scl_loss

        optim.zero_grad()
        total.backward()
        optim.step()
        sched.step()

        running['total'] += float(total)
        running['cls']   += float(cls_loss)
        running['span'] += float(span_loss)
        running['scl']  += float(scl_loss)
        n += 1
    return {k: v / n for k, v in running.items()}


# ── Evaluation (extends Train_Join.evaluate with optional LI decode) ─────────

def evaluate_v2(model, loader, examples, device, split_name, args, tokenizer):
    model.eval()
    all_cls_preds, all_cls_labels = [], []
    lang_exact, lang_f1 = defaultdict(list), defaultdict(list)
    lang_cls_preds, lang_cls_labels = defaultdict(list), defaultdict(list)
    per_example_preds = []
    total = 0

    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(loader, desc=f'eval {split_name}', leave=False)):
            input_ids      = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            token_type_ids = batch['token_type_ids'].to(device)

            cls_logits, start_logits, end_logits, _ = model.forward_with_repr(
                input_ids, attention_mask, token_type_ids)

            if args.use_li:
                pred_s, pred_e = lateral_inhibition_decode(
                    start_logits, end_logits, attention_mask,
                    top_k=args.li_top_k, iou_threshold=args.li_iou_threshold)
            else:
                pred_s = torch.argmax(start_logits, dim=-1)
                pred_e = torch.argmax(end_logits,   dim=-1)
                pred_e = torch.maximum(pred_e, pred_s)

            cls_preds = torch.argmax(cls_logits, dim=-1)

            for i in range(len(cls_preds)):
                ex_idx = batch_idx * loader.batch_size + i
                if ex_idx >= len(examples): break
                ex   = examples[ex_idx]
                lang = ex['language']

                gold_s = int(batch['start_positions'][i])
                gold_e = int(batch['end_positions'][i])
                ps, pe = int(pred_s[i]), int(pred_e[i])

                exact = int(ps == gold_s and pe == gold_e)
                f1    = compute_overlap_f1(ps, pe, gold_s, gold_e)

                lang_exact[lang].append(exact); lang_f1[lang].append(f1)
                all_cls_preds.append(int(cls_preds[i]))
                all_cls_labels.append(int(batch['cls_labels'][i]))
                lang_cls_preds[lang].append(int(cls_preds[i]))
                lang_cls_labels[lang].append(int(batch['cls_labels'][i]))
                total += 1

                per_example_preds.append({
                    **{k: ex.get(k) for k in (
                        'language', 'idiom_id', 'idiom', 'meaning_id', 'sense_number',
                        'idiomaticity', 'sentence', 'span_start', 'span_end',
                        'matched_span')},
                    'pred_cls':        ID2LABEL[int(cls_preds[i])],
                    'pred_token_start': ps,
                    'pred_token_end':   pe,
                    'span_exact_match': exact,
                    'span_overlap_f1':  round(f1, 4),
                })

    macro_f1 = f1_score(all_cls_labels, all_cls_preds, average='macro')
    em       = np.mean([v for vs in lang_exact.values() for v in vs])
    of1      = np.mean([v for vs in lang_f1.values()    for v in vs])

    print(f"\n── {split_name}  cls_macro_F1={macro_f1:.4f}  span_EM={em:.4f}  span_F1={of1:.4f}  N={total} ──")
    for lang in sorted(lang_exact.keys()):
        lcm = f1_score(lang_cls_labels[lang], lang_cls_preds[lang], average='macro') \
              if lang_cls_labels[lang] else 0.0
        print(f"  {lang:<12} cls_F1={lcm:.4f}  EM={np.mean(lang_exact[lang]):.4f}  "
              f"F1={np.mean(lang_f1[lang]):.4f}  N={len(lang_exact[lang])}")

    return macro_f1, em, of1, per_example_preds


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    torch.manual_seed(args.seed); np.random.seed(args.seed); random.seed(args.seed)
    device = torch.device(args.device) if args.device else (
        torch.device('cuda') if torch.cuda.is_available()
        else torch.device('mps') if torch.backends.mps.is_available()
        else torch.device('cpu'))
    print(f"Device: {device}")
    print(f"Flags: use_scl={args.use_scl}  use_hnr={args.use_hnr}  use_li={args.use_li}")

    output_dir = Path(args.output_dir); output_dir.mkdir(parents=True, exist_ok=True)
    json.dump(vars(args), open(output_dir / 'config.json', 'w'), indent=2)

    # ── Data ──
    from Train_Join import build_dataset_for_split
    train_ex = build_dataset_for_split('train', args.data_dir, args.langs, args.seed)
    dev_ex   = build_dataset_for_split('dev',   args.data_dir, args.langs, args.seed)
    test_langs = args.test_langs or args.langs
    test_ex  = build_dataset_for_split('test',  args.data_dir, test_langs, args.seed)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    train_ds  = JointDatasetV2(train_ex, tokenizer, args.max_len)
    dev_ds    = JointDatasetV2(dev_ex,   tokenizer, args.max_len)
    test_ds   = JointDatasetV2(test_ex,  tokenizer, args.max_len)
    print(f"Train: {len(train_ds)}  Dev: {len(dev_ds)}  Test: {len(test_ds)}")

    if args.use_scl:
        sampler = IdiomBalancedBatchSampler(train_ds.valid_examples,
                                            batch_size=args.batch_size,
                                            min_idioms_per_batch=args.scl_min_idioms_per_batch,
                                            seed=args.seed)
        train_loader = DataLoader(train_ds, batch_sampler=sampler, collate_fn=collate_v2)
    else:
        train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                                  shuffle=True, collate_fn=collate_v2)
    dev_loader   = DataLoader(dev_ds,  batch_size=args.batch_size, collate_fn=collate_v2)
    test_loader  = DataLoader(test_ds, batch_size=args.batch_size, collate_fn=collate_v2)

    # ── Model ──
    model = IdiomBERTv2(args.model_name, scl_proj_dim=args.scl_proj_dim,
                        use_scl=args.use_scl).to(device)

    # ── Optimizer + schedule ──
    optim = AdamW(model.parameters(), lr=args.lr)
    total_steps = len(train_loader) * args.epochs
    sched = get_linear_schedule_with_warmup(
        optim, num_warmup_steps=int(args.warmup_ratio * total_steps),
        num_training_steps=total_steps)
    cls_weights = compute_cls_loss_weights(train_ds.valid_examples, device)
    cls_criterion_base = torch.nn.CrossEntropyLoss(weight=cls_weights)

    # ── Train ──
    best_dev_f1 = -1.0
    for epoch in range(1, args.epochs + 1):
        losses = train_one_epoch(model, train_loader, optim, sched, device, args,
                                 cls_criterion_base, epoch)
        print(f"Epoch {epoch}: " + " ".join(f"{k}={v:.4f}" for k, v in losses.items()))
        _, dev_em, dev_f1, _ = evaluate_v2(model, dev_loader, dev_ds.valid_examples,
                                            device, 'dev', args, tokenizer)
        if dev_f1 > best_dev_f1:
            best_dev_f1 = dev_f1
            torch.save({'bert': model.bert.state_dict(),
                        'cls_head':   model.cls_head.state_dict(),
                        'start_head': model.start_head.state_dict(),
                        'end_head':   model.end_head.state_dict(),
                        **({'scl_proj': model.scl_proj.state_dict()} if args.use_scl else {})},
                       output_dir / 'best_model.pt')
            print(f"  ↑ new best dev F1 {best_dev_f1:.4f}")

    # ── Test ──
    print("\nLoading best model for final test eval ...")
    ckpt = torch.load(output_dir / 'best_model.pt', map_location=device)
    model.bert.load_state_dict(ckpt['bert'])
    model.cls_head.load_state_dict(ckpt['cls_head'])
    model.start_head.load_state_dict(ckpt['start_head'])
    model.end_head.load_state_dict(ckpt['end_head'])
    if args.use_scl and 'scl_proj' in ckpt:
        model.scl_proj.load_state_dict(ckpt['scl_proj'])
    macro_f1, em, of1, per_ex = evaluate_v2(
        model, test_loader, test_ds.valid_examples, device, 'test', args, tokenizer)

    # ── Save test predictions + metrics ──
    with open(output_dir / 'test_predictions.jsonl', 'w', encoding='utf-8') as f:
        for ex in per_ex:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")
    json.dump({'test_cls_macro_f1': macro_f1, 'test_span_em': em,
               'test_span_f1': of1, 'best_dev_f1': best_dev_f1},
              open(output_dir / 'metrics.json', 'w'), indent=2)
    print(f"\n  Wrote {output_dir / 'test_predictions.jsonl'} and metrics.json")


if __name__ == '__main__':
    main()
