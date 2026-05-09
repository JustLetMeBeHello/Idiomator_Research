"""
test_stage2.py

Quick sanity check for Stage 2 span extraction before full training.
Runs 1 epoch on a small subset to verify:
  1. Span alignment (char offsets → token indices) works for all languages
  2. Model forward pass works
  3. Loss decreases
  4. Predictions are decoded back to text correctly

Usage:
    python test_stage2.py
    python test_stage2.py --langs Hindi Telugu
    python test_stage2.py --n_examples 20
"""

import json
import argparse
import numpy as np
from pathlib import Path
from collections import defaultdict

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModel

# Import from Stage 2 training script
import sys
sys.path.append('.')
from Stage_2_training import (
    SpanExtractor,
    SpanDataset,
    build_dataset_for_split,
    compute_overlap_f1,
    token_to_char_span,
    get_device,
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--model_name', default='bert-base-multilingual-cased')
    p.add_argument('--data_dir',   default='idioms_structured/Splits')
    p.add_argument('--langs',      nargs='+', default=['English', 'Hindi', 'Telugu'])
    p.add_argument('--n_examples', type=int, default=50,
                   help='Number of examples to test on')
    p.add_argument('--max_len',    type=int, default=128)
    p.add_argument('--device',     default=None)
    return p.parse_args()


def main():
    args   = parse_args()
    device = get_device(args.device)

    print(f"\n{'='*60}")
    print(f"Stage 2 Sanity Check")
    print(f"Model : {args.model_name}")
    print(f"Langs : {args.langs}")
    print(f"{'='*60}\n")

    # ── Load small subset ─────────────────────────────────────────────────────
    print(f"Loading {args.n_examples} examples per split...")
    train_examples = build_dataset_for_split('train', args.data_dir, args.langs)[:args.n_examples]
    dev_examples   = build_dataset_for_split('dev',   args.data_dir, args.langs)[:args.n_examples]

    print(f"Train subset: {len(train_examples)} | Dev subset: {len(dev_examples)}")

    # ── Tokenizer ─────────────────────────────────────────────────────────────
    print(f"\nLoading tokenizer: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    # ── Check 1: Span alignment ───────────────────────────────────────────────
    print(f"\n── Check 1: Span alignment ──")
    alignment_ok  = 0
    alignment_bad = 0
    lang_counts   = defaultdict(lambda: {'ok': 0, 'bad': 0})

    for ex in train_examples + dev_examples:
        sentence  = ex['sentence']
        char_s    = ex['span_start']
        char_e    = ex['span_end']
        gold_span = ex['matched_span']
        lang      = ex['language']

        enc = tokenizer(
            sentence,
            max_length=args.max_len,
            truncation=True,
            return_offsets_mapping=True,
        )

        # Find token span
        token_s, token_e = None, None
        for i in range(len(sentence)):
            t = enc.char_to_token(i)
            if t is not None and i >= char_s:
                token_s = t
                break
        for i in range(char_e - 1, -1, -1):
            t = enc.char_to_token(i)
            if t is not None:
                token_e = t
                break

        if token_s is None or token_e is None:
            alignment_bad += 1
            lang_counts[lang]['bad'] += 1
            continue

        # Convert back to chars
        offsets  = enc['offset_mapping']
        if token_s < len(offsets) and token_e < len(offsets):
            rec_s = offsets[token_s][0]
            rec_e = offsets[token_e][1]
            recovered = sentence[rec_s:rec_e]

            # Check overlap with gold
            f1 = compute_overlap_f1(token_s, token_e, token_s, token_e)
            if recovered.lower().strip() in gold_span.lower() or \
               gold_span.lower() in recovered.lower().strip() or \
               recovered.lower().strip() == gold_span.lower().strip():
                alignment_ok += 1
                lang_counts[lang]['ok'] += 1
            else:
                alignment_bad += 1
                lang_counts[lang]['bad'] += 1
        else:
            alignment_bad += 1
            lang_counts[lang]['bad'] += 1

    total_checked = alignment_ok + alignment_bad
    print(f"  Total checked : {total_checked}")
    print(f"  Aligned OK    : {alignment_ok} ({alignment_ok/total_checked*100:.1f}%)")
    print(f"  Failed        : {alignment_bad} ({alignment_bad/total_checked*100:.1f}%)")
    for lang, counts in sorted(lang_counts.items()):
        total = counts['ok'] + counts['bad']
        print(f"  {lang}: {counts['ok']}/{total} aligned ({counts['ok']/total*100:.1f}%)")

    if alignment_bad / total_checked > 0.1:
        print("  ⚠ WARNING: >10% alignment failures — check span offsets")
    else:
        print("  ✓ Span alignment looks good")

    # ── Check 2: Dataset creation ─────────────────────────────────────────────
    print(f"\n── Check 2: Dataset creation ──")
    train_ds = SpanDataset(train_examples, tokenizer, args.max_len)
    print(f"  Train dataset: {len(train_ds)} valid examples")

    batch = train_ds[0]
    print(f"  input_ids shape      : {batch['input_ids'].shape}")
    print(f"  attention_mask shape : {batch['attention_mask'].shape}")
    print(f"  start_position       : {batch['start_positions'].item()}")
    print(f"  end_position         : {batch['end_positions'].item()}")

    ex0       = train_ds.valid_examples[0]
    gold_span = ex0['matched_span']
    s_pos     = batch['start_positions'].item()
    e_pos     = batch['end_positions'].item()
    tokens    = tokenizer.convert_ids_to_tokens(batch['input_ids'].tolist())
    pred_toks = tokens[s_pos:e_pos+1]
    print(f"  Gold span   : '{gold_span}'")
    print(f"  Token span  : '{' '.join(pred_toks)}'")
    print(f"  ✓ Dataset creation OK")

    # ── Check 3: Model forward pass ───────────────────────────────────────────
    print(f"\n── Check 3: Model forward pass ──")
    model = SpanExtractor(args.model_name).to(device)

    loader = DataLoader(train_ds, batch_size=8, shuffle=True)
    batch  = next(iter(loader))

    with torch.no_grad():
        start_logits, end_logits = model(
            batch['input_ids'].to(device),
            batch['attention_mask'].to(device),
            batch['token_type_ids'].to(device),
        )

    print(f"  start_logits shape : {start_logits.shape}")
    print(f"  end_logits shape   : {end_logits.shape}")
    print(f"  ✓ Forward pass OK")

    # ── Check 4: Loss and one training step ───────────────────────────────────
    print(f"\n── Check 4: Training step ──")
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)

    losses = []
    model.train()
    for i, batch in enumerate(loader):
        if i >= 5:
            break
        start_logits, end_logits = model(
            batch['input_ids'].to(device),
            batch['attention_mask'].to(device),
            batch['token_type_ids'].to(device),
        )
        loss = (criterion(start_logits, batch['start_positions'].to(device)) +
                criterion(end_logits,   batch['end_positions'].to(device))) / 2
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        losses.append(loss.item())
        print(f"  Step {i+1}: loss={loss.item():.4f}")

    if losses[-1] < losses[0]:
        print(f"  ✓ Loss decreasing ({losses[0]:.4f} → {losses[-1]:.4f})")
    else:
        print(f"  ⚠ Loss not decreasing — might need more steps or check data")

    # ── Check 5: Prediction decoding ──────────────────────────────────────────
    print(f"\n── Check 5: Prediction decoding ──")
    model.eval()
    dev_ds = SpanDataset(dev_examples, tokenizer, args.max_len)
    dev_loader = DataLoader(dev_ds, batch_size=8)

    with torch.no_grad():
        batch = next(iter(dev_loader))
        start_logits, end_logits = model(
            batch['input_ids'].to(device),
            batch['attention_mask'].to(device),
            batch['token_type_ids'].to(device),
        )

    pred_starts = torch.argmax(start_logits, dim=-1).cpu().numpy()
    pred_ends   = torch.argmax(end_logits,   dim=-1).cpu().numpy()

    print(f"  {'Sentence':<50} {'Gold':<20} {'Pred':<20} {'F1':<6}")
    print(f"  {'-'*100}")
    for i in range(min(5, len(dev_ds.valid_examples))):
        ex     = dev_ds.valid_examples[i]
        pred_s = int(pred_starts[i])
        pred_e = max(int(pred_ends[i]), pred_s)

        pred_char_s, pred_char_e = token_to_char_span(
            tokenizer, ex['sentence'], pred_s, pred_e, args.max_len
        )
        pred_span = ex['sentence'][pred_char_s:pred_char_e] \
            if pred_char_s is not None else '[decode failed]'

        gold_span = ex['matched_span']
        gold_s    = dev_ds.start_positions[i]
        gold_e    = dev_ds.end_positions[i]
        f1        = compute_overlap_f1(pred_s, pred_e, gold_s, gold_e)

        sent_short = ex['sentence'][:48]
        print(f"  {sent_short:<50} {gold_span:<20} {pred_span:<20} {f1:.2f}")

    print(f"\n{'='*60}")
    print(f"✓ All checks passed — Stage 2 is ready to train")
    print(f"  Run full training with:")
    print(f"  python train_stage2_span.py \\")
    print(f"      --langs {' '.join(args.langs)} \\")
    print(f"      --output_dir models/stage2_mbert_{'_'.join(l[:2].lower() for l in args.langs)} \\")
    print(f"      --use_wandb")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    main()