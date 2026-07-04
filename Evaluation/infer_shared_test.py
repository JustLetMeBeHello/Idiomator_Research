import json
import argparse
import numpy as np
from pathlib import Path
from collections import defaultdict

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModel
from sklearn.metrics import classification_report, f1_score
from tqdm import tqdm

# ── Constants ─────────────────────────────────────────────────────────────────

LABEL2ID = {'literal': 0, 'idiomatic': 1}
ID2LABEL  = {0: 'literal', 1: 'idiomatic'}

# ── Args ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--test_path',   default='data/idioms_structured/Splits/test.jsonl')
    p.add_argument('--stage1_dir',  default='models/stage1_mbert_en_hi_te')
    p.add_argument('--joint_dir',   default='models/joint_mbert_en_hi_te')
    p.add_argument('--stage2_dir',  default='models/stage2_mbert_en_hi_te') # Added Stage 2 Path
    p.add_argument('--max_len',     type=int, default=128)
    p.add_argument('--batch_size',  type=int, default=32)
    p.add_argument('--device',      default=None)
    p.add_argument('--joint_only',  action='store_true',
                    help='Skip stage1/stage2 runners (no stage1_dir/stage2_dir checkpoints needed)')
    return p.parse_args()

# ── Device ────────────────────────────────────────────────────────────────────

def get_device(forced=None):
    if forced:
        return torch.device(forced)
    if torch.cuda.is_available():
        return torch.device('cuda')
    if torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')

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
    pred_set = set(range(int(pred_start), int(pred_end) + 1))
    gold_set = set(range(int(gold_start), int(gold_end) + 1))
    if not pred_set or not gold_set:
        return 0.0
    overlap = len(pred_set & gold_set)
    if overlap == 0:
        return 0.0
    p = overlap / len(pred_set)
    r = overlap / len(gold_set)
    return 2 * p * r / (p + r)

# ── Datasets ──────────────────────────────────────────────────────────────────

class ClsDataset(Dataset):
    """For Stage 1 — classification only."""
    def __init__(self, examples, tokenizer, max_len):
        self.examples  = examples
        self.labels    = [LABEL2ID[ex['idiomaticity']] for ex in examples]
        self.encodings = tokenizer(
            [ex['sentence'] for ex in examples],
            max_length=max_len, padding='max_length',
            truncation=True, return_tensors='pt',
        )

    def __len__(self): return len(self.labels)

    def __getitem__(self, idx):
        enc = self.encodings
        return {
            'input_ids':      enc['input_ids'][idx],
            'attention_mask': enc['attention_mask'][idx],
            'token_type_ids': enc.get('token_type_ids',
                              torch.zeros_like(enc['input_ids']))[idx],
            'labels': torch.tensor(self.labels[idx], dtype=torch.long),
        }

class SpanDataset(Dataset):
    """For Stage 2 — span identification only."""
    def __init__(self, examples, tokenizer, max_len):
        self.valid_examples  = []
        self.input_ids       = []
        self.attention_masks = []
        self.token_type_ids  = []
        self.start_positions = []
        self.end_positions   = []

        skipped = 0
        for ex in examples:
            sentence   = ex['sentence']
            char_start = ex['span_start']
            char_end   = ex['span_end']

            encoding = tokenizer(
                sentence, max_length=max_len, padding='max_length',
                truncation=True, return_tensors='pt',
            )
            enc_map = tokenizer(
                sentence, max_length=max_len, truncation=True,
                return_offsets_mapping=True,
            )

            token_start, token_end = char_to_token_span(
                enc_map, char_start, char_end, sentence
            )
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
            'start_positions': torch.tensor(self.start_positions[idx], dtype=torch.long),
            'end_positions':   torch.tensor(self.end_positions[idx],   dtype=torch.long),
        }

# ── Model Classes ─────────────────────────────────────────────────────────────

class JointIdiomModel(torch.nn.Module):
    def __init__(self, model_name):
        super().__init__()
        self.bert       = AutoModel.from_pretrained(model_name)
        hidden_size     = self.bert.config.hidden_size
        self.cls_head   = torch.nn.Linear(hidden_size, 2)
        self.start_head = torch.nn.Linear(hidden_size, 1)
        self.end_head   = torch.nn.Linear(hidden_size, 1)

    def forward(self, input_ids, attention_mask, token_type_ids):
        outputs      = self.bert(input_ids=input_ids,
                                 attention_mask=attention_mask,
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

class SpanOnlyModel(torch.nn.Module):
    """Stand-alone Stage 2 Span Model."""
    def __init__(self, model_name):
        super().__init__()
        self.bert       = AutoModel.from_pretrained(model_name)
        hidden_size     = self.bert.config.hidden_size
        self.start_head = torch.nn.Linear(hidden_size, 1)
        self.end_head   = torch.nn.Linear(hidden_size, 1)

    def forward(self, input_ids, attention_mask, token_type_ids):
        outputs      = self.bert(input_ids=input_ids,
                                 attention_mask=attention_mask,
                                 token_type_ids=token_type_ids)
        seq_output   = outputs.last_hidden_state
        start_logits = self.start_head(seq_output).squeeze(-1)
        end_logits   = self.end_head(seq_output).squeeze(-1)
        mask         = attention_mask.bool()
        start_logits = start_logits.masked_fill(~mask, float('-inf'))
        end_logits   = end_logits.masked_fill(~mask,   float('-inf'))
        return start_logits, end_logits

BASE_ENCODER = 'bert-base-multilingual-cased'  # best_model/ only stores fine-tuned task heads, not encoder weights

def load_joint_model(joint_dir, device):
    best = Path(joint_dir) / 'best_model'
    model = JointIdiomModel(BASE_ENCODER)
    heads = torch.load(best / 'task_heads.pt', map_location='cpu', weights_only=True)
    model.cls_head.load_state_dict(heads['cls_head'])
    model.start_head.load_state_dict(heads['start_head'])
    model.end_head.load_state_dict(heads['end_head'])
    return model.to(device)

def load_span_model(stage2_dir, device):
    best = Path(stage2_dir) / 'best_model'
    model = SpanOnlyModel(BASE_ENCODER)
    heads = torch.load(best / 'span_heads.pt', map_location='cpu', weights_only=True)
    model.start_head.load_state_dict(heads['start_head'])
    model.end_head.load_state_dict(heads['end_head'])
    return model.to(device)

# ── Inference runners ─────────────────────────────────────────────────────────

def run_stage1(examples, args, device):
    print("\n" + "="*60)
    print("STAGE 1 mBERT — classification on shared test.jsonl")
    print("="*60)

    model_dir = Path(args.stage1_dir) / 'best_model'
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
    model     = AutoModelForSequenceClassification.from_pretrained(str(model_dir)).to(device)
    model.eval()

    dataset = ClsDataset(examples, tokenizer, args.max_len)
    loader  = DataLoader(dataset, batch_size=args.batch_size)

    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in tqdm(loader, desc='Stage 1 inference'):
            out   = model(input_ids=batch['input_ids'].to(device),
                          attention_mask=batch['attention_mask'].to(device),
                          token_type_ids=batch['token_type_ids'].to(device))
            preds = torch.argmax(out.logits, dim=-1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(batch['labels'].numpy())

    macro_f1 = f1_score(all_labels, all_preds, average='macro')
    print(classification_report(all_labels, all_preds,
                                 target_names=['literal', 'idiomatic'], digits=4))
    
    out_path = Path(args.stage1_dir) / 'test_predictions_shared.jsonl'
    with open(out_path, 'w', encoding='utf-8') as f:
        for ex, pred, label in zip(examples, all_preds, all_labels):
            f.write(json.dumps({
                **ex,
                'pred_idiomaticity': ID2LABEL[int(pred)],
                'correct':           bool(pred == label),
            }, ensure_ascii=False) + '\n')
    print(f"Saved → {out_path}")
    return macro_f1

def run_stage2(examples, args, device):
    print("\n" + "="*60)
    print("STAGE 2 mBERT — span extraction only on shared test.jsonl")
    print("="*60)

    best      = Path(args.stage2_dir) / 'best_model'
    tokenizer = AutoTokenizer.from_pretrained(str(best))
    model     = load_span_model(args.stage2_dir, device)
    model.eval()

    dataset = SpanDataset(examples, tokenizer, args.max_len)
    loader  = DataLoader(dataset, batch_size=args.batch_size)
    valid   = dataset.valid_examples

    span_results = []
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(loader, desc='Stage 2 inference')):
            start_logits, end_logits = model(
                input_ids=batch['input_ids'].to(device),
                attention_mask=batch['attention_mask'].to(device),
                token_type_ids=batch['token_type_ids'].to(device),
            )
            pred_starts = torch.argmax(start_logits, dim=-1).cpu().numpy()
            pred_ends   = torch.argmax(end_logits,   dim=-1).cpu().numpy()

            batch_start = batch_idx * args.batch_size
            for i in range(len(pred_starts)):
                ex_idx = batch_start + i
                if ex_idx >= len(valid): break
                ex = valid[ex_idx]

                pred_s = int(pred_starts[i])
                pred_e = int(pred_ends[i])
                if pred_e < pred_s: pred_e = pred_s

                pred_char_s, pred_char_e = token_to_char_span(
                    tokenizer, ex['sentence'], pred_s, pred_e, args.max_len
                )
                if pred_char_s is None: pred_char_s, pred_char_e = 0, 0

                span_results.append((pred_char_s, pred_char_e, ex['span_start'], ex['span_end'], ex))

    exact_matches = sum(1 for r in span_results if r[0] == r[2] and r[1] == r[3])
    overlap_f1s   = [compute_overlap_f1(r[0], r[1], r[2], r[3]) for r in span_results]
    print(f"Span exact match: {exact_matches/len(span_results):.4f}")
    print(f"Overlap F1:       {np.mean(overlap_f1s):.4f}")

    out_path = Path(args.stage2_dir) / 'test_predictions_shared.jsonl'
    with open(out_path, 'w', encoding='utf-8') as f:
        for pred_char_s, pred_char_e, gold_s, gold_e, ex in span_results:
            f.write(json.dumps({
                **ex,
                'pred_span_start': pred_char_s,
                'pred_span_end':   pred_char_e,
                'exact_match':     bool(pred_char_s == gold_s and pred_char_e == gold_e),
                'overlap_f1':      round(compute_overlap_f1(pred_char_s, pred_char_e, gold_s, gold_e), 4),
            }, ensure_ascii=False) + '\n')
    print(f"Saved → {out_path}")

def run_joint(examples, args, device):
    print("\n" + "="*60)
    print("JOINT mBERT — cls + span on shared test.jsonl")
    print("="*60)

    best      = Path(args.joint_dir) / 'best_model'
    tokenizer = AutoTokenizer.from_pretrained(str(best))
    model     = load_joint_model(args.joint_dir, device)
    model.eval()

    dataset = SpanDataset(examples, tokenizer, args.max_len)
    loader  = DataLoader(dataset, batch_size=args.batch_size)
    valid   = dataset.valid_examples
    labels  = [LABEL2ID[ex['idiomaticity']] for ex in valid]

    records = []
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(loader, desc='Joint inference')):
            cls_logits, start_logits, end_logits = model(
                input_ids=batch['input_ids'].to(device),
                attention_mask=batch['attention_mask'].to(device),
                token_type_ids=batch['token_type_ids'].to(device),
            )
            cls_preds   = torch.argmax(cls_logits, dim=-1).cpu().numpy()
            pred_starts = torch.argmax(start_logits, dim=-1).cpu().numpy()
            pred_ends   = torch.argmax(end_logits,   dim=-1).cpu().numpy()

            batch_start = batch_idx * args.batch_size
            for i in range(len(cls_preds)):
                ex_idx = batch_start + i
                if ex_idx >= len(valid):
                    break
                ex = valid[ex_idx]

                pred_s = int(pred_starts[i])
                pred_e = int(pred_ends[i])
                if pred_e < pred_s:
                    pred_e = pred_s

                pred_char_s, pred_char_e = token_to_char_span(
                    tokenizer, ex['sentence'], pred_s, pred_e, args.max_len
                )
                if pred_char_s is None:
                    pred_char_s, pred_char_e = 0, 0

                records.append({
                    **ex,
                    'pred_idiomaticity': ID2LABEL[int(cls_preds[i])],
                    'correct':           bool(cls_preds[i] == labels[ex_idx]),
                    'pred_span_start':   pred_char_s,
                    'pred_span_end':     pred_char_e,
                    'exact_match':       bool(pred_char_s == ex['span_start'] and pred_char_e == ex['span_end']),
                    'overlap_f1':        round(compute_overlap_f1(pred_char_s, pred_char_e, ex['span_start'], ex['span_end']), 4),
                })

    cls_acc  = np.mean([r['correct'] for r in records])
    exact    = np.mean([r['exact_match'] for r in records])
    overlap  = np.mean([r['overlap_f1'] for r in records])
    print(f"Cls accuracy:     {cls_acc:.4f}")
    print(f"Span exact match: {exact:.4f}")
    print(f"Overlap F1:       {overlap:.4f}")

    joint_path = Path(args.joint_dir) / 'test_predictions_shared.jsonl'
    with open(joint_path, 'w', encoding='utf-8') as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')
    print(f"Saved → {joint_path}")

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args   = parse_args()
    device = get_device(args.device)
    print(f"Device: {device}")

    examples = [json.loads(l) for l in open(args.test_path, encoding='utf-8')]
    print(f"Shared test set: {len(examples)} examples")

    if not args.joint_only:
        run_stage1(examples, args, device)
        run_stage2(examples, args, device) # Now runs the actual Stage 2 model
    run_joint(examples, args, device)

    print("\n" + "="*60)
    print("Done. Evaluation files updated.")
    print("="*60)

if __name__ == '__main__':
    main()