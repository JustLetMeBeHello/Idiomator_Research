"""
run_09_semeval_eval.py

Zero-shot evaluation of System E (Joint mBERT, EN+TE training) on
SemEval-2022 Task 2 Subtask A idiomaticity detection.

Languages:
  EN  — full pipeline: CLS F1 + Span Exact Match + Span Overlap F1
        (span derived from MWE string search in sentence)
  PT  — CLS F1 only (no span annotations)
  GL  — CLS F1 only

Model is truly zero-shot on SemEval — trained on MultiIdiom EN+TE only,
never fine-tuned on any SemEval data.

SemEval-2022 Task 2 data:
  https://github.com/semeval-2022/semeval-2022-task-2-idiomaticity

Step 1 — Train System E on Colab (EN+TE):
  !python training/Train_Join.py \\
      --langs English Telugu \\
      --output_dir /content/drive/MyDrive/IdiomatorModels/system_e_en_te \\
      --epochs 7 --batch_size 32 --lr 2e-5 --seed 42

Step 2 — Clone SemEval data on Colab:
  !git clone https://github.com/semeval-2022/semeval-2022-task-2-idiomaticity /tmp/semeval2022

Step 3 — Run this script:
  !python experiments/rigor/run_09_semeval_eval.py \\
      --data_dir   /tmp/semeval2022 \\
      --model_dir  /content/drive/MyDrive/IdiomatorModels/system_e_en_te \\
      --output_dir /content/drive/MyDrive/IdiomatorRigor/semeval_eval

SemEval-2022 supervised toplines for context (no SemEval training data used here):
  EN zero-shot supervised best: 0.9016 (clay team)
  EN one-shot  supervised best: 0.9639 (HIT team)
  Our system is more zero-shot than SemEval "zero-shot" — they trained on SemEval data;
  we trained only on MultiIdiom.

Outputs (incremental, Drive-safe):
  semeval_system_e_preds.jsonl   — per-row predictions
  semeval_results.json           — aggregate metrics
  semeval_results_summary.txt    — human-readable table
"""

import os
import re
import sys
import json
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics import f1_score, classification_report
from tqdm import tqdm


# ── Constants ──────────────────────────────────────────────────────────────────

LABEL2ID = {'literal': 0, 'idiomatic': 1}
ID2LABEL  = {0: 'literal', 1: 'idiomatic'}

LANG_MAP = {'EN': 'English', 'PT': 'Portuguese', 'GL': 'Galician'}

# EN has MWE-derived span; PT/GL are CLS-only
SPAN_LANGS = {'English'}


# ── Args ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--data_dir',   required=True,
                   help='Root of cloned H-TayyarMadabushi/SemEval_2022_Task2-idiomaticity repo')
    p.add_argument('--model_dir',  required=True,
                   help='System E model dir (contains best_model/ with encoder + task_heads.pt)')
    p.add_argument('--output_dir', default='experiments/rigor/results/semeval_eval')
    p.add_argument('--split',      default='dev', choices=['dev'],
                   help='Only dev has gold labels; test labels are withheld by organizers')
    p.add_argument('--langs',      nargs='+', default=['EN'],
                   help='Languages to evaluate. EN has span eval; PT is CLS-only. Default: EN only.')
    p.add_argument('--max_len',    type=int, default=128)
    p.add_argument('--batch_size', type=int, default=32)
    p.add_argument('--device',     default=None)
    p.add_argument('--force',      action='store_true',
                   help='Overwrite existing prediction file')
    return p.parse_args()


def get_device(forced=None):
    if forced:
        return torch.device(forced)
    if torch.cuda.is_available():
        return torch.device('cuda')
    if torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


# ── Data loading ───────────────────────────────────────────────────────────────

def locate_span(sentence: str, mwe: str) -> tuple[int | None, int | None]:
    idx = sentence.lower().find(mwe.lower())
    if idx != -1:
        return idx, idx + len(mwe)
    words   = mwe.lower().split()
    pattern = r'\b' + r'\W+'.join(re.escape(w) for w in words) + r'\b'
    m       = re.search(pattern, sentence.lower())
    if m:
        return m.start(), m.end()
    return None, None


def load_semeval_data(data_dir: Path, split: str, lang_codes: list[str]) -> dict[str, list[dict]]:
    """Load SemEval-2022 Task 2 SubTaskA data.

    Repo layout (H-TayyarMadabushi):
      SubTaskA/Data/dev.csv          — sentences (ID, Language, MWE, Previous, Target, Next)
      SubTaskA/Data/dev_gold.csv     — labels    (ID, DataID, Language, Label)

    Join on ID. Use Target as sentence. Label: 0=literal, 1=idiomatic.
    """
    base = data_dir / 'SubTaskA' / 'Data'

    sents_csv = base / f'{split}.csv'
    gold_csv  = base / f'{split}_gold.csv'

    if not sents_csv.exists():
        raise FileNotFoundError(f"Sentences file not found: {sents_csv}")
    if not gold_csv.exists():
        raise FileNotFoundError(f"Gold labels file not found: {gold_csv}")

    sents = pd.read_csv(sents_csv)
    gold  = pd.read_csv(gold_csv)

    # Join on ID
    merged = sents.merge(gold[['ID', 'Label']], on='ID', how='inner')
    print(f"  Loaded {len(merged)} labeled examples (split={split})")

    all_examples: dict[str, list[dict]] = {}
    for lang_code in lang_codes:
        lang_label = LANG_MAP.get(lang_code, lang_code)
        subset     = merged[merged['Language'] == lang_code]
        if subset.empty:
            print(f"  [{lang_label}] No examples found")
            continue

        examples = []
        n_fallback = 0
        for _, row in subset.iterrows():
            sentence = str(row['Target'])
            mwe      = str(row['MWE'])
            label    = 'idiomatic' if int(row['Label']) == 1 else 'literal'

            ex = {
                'semeval_id':   str(row['ID']),
                'idiom':        mwe,
                'language':     lang_label,
                'idiomaticity': label,
                'sentence':     sentence,
                'span_start':   None,
                'span_end':     None,
            }

            if lang_label in SPAN_LANGS:
                s, e = locate_span(sentence, mwe)
                if s is not None:
                    ex['span_start'] = s
                    ex['span_end']   = e
                else:
                    ex['span_start']    = 0
                    ex['span_end']      = len(sentence)
                    ex['span_fallback'] = True
                    n_fallback         += 1

            examples.append(ex)

        dist = {k: sum(1 for e in examples if e['idiomaticity'] == k)
                for k in ('idiomatic', 'literal')}
        n_span = sum(1 for e in examples if e.get('span_start') is not None
                     and not e.get('span_fallback'))
        print(f"  [{lang_label}] {len(examples)} examples | {dist} | "
              f"span_located={n_span} fallback={n_fallback}")
        all_examples[lang_label] = examples

    return all_examples


# ── Span utilities ─────────────────────────────────────────────────────────────

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


def token_to_char_span(tokenizer, sentence, tok_s, tok_e, max_len):
    enc     = tokenizer(sentence, max_length=max_len, truncation=True, return_offsets_mapping=True)
    offsets = enc['offset_mapping']
    if tok_s >= len(offsets) or tok_e >= len(offsets):
        return None, None
    return offsets[tok_s][0], offsets[tok_e][1]


def overlap_f1(ps, pe, gs, ge):
    if ps is None or pe is None:
        return 0.0
    pred = set(range(int(ps), int(pe)))
    gold = set(range(int(gs), int(ge)))
    if not pred or not gold:
        return 0.0
    ov = len(pred & gold)
    if ov == 0:
        return 0.0
    return 2 * (ov / len(pred)) * (ov / len(gold)) / (ov / len(pred) + ov / len(gold))


# ── Dataset ────────────────────────────────────────────────────────────────────

class SemEvalDataset(torch.utils.data.Dataset):
    """All examples. Span positions set to 0 when unavailable (span head output ignored)."""
    def __init__(self, examples, tokenizer, max_len):
        self.examples        = []
        self.input_ids       = []
        self.attention_masks = []
        self.token_type_ids  = []
        self.start_positions = []
        self.end_positions   = []

        skipped = 0
        for ex in examples:
            sentence = ex['sentence']
            encoding = tokenizer(
                sentence, max_length=max_len, padding='max_length',
                truncation=True, return_tensors='pt',
            )
            seq_len = encoding['input_ids'].shape[1]

            if ex.get('span_start') is not None:
                enc_map = tokenizer(sentence, max_length=max_len, truncation=True,
                                    return_offsets_mapping=True)
                tok_s, tok_e = char_to_token_span(enc_map, ex['span_start'], ex['span_end'], sentence)
                if tok_s is None:
                    tok_s, tok_e = 0, 0
                tok_s = min(tok_s, seq_len - 1)
                tok_e = min(tok_e, seq_len - 1)
            else:
                tok_s, tok_e = 0, 0

            self.examples.append(ex)
            self.input_ids.append(encoding['input_ids'].squeeze(0))
            self.attention_masks.append(encoding['attention_mask'].squeeze(0))
            tid = encoding.get('token_type_ids')
            self.token_type_ids.append(
                tid.squeeze(0) if tid is not None
                else torch.zeros(seq_len, dtype=torch.long)
            )
            self.start_positions.append(tok_s)
            self.end_positions.append(tok_e)

    def __len__(self): return len(self.examples)

    def __getitem__(self, idx):
        return {
            'input_ids':       self.input_ids[idx],
            'attention_mask':  self.attention_masks[idx],
            'token_type_ids':  self.token_type_ids[idx],
            'start_positions': torch.tensor(self.start_positions[idx], dtype=torch.long),
            'end_positions':   torch.tensor(self.end_positions[idx],   dtype=torch.long),
        }


# ── Model ──────────────────────────────────────────────────────────────────────

class JointIdiomModel(torch.nn.Module):
    def __init__(self, model_name):
        super().__init__()
        self.bert       = AutoModel.from_pretrained(model_name)
        h               = self.bert.config.hidden_size
        self.cls_head   = torch.nn.Linear(h, 2)
        self.start_head = torch.nn.Linear(h, 1)
        self.end_head   = torch.nn.Linear(h, 1)

    def forward(self, input_ids, attention_mask, token_type_ids):
        out  = self.bert(input_ids=input_ids, attention_mask=attention_mask,
                         token_type_ids=token_type_ids)
        seq  = out.last_hidden_state
        mask = attention_mask.bool()
        cls_logits = self.cls_head(seq[:, 0, :])
        start_l    = self.start_head(seq).squeeze(-1).masked_fill(~mask, float('-inf'))
        end_l      = self.end_head(seq).squeeze(-1).masked_fill(~mask,   float('-inf'))
        return cls_logits, start_l, end_l


def load_system_e(model_dir: str, device):
    best  = Path(model_dir) / 'best_model'
    model = JointIdiomModel(str(best))
    model.bert = AutoModel.from_pretrained(str(best))
    heads = torch.load(best / 'task_heads.pt', map_location='cpu', weights_only=True)
    model.cls_head.load_state_dict(heads['cls_head'])
    model.start_head.load_state_dict(heads['start_head'])
    model.end_head.load_state_dict(heads['end_head'])
    return model.to(device)


# ── Inference ──────────────────────────────────────────────────────────────────

def run_inference(examples: list[dict], model_dir: str,
                  device, max_len: int, batch_size: int) -> list[dict]:
    best      = Path(model_dir) / 'best_model'
    tokenizer = AutoTokenizer.from_pretrained(str(best))
    model     = load_system_e(model_dir, device)
    model.eval()

    dataset = SemEvalDataset(examples, tokenizer, max_len)
    loader  = DataLoader(dataset, batch_size=batch_size)

    results = []
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(loader, desc='System E inference')):
            cls_logits, start_logits, end_logits = model(
                input_ids=batch['input_ids'].to(device),
                attention_mask=batch['attention_mask'].to(device),
                token_type_ids=batch['token_type_ids'].to(device),
            )
            cls_preds   = torch.argmax(cls_logits,   dim=-1).cpu().numpy()
            start_preds = torch.argmax(start_logits, dim=-1).cpu().numpy()
            end_preds   = torch.argmax(end_logits,   dim=-1).cpu().numpy()

            batch_start = batch_idx * batch_size
            for i in range(len(cls_preds)):
                ex_idx = batch_start + i
                if ex_idx >= len(dataset.examples):
                    break
                ex = dataset.examples[ex_idx]

                pred_cls = ID2LABEL[int(cls_preds[i])]
                tok_s    = int(start_preds[i])
                tok_e    = int(end_preds[i])
                if tok_e < tok_s:
                    tok_e = tok_s

                out = {**ex, 'pred_idiomaticity': pred_cls}

                if ex.get('span_start') is not None and not ex.get('span_fallback'):
                    char_s, char_e = token_to_char_span(tokenizer, ex['sentence'], tok_s, tok_e, max_len)
                    if char_s is None:
                        char_s, char_e = 0, 0
                    out['pred_span_start'] = char_s
                    out['pred_span_end']   = char_e

                results.append(out)

    return results


# ── Metrics ────────────────────────────────────────────────────────────────────

def compute_metrics(preds: list[dict]) -> dict:
    gold  = [LABEL2ID[p['idiomaticity']]     for p in preds]
    pred  = [LABEL2ID[p['pred_idiomaticity']] for p in preds]
    macro = f1_score(gold, pred, average='macro', zero_division=0)
    rep   = classification_report(gold, pred, target_names=['literal', 'idiomatic'],
                                   output_dict=True, zero_division=0)
    result = {
        'macro_f1':     round(macro, 4),
        'literal_f1':   round(rep['literal']['f1-score'],   4),
        'idiomatic_f1': round(rep['idiomatic']['f1-score'], 4),
        'n': len(gold),
    }

    span_preds = [p for p in preds
                  if p.get('pred_span_start') is not None
                  and p.get('span_start')     is not None
                  and not p.get('span_fallback')]
    if span_preds:
        exact  = [int(p['pred_span_start'] == p['span_start'] and
                      p['pred_span_end']   == p['span_end'])
                  for p in span_preds]
        ov_f1s = [overlap_f1(p['pred_span_start'], p['pred_span_end'],
                              p['span_start'],      p['span_end'])
                  for p in span_preds]
        result['span_exact_match'] = round(np.mean(exact),  4)
        result['span_overlap_f1']  = round(np.mean(ov_f1s), 4)
        result['n_span']           = len(span_preds)

    return result


# ── Persistence helpers ────────────────────────────────────────────────────────

def save_jsonl(records, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')
    actual = sum(1 for _ in open(path, encoding='utf-8'))
    assert actual == len(records), f"Persistence check FAILED: {path} wrote {len(records)}, read back {actual}"
    print(f"  ✓ {actual} rows → {path}")


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    args    = parse_args()
    device  = get_device(args.device)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Device : {device}")
    print(f"Split  : {args.split}")
    print(f"Model  : {args.model_dir}")
    print(f"Output : {out_dir}")

    # Verify output dir is durable
    probe = out_dir / '.write_test'
    probe.write_text('ok')
    assert probe.read_text() == 'ok', "Output dir not readable after write"
    probe.unlink()
    print(f"  ✓ Output dir durable")

    # ── Load data ─────────────────────────────────────────────────────────────
    data_dir     = Path(args.data_dir)
    all_examples = load_semeval_data(data_dir, args.split, args.langs)

    if not all_examples:
        print("No data found. Check --data_dir and --langs.")
        sys.exit(1)

    flat = [ex for exs in all_examples.values() for ex in exs]

    # ── Run inference ─────────────────────────────────────────────────────────
    pred_path = out_dir / 'semeval_system_e_preds.jsonl'

    if pred_path.exists() and not args.force:
        print(f"\nPredictions exist ({pred_path}). Use --force to rerun.")
        preds = [json.loads(l) for l in open(pred_path, encoding='utf-8')]
    else:
        print(f"\nRunning System E inference on {len(flat)} examples...")
        preds = run_inference(flat, args.model_dir, device, args.max_len, args.batch_size)
        save_jsonl(preds, pred_path)

    # ── Compute metrics ───────────────────────────────────────────────────────
    all_results = {}
    id_map      = {ex['semeval_id']: ex for ex in flat}

    print(f"\n{'='*55}")
    print(f"System E (Joint mBERT, EN+TE) — SemEval-2022 Task 2")
    print(f"{'='*55}")

    for lang, exs in all_examples.items():
        ids        = {ex['semeval_id'] for ex in exs}
        lang_preds = [p for p in preds if p['semeval_id'] in ids]
        m          = compute_metrics(lang_preds)
        all_results[lang] = m

        print(f"\n  [{lang}]  n={m['n']}")
        print(f"    CLS macro F1 : {m['macro_f1']:.4f}  "
              f"(lit={m['literal_f1']:.4f}, idiom={m['idiomatic_f1']:.4f})")
        if 'span_exact_match' in m:
            print(f"    Span exact   : {m['span_exact_match']:.4f}  "
                  f"Overlap F1: {m['span_overlap_f1']:.4f}  (n={m['n_span']})")

    # SemEval supervised context
    print(f"\n  SemEval-2022 supervised toplines (trained on SemEval data):")
    print(f"    EN zero-shot setting: 0.9016 (clay)  one-shot: 0.9639 (HIT)")
    print(f"    Our system: truly zero-shot (MultiIdiom only, no SemEval data)")

    # ── Save aggregate results ─────────────────────────────────────────────────
    results_path = out_dir / 'semeval_results.json'
    results_path.write_text(json.dumps(all_results, indent=2))
    assert json.loads(results_path.read_text()) == all_results
    print(f"\n  ✓ Results JSON → {results_path}")

    summary_lines = [
        "SemEval-2022 Task 2 — System E (Joint mBERT, EN+TE) Zero-Shot",
        "=" * 55,
        f"Split: {args.split}",
        "",
    ]
    for lang, m in all_results.items():
        summary_lines.append(f"[{lang}]  n={m['n']}")
        summary_lines.append(f"  CLS macro F1 : {m['macro_f1']:.4f}"
                              f"  (lit={m['literal_f1']:.4f}, idiom={m['idiomatic_f1']:.4f})")
        if 'span_exact_match' in m:
            summary_lines.append(f"  Span exact   : {m['span_exact_match']:.4f}"
                                  f"  Overlap F1: {m['span_overlap_f1']:.4f}"
                                  f"  (n={m['n_span']})")
        summary_lines.append("")
    summary_lines += [
        "SemEval supervised toplines (trained on SemEval data):",
        "  EN zero-shot setting best: 0.9016 (clay)",
        "  EN one-shot  setting best: 0.9639 (HIT)",
    ]
    summary_path = out_dir / 'semeval_results_summary.txt'
    summary_path.write_text('\n'.join(summary_lines))
    print(f"  ✓ Summary → {summary_path}")
    print("\n✓ Done.")


if __name__ == '__main__':
    main()
