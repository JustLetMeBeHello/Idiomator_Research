"""
run_10_id10m_eval.py

Zero-shot evaluation of System E (Joint mBERT, EN+TE) on ID10M
(Idiom Identification in 10 Languages, NAACL 2022).

Languages evaluated:
  EN  — dataset transfer: MultiIdiom EN → ID10M EN
        (same language, different corpus, same literal-type: physical/neutral context)
  ES  — dataset + language transfer: model never trained on ES
        (true zero-shot cross-lingual from mBERT pretraining)

ID10M literal cases are physical/neutral-context (e.g. "rock broke the ice")
unlike MAGPIE's domain-embedded literals. Transfer expected to work.

Data format: BIO TSV (one token per line, blank lines between sentences)
  token  O
  token  B-IDIOM
  token  I-IDIOM

Binary label derived from BIO tags:
  sentence with ≥1 B-IDIOM → idiomaticity = 'idiomatic'
  sentence with all O      → idiomaticity = 'literal'

Multi-idiom sentences: first span only (most sentences have exactly one).

Data repo: https://github.com/Babelscape/ID10M
  Clone: !git clone https://github.com/Babelscape/ID10M /tmp/id10m

Usage (Colab):
  !git clone https://github.com/Babelscape/ID10M /tmp/id10m
  !python3 -u Additional_Rigor_Experiments/run_10_id10m_eval.py \\
      --data_dir   /tmp/id10m \\
      --model_dir  /content/drive/MyDrive/IdiomatorModels/system_e_en_te \\
      --output_dir /content/drive/MyDrive/IdiomatorRigor/id10m_eval \\
      --langs EN ES

ID10M reported toplines (supervised on ID10M training data):
  EN F1: ~0.80 (from paper Table 3)  — check paper for exact number
  ES F1: ~0.77 (from paper Table 3)  — check paper for exact number

Outputs (incremental, Drive-safe):
  id10m_system_e_{lang}_preds.jsonl  — per-sentence predictions
  id10m_results.json                 — aggregate metrics
  id10m_results_summary.txt          — human-readable table

Metric integrity: no numeric baselines hardcoded (CLAUDE.md rule).
ID10M toplines must be read from the paper and filled in manually.
"""

import re
import sys
import json
import argparse
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics import f1_score, classification_report
from tqdm import tqdm


# ── Constants ──────────────────────────────────────────────────────────────────

LABEL2ID = {'literal': 0, 'idiomatic': 1}
ID2LABEL  = {0: 'literal', 1: 'idiomatic'}

ID10M_LANG_MAP = {
    'EN': 'English',  'ES': 'Spanish',  'DE': 'German',
    'FR': 'French',   'IT': 'Italian',  'PT': 'Portuguese',
    'ZH': 'Chinese',  'JA': 'Japanese', 'NL': 'Dutch',  'PL': 'Polish',
}

ID10M_DIR_MAP = {
    'EN': 'english',  'ES': 'spanish',  'DE': 'german',
    'FR': 'french',   'IT': 'italian',  'PT': 'portuguese',
    'ZH': 'chinese',  'JA': 'japanese', 'NL': 'dutch',  'PL': 'polish',
}

PUNCT = set('.,;:!?)]\'"…—–')


# ── Args ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--data_dir',   required=True,
                   help='Root of cloned Babelscape/ID10M repo')
    p.add_argument('--model_dir',  required=True,
                   help='System E model dir (contains best_model/)')
    p.add_argument('--output_dir', default='Additional_Rigor_Experiments/results/id10m_eval')
    p.add_argument('--langs',      nargs='+', default=['EN', 'ES'])
    p.add_argument('--split',      default='test', choices=['test', 'train'])
    p.add_argument('--max_len',    type=int, default=128)
    p.add_argument('--batch_size', type=int, default=32)
    p.add_argument('--device',     default=None)
    p.add_argument('--force',      action='store_true')
    return p.parse_args()


def get_device(forced=None):
    if forced:
        return torch.device(forced)
    if torch.cuda.is_available():
        return torch.device('cuda')
    if torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


# ── BIO data loading ───────────────────────────────────────────────────────────

def reconstruct_sentence(tokens: list[str]) -> tuple[str, list[int]]:
    """Join tokens into sentence string, return (sentence, token_char_starts).

    Handles punctuation attachment: no space before .,;:!?)]'"
    Returns char start position of each token in the reconstructed sentence.
    """
    sentence   = ''
    tok_starts = []
    for tok in tokens:
        if sentence and tok not in PUNCT and not tok.startswith("'"):
            sentence += ' '
        tok_starts.append(len(sentence))
        sentence += tok
    return sentence, tok_starts


def parse_bio_file(tsv_path: Path, lang_label: str) -> list[dict]:
    """Parse a BIO TSV file into internal example format.

    Each sentence becomes one example. Label derived from BIO tags:
      ≥1 B-IDIOM → idiomatic
      all O      → literal

    Multi-idiom sentences: first span used; 'multi_idiom' flag set.
    """
    examples = []
    tokens: list[str] = []
    tags:   list[str] = []
    sent_id = 0

    def flush(tokens, tags, sent_id):
        if not tokens:
            return None
        sentence, tok_starts = reconstruct_sentence(tokens)

        # Find all idiom spans
        spans = []
        span_s = None
        for i, tag in enumerate(tags):
            if tag == 'B-IDIOM':
                span_s = i
            if tag == 'O' and span_s is not None:
                spans.append((span_s, i - 1))
                span_s = None
        if span_s is not None:
            spans.append((span_s, len(tags) - 1))

        label = 'idiomatic' if spans else 'literal'

        ex = {
            'id10m_id':      f'{lang_label}_{sent_id}',
            'language':      lang_label,
            'idiomaticity':  label,
            'sentence':      sentence,
            'span_start':    None,
            'span_end':      None,
            'multi_idiom':   len(spans) > 1,
        }

        if spans:
            # Use first span
            s_tok, e_tok = spans[0]
            char_s = tok_starts[s_tok]
            # char_end = end of last token in span
            char_e = tok_starts[e_tok] + len(tokens[e_tok])
            ex['span_start'] = char_s
            ex['span_end']   = char_e
            # Reconstruct MWE string for reference
            ex['idiom'] = sentence[char_s:char_e]
        else:
            ex['idiom'] = ''

        return ex

    with open(tsv_path, encoding='utf-8') as f:
        for line in f:
            line = line.rstrip('\n')
            if not line.strip():
                ex = flush(tokens, tags, sent_id)
                if ex is not None:
                    examples.append(ex)
                    sent_id += 1
                tokens, tags = [], []
            else:
                parts = line.split('\t')
                if len(parts) >= 2:
                    tokens.append(parts[0])
                    tags.append(parts[1].strip())

    # Flush last sentence if no trailing blank line
    ex = flush(tokens, tags, sent_id)
    if ex is not None:
        examples.append(ex)

    dist       = {k: sum(1 for e in examples if e['idiomaticity'] == k)
                  for k in ('idiomatic', 'literal')}
    n_multi    = sum(1 for e in examples if e.get('multi_idiom'))
    n_span     = sum(1 for e in examples if e.get('span_start') is not None)
    print(f"  [{lang_label}] {len(examples)} sentences | {dist} | "
          f"span={n_span} multi_idiom={n_multi}")
    return examples


def load_id10m_data(data_dir: Path, lang_codes: list[str], split: str) -> dict[str, list[dict]]:
    all_examples: dict[str, list[dict]] = {}
    for code in lang_codes:
        lang_label = ID10M_LANG_MAP.get(code, code)
        lang_dir   = ID10M_DIR_MAP.get(code, code.lower())
        tsv_path   = data_dir / 'resources' / 'bio_format' / lang_dir / f'{split}_{lang_dir}.tsv'
        if not tsv_path.exists():
            print(f"  [{lang_label}] Not found: {tsv_path}")
            continue
        examples = parse_bio_file(tsv_path, lang_label)
        if examples:
            all_examples[lang_label] = examples
    return all_examples


# ── Span utilities (shared with run_09) ────────────────────────────────────────

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

class ID10MDataset(Dataset):
    def __init__(self, examples, tokenizer, max_len):
        self.examples        = []
        self.input_ids       = []
        self.attention_masks = []
        self.token_type_ids  = []
        self.start_positions = []
        self.end_positions   = []

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
                tok_s, tok_e = char_to_token_span(enc_map, ex['span_start'],
                                                   ex['span_end'], sentence)
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


# ── Model (identical to run_09) ────────────────────────────────────────────────

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
                  device, max_len: int, batch_size: int,
                  lang_label: str) -> list[dict]:
    best      = Path(model_dir) / 'best_model'
    tokenizer = AutoTokenizer.from_pretrained(str(best))
    model     = load_system_e(model_dir, device)
    model.eval()

    dataset = ID10MDataset(examples, tokenizer, max_len)
    loader  = DataLoader(dataset, batch_size=batch_size)

    results = []
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(loader, desc=f'System E [{lang_label}]')):
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

                if ex.get('span_start') is not None:
                    char_s, char_e = token_to_char_span(
                        tokenizer, ex['sentence'], tok_s, tok_e, max_len
                    )
                    if char_s is None:
                        char_s, char_e = 0, 0
                    out['pred_span_start'] = char_s
                    out['pred_span_end']   = char_e

                results.append(out)

    return results


# ── Metrics ────────────────────────────────────────────────────────────────────

def compute_metrics(preds: list[dict]) -> dict:
    gold  = [LABEL2ID[p['idiomaticity']]      for p in preds]
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

    # Span metrics — idiomatic examples with gold span only (exclude literal, no span expected)
    idiomatic_preds = [p for p in preds
                       if p['idiomaticity'] == 'idiomatic'
                       and p.get('pred_span_start') is not None
                       and p.get('span_start')      is not None]
    if idiomatic_preds:
        exact  = [int(p['pred_span_start'] == p['span_start'] and
                      p['pred_span_end']   == p['span_end'])
                  for p in idiomatic_preds]
        ov_f1s = [overlap_f1(p['pred_span_start'], p['pred_span_end'],
                              p['span_start'],      p['span_end'])
                  for p in idiomatic_preds]
        result['span_exact_match'] = round(np.mean(exact),  4)
        result['span_overlap_f1']  = round(np.mean(ov_f1s), 4)
        result['n_span']           = len(idiomatic_preds)

    return result


# ── Persistence ────────────────────────────────────────────────────────────────

def save_jsonl(records, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')
    actual = sum(1 for _ in open(path, encoding='utf-8'))
    assert actual == len(records), f"Persistence FAILED: wrote {len(records)}, read {actual}"
    print(f"  ✓ {actual} rows → {path}")


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    args    = parse_args()
    device  = get_device(args.device)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Device : {device}")
    print(f"Split  : {args.split}")
    print(f"Langs  : {args.langs}")
    print(f"Model  : {args.model_dir}")
    print(f"Output : {out_dir}")

    # Verify Drive durability
    probe = out_dir / '.write_test'
    probe.write_text('ok')
    assert probe.read_text() == 'ok'
    probe.unlink()
    print(f"  ✓ Output dir durable")

    # ── Load data ─────────────────────────────────────────────────────────────
    data_dir     = Path(args.data_dir)
    all_examples = load_id10m_data(data_dir, args.langs, args.split)

    if not all_examples:
        print("No data found. Check --data_dir and --langs.")
        sys.exit(1)

    # ── Inference per language ────────────────────────────────────────────────
    all_results: dict[str, dict] = {}

    for lang_label, examples in all_examples.items():
        pred_path = out_dir / f'id10m_system_e_{lang_label.lower()}_preds.jsonl'

        if pred_path.exists() and not args.force:
            print(f"\n[{lang_label}] Predictions exist. Use --force to rerun.")
            preds = [json.loads(l) for l in open(pred_path, encoding='utf-8')]
        else:
            print(f"\nRunning inference on {len(examples)} [{lang_label}] examples...")
            preds = run_inference(examples, args.model_dir, device,
                                   args.max_len, args.batch_size, lang_label)
            save_jsonl(preds, pred_path)

        m = compute_metrics(preds)
        all_results[lang_label] = m

        print(f"\n  [{lang_label}]  n={m['n']}")
        print(f"    CLS macro F1 : {m['macro_f1']:.4f}  "
              f"(lit={m['literal_f1']:.4f}, idiom={m['idiomatic_f1']:.4f})")
        if 'span_exact_match' in m:
            print(f"    Span exact   : {m['span_exact_match']:.4f}  "
                  f"Overlap F1: {m['span_overlap_f1']:.4f}  (n={m['n_span']})")

    print(f"\n  Note: ID10M supervised toplines must be read from paper Table 3.")
    print(f"  Our system: zero-shot (no ID10M training data, no fine-tuning).")

    # ── Save results ──────────────────────────────────────────────────────────
    results_path = out_dir / 'id10m_results.json'
    results_path.write_text(json.dumps(all_results, indent=2))
    assert json.loads(results_path.read_text()) == all_results
    print(f"\n  ✓ Results → {results_path}")

    summary_lines = [
        "ID10M — System E (Joint mBERT, EN+TE) Zero-Shot",
        "=" * 50,
        f"Split: {args.split}  |  Model: EN+TE training only",
        "",
    ]
    for lang, m in all_results.items():
        summary_lines.append(f"[{lang}]  n={m['n']}")
        summary_lines.append(f"  CLS macro F1 : {m['macro_f1']:.4f}  "
                              f"(lit={m['literal_f1']:.4f}, idiom={m['idiomatic_f1']:.4f})")
        if 'span_exact_match' in m:
            summary_lines.append(f"  Span exact   : {m['span_exact_match']:.4f}  "
                                  f"Overlap F1: {m['span_overlap_f1']:.4f}  (n={m['n_span']})")
        summary_lines.append("")
    summary_lines.append("ID10M supervised toplines: see paper Table 3 (fill in manually).")

    summary_path = out_dir / 'id10m_results_summary.txt'
    summary_path.write_text('\n'.join(summary_lines))
    print(f"  ✓ Summary → {summary_path}")
    print("\n✓ Done.")


if __name__ == '__main__':
    main()
