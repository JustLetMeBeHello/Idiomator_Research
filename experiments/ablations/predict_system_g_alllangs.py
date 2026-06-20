"""Regenerate System G (mBERT BIO) predictions for ALL test languages
(EN/ES/HI/TE/ID) using the fixed-encoder best_model, replicating the exact
prediction loop in BiO_Task_mBERT_train.py:634-686 with the patched decoder.

EN/ES/HI/TE rows are cross-checked against the known-good fixed 632-row file;
Indonesian rows are the new fixed-encoder zero-shot predictions.
"""
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent))
from BiO_Task_mBERT_train import (
    load_split, BIODataset, load_best_model, decode_bio_to_char_span,
    compute_overlap_f1, ID2LABEL, IGNORE_IDX, get_device,
)
from transformers import AutoTokenizer

MODEL_NAME = "bert-base-multilingual-cased"
OUTPUT_DIR = Path("models/bio_tagger_en_hi_te")
DATA_DIR = "data/idioms_structured/Splits"
ALL_LANGS = ["English", "Spanish", "Hindi", "Telugu", "Indonesian"]
MAX_LEN = 128
BATCH_SIZE = 32
OUT_PATH = OUTPUT_DIR / "test_predictions_alllangs.jsonl"


def main():
    device = get_device(None)
    print(f"Device: {device}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    test_examples = load_split(DATA_DIR, "test", ALL_LANGS)
    test_ds = BIODataset(test_examples, tokenizer, MAX_LEN)
    print(f"Valid test examples: {len(test_ds)} / {len(test_examples)}")
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

    model = load_best_model(MODEL_NAME, OUTPUT_DIR, device)
    model.eval()

    preds_out = []
    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            token_type_ids = batch["token_type_ids"].to(device)

            logits = model(input_ids, attention_mask, token_type_ids)
            preds = torch.argmax(logits, dim=-1).cpu()

            batch_start = batch_idx * test_loader.batch_size
            for i in range(len(preds)):
                ex_idx = batch_start + i
                if ex_idx >= len(test_ds.valid_examples):
                    break
                ex = test_ds.valid_examples[ex_idx]
                sentence = ex["sentence"]
                gold_s = ex["span_start"]
                gold_e = ex["span_end"]

                enc = tokenizer(
                    sentence, max_length=MAX_LEN,
                    truncation=True, return_offsets_mapping=True,
                )
                pred_bio = preds[i].tolist()
                pred_char_s, pred_char_e = decode_bio_to_char_span(
                    pred_bio, enc, sentence, MAX_LEN
                )
                if pred_char_s is None:
                    pred_char_s, pred_char_e = 0, 0

                exact = bool(pred_char_s == gold_s and pred_char_e == gold_e)
                overlap = compute_overlap_f1(pred_char_s, pred_char_e, gold_s, gold_e)

                preds_out.append({
                    **ex,
                    "pred_span_start": pred_char_s,
                    "pred_span_end": pred_char_e,
                    "pred_matched_span": sentence[pred_char_s:pred_char_e],
                    "pred_bio_tags": [ID2LABEL.get(l, "O") for l in pred_bio
                                      if l != IGNORE_IDX],
                    "span_exact_match": exact,
                    "span_overlap_f1": round(overlap, 4),
                })

    with open(OUT_PATH, "w", encoding="utf-8") as f:
        for p in preds_out:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    print(f"Wrote {len(preds_out)} predictions -> {OUT_PATH}")


if __name__ == "__main__":
    main()
