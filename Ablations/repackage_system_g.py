"""
Repackage System G (BIOTagger) as a standard BertForTokenClassification artifact.

The original `bio_tagger_en_hi_te/best_model/` only contains the BIO head
state_dict (`bio_head.pt`), the tokenizer, and a config that points the encoder
back at the HF hub. To deploy it through the website's standard
`AutoModelForTokenClassification.from_pretrained(...)` path, we need a single
self-contained directory.

This script:
  1. Loads `bert-base-multilingual-cased` as a `BertForTokenClassification`
     (encoder + dropout + linear head with 3 output classes, head initially
     random).
  2. Loads `bio_head.pt` (the trained BIO head weights) into `model.classifier`.
  3. Loads the tokenizer that shipped with the BIO checkpoint.
  4. Saves the full thing via `save_pretrained()` → a directory containing
     `config.json`, `model.safetensors` (or `pytorch_model.bin`),
     `tokenizer.json`, `tokenizer_config.json`, etc.

Usage:
    python repackage_system_g.py \\
        --src ../models/bio_tagger_en_hi_te/best_model \\
        --dst ~/Desktop/Language_Learning_BaseWebsite/Backend/checkpoints_system_g
"""

import argparse
from pathlib import Path

import torch
from transformers import (
    AutoTokenizer,
    BertForTokenClassification,
)

LABEL2ID = {"O": 0, "B-IDIOM": 1, "I-IDIOM": 2}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--src",
        required=True,
        type=Path,
        help="Source checkpoint dir containing bio_head.pt + tokenizer files.",
    )
    p.add_argument(
        "--dst",
        required=True,
        type=Path,
        help="Destination dir for the repackaged BertForTokenClassification.",
    )
    p.add_argument(
        "--base_model",
        default="bert-base-multilingual-cased",
        help="HF base model name to use as the encoder.",
    )
    p.add_argument(
        "--dropout",
        type=float,
        default=0.239431179668018881,
        help="Classifier dropout — matches the value used during System G training.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    src: Path = args.src.expanduser().resolve()
    dst: Path = args.dst.expanduser().resolve()

    head_pt = src / "bio_head.pt"
    if not head_pt.exists():
        raise FileNotFoundError(f"bio_head.pt not found at {head_pt}")

    print(f"[•] Source     : {src}")
    print(f"[•] Destination: {dst}")
    print(f"[•] Base model : {args.base_model}")

    # 1. Build a BertForTokenClassification with the right shape.
    print("[•] Loading base model as BertForTokenClassification ...")
    model = BertForTokenClassification.from_pretrained(
        args.base_model,
        num_labels=len(LABEL2ID),
        id2label=ID2LABEL,
        label2id=LABEL2ID,
        classifier_dropout=args.dropout,
    )

    # 2. Load the trained BIO head into the classifier.
    print(f"[•] Loading BIO head weights from {head_pt} ...")
    head_state = torch.load(head_pt, map_location="cpu")

    # Sanity-check shapes before assignment.
    expected_w = (model.config.num_labels, model.config.hidden_size)
    expected_b = (model.config.num_labels,)
    if "weight" not in head_state or "bias" not in head_state:
        raise RuntimeError(
            f"bio_head.pt is missing weight/bias keys: got {list(head_state.keys())}"
        )
    if tuple(head_state["weight"].shape) != expected_w:
        raise RuntimeError(
            f"weight shape mismatch: bio_head={tuple(head_state['weight'].shape)} "
            f"expected={expected_w}"
        )
    if tuple(head_state["bias"].shape) != expected_b:
        raise RuntimeError(
            f"bias shape mismatch: bio_head={tuple(head_state['bias'].shape)} "
            f"expected={expected_b}"
        )

    missing, unexpected = model.classifier.load_state_dict(head_state, strict=True)
    if missing or unexpected:
        # strict=True would have raised, but belt-and-braces for older torch.
        raise RuntimeError(
            f"Head load had missing={missing} unexpected={unexpected}"
        )

    # 3. Load the tokenizer that shipped with System G.
    print(f"[•] Loading tokenizer from {src} ...")
    tokenizer = AutoTokenizer.from_pretrained(str(src))

    # 4. Save the full self-contained artifact.
    dst.mkdir(parents=True, exist_ok=True)
    print(f"[•] Saving repackaged model to {dst} ...")
    model.save_pretrained(dst)
    tokenizer.save_pretrained(dst)

    # 5. Verify the round-trip by re-loading the saved dir.
    print("[•] Round-trip verification ...")
    from transformers import AutoModelForTokenClassification
    reloaded = AutoModelForTokenClassification.from_pretrained(str(dst))
    if reloaded.config.num_labels != len(LABEL2ID):
        raise RuntimeError(
            f"Reloaded model has wrong num_labels: {reloaded.config.num_labels}"
        )

    # Confirm head weights survived the round-trip.
    orig_w = head_state["weight"]
    reloaded_w = reloaded.classifier.weight.detach().cpu()
    if not torch.allclose(orig_w, reloaded_w, atol=1e-6):
        raise RuntimeError("Head weights changed during save/reload!")

    print()
    print(f"[✓] Repackaged System G saved to {dst}")
    print(f"[✓] num_labels={reloaded.config.num_labels} id2label={reloaded.config.id2label}")
    print(f"[✓] Head weights survived round-trip (allclose @ 1e-6)")


if __name__ == "__main__":
    main()
