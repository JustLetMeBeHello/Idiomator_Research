# Idiomator Training

Fine-tuning mBERT and monolingual BERT for multilingual idiomaticity classification and span extraction.

## Repo structure

```
Idiomator_Training/
├── data/
│   ├── Splits/                        ← train/dev/test.jsonl (from prepare_dataset.py)
│   └── Span_tagged_data/              ← raw per-language JSONL files
│       ├── Final_English_MERGED_normalized.jsonl
│       ├── Final_Hindi_MERGED.jsonl
│       └── Final_Telugu_MERGED.jsonl
├── models/                            ← saved checkpoints (gitignored)
│   └── sweep/                         ← one dir per sweep run
├── scripts/
│   ├── prepare_dataset.py             ← sampling + train/dev/test split
│   ├── train_stage1_classifier.py     ← Stage 1 training script
│   ├── sweep_stage1.sh                ← hyperparameter sweep (run on Colab)
│   └── check_mps.py                   ← MPS diagnostic (local Mac only)
├── results/
│   └── sweep_stage1_results.csv       ← auto-populated by sweep script
├── requirements.txt
└── README.md
```

## Setup

```bash
git clone https://github.com/YOUR_USERNAME/Idiomator_Training.git
cd Idiomator_Training
pip install -r requirements.txt
```

## Data

Hindi and Telugu use **all available examples** (low-resource setting).
English uses a 2x sample relative to Hindi+Telugu combined.
All splits use an **unseen idiom split** (80/10/10) — test idioms never appear in train.

## Running on Google Colab

1. Mount Google Drive:
```python
from google.colab import drive
drive.mount('/content/drive')
```

2. Clone repo into Drive:
```bash
cd /content/drive/MyDrive
git clone https://github.com/YOUR_USERNAME/Idiomator_Training.git
```

3. Install dependencies:
```bash
pip install -r /content/drive/MyDrive/Idiomator_Training/requirements.txt
```

4. Run hyperparameter sweep:
```bash
DRIVE_ROOT=/content/drive/MyDrive/Idiomator_Training bash scripts/sweep_stage1.sh
```

## Running a single experiment

```bash
# mBERT — all three languages
python scripts/train_stage1_classifier.py \
    --output_dir models/stage1_mbert_en_hi_te \
    --langs English Hindi Telugu

# mBERT — Hindi + Telugu only
python scripts/train_stage1_classifier.py \
    --output_dir models/stage1_mbert_hi_te \
    --langs Hindi Telugu

# Monolingual BERT — English only
python scripts/train_stage1_classifier.py \
    --model_name bert-base-uncased \
    --output_dir models/stage1_bert_en \
    --langs English
```

## Experiments

### Stage 1: Idiomaticity Classification

| Run | Model | Languages | Status |
|-----|-------|-----------|--------|
| 1 | mBERT | EN+HI+TE | ✅ test_f1=0.7693 |
| 2 | mBERT | HI+TE | 🔄 |
| 3 | mBERT | EN+TE | 🔄 |
| 4 | mBERT | EN+HI | 🔄 |
| 5 | mBERT | EN only | 🔄 |
| 6 | mBERT | HI only | 🔄 |
| 7 | mBERT | TE only | 🔄 |
| 8 | BERT (monolingual) | EN only | 🔄 |
| 9 | GPT-4o zero-shot | EN+HI+TE | 🔄 |

### Stage 2: Span Extraction

Coming after Stage 1 hyperparameter tuning is complete.

## Results

See `results/sweep_stage1_results.csv` for full sweep results.