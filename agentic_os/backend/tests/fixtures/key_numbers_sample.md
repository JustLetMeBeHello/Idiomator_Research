## IdiomBERT — Full EN+ES+HI+TE Training, Joint F1 (macro-avg)
D=0.7515, A=0.7486, F=0.7440, E=0.7381, C=0.6977, B4=0.6966, B=0.6915, C4=0.5118, G=0.4750

## Stability Score (mean − std, in-distribution languages)
D=0.7061, A=0.7049, F=0.6843, E=0.6774, C=0.6148, B4=0.6247, B=0.5999, C4=0.4442, G=0.4658

## Classification F1 — full training
A/D: 0.7823 overall; E: 0.7760; F: 0.7741; B: 0.7599; C: 0.7268; B4: 0.7278; C4: 0.6668

## System G Language Ablation Matrix — E2E Span Overlap F1 (idiomatic, overall)

| Combo | E2E Span F1 | Indo held-out F1 | Stability (mean−std) |
|---|---|---|---|
| en | 0.7348 | 0.6474 | 0.4427 |
| en_es | 0.7824 | 0.6725 | 0.4522 |
| **en_es_hi_te** | **0.8072** | **0.7209** | **0.4699** |
