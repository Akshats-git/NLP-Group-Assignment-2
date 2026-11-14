# Q1 evaluation scores (test set, 2,393 sentences, seed 42)

Recomputed from `test_predictions.jsonl` with sacrebleu 2.6.0; identical to `test_bleu.json` written by evaluate.py.

## Overall

| metric | baseline (A) | attention (B) |
|---|---|---|
| n_sentences | 2,393 | 2,393 |
| BLEU | 0.5468 | 2.3396 |
| BLEU_p1 | 15.8863 | 23.9791 |
| BLEU_p2 | 2.1689 | 5.3346 |
| BLEU_p3 | 0.1900 | 1.1264 |
| BLEU_p4 | 0.0137 | 0.2080 |
| brevity_penalty | 1.0000 | 1.0000 |
| length_ratio | 1.0332 | 1.0872 |
| hyp_tokens | 58,459 | 61,516 |
| ref_tokens | 56,583 | 56,583 |
| chrF | 12.2746 | 17.9086 |
| chrF++ | 11.5409 | 17.3667 |
| TER | 104.0272 | 99.1188 |
| test_loss | 5.0778 | 4.6649 |
| test_perplexity | 160.4165 | 106.1505 |
| test_token_acc | 0.2428 | 0.2902 |
| best_epoch | 4 | 4 |
| best_val_loss | 5.1801 | 4.7542 |
| best_val_acc | 0.2494 | 0.2980 |
| best_epoch_train_loss | 4.5424 | 3.6331 |
| best_epoch_train_acc | 0.2478 | 0.3250 |
| parameters | 38,334,930 | 71,896,274 |
| seconds_per_epoch | 48.2818 | 221.6276 |

## By English sentence length: A: baseline (no attention)

| length | n | BLEU | p1 | p2 | p3 | p4 | BP | len ratio | chrF | chrF++ | TER |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1-5 | 13 | 3.07 | 20.5 | 5.7 | 1.8 | 1.1 | 0.786 | 0.806 | 7.61 | 8.46 | 91.3 |
| 6-10 | 282 | 0.41 | 14.2 | 2.1 | 0.0 | 0.0 | 0.969 | 0.970 | 8.96 | 8.71 | 102.2 |
| 11-15 | 481 | 0.48 | 14.1 | 2.3 | 0.1 | 0.0 | 1.000 | 1.119 | 10.74 | 10.31 | 108.5 |
| 16+ | 1617 | 0.57 | 16.3 | 2.2 | 0.2 | 0.0 | 1.000 | 1.024 | 12.73 | 11.91 | 103.5 |

## By English sentence length: B: attention (Bahdanau)

| length | n | BLEU | p1 | p2 | p3 | p4 | BP | len ratio | chrF | chrF++ | TER |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1-5 | 13 | 2.13 | 26.7 | 4.8 | 1.0 | 0.7 | 0.688 | 0.728 | 12.77 | 12.59 | 84.5 |
| 6-10 | 282 | 2.73 | 27.7 | 6.9 | 1.5 | 0.3 | 0.931 | 0.933 | 16.03 | 16.10 | 88.9 |
| 11-15 | 481 | 2.74 | 25.9 | 6.3 | 1.3 | 0.3 | 1.000 | 1.034 | 17.13 | 16.99 | 94.0 |
| 16+ | 1617 | 2.26 | 23.5 | 5.1 | 1.1 | 0.2 | 1.000 | 1.106 | 18.16 | 17.51 | 100.6 |
