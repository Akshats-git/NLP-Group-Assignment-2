Raters (4): Ayush, Keshav, Rishi Kharya, Shivam Singh. Ratings: 80. Stories: 20.

## Mean rating by model (1-5, mean over stories of the mean over raters)

| model | stories | coherence | grammar | relevance | creativity | overall | val ppl |
|---|---|---|---|---|---|---|---|
| LSTM | 10 | 3.00 ± 0.57 | 3.83 ± 0.43 | 2.73 ± 0.52 | 3.42 ± 0.59 | 3.24 | 12.12 |
| Vanilla RNN | 10 | 2.70 ± 0.51 | 4.08 ± 0.39 | 2.45 ± 0.52 | 3.55 ± 0.52 | 3.19 | 16.44 |

## By model and decoding configuration

| model | decoding | stories | coherence | grammar | relevance | creativity | overall |
|---|---|---|---|---|---|---|---|
| LSTM | greedy | 5 | 3.10 | 3.95 | 2.70 | 3.10 | 3.21 |
| LSTM | top-k=40 | 5 | 2.90 | 3.70 | 2.75 | 3.75 | 3.27 |
| Vanilla RNN | greedy | 5 | 2.95 | 4.05 | 2.45 | 3.25 | 3.17 |
| Vanilla RNN | top-k=40 | 5 | 2.45 | 4.10 | 2.45 | 3.85 | 3.21 |

## Paired comparison, LSTM minus RNN (10 matched pairs: same opening, same decoding)

| criterion | mean diff | 95% bootstrap CI | LSTM wins / ties / RNN wins |
|---|---|---|---|
| coherence | +0.30 | [-0.23, +0.78] | 6 / 1 / 3 |
| grammar | -0.25 | [-0.57, +0.03] | 1 / 4 / 5 |
| relevance | +0.28 | [-0.15, +0.80] | 5 / 4 / 1 |
| creativity | -0.12 | [-0.38, +0.15] | 3 / 1 / 6 |
| overall | +0.05 | [-0.22, +0.36] | 4 / 1 / 5 |

## Inter-rater agreement

| criterion | Krippendorff alpha (ordinal) | mean pairwise weighted kappa | exact agreement | within 1 point | mean abs difference |
|---|---|---|---|---|---|
| coherence | -0.14 | 0.01 | 25% | 48% | 1.73 |
| grammar | -0.14 | -0.06 | 21% | 62% | 1.37 |
| relevance | -0.17 | 0.05 | 15% | 46% | 1.86 |
| creativity | -0.14 | 0.02 | 26% | 58% | 1.44 |

## Each rater's mean rating

| rater | n | coherence | grammar | relevance | creativity |
|---|---|---|---|---|---|
| Ayush | 20 | 4.40 | 4.70 | 4.35 | 4.60 |
| Keshav | 20 | 2.15 | 3.65 | 1.25 | 2.50 |
| Rishi Kharya | 20 | 2.15 | 4.65 | 2.10 | 3.70 |
| Shivam Singh | 20 | 2.70 | 2.80 | 2.65 | 3.15 |
