## Plot-twist proxies (5 openings; all samples, per model and decoding)

| model | decoding | stories | prompt coverage | new content words | repetition | max trigram repeat | name kept | ended |
|---|---|---|---|---|---|---|---|---|
| LSTM | greedy | 5 | 0.12 | 13.80 | 0.52 | 11.40 | 0.50 | 0.80 |
| LSTM | temp=0.5 | 15 | 0.12 | 26.53 | 0.12 | 3.13 | 0.67 | 1.00 |
| LSTM | temp=0.8 | 15 | 0.09 | 34.73 | 0.02 | 1.93 | 0.67 | 1.00 |
| LSTM | temp=1.2 | 15 | 0.11 | 52.87 | 0.00 | 1.33 | 0.67 | 1.00 |
| LSTM | top-k=10 | 15 | 0.08 | 33.20 | 0.02 | 2.27 | 0.67 | 0.93 |
| LSTM | top-k=40 | 15 | 0.10 | 40.33 | 0.01 | 1.60 | 0.50 | 1.00 |
| Vanilla RNN | greedy | 5 | 0.00 | 12.80 | 0.30 | 5.20 | 0.00 | 0.80 |
| Vanilla RNN | temp=0.5 | 15 | 0.05 | 25.67 | 0.06 | 2.27 | 0.00 | 0.93 |
| Vanilla RNN | temp=0.8 | 15 | 0.06 | 37.07 | 0.02 | 1.93 | 0.00 | 1.00 |
| Vanilla RNN | temp=1.2 | 15 | 0.08 | 66.33 | 0.00 | 1.27 | 0.00 | 0.67 |
| Vanilla RNN | top-k=10 | 15 | 0.03 | 31.67 | 0.02 | 2.00 | 0.00 | 0.93 |
| Vanilla RNN | top-k=40 | 15 | 0.05 | 48.87 | 0.01 | 1.73 | 0.00 | 0.87 |
