| model | BLEU | chrF | test token acc | test loss | best epoch | val loss | val acc |
|---|---|---|---|---|---|---|---|
| baseline | 0.55 | 12.3 | 0.243 | 5.078 | 4 | 5.180 | 0.249 |
| attention | 2.34 | 17.9 | 0.290 | 4.665 | 4 | 4.754 | 0.298 |

| source length | sentences | baseline BLEU | baseline chrF | attention BLEU | attention chrF |
|---|---|---|---|---|---|
| 1-5 | 13 | 3.07 | 7.6 | 2.13 | 12.8 |
| 6-10 | 282 | 0.41 | 9.0 | 2.73 | 16.0 |
| 11-15 | 481 | 0.48 | 10.7 | 2.74 | 17.1 |
| 16+ | 1617 | 0.57 | 12.7 | 2.26 | 18.2 |
