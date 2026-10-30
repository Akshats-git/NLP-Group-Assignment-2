# Question 1: English-to-Hindi NMT with LSTM Encoder-Decoder

**Owner:** Rishi Kharya (@krish1809)

English-to-Hindi neural machine translation using an LSTM encoder-decoder architecture.
Two models are compared: a baseline without attention, and the same architecture augmented
with Bahdanau (additive) attention. We train both, compare performance, and visualise the
attention weights to understand the model's behaviour.

## Data

**IIT Bombay English-Hindi Parallel Corpus** ([link](https://www.cfilt.iitb.ac.in/iitb_parallel/),
loaded through HuggingFace `cfilt/iitb-english-hindi`)

- 50,000 training pairs (seeded random draw from 1.66M, at least 5 English tokens each)
- Official dev (514) and test (2,393) splits after cleaning
- Max sentence length: 50 tokens
- Tokenisation: lowercased English, NFC-normalised Hindi, punctuation split off, whitespace split
- Vocabulary: built from training data only, `min_freq=2` (21,433 English, 24,274 Hindi)

Full statistics and the reasoning for the filters are in [REPORT.md](REPORT.md).

## Models

| | Baseline (Model A) | Attention (Model B) |
|---|---|---|
| Encoder | Bidirectional LSTM, packed sequences | same |
| Decoder | LSTM, initial state from encoder only | LSTM + Bahdanau attention (own implementation) |
| Embedding dim | 256 | 256 |
| Hidden dim | 512 | 512 |
| Layers | 2 | 2 |
| Dropout | 0.3 | 0.3 |
| Parameters | 38.3M | 71.9M |

Decoding is greedy with `<pad>`, `<unk>` and `<sos>` banned for both models.

## Results (test set, seed 42)

| | BLEU | chrF | token acc |
|---|---|---|---|
| Baseline | 0.55 | 12.3 | 0.243 |
| Attention | 2.34 | 17.9 | 0.290 |

## Metrics

- **BLEU** (corpus-level, via sacrebleu) and **chrF** — translation metrics
- **Token accuracy** — training/validation monitoring metric

## Setup

```bash
# from repo root
git checkout q1/attention-nmt
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pip install -r Question-1/requirements.txt

# for Hindi text in plots (system-level):
# sudo apt-get install fonts-noto-core
```

## Reproduction

```bash
cd Question-1

# 1. Train both models (runs data download on first run)
python train.py --model both --epochs 20 --patience 4

# 2. Evaluate on test set (BLEU, chrF, length buckets, predictions)
python evaluate.py

# 3. Baseline limitation, attention statistics, error cases
python analysis.py

# 4. Figures and tables
python visualize.py

# Quick smoke test (2 epochs, small subset)
python train.py --model baseline --epochs 2 --max-train 1000 --min-len 1
```

## Output Structure

```
Question-1/
├── data.py              # Data loading, cleaning, vocabularies
├── models.py            # BaselineSeq2Seq, AttentionSeq2Seq
├── train.py             # Training loop
├── evaluate.py          # Test BLEU / chrF, length buckets, predictions
├── analysis.py          # Baseline limitation, attention statistics, error cases
├── visualize.py         # Curves, heatmaps, length plot, tables
├── REPORT.md            # Q1 section of the combined report
├── requirements.txt
├── README.md
├── data/                # (gitignored) cached splits, vocabularies, attention weights
└── outputs/
    ├── results/         # metrics, predictions, error cases, run info
    ├── figures/         # all report figures
    └── checkpoints/     # (gitignored) model weights
```

## Deliverables

- [x] Data preprocessing code
- [x] Baseline encoder-decoder code
- [x] Attention NMT code (custom Bahdanau attention)
- [x] Training and validation plots
- [x] Translation results
- [x] Attention heatmaps
- [x] Sentence-length analysis
- [x] Error analysis
- [x] Baseline vs attention comparison
- [x] Short technical report ([REPORT.md](REPORT.md))
