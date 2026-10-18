# NLP Group Assignment 2

DSL 504, Natural Language Processing. Assignment brief: [`NLP_Group_Assignment_2.pdf`](NLP_Group_Assignment_2.pdf).

Four independent NLP systems, one per question. Each lives in its own folder with its own
code and setup notes. At the end we hand in one ZIP with everything and one combined
technical report.

## Layout

| Question | What we build | Issue |
|---|---|---|
| [`Question-1/`](Question-1/) | English to Hindi NMT: LSTM encoder-decoder with and without attention, attention heatmaps, error and sentence-length analysis. IIT Bombay English-Hindi corpus. | [#1](https://github.com/Akshats-git/NLP-Group-Assignment-2/issues/1) |
| [`Question-2/`](Question-2/) | 5-class Amazon review rating prediction: fine-tuned RoBERTa (mean pooling) vs a BiGRU with additive attention pooling. Accuracy, macro-F1, MAE, one ablation. | [#2](https://github.com/Akshats-git/NLP-Group-Assignment-2/issues/2) |
| [`Question-3/`](Question-3/) | Abstractive summarisation on CNN/DailyMail with a Transformer encoder-decoder written from scratch. ROUGE and cross-attention heatmap. | [#3](https://github.com/Akshats-git/NLP-Group-Assignment-2/issues/3) |
| [`Question-4/`](Question-4/) | Plot-Twist Generator on TinyStories: Vanilla RNN vs LSTM language models, greedy / temperature / top-k decoding, diversity metrics and blinded human evaluation. | [#4](https://github.com/Akshats-git/NLP-Group-Assignment-2/issues/4) |

Each issue has the full breakdown, the work split, the deliverables checklist and the
marking scheme. Tick things off there as they get done.

## Team

| Name | Roll No. | Email | GitHub |
|---|---|---|---|
| Shivam Singh | 12342020 | shivamsgh@iitbhilai.ac.in | [@Sh3001](https://github.com/Sh3001) |
| Ayush Khelwal | 12340430 | ayushkl@iitbhilai.ac.in | [@Ayush-khelwal2003](https://github.com/Ayush-khelwal2003) |
| Keshav Mishra | 12341140 | keshavm@iitbhilai.ac.in | [@keshavm21](https://github.com/keshavm21) |
| Rishi Kharya | 12341790 | rishik@iitbhilai.ac.in | [@krish1809](https://github.com/krish1809) |
| Akshat Gupta | 12340160 | akshatg@iitbhilai.ac.in | [@Akshats-git](https://github.com/Akshats-git) |

## Work split

Five people, four questions, so everyone has one main piece plus a smaller one.
Slots M1 to M5 are placeholders until we decide who takes which.

| Slot | Main | Secondary | Owner |
|---|---|---|---|
| M1 | Q1 core: data prep, baseline LSTM, attention decoder, training | Q4: data prep, RNN/LSTM models, training | _TBD_ |
| M2 | Q3 data and evaluation: tokenizer, ROUGE, qualitative cases, attention heatmap | Q1 analysis: heatmaps, error analysis, length analysis | _TBD_ |
| M3 | Q2 baseline and evaluation: both preprocessing pipelines, BiGRU, metrics and confusion matrices | Q4: decoding strategies and Plot-Twist generation | _TBD_ |
| M4 | Q2 RoBERTa: fine-tuning, multi-seed runs, ablation | Q4: diversity metrics and blinded human evaluation | _TBD_ |
| M5 | Q3 model: attention, encoder/decoder, full model, training loop | none | _TBD_ |

Everyone also writes the report sections for the parts they built, and reviews one
teammate's code.

## Working on it

- Branch per piece of work, named like `q1/attention-decoder` or `q3/rouge-eval`. Open a PR
  into `main` and get one teammate to look at it before merging. Don't push straight to `main`.
- Commits: short and plain, say what changed. `add baseline lstm`, `fix pad mask in loss`.
- Keep code for each question inside its own `Question-N/` folder. Shared helpers go in
  that question's folder too, so the four submissions stay independent.
- Fix random seeds and write them down. The brief asks for reproducible results in Q2 and Q4.
- Don't commit datasets. Large checkpoints stay out of git too (see packaging below).

## Setup

Python 3.10+ and a CUDA GPU for Q2 and Q3 (Colab or Kaggle is fine). Each question will get
its own `requirements.txt` and run instructions in its folder README as the code lands.

```bash
git clone https://github.com/Akshats-git/NLP-Group-Assignment-2.git
cd NLP-Group-Assignment-2
python -m venv .venv && source .venv/bin/activate
pip install -r Question-N/requirements.txt   # once it exists
```

## Submission

One member submits one ZIP for the whole group. It contains the source code, notebooks,
results, visualisations, a README with reproduction steps, and one combined technical
report covering all four questions.

The brief is not identical across questions, so check these when packaging:

| | Datasets | Checkpoints | Report |
|---|---|---|---|
| Q1 | no | not required | short technical report |
| Q2 | no | not required | max 6 pages |
| Q3 | no | **no** | must state tokenizer, truncation lengths, model config, training-set size, decoding method, ROUGE-1/2/L/Lsum |
| Q4 | no (document the subset) | **yes**, plus vocab/tokenizer | methodology, results, conclusions, exact commands |

Before submitting: every member has contributed to code, experiments, evaluation and the
report; the README reproduces the results from a clean environment; no dataset is in the
archive.

## Status

- [ ] Q1: attention NMT
- [ ] Q2: RoBERTa vs BiGRU
- [ ] Q3: Transformer summariser
- [ ] Q4: Plot-Twist Generator
- [ ] Combined report
- [ ] Final ZIP
