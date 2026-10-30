# NLP Group Assignment 2

DSL 504, Natural Language Processing. Assignment brief: [`NLP_Group_Assignment_2.pdf`](NLP_Group_Assignment_2.pdf).

Four independent NLP systems, one per question. Each lives in its own folder with its own
code and setup notes. At the end we hand in one ZIP with everything and one combined
technical report.

## Layout

| Folder | What we build | Owner | Issue |
|---|---|---|---|
| [`Question-1/`](Question-1/) | English to Hindi NMT: LSTM encoder-decoder with and without attention, attention heatmaps, error and sentence-length analysis. IIT Bombay English-Hindi corpus. | Rishi | [#1](https://github.com/Akshats-git/NLP-Group-Assignment-2/issues/1) |
| [`Question-2/`](Question-2/) | 5-class Amazon review rating prediction: fine-tuned RoBERTa (mean pooling) vs a BiGRU with additive attention pooling. Accuracy, macro-F1, MAE, one ablation. | Keshav | [#2](https://github.com/Akshats-git/NLP-Group-Assignment-2/issues/2) |
| [`Question-3/`](Question-3/) | Abstractive summarisation on CNN/DailyMail with a Transformer encoder-decoder written from scratch. ROUGE and cross-attention heatmap. | Shivam, Ayush | [#3](https://github.com/Akshats-git/NLP-Group-Assignment-2/issues/3) |
| [`Question-4/`](Question-4/) | Plot-Twist Generator on TinyStories: Vanilla RNN vs LSTM language models, greedy / temperature / top-k decoding, diversity metrics and blinded human evaluation. | Akshat | [#4](https://github.com/Akshats-git/NLP-Group-Assignment-2/issues/4) |
| [`common/`](common/) | Shared helpers every question imports: seeding, devices and mixed precision, vocab and padding, early stopping and checkpoints, decoding, report figures. | everyone | |
| [`tests/`](tests/) | Tests for `common/`. | | |

Each issue has the full breakdown, the deliverables checklist and the marking scheme.
Tick things off there as they get done.

## Team

| Name | Roll No. | Email | GitHub |
|---|---|---|---|
| Shivam Singh | 12342020 | shivamsgh@iitbhilai.ac.in | [@Sh3001](https://github.com/Sh3001) |
| Ayush Khelwal | 12340430 | ayushkl@iitbhilai.ac.in | [@Ayush-khelwal2003](https://github.com/Ayush-khelwal2003) |
| Keshav Mishra | 12341140 | keshavm@iitbhilai.ac.in | [@keshavm21](https://github.com/keshavm21) |
| Rishi Kharya | 12341790 | rishik@iitbhilai.ac.in | [@krish1809](https://github.com/krish1809) |
| Akshat Gupta | 12340160 | akshatg@iitbhilai.ac.in | [@Akshats-git](https://github.com/Akshats-git) |

## Work split

One question each, with two people on Question 3 because it is the biggest: the whole
Transformer has to be written from scratch and evaluated on the full CNN/DM test set.

| Question | Owner | Notes |
|---|---|---|
| Q1: Attention NMT | Rishi Kharya | Hindi tick labels in the heatmaps need the font setup in [`common/README.md`](common/README.md#hindi-in-plots). |
| Q2: RoBERTa vs BiGRU | Keshav Mishra | Heaviest GPU job (200k reviews, two seeds per model). Start the RoBERTa runs early. |
| Q3: Transformer summariser | Shivam Singh, Ayush Khelwal | Natural split: one on the model, training and decoding, the other on tokenizer, data, ROUGE, qualitative cases and the attention heatmap. Agree on tensor shapes and the mask convention first. |
| Q4: Plot-Twist Generator | Akshat Gupta | The blinded human evaluation needs at least three raters, so everyone rates Q4 stories. |

Every owner writes their question's section of the combined report. On top of that,
everyone reviews at least one other question's PRs and helps with the Q4 ratings, so
each member has touched implementation, experiments, evaluation and the report.

## Working on it

- Branch per piece of work, named like `q1/attention-decoder` or `q3/rouge-eval`. Open a PR
  into `main` and get one teammate to look at it before merging. Don't push straight to `main`.
- Commits: short and plain, say what changed. `add baseline lstm`, `fix pad mask in loss`.
- Keep code for each question inside its own `Question-N/` folder.
- Before writing a helper, check [`common/`](common/README.md). Seeding, padding,
  vocab, early stopping, checkpoints, decoding and plots are already there. If something
  would help more than one question, add it to `common/` with a test instead of copying it.
- `common/` is shared, so changes to existing functions go through a PR that the people
  using them have seen, and `pytest -q` must pass.
- Use `set_seed` from `common` and keep the seeds in `run_info.json` (`save_run_info`).
  The brief asks for reproducible results in Q2 and Q4.
- Results and figures go to `Question-N/outputs/` (see the layout in
  [`common/README.md`](common/README.md#conventions-everyone-follows)). Don't commit
  datasets. Checkpoints stay out of git, except Q4's, which the brief says to submit.

## Setup

Python 3.11+ and a CUDA GPU for Q2 and Q3 (Colab or Kaggle is fine). Each question will get
its own `requirements.txt` and run instructions in its folder README as the code lands.

```bash
git clone https://github.com/Akshats-git/NLP-Group-Assignment-2.git
cd NLP-Group-Assignment-2
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"                       # installs common/ so any notebook can import it
pytest -q                                     # check common/ works on your machine
pip install -r Question-N/requirements.txt    # once it exists
```

On Colab or Kaggle, clone the repo, `%cd` into it and run `import sys; sys.path.insert(0, ".")`
before `import common`. The [`common/README.md`](common/README.md) has the full snippet and examples.

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
