# Question 2: RoBERTa vs BiGRU for Amazon review ratings

5-class star rating on `SetFit/amazon_reviews_multi_en` (train 200k, validation 5k, test 5k).
A BiGRU with additive attention pooling over GloVe embeddings is compared with fine-tuned
`roberta-base` using masked mean pooling. The ablation compares mean pooling with first-token
(`<s>`) pooling for RoBERTa. Every model is trained with seeds 42 and 43.

## Layout

```
Question-2/
  01_preprocessing.ipynb   both input pipelines, dataset and length statistics
  03_roberta.ipynb         RoBERTa fine-tuning (4 runs), pooling ablation, curves
  q2/                      code the notebooks import
    data.py                loading, title + body merge, length statistics
    classic.py             BiGRU pipeline: cleaning, spaCy tokens, vocab, GloVe matrix, loaders
    roberta_data.py        RoBERTa pipeline: tokenisation, truncation, dynamic padding, loaders
    batching.py            length-grouped batch sampler
    models.py              RobertaClassifier with mean or first-token pooling
    metrics.py             accuracy, macro / per-class F1, MAE, adjacent vs distant errors
    train_roberta.py       fine-tuning loop, also a command-line script
  tests/                   offline tests for q2/
  outputs/results/         metrics and statistics (committed)
  outputs/figures/         report figures (committed)
  data/                    cached datasets, GloVe, vocab (gitignored)
```

## Data notes

- **Titles.** The SetFit copy only kept the review body. `mteb/amazon_reviews_multi` (config `en`)
  is the same data with the same ids and splits and stores `title + "\n\n" + body`. Splits and labels
  come from SetFit, as the brief asks; titles are matched by id and only attached when the two bodies
  agree. The model input is `"<title>. <body>"`.
- **Labels** are 0-4 and stars are label + 1. MAE is reported in stars.
- **Order.** The raw splits are sorted by label, so training loaders always shuffle and `limit`
  takes a seeded random subset.
- **Test split.** Nothing loads `test` except the final evaluation (`load_reviews(("test",))`).

## Running on Kaggle

1. **File → Import Notebook → GitHub**, and paste the notebook's GitHub link, e.g.
   `https://github.com/Akshats-git/NLP-Group-Assignment-2/blob/main/Question-2/03_roberta.ipynb`.
2. In the first code cell set `BRANCH` to the branch with the code you want, e.g. `"q2/roberta"` until it
   is merged. When the notebook finds no repo around it, it clones that branch and moves into `Question-2/`.
   The libraries it needs (transformers, datasets, spaCy, scikit-learn) are already installed on Kaggle.
3. Side panel: turn **Internet on** (it needs a phone-verified account). The preprocessing notebook
   runs on CPU. Training needs **GPU T4 x2**.
4. For the BiGRU work, **Add Input** a GloVe dataset that contains `glove.6B.300d.txt` (search "glove 6B 300d").
   `find_glove` picks it up from `/kaggle/input/` on its own. Without it, the 822 MB Stanford zip is downloaded.

`01_preprocessing.ipynb` writes `outputs/results/preprocessing_stats.json`, `outputs/figures/length_distribution.png`,
and the vocabulary and embedding matrix to `data/classic/`.

### RoBERTa training

Use **GPU T4 x2**: both GPUs are used through `DataParallel`. Import `03_roberta.ipynb` as above and use
**Save Version → Save & Run All**, which runs in the background for up to 12 hours, so the browser can
be closed. The notebook trains four runs, mean and first-token pooling with seeds 42 and 43, each for
2 epochs. Every run prints validation scores and elapsed minutes twice per epoch.

When the version finishes, download `outputs/results/` and `outputs/figures/` from its Output tab
and copy them into `Question-2/outputs/` in the repo. The checkpoints in `outputs/checkpoints/`
(about 500 MB each) stay on Kaggle. Add this notebook's output as an input to the evaluation
notebook later.

Runs that already have a `summary.json` are skipped. A new Kaggle version starts from a fresh clone, so
to continue from an earlier version, attach its output (**Add Input → Your Work → Notebooks**). The
notebook copies the finished runs back first (`q2.kaggle.import_finished_runs`) and only trains the rest.
A single run from a terminal:

```bash
python -m q2.train_roberta --pooling mean --seed 42                 # from Question-2/
python -m q2.train_roberta --limit 2000 --epochs 1 --device cpu     # quick check, writes *_limit2000 folders
```

On an Apple Silicon Mac, pass `--device cpu`: the MPS backend stalls on RoBERTa training.

## Running locally

From the repo root:

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]" -r Question-2/requirements.txt
pytest -q Question-2/tests                                  # offline, a few seconds
Q2_LIMIT=2000 jupyter nbconvert --to notebook --execute Question-2/01_preprocessing.ipynb --output /tmp/q2_check.ipynb
```

`Q2_LIMIT` runs the notebook on a random subset as a smoke test. It writes no results, figures or vocabulary.
