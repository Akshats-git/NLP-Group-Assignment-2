# Question 2: RoBERTa vs BiGRU for Amazon review ratings

5-class star rating on `SetFit/amazon_reviews_multi_en` (train 200k, validation 5k, test 5k).
A BiGRU with additive attention pooling over GloVe embeddings is compared with fine-tuned
`roberta-base` using masked mean pooling. The ablation compares mean pooling with first-token
(`<s>`) pooling for RoBERTa. Every model is trained with seeds 42 and 43.

## Layout

```
Question-2/
  01_preprocessing.ipynb   both input pipelines, dataset and length statistics
  q2/                      code the notebooks import
    data.py                loading, title + body merge, length statistics
    classic.py             BiGRU pipeline: cleaning, spaCy tokens, vocab, GloVe matrix, loaders
    roberta_data.py        RoBERTa pipeline: tokenisation, truncation, dynamic padding, loaders
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

1. New notebook. In the side panel turn **Internet on** (it needs a phone-verified account).
   The preprocessing notebook runs on CPU. Training will need **GPU T4 x2** or **P100**.
2. **Add Input**: a GloVe dataset that contains `glove.6B.300d.txt` (search "glove 6B 300d").
   `find_glove` picks it up from `/kaggle/input/` on its own. Without it, the 822 MB Stanford zip is downloaded.
3. In the first cell:

   ```python
   !git clone -b <branch> https://github.com/Akshats-git/NLP-Group-Assignment-2.git
   %cd NLP-Group-Assignment-2/Question-2
   !pip install -q -r requirements.txt
   ```

4. Open or import `01_preprocessing.ipynb` and run it. It writes `outputs/results/preprocessing_stats.json`,
   `outputs/figures/length_distribution.png`, and the vocabulary and embedding matrix to `data/classic/`.

## Running locally

From the repo root:

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]" -r Question-2/requirements.txt
pytest -q Question-2/tests                                  # offline, a few seconds
Q2_LIMIT=2000 jupyter nbconvert --to notebook --execute Question-2/01_preprocessing.ipynb --output /tmp/q2_check.ipynb
```

`Q2_LIMIT` runs the notebook on a random subset as a smoke test. It writes no results, figures or vocabulary.
