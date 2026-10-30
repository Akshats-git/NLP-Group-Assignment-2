# Question 3: Transformer summariser written from scratch

Abstractive summarisation of CNN / Daily Mail 3.0.0 news articles with a Transformer encoder-decoder
("Attention Is All You Need") whose attention, feed-forward, positional-encoding, encoder-layer and
decoder-layer code is written by hand, plus a copy (pointer) switch so the model can take names and numbers
straight from the article. Evaluated with ROUGE-1/2/L/Lsum on the full test set (11,490 pairs),
with qualitative examples and encoder-decoder attention heatmaps.

## Files

| File | What it contains |
|---|---|
| `Question_3.ipynb` | Runs everything in order: data, checks, training, evaluation, examples, attention, report |
| `q3_model.py` | The model from scratch: positional encoding, multi-head attention, feed-forward, encoder and decoder layers, masks, copy switch, greedy decoding |
| `q3_data.py` | Loading CNN/DM, lowercasing, tokenisation (`bert-base-uncased` + `[NL]`), truncation, batching, ids back to text |
| `q3_train.py` | Loss that ignores padding (with label smoothing and copying), the paper's learning-rate schedule, validation loss, right-vs-wrong-article check, batched generation, saving and resuming the training state |
| `q3_eval.py` | ROUGE with bootstrap intervals, Lead-3, checks for repetition / unsupported numbers / novelty, example picking |
| `q3_report.py` | Writes `REPORT.md` from the numbers of the run |
| `requirements.txt` | Libraries |
| `../tests/test_q3.py` | Offline tests: masks, padding, loss, attention formula, decoding, no banned layers |

After a run: `REPORT.md` (the Q3 report) and `outputs/results/` + `outputs/figures/` (see the end of this file).

## Setup used

| | |
|---|---|
| Tokenizer | `bert-base-uncased` WordPiece (pre-trained tokenizer only), lowercased text, plus `[NL]` for sentence breaks; shared vocabulary of 30,523 |
| Start / end / pad | `[CLS]` / `[SEP]` / `[PAD]` (id 0) |
| Truncation | article 400 tokens, summary 128 tokens (incl. start/end) |
| Model | d_model 512, 8 heads, 4 encoder + 4 decoder layers, d_ff 2048, dropout 0.1, post-norm, tied embeddings; 45.05M parameters |
| Copy switch | `P(w) = p_gen · P_vocab(w) + (1 − p_gen) · attention on article positions holding w` (See et al., 2017), using the last decoder layer's cross-attention; `copy_mechanism=False` gives the plain Transformer |
| Training | all 287,113 training pairs; cross-entropy ignoring padding, label smoothing 0.1; Adam (0.9, 0.98, 1e-9) with warm-up 4,000 then 1/sqrt(step) decay; clipping 1.0; fp16 |
| Early stopping | validation loss (all 13,368 pairs) every 3,000 steps, patience 3, best checkpoint restored; 3.25 h time budget |
| Decoding | greedy from `[CLS]` until `[SEP]` or 128 tokens, article encoded once; greedy + trigram blocking also evaluated, choice made on validation |
| ROUGE | `rouge_score`, `use_stemmer=True`, F1, full test set; ROUGE-L and ROUGE-Lsum reported separately |

## Run it on Kaggle (about 4 to 4.5 h)

1. **Get the code onto Kaggle**, one of:
   - *Upload it as a dataset (no git push needed):* Kaggle → Datasets → New dataset → upload
     `Question-3-kaggle-code.zip` (it holds `common/` and the Question-3 code). Or zip the repository folder
     yourself without `data/`, `outputs/` and `.git/`.
   - *Or push to GitHub:* push the Question-3 code; if it is not on `main`, set `REPO_BRANCH` in the first
     code cell. The notebook clones the repo when no uploaded copy is found.
2. Kaggle → Code → New notebook → File → **Import notebook** → `Question_3.ipynb`.
   If you uploaded the dataset: **Add Input** → your dataset.
3. Settings: **Accelerator: GPU T4 x2**, **Internet: on** (needs a phone-verified account).
   A single P100 also works (one GPU, batch 32, fewer steps in the same time).
4. **Save Version → Save & Run All (Commit)**. Use this, not the interactive editor: a commit runs on Kaggle's
   side, so closing the tab, losing Wi-Fi or the laptop sleeping does not stop it. An interactive session stops
   when it is idle or the browser disconnects, and its files are gone. The cell after the first 200 training
   steps prints the speed and how many epochs fit in the budget.
5. When it finishes, open the version → **Output** and download `q3_outputs.zip`, plus the executed notebook.
6. Unzip `q3_outputs.zip` into `Question-3/`. It holds `REPORT.md` and `outputs/` (results and figures), with
   no checkpoints and no data. Replace `Question_3.ipynb` with the executed notebook so the outputs are visible.

To train longer, raise `train_hours` in the `CFG` cell (everything else adapts); Kaggle stops a session
after 12 h, and evaluation takes about 30 to 45 min after training.

### If a run is interrupted

After every validation check (about every 20 minutes) the notebook saves the full training state to
`outputs/checkpoints/last.pt`. Running the notebook again continues from there: it prints
`RESUMING: … steps and … h of training already done`, keeps the history and the time already used, and goes
straight to evaluation if training had finished. At most one check interval of training is lost.

- **Interrupted commit (version shows an error or timeout):** open the notebook in the editor →
  **Add Input → Notebook Output** → choose that failed version. The notebook finds
  `…/Question-3/outputs/checkpoints/last.pt` there and resumes. Then *Save & Run All* again.
- **Interactive session that stopped:** if the session's files are still there (Kaggle's *Persistence: Files*
  setting keeps `/kaggle/working` between sessions), just run all cells again. If the files are gone, nothing
  was saved, so start again as a commit (step 4).
- **To start from scratch** (for example after changing the model size), set `resume=False` in `CFG` or delete
  `outputs/checkpoints/last.pt`.

## Reading the training log

Every validation check prints three lines:

```
step  6,000 | epoch 1.34 | train loss 3.912 | val loss 3.750 | val R-1/2/Lsum 31.2/11.0/28.4 | 26 min  <- best
    reads the article? loss with right article 3.70 vs wrong article 6.10 (gap 2.40), p_gen 0.38
    generated: <greedy summary of the first monitored validation article>
```

(The numbers above only show the format; they are not results.)

- **gap** = how much worse the reference summaries score against a *different* article. A large and growing
  gap means the model reads the article. If it stays near 0, the decoder is only writing generic text.
- **p_gen** = how much of each word comes from generating rather than copying (0 = always copy).
- Early on (the first checks, around the end of the 4,000-step warm-up) a summariser trained from scratch writes
  fluent but generic sentences, for example the same football sentence for every article. That is the normal
  first phase, not a bug. With the copy switch it should pass quickly; watch the gap and ROUGE-2 rise.

## Run it locally

```bash
pip install -e ".[dev]"                         # from the repo root: makes `common` importable
pip install -r Question-3/requirements.txt
pytest -q                                       # common/ and Question-3 tests, no download needed
jupyter notebook Question-3/Question_3.ipynb    # set SMOKE = True for a quick CPU run
```

`SMOKE = True` uses a tiny model and small subsets and writes to `outputs_smoke/` (gitignored); it only
checks that every cell works. Real results need a GPU and `SMOKE = False`.

## Outputs and the brief's deliverables

| Deliverable | Where |
|---|---|
| Source code, README, libraries | this folder, `requirements.txt` |
| Report with tokenizer, truncation lengths, model configuration, training-set size, decoding method, ROUGE-1/2/L/Lsum | `REPORT.md` |
| Data statistics and settings | `outputs/results/data_info.json` |
| Model configuration, parameter count, seed, environment | `outputs/results/run_info.json` |
| Checks for the pitfalls | `outputs/results/sanity_checks.json` |
| Training / validation loss curves, ROUGE during training | `outputs/figures/loss_curves.png`, `val_rouge.png`, `outputs/results/history.json` |
| Test ROUGE (with Lead-3) | `outputs/results/rouge.json`, `outputs/figures/rouge_comparison.png` |
| Generated test summaries | `outputs/results/test_predictions.jsonl` (ids, summaries and scores; no dataset text) |
| Decoding choice on validation | `outputs/results/decoding_choice_validation.json` |
| At least five qualitative examples incl. failures | `outputs/results/qualitative_examples.json`, `REPORT.md` section 6 |
| Cross-attention heatmaps (last decoder layer, generation time, head mean and per head) | `outputs/figures/cross_attention_*.png`, `attention_by_position.png` |
| Checklist | `outputs/results/deliverables.json` |

Not submitted: the dataset (downloaded at run time) and checkpoints (`outputs/checkpoints/`, gitignored).

## How the brief's pitfalls are avoided

1. **No built-in Transformer parts**: all layers in `q3_model.py`; a check scans the code and the model.
2. **Causal mask**: lower-triangular mask combined with the target padding mask; tested by changing future tokens.
3. **Padding mask**: source padding masked in encoder self-attention and cross-attention; tested.
4. **Padding in the loss**: only non-padding target tokens count; tested against `F.cross_entropy(ignore_index=0)`.
5. **ROUGE on generated text only**: summaries come from free-running greedy decoding, never teacher forcing.
6. **ROUGE-L vs ROUGE-Lsum**: both reported and labelled; sentence breaks come from `[NL]`.
7. **No tuning on test, best checkpoint**: early stopping and the decoding choice use validation; the best
   checkpoint is restored before testing; the test set is used once.
