# Question 4: The Plot-Twist Generator

Vanilla RNN vs LSTM story language models on TinyStories, three decoding strategies, diversity
metrics, and a blinded human evaluation. Owner: Akshat. Issue: [#4](https://github.com/Akshats-git/NLP-Group-Assignment-2/issues/4).
The technical report is [`report/report.tex`](report/report.tex).

## What is here

| Path | What it is |
|---|---|
| `prepare_data.py` | Part 1. Draws the TinyStories subset, tokenises, builds the vocabulary, writes data statistics and a next-token example. |
| `train.py` | Part 2. Trains one model (`--cell rnn` or `--cell lstm`) with next-token cross-entropy and early stopping. |
| `generate_stories.py` | Parts 3 and 4. 10 prompts x 2 models x 6 decoding configurations, unedited output. |
| `evaluate.py` | Part 5, quantitative. Perplexity, Distinct-1/2, repetition rate, length and early-stop statistics. |
| `build_human_eval.py`, `analyze_human_eval.py` | Part 5, human. Builds the blinded rating sheet, then analyses the ratings (agreement, paired comparison). |
| `make_figures.py` | Loss curves and decoding-comparison figures. |
| `llm_baseline.py` | Optional extension: a small open-source LLM, zero-shot, on the same openings. |
| `plot_twist/` | The library the scripts share: `data`, `models`, `generation`, `metrics`, `human_eval`, `prompts`, `config`. |
| `outputs/vocab.json` | The vocabulary. Needed to use the checkpoints. |
| `outputs/checkpoints/` | Trained weights: `rnn_seed42.pt`, `lstm_seed42.pt` (+ seed 43). Shipped, as the brief requires. |
| `outputs/results/` | Metrics, histories, generations, tables. Each run has `run_info.json` with config, seed and environment. |
| `outputs/figures/` | Report figures. |
| `outputs/human_eval/` | The rating form (`rate.html`), blinded samples, collected ratings. |
| `../tests/test_q4.py` | Unit tests (masking, splits, metrics, seeding, agreement statistics). |

## Reproduce everything

Python 3.11+. A GPU helps but is not required (the models are small; training on the GTX 1650 used for the reported
runs takes about 1 hour for both models, on a CPU expect several hours).

```bash
git clone https://github.com/Akshats-git/NLP-Group-Assignment-2.git && cd NLP-Group-Assignment-2
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"                    # shared helpers in common/ (installs torch and matplotlib)
pip install -r Question-4/requirements.txt # datasets, transformers (optional extension)
pytest -q                                  # all green

python Question-4/prepare_data.py          # downloads the TinyStories subset, builds the vocabulary
for seed in 42 43; do                      # two seeds; 42 is the headline run
  python Question-4/train.py --cell rnn  --seed $seed
  python Question-4/train.py --cell lstm --seed $seed
done
python Question-4/evaluate.py              # perplexity, Distinct-n, repetition, lengths
python Question-4/generate_stories.py      # the 10 prompts under every decoding configuration
python Question-4/build_human_eval.py      # blinded rating sheet for the human evaluation
python Question-4/analyze_human_eval.py    # after the ratings are in outputs/human_eval/ratings/
python Question-4/make_figures.py
```

Everything is seeded (`set_seed`, seeded loaders, a sampling seed derived per prompt and decoding configuration), and
`run_info.json` records the seeds. Rerunning with a seed and the same hardware gives the same numbers up to GPU
non-determinism in cuDNN (`deterministic=True` in `set_seed` makes it exact but slower). The checkpoints and
`generations_seed42.jsonl` let you check the shipped results without retraining.

Only the stories needed for the subset are downloaded (the train split is streamed). No dataset files are part of the
submission; `prepare_data.py` rebuilds them from the seed, and `outputs/results/data_summary.json` stores a hash of
each split so you can check you got the same stories.

> **The dataset id.** The brief spells it `roneldan/TinyStories`. The Hugging Face id is `roneneldan/TinyStories`
> (with `ne`); the first form returns "not found". The code uses the correct id.

## Design decisions in one place

- **Data.** 20,000 train / 1,000 validation / 1,000 test stories, as suggested in the brief. Train is a seeded sample
  from the first 3x20,000 filtered stories of the official train split; validation and test are disjoint seeded samples of
  the official validation split, with exact duplicates of training stories removed. Stories over 400 tokens are dropped
  (not truncated), so every training sequence is a complete story with its real ending.
- **Tokenisation.** Lowercase word-level tokens with punctuation as separate tokens; contractions stay whole. One
  vocabulary from the training split only, words seen fewer than 5 times become `<unk>`. Both models use the same tokens.
- **Sequences.** One story is one sequence: `<sos> ... <eos>`. Input is the sequence without its last token, target without
  its first. No window crosses a story boundary, and padding is on the right with `ignore_index=<pad>` in the loss.
- **Models.** `Embedding(256) -> {tanh RNN | LSTM}(512, 1 layer) -> Dropout -> Linear -> logits`. Same class, so only the
  recurrent cell differs. The LSTM has about 1.2M more parameters because it has four gates (see the report).
- **Training.** Adam, lr 2e-3 with ReduceLROnPlateau, gradient clipping at 1.0, batches of 64 stories, early stopping on
  validation loss with patience 3, best epoch restored. fp32 (mixed precision was 5x *slower* on the GTX 1650).
- **Decoding.** `common.generate` with greedy, temperature 0.5 / 0.8 / 1.2, and top-k 10 / 40. `<unk>`, `<pad>`, `<sos>` are
  never generated. Cap: 250 new tokens everywhere.

## Human evaluation (needs the rest of the group)

The brief asks for at least three raters who do not know which model wrote what. `outputs/human_eval/rate.html` is a
self-contained form: open it in any browser, enter a name, rate the stories 1 to 5 on four questions, and download a CSV.
Put each rater's CSV in `outputs/human_eval/ratings/` and run `analyze_human_eval.py`. Raters should not open
`outputs/results/generations*.jsonl`, `generated_stories*.md` or the report before they have finished.
Rating is finished (4 raters, in `ratings/`), so the answer key `key.json` is now committed and `analyze_human_eval.py` can be rerun.
