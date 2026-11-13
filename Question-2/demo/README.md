# Q2 live demo

A Streamlit app that runs the two trained Q2 models (BiGRU + attention and fine-tuned RoBERTa) on any review typed
into it, explains their decisions, and shows the test-set results. It runs fully offline on a laptop.

```
demo/
  app.py               the Streamlit app (4 tabs)
  inference.py         both models on one review, attention + integrated gradients (reuses q2/)
  results.py           metrics, saved test predictions, curves and examples from outputs/results/
  find_checkpoint.py   identifies a downloaded best.pt by its validation predictions, --install copies it in place
  report_figures.py    the Live tab's output for three reviews as figures and numbers for the report (appendix A)
  run_demo.sh          starts the app
```

## Before the presentation

1. Checkpoints: `Question-2/outputs/checkpoints/<run>/best.pt` for the six runs (gitignored, 1.9 GB). They come
   from the Kaggle notebook `04-evaluation-1`. To check or reinstall downloaded files (from `Question-2/`):
   `../.venv/bin/python demo/find_checkpoint.py <folder> --install`
2. Install Streamlit once: `uv pip install --python .venv/bin/python -r Question-2/demo/requirements.txt`
3. Start it about a minute before you need it (from the repo root):

   ```bash
   Question-2/demo/run_demo.sh            # opens http://localhost:8501, light theme for projectors
   ```

   The first load takes about 15 s while the models load; after that a prediction takes well under a second and an
   integrated-gradients explanation about 6 s (cached afterwards). Pick example A once beforehand, so its explanation
   is already cached when you show it.
4. Plug in the charger: the models run on the Mac's GPU (MPS).

## What is in the app

| Tab | What it shows |
|---|---|
| Live demo | Type or pick a review. Both models predict side by side (stars, confidence, probabilities). The BiGRU's attention weights and RoBERTa's integrated gradients are shaded on the words; words the BiGRU does not know are underlined. Pooling ablation (mean vs `<s>`) and the tokens each model reads. |
| Test-set results | Metrics table (mean ± std over two seeds), per-class F1, error distance, confusion matrices, training curves. |
| Browse test predictions | Any test review with all six runs' **saved** predictions, filterable (e.g. "RoBERTa right, BiGRU wrong"); "Try this review live" sends it to the first tab. Nothing is re-predicted. |
| How it works | Architecture diagram, training setup, protocol. |

Sidebar: seed (42 / 43), integrated-gradient steps (256 = the report's setting), and toggles for the explanation
and the ablation.

## A five-minute walkthrough

1. **Headline (top row).** RoBERTa beats the BiGRU on every metric: +3.3 macro-F1, lower MAE, and about half the
   1★↔5★ swaps.
2. **Example A, negated negatives.** Pick *Test example A*. The BiGRU says 1★, RoBERTa 5★ (true 5★). Show the
   highlights: the BiGRU's attention piles onto "do n't", while integrated gradients show RoBERTa reads
   "don't" and "horrible" as evidence *against* 5★ but "gone!!" at the end outweighs them. The numbers match
   Figure 4 of the report (+1.09 for "gone!!", logit change 4.24).
3. **Ask "why not 1★?"** Switch the explained star to 1★ and the colours flip.
4. **Let the evaluator type something.** Good probes are in the drop-down: *Typos and slang* (the BiGRU underlines
   words it has never seen, while RoBERTa's BPE splits them into known pieces), *Sarcasm*, *Twist at the end*.
5. **Be honest about its weak spots.** Examples G–I are 1★ reviews with a concession ("probably very good, but…")
   where RoBERTa rates too high and the BiGRU wins.
6. **Test-set results tab.** Confusion matrices: the gain is on 2★/3★ and on far-off errors, while 4★ vs 5★ stays
   hard for both. Training curves: the BiGRU overfits once its embeddings are unfrozen.
7. **Browse tab.** Filter "RoBERTa right, BiGRU wrong", show a random one, then send it to the live tab.

## Likely questions

- *Are these the same models as in the report?* Yes: the checkpoints were checked against the committed validation
  predictions (99.8–100% agreement), and the live demo reproduces examples A–I and Figure 4 exactly.
- *Doesn't showing test reviews break the protocol?* The browse tab only reads the predictions saved by the single
  test evaluation. The live tab can run on test reviews, but that is a demonstration, not a measurement; no number in
  the report comes from it.
- *Why integrated gradients for RoBERTa but attention for the BiGRU?* The BiGRU's attention weights are part of the
  model and show what it pooled. RoBERTa has 144 attention heads with no single pooled weight, so we attribute the
  logit to the input words instead; the scores add up to the logit change (completeness), which attention does not
  guarantee.
- *Why is the gap at 128 steps larger than at 64?* The midpoint rule does not converge monotonically on every
  review; 256 steps (the report's setting) keeps it small.

## Troubleshooting

- **"The trained checkpoints are missing"**: install them with `find_checkpoint.py` (see above).
- **Port in use**: `Question-2/demo/run_demo.sh --server.port 8502`.
- **Dark theme**: `Question-2/demo/run_demo.sh --theme.base dark`.
- **No internet**: fine. The app sets Hugging Face offline mode and loads `roberta-base` from the local cache.
