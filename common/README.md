# common/

Shared helpers that all four questions import. The aim is that nobody writes the same
seed / padding / early-stopping / plotting boilerplate five times, and that numbers,
files and figures look the same in every section of the combined report.

**This folder is shared.** Changing a function signature here can break someone else's
notebook. Add new things freely, but change existing behaviour only through a PR that
everyone affected has seen, and run `pytest -q` first.

## Getting it into your code

Local (once per venv, from the repo root):

```bash
pip install -e ".[dev]"
pytest -q            # should be all green
```

After that `import common` works from any folder, including notebooks in `Question-N/`.

Colab / Kaggle:

```python
!git clone https://github.com/Akshats-git/NLP-Group-Assignment-2.git
%cd NLP-Group-Assignment-2
import sys; sys.path.insert(0, ".")   # quicker than pip install -e, which needs a runtime restart
import common
```

## What's in it

| Module | What it gives you | Who uses it |
|---|---|---|
| `seed` | `set_seed(seed)`, `DEFAULT_SEEDS = (42, 43)`, `dataloader_kwargs(seed)` for seeded shuffling | everyone |
| `device` | `get_device()`, `autocast(device)`, `grad_scaler(device)` (fp16 vs bf16 picked for the GPU), `to_device(batch)` | everyone |
| `paths` | `data_dir(q)`, `results_dir(q)`, `figures_dir(q)`, `checkpoints_dir(q)` | everyone |
| `files` | `save_json` / `load_json`, `save_jsonl` for per-example predictions, `markdown_table` for the report, `save_run_info` (config, seed, versions, GPU, git commit) | everyone |
| `text` | `normalize` (NFC + whitespace + optional lowercase), `strip_html_and_urls` | Q1, Q2, Q4 |
| `vocab` | `Vocab.build / encode / decode / save / load / oov_rate` with fixed special ids | Q1, Q2 baseline, Q4 |
| `sequences` | `pad_sequences`, `pad_collate`, `padding_mask`, `causal_mask`, `lengths_to_mask`, `teacher_forcing_pair` | Q1, Q2, Q3, Q4 |
| `training` | `History`, `EarlyStopping` (restores best weights), `save_checkpoint` / `load_checkpoint`, `AverageMeter`, `Timer`, `count_parameters` | everyone |
| `metrics` | `masked_accuracy`, `perplexity`, `mean_std` / `format_mean_std` / `summarize_runs` for multi-seed results, length buckets (`bucket_indices`, `bucket_labels`) | Q1, Q2, Q3, Q4 |
| `decoding` | `generate(step_fn, ...)` with `Decoding` configs for greedy, temperature, top-k and top-p | Q1, Q3, Q4 |
| `plotting` | `plot_history`, `plot_attention` + `attention_window`, `plot_grouped_bars`, `plot_confusion_matrix`, shared colours | everyone |

### Deliberately not in here

- **Anything Q3 has to write from scratch.** No attention layers, no Transformer blocks.
  There is a test that fails if `nn.Transformer*`, `nn.MultiheadAttention` or
  `scaled_dot_product_attention` ever appear in `common/`.
- **Metrics only one question needs.** BLEU (Q1), accuracy / F1 / MAE (Q2), ROUGE (Q3),
  Distinct-n and repetition rate (Q4) live in their question's folder.
- **Tokenisers and dataset loaders.** Hindi, Amazon reviews, CNN/DM and TinyStories all
  need different handling, so each question owns its own.
- **A generic `train()` function.** Seq2seq, classification and LM training loops differ
  enough that a shared loop would end up full of flags. Use the building blocks instead.

## Conventions everyone follows

**Special tokens.** `<pad>=0, <unk>=1, <sos>=2, <eos>=3` in every `Vocab`, so
`nn.Embedding(..., padding_idx=0)` and `CrossEntropyLoss(ignore_index=0)` are always right.

**Masks.** `True` means real token / allowed to attend. Apply with
`scores.masked_fill(~mask, torch.finfo(scores.dtype).min)` (not `-inf`, which gives NaN
on fully masked rows and breaks under fp16).

**Metric names** in `History`: `train_loss, val_loss, train_acc, val_acc, val_macro_f1,
val_ppl, lr, time_s`. `plot_history` looks for `train_<metric>` and `val_<metric>`.

**Seeds.** `DEFAULT_SEEDS = (42, 43)`. Call `set_seed` before building the model and
the DataLoaders, and pass `**dataloader_kwargs(seed)` to shuffled loaders.

**Where files go.** Inside each question:

```
Question-N/
  data/                 cached datasets            gitignored, never in the ZIP
  outputs/results/      metrics json, predictions  committed
  outputs/figures/      report figures             committed
  outputs/checkpoints/  weights                    gitignored (Q4's are committed: it must ship them)
```

**Every training run** writes `save_run_info(results_dir(q) / run_name, config, seed)`
at the start and `history.save(...)` at the end. That is what makes the README's
reproduction steps honest.

**Figures.** Use the `plotting` helpers rather than raw matplotlib so colours match
across the report. List the baseline first: it always gets blue, the second model
orange. Train curves are dashed, validation solid.

## Examples

A training loop with the pieces wired together:

```python
from common import (set_seed, get_device, autocast, grad_scaler, to_device, History,
                    EarlyStopping, AverageMeter, save_run_info, results_dir, checkpoints_dir,
                    dataloader_kwargs)

set_seed(seed)
device = get_device()
run_dir = results_dir(1) / f"attention_seed{seed}"
save_run_info(run_dir, config, seed=seed)

loader = DataLoader(train_ds, batch_size=64, shuffle=True,
                    collate_fn=pad_collate(vocab.pad_id), **dataloader_kwargs(seed))
scaler = grad_scaler(device)
history = History()
stopper = EarlyStopping(mode="min", patience=3, checkpoint_path=checkpoints_dir(1) / "attention_best.pt")

for epoch in range(1, config["epochs"] + 1):
    model.train()
    train_loss = AverageMeter()
    for batch in loader:
        batch = to_device(batch, device)
        with autocast(device):
            loss = compute_loss(model, batch)
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        train_loss.update(loss.item(), n=batch["src"].size(0))
    val_loss = evaluate(model, val_loader)
    history.log(epoch=epoch, train_loss=train_loss.avg, val_loss=val_loss)
    stopper.step(val_loss, model)
    if stopper.should_stop:
        break

stopper.restore(model)            # evaluate the best epoch, not the last
history.save(run_dir / "history.json")
```

Greedy translation with attention weights collected for the heatmap (Q1):

```python
from common import generate, Decoding
from common.plotting import plot_attention

model.eval()
enc_out, hidden0 = model.encode(src_ids)                # encode the source once

def step_fn(tokens, state):
    hidden = hidden0 if state is None else state
    logits, hidden, alpha = model.decode_step(tokens[:, -1], hidden, enc_out)
    return logits, hidden, alpha                        # alpha: [batch, src_len]

out = generate(step_fn, torch.full((1, 1), vocab_hi.sos_id), max_new_tokens=50,
               eos_id=vocab_hi.eos_id, pad_id=vocab_hi.pad_id)
weights = torch.stack(out.extras, dim=1)[0]             # [tgt_len, src_len]
hindi = vocab_hi.decode(out.new_tokens[0])
plot_attention(weights[: len(hindi)], english_tokens, hindi, out_path=figures_dir(1) / "attn_01.png")
```

Story sampling with a recorded configuration (Q4):

```python
from common import Decoding, generate, make_generator

configs = [Decoding(), Decoding("sample", temperature=0.5), Decoding("sample", temperature=0.8),
           Decoding("sample", temperature=1.2), Decoding("sample", top_k=40)]
for cfg in configs:
    out = generate(step_fn, prompt_ids, max_new_tokens=150, eos_id=vocab.eos_id,
                   decoding=cfg, generator=make_generator(42, device))
    rows.append({"model": "lstm", "decoding": cfg.to_dict(), "label": cfg.label,
                 "text": " ".join(vocab.decode(out.new_tokens[0]))})
```

Multi-seed reporting (Q2):

```python
from common import DEFAULT_SEEDS, format_mean_std, markdown_table
from common.metrics import summarize_runs

runs = [evaluate_seed(s) for s in DEFAULT_SEEDS]       # each returns {"accuracy": .., "macro_f1": .., "mae": ..}
summary = summarize_runs(runs)
print(markdown_table([{"model": "RoBERTa", **{k: format_mean_std(v["values"], scale=100) for k, v in summary.items()}}]))
```

## Hindi in plots

Two things are needed for Hindi tick labels to come out right:

1. **A Devanagari font.** `plotting` finds an installed one (Noto Sans Devanagari, Lohit,
   Nirmala UI, Mangal) and uses it as a fallback. Without one, Hindi shows as boxes.
2. **matplotlib 3.11 or newer.** Older versions draw the letters but do not shape them, so
   `हिंदी` comes out with the ि after the consonant and conjuncts like `क्ष` fall apart.
   We checked this side by side: 3.10 is wrong, 3.11 matches proper text shaping.

On Colab, before importing matplotlib:

```python
!apt-get -qq install fonts-noto-core
!pip install -q -U "matplotlib>=3.11"
# then Runtime > Restart session
```

`plot_attention` warns if either one is missing.
