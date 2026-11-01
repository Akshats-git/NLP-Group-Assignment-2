"""Part 5 (quantitative): perplexity, Distinct-1/2, repetition rate, length statistics.

    python Question-4/evaluate.py

For each trained model and each decoding configuration, every held-out test story is cut after its
first N tokens (GENERATION["test_prefix_tokens"]) and the model writes the rest. The same prefixes and the
same sampling seeds are used for both models. The real continuations of those test stories are scored
with the same metrics as a human-written reference row.

Reads the checkpoints in outputs/checkpoints and the run metrics in outputs/results/<run>/metrics.json.
Writes outputs/results/quantitative.json and outputs/results/quantitative_tables.md
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import torch

from common import Vocab, format_mean_std, get_device, load_json, load_jsonl, markdown_table, save_json
from common.metrics import mean_std
from plot_twist import config as cfg
from plot_twist.generation import DECODING_CONFIGS, continue_story, derive_seed
from plot_twist.data import tokenize
from plot_twist.metrics import summarize_group
from plot_twist.models import load_model

BATCH = 250


def test_prefixes(vocab: Vocab, stories: list[str], n_prefix: int):
    """Tokenised test stories cut after n_prefix tokens. Stories shorter than that are skipped."""
    prefixes, reals = [], []
    for story in stories:
        toks = tokenize(story)
        if len(toks) > n_prefix + 1:
            prefixes.append(vocab.encode(toks[:n_prefix], add_sos=True))
            reals.append(toks[n_prefix:])
    return torch.tensor(prefixes), reals


def run_group(model, vocab, prefixes, decoding, device, max_new_tokens, label_key):
    stories, finished = [], []
    for start in range(0, prefixes.size(0), BATCH):
        seed = derive_seed(cfg.GENERATION["seed"], "test_prefix", label_key, str(start))
        for row in continue_story(model, vocab, prefixes[start:start + BATCH], decoding, seed, max_new_tokens, device):
            stories.append(row["tokens"])
            finished.append(row["finished"])
    return stories, finished


def main():
    p, g = cfg.paths(), cfg.GENERATION
    g_cfg = g
    device = get_device()
    vocab = Vocab.load(p["results"].parent / "vocab.json")
    test_stories = [r["text"] for r in load_jsonl(p["data"] / "test.jsonl")]
    prefixes, real_tail = test_prefixes(vocab, test_stories, g["test_prefix_tokens"])
    print(f"{len(prefixes)} test prefixes of {g['test_prefix_tokens']} tokens")

    # 1. Perplexity and training cost, from the training runs.
    runs = {}
    for cell in cfg.CELLS:
        for seed in cfg.TRAIN["seeds"]:
            f = p["results"] / f"{cell}_seed{seed}" / "metrics.json"
            if f.exists():
                runs.setdefault(cell, []).append(load_json(f))

    ppl_rows = []
    for cell, ms in runs.items():
        head = next(m for m in ms if m["seed"] == 42)
        ppl_rows.append({
            "model": cfg.MODEL_NAMES[cell], "params": f"{head['parameters']['total']:,}",
            "val ppl (seed 42)": head["val_ppl"], "test ppl (seed 42)": head["test_ppl"],
            "val ppl (mean ± std)": format_mean_std([m["val_ppl"] for m in ms]),
            "test ppl (mean ± std)": format_mean_std([m["test_ppl"] for m in ms]),
            "seeds": ", ".join(str(m["seed"]) for m in ms), "best epoch": head["best_epoch"],
            "train time (min)": head["train_time_s"] / 60,
        })

    # 2. Generation metrics per (model, seed, decoding).
    results = {"perplexity": runs, "generation": {}, "settings": {"test_prefix_tokens": g["test_prefix_tokens"],
              "max_new_tokens": g["max_new_tokens"], "short_below": g["short_below"], "n_prefixes": len(prefixes)}}
    # Human stories judged by the same rules: "did not finish" = longer than the cap the models are given.
    reference = summarize_group(real_tail, [len(t) < g["max_new_tokens"] for t in real_tail], g["max_new_tokens"], g["short_below"])
    results["generation"]["human_reference"] = reference

    for cell in cfg.CELLS:
        for seed in cfg.TRAIN["seeds"]:
            ckpt = p["checkpoints"] / f"{cell}_seed{seed}.pt"
            if not ckpt.exists():
                continue
            model = load_model(ckpt, device)[0]
            for decoding in DECODING_CONFIGS:
                stories, finished = run_group(model, vocab, prefixes, decoding, device, g["max_new_tokens"], decoding.label)
                results["generation"][f"{cell}|seed{seed}|{decoding.label}"] = summarize_group(
                    stories, finished, g["max_new_tokens"], g["short_below"])
                print(cell, seed, decoding.label, "done")

    # 2b. The ten hand-written openings: the same metrics for the RNN, the LSTM and (if it was run) the optional LLM.
    ten = []
    gens = p["results"] / "generations_seed42.jsonl"
    llm_file = p["results"] / "llm_generations.jsonl"
    if gens.exists() and llm_file.exists():
        own, llm = load_jsonl(gens), load_jsonl(llm_file)
        for label in ("greedy", "temp=0.8", "top-k=40"):
            for name, rows in (("Vanilla RNN", [r for r in own if r["model"] == "rnn"]), ("LSTM", [r for r in own if r["model"] == "lstm"]),
                               ("SmolLM2-360M-Instruct (zero-shot)", llm)):
                g = [r for r in rows if r["decoding_label"] == label]
                tokens = [r["tokens"] if "tokens" in r else tokenize(r["text"]) for r in g]
                ten.append({"model": name, "decoding": label,
                            **{k: v for k, v in summarize_group(tokens, [r["finished"] for r in g], g_cfg["max_new_tokens"], g_cfg["short_below"]).items()}})
        results["ten_openings"] = ten

    save_json(results, p["results"] / "quantitative.json")

    # 3. Markdown tables for the report.
    md = ["## Perplexity and training cost", "", markdown_table(ppl_rows, floatfmt=".2f"), ""]
    for seed in cfg.TRAIN["seeds"]:
        rows = [{"model": "human (test stories)", "decoding": "-", **{k: reference[k] for k in COLS}}]
        for cell in cfg.CELLS:
            for decoding in DECODING_CONFIGS:
                r = results["generation"].get(f"{cell}|seed{seed}|{decoding.label}")
                if r:
                    rows.append({"model": cfg.MODEL_NAMES[cell], "decoding": decoding.label, **{k: r[k] for k in COLS}})
        if len(rows) > 1:
            md += [f"## Generation metrics, models trained with seed {seed}", "", markdown_table(rows, floatfmt=".3f"), ""]
    # Mean and spread over the two training seeds, headline table.
    rows = []
    for cell in cfg.CELLS:
        for decoding in DECODING_CONFIGS:
            per_seed = [results["generation"][k] for k in (f"{cell}|seed{s}|{decoding.label}" for s in cfg.TRAIN["seeds"])
                        if k in results["generation"]]
            if len(per_seed) > 1:
                rows.append({"model": cfg.MODEL_NAMES[cell], "decoding": decoding.label,
                             **{k: format_mean_std([r[k] for r in per_seed], digits=3) for k in COLS}})
    if rows:
        md += ["## Generation metrics, mean ± std over training seeds", "", markdown_table(rows), ""]
    if ten:
        md += ["## The ten openings: RNN vs LSTM vs optional LLM (all samples; the LLM's cap counts sub-word tokens)", "",
               markdown_table([{k: r[k] for k in ("model", "decoding", "n_stories", *COLS)} for r in ten], floatfmt=".3f"), ""]
    (p["results"] / "quantitative_tables.md").write_text("\n".join(md), encoding="utf-8")
    print("\n".join(md))


COLS = ["distinct_1", "distinct_2", "distinct_1_story", "distinct_2_story", "repetition_rate", "degenerate_share", "avg_length", "too_short_share", "too_long_share"]

if __name__ == "__main__":
    main()
