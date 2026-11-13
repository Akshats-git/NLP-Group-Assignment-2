"""Figures for the Q1 report: training curves, attention heatmaps, length analysis.

Run after train.py and evaluate.py (analysis.py makes the error-case heatmaps):

    python visualize.py
"""

import argparse
import random
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common import (
    History, load_json, load_jsonl, save_json, markdown_table,
    results_dir, figures_dir, bucket_indices, bucket_labels,
)
from common.plotting import plot_history, plot_attention, plot_grouped_bars
from evaluate import attn_path


def plot_training_curves(seed: int):
    """Train/val loss and accuracy for both models."""
    histories = {}
    for name in ("baseline", "attention"):
        path = results_dir(1) / f"{name}_seed{seed}" / "history.json"
        if path.exists():
            histories[name.title()] = History.load(path)
            print(f"  loaded {name} history ({len(histories[name.title()])} epochs)")
    if not histories:
        print("  no training histories found")
        return
    fig_dir = figures_dir(1)
    plot_history(histories, metrics=("loss", "acc"), out_path=fig_dir / "training_curves.png",
                 title="Training and validation curves")
    plot_history(histories, metrics=("loss",), out_path=fig_dir / "loss_curves.png", title="Loss curves")
    plot_history(histories, metrics=("acc",), out_path=fig_dir / "accuracy_curves.png",
                 title="Token accuracy curves")


def generate_heatmaps(seed: int, n_examples: int = 12):
    """Heatmaps of the attention model on test sentences, spread over the four length buckets.

    Within a bucket, sentences are drawn at random but only from those with a sentence
    chrF above 25, so the first figures show attention on reasonable translations; the
    failures are plotted by analysis.py.
    """
    res = results_dir(1)
    att = load_jsonl(res / f"attention_seed{seed}" / "test_predictions.jsonl")
    weights = torch.load(attn_path(seed), weights_only=True)
    import sacrebleu

    rng = random.Random(seed)
    buckets = bucket_indices([r["src_len"] for r in att])
    per_bucket = max(1, n_examples // 4)
    chosen = []
    for label in bucket_labels():
        good = [i for i in buckets[label]
                if len(att[i]["hyp_tokens"]) >= 2 and att[i]["src_len"] >= 3
                and sacrebleu.sentence_chrf(att[i]["hyp_text"], [att[i]["ref_text"]]).score > 25]
        chosen += rng.sample(good, min(per_bucket, len(good)))

    meta = []
    for k, i in enumerate(chosen, 1):
        r, w = att[i], weights[i]
        src_tokens = ["<sos>"] + r["src_text"].split() + ["<eos>"]
        n = len(r["hyp_tokens"])
        name = f"attention_heatmap_{k:02d}.png"
        plot_attention(w[:n], src_tokens, r["hyp_tokens"], out_path=figures_dir(1) / name,
                       title=f"Test #{i} ({r['src_len']} English tokens)")
        meta.append({"figure": name, "test_idx": i, "src_text": r["src_text"],
                     "ref_text": r["ref_text"], "hyp_text": r["hyp_text"]})
        print(f"  saved {name}")
    save_json(meta, res / "heatmap_examples.json")


def plot_length_analysis(seed: int):
    """BLEU per source-length bucket for both models."""
    data = {}
    for name in ("baseline", "attention"):
        path = results_dir(1) / f"{name}_seed{seed}" / "bucket_bleu.json"
        if path.exists():
            data[name.title()] = load_json(path)
    if not data:
        print("  no bucket BLEU found")
        return
    plot_grouped_bars(data, ylabel="BLEU", out_path=figures_dir(1) / "length_analysis.png",
                      title="BLEU by source sentence length", xlabel="English tokens per sentence")
    print("  saved length_analysis.png")


def comparison_tables(seed: int):
    """Markdown tables for the report: overall metrics and per-bucket scores."""
    res = results_dir(1)
    runs = {n: res / f"{n}_seed{seed}" for n in ("baseline", "attention")}
    if not all((d / "test_bleu.json").exists() for d in runs.values()):
        return
    overall, bucket_rows = [], []
    stats = {n: load_json(d / "bucket_stats.json") for n, d in runs.items()}
    for n, d in runs.items():
        m, h = load_json(d / "test_bleu.json"), History.load(d / "history.json")
        best = h.best("val_loss")
        overall.append({"model": n, "BLEU": f"{m['bleu']:.2f}", "chrF": f"{m['chrf']:.1f}",
                        "test token acc": f"{m['test_token_acc']:.3f}", "test loss": f"{m['test_loss']:.3f}",
                        "best epoch": best["epoch"], "val loss": f"{best['val_loss']:.3f}",
                        "val acc": f"{best['val_acc']:.3f}"})
    for label in bucket_labels():
        row = {"source length": label, "sentences": stats["baseline"][label].get("count", 0)}
        for n in runs:
            s = stats[n][label]
            row[f"{n} BLEU"] = f"{s.get('bleu', 0):.2f}"
            row[f"{n} chrF"] = f"{s.get('chrf', 0):.1f}"
        bucket_rows.append(row)
    text = markdown_table(overall) + "\n\n" + markdown_table(bucket_rows) + "\n"
    (res / "comparison_tables.md").write_text(text, encoding="utf-8")
    print(text)


def main():
    p = argparse.ArgumentParser(description="Generate Q1 NMT figures")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n-heatmaps", type=int, default=12)
    args = p.parse_args()

    print("1. Training curves");  plot_training_curves(args.seed)
    print("2. Attention heatmaps"); generate_heatmaps(args.seed, args.n_heatmaps)
    print("3. Length analysis");  plot_length_analysis(args.seed)
    print("4. Comparison tables"); comparison_tables(args.seed)
    print(f"\nFigures in {figures_dir(1)}")


if __name__ == "__main__":
    main()
