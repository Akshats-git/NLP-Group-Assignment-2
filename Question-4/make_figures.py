"""Report figures: training curves and generation metrics across decoding configurations.

    python Question-4/make_figures.py

Reads outputs/results/<run>/history.json and outputs/results/quantitative.json, writes PNGs to outputs/figures/.
The baseline (Vanilla RNN) is always listed first, so it is blue and the LSTM orange, as in the other questions.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from common import History, load_json
from common.plotting import plot_grouped_bars, plot_history
from plot_twist import config as cfg
from plot_twist.generation import DECODING_CONFIGS

METRIC_PANELS = {
    "repetition_rate": ("repetition rate (repeated trigrams)", "{:.2f}"),
    "distinct_2": ("Distinct-2 (pooled over all stories)", "{:.2f}"),
    "distinct_2_story": ("Distinct-2 (within each story)", "{:.2f}"),
    "too_long_share": ("share of stories with no ending", "{:.2f}"),
    "avg_length": ("average generated length (tokens)", "{:.0f}"),
}


def main():
    p = cfg.paths()
    names = cfg.MODEL_NAMES

    for seed in cfg.TRAIN["seeds"]:
        hist = {names[c]: History.load(p["results"] / f"{c}_seed{seed}" / "history.json")
                for c in cfg.CELLS if (p["results"] / f"{c}_seed{seed}" / "history.json").exists()}
        if hist:
            plot_history(hist, metrics=("loss", "ppl"), out_path=p["figures"] / f"training_curves_seed{seed}.png",
                         title=f"Next-token training, seed {seed}")

    q = p["results"] / "quantitative.json"
    if not q.exists():
        return
    gen = load_json(q)["generation"]
    labels = [d.label for d in DECODING_CONFIGS]
    for seed in cfg.TRAIN["seeds"]:
        for metric, (ylabel, fmt) in METRIC_PANELS.items():
            data = {names[c]: {lab: gen[f"{c}|seed{seed}|{lab}"][metric] for lab in labels}
                    for c in cfg.CELLS if f"{c}|seed{seed}|{labels[0]}" in gen}
            if data:
                ref = gen["human_reference"][metric]
                plot_grouped_bars(data, ylabel=ylabel, value_format=fmt, xlabel=f"decoding (human stories: {ref:{fmt[2:-1]}})",
                                  out_path=p["figures"] / f"decoding_{metric}_seed{seed}.png")
    print("figures written to", p["figures"])


if __name__ == "__main__":
    main()
