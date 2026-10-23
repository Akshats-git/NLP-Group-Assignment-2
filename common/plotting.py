"""Report figures with one shared look across all four questions.

    from common.plotting import plot_history, plot_attention, plot_grouped_bars, plot_confusion_matrix

Rules the helpers follow, so figures from different people match:
- Colours follow the model, in a fixed order: the first model passed gets SERIES[0],
  the second SERIES[1], and so on. List the baseline first in every figure.
- Train curves are dashed and validation curves solid, in the model's colour.
- Different metrics go in separate panels, never on a second y-axis.
- Heatmaps (attention, confusion matrices) use one blue scale, light = low.
- Each series also gets its own marker, so figures survive greyscale printing.

Hindi tick labels need a Devanagari font and matplotlib >= 3.11 (older versions put
vowel signs and conjuncts in the wrong place). set_style() finds the font if it is
installed. On Colab:
    !apt-get -qq install fonts-noto-core && pip install -q -U "matplotlib>=3.11"
then restart the runtime.
"""

import warnings
from collections.abc import Mapping, Sequence
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D

# Categorical slots, in this order. The order is chosen so neighbouring colours stay
# distinguishable for colour-blind readers; do not reshuffle or cycle past eight.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
MARKERS = ["o", "s", "^", "D", "v", "P", "X", "*"]

INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
SURFACE = "#ffffff"

BLUE_RAMP = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
SEQUENTIAL = LinearSegmentedColormap.from_list("nlp_blue", BLUE_RAMP)

_DEVANAGARI_CANDIDATES = ("Noto Sans Devanagari", "Lohit Devanagari", "Nirmala UI", "Mangal", "Noto Serif Devanagari")
_style_applied = False


def find_devanagari_font() -> str | None:
    installed = {f.name for f in font_manager.fontManager.ttflist}
    for name in _DEVANAGARI_CANDIDATES:
        if name in installed:
            return name
    return next((name for name in sorted(installed) if "Devanagari" in name), None)


def set_style() -> None:
    """Apply the shared matplotlib style. The plot helpers call this themselves."""
    global _style_applied
    families = ["DejaVu Sans"]
    devanagari = find_devanagari_font()
    if devanagari:
        families.append(devanagari)  # matplotlib falls back glyph by glyph through this list
    plt.rcParams.update({
        "font.family": families,
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.titleweight": "bold",
        "axes.labelsize": 10,
        "axes.edgecolor": AXIS,
        "axes.labelcolor": INK_SECONDARY,
        "axes.titlecolor": INK,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.color": GRID,
        "grid.linewidth": 0.6,
        "xtick.color": INK_MUTED,
        "ytick.color": INK_MUTED,
        "xtick.labelcolor": INK_SECONDARY,
        "ytick.labelcolor": INK_SECONDARY,
        "legend.frameon": False,
        "legend.fontsize": 9,
        "lines.linewidth": 2,
        "lines.markersize": 5,
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "figure.dpi": 100,
    })
    _style_applied = True


def _label_color(rgba) -> str:
    """White or near-black, whichever contrasts more with the cell colour behind the text."""
    def channel(c):
        return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4
    r, g, b = (channel(c) for c in rgba[:3])
    luminance = 0.2126 * r + 0.7152 * g + 0.0722 * b
    return SURFACE if (1.05 / (luminance + 0.05)) > ((luminance + 0.05) / 0.05) else INK


def _ensure_style() -> None:
    if not _style_applied:
        set_style()


def _warn_if_no_devanagari(labels: Sequence[str]) -> None:
    if not any("ऀ" <= ch <= "ॿ" for label in labels for ch in str(label)):
        return
    if find_devanagari_font() is None:
        warnings.warn(
            "Hindi labels but no Devanagari font installed; they will show as boxes. "
            "Install Noto Sans Devanagari (Colab: !apt-get -qq install fonts-noto-core) and restart.",
            stacklevel=3,
        )
    if tuple(int(p) for p in matplotlib.__version__.split(".")[:2]) < (3, 11):
        warnings.warn(
            f"matplotlib {matplotlib.__version__} cannot shape Devanagari: vowel signs and conjuncts "
            "will be drawn in the wrong place. pip install -U 'matplotlib>=3.11' and restart.",
            stacklevel=3,
        )


def savefig(fig, path: str | Path, dpi: int = 200) -> Path:
    """Save to path (.png for the markdown report, .pdf for LaTeX) and close the figure."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return path


def _finish(fig, out_path):
    if out_path is not None:
        savefig(fig, out_path)
    return fig


def _records(history) -> list[dict]:
    if hasattr(history, "records"):
        return history.records
    if isinstance(history, Mapping):  # {"train_loss": [...], "val_loss": [...]}
        length = max(len(v) for v in history.values())
        return [{k: (v[i] if i < len(v) else None) for k, v in history.items()} for i in range(length)]
    return list(history)


_METRIC_NAMES = {"loss": "Loss", "acc": "Accuracy", "macro_f1": "Macro F1", "ppl": "Perplexity", "bleu": "BLEU"}


def plot_history(
    histories,
    metrics: Sequence[str] = ("loss",),
    out_path: str | Path | None = None,
    title: str | None = None,
    x: str = "epoch",
):
    """Training curves, one panel per metric.

    histories is a History (or list of epoch dicts), or {"Model A": history_a, "Model B": history_b}.
    For metric "loss" it draws train_loss (dashed) and val_loss (solid) when present.

        plot_history({"Baseline": hist_a, "Attention": hist_b}, metrics=("loss", "acc"),
                     out_path=figures_dir(1) / "curves.png")
    """
    _ensure_style()
    if not isinstance(histories, Mapping) or any(k.startswith(("train_", "val_")) for k in histories):
        histories = {"": histories}
    if len(histories) > len(SERIES):
        raise ValueError(f"at most {len(SERIES)} models per figure")

    fig, axes = plt.subplots(1, len(metrics), figsize=(4.2 * len(metrics), 3.2), squeeze=False)
    splits_seen = set()
    for ax, metric in zip(axes[0], metrics):
        for i, (name, history) in enumerate(histories.items()):
            records = _records(history)
            xs = [r.get(x, j + 1) for j, r in enumerate(records)]
            for split, style in (("train", "--"), ("val", "-")):
                ys = [r.get(f"{split}_{metric}") for r in records]
                points = [(a, b) for a, b in zip(xs, ys) if b is not None]
                if not points:
                    continue
                splits_seen.add(split)
                px, py = zip(*points)
                ax.plot(px, py, linestyle=style, color=SERIES[i], marker=MARKERS[i],
                        markevery=max(1, len(px) // 10), label=f"{name} {split}".strip())
        ax.set_title(_METRIC_NAMES.get(metric, metric))
        ax.set_xlabel(x)

    handles = []
    if len(histories) > 1:
        handles += [Line2D([], [], color=SERIES[i], marker=MARKERS[i], label=name) for i, name in enumerate(histories)]
    color = SERIES[0] if len(histories) == 1 else INK_SECONDARY
    handles += [Line2D([], [], color=color, linestyle=s, label=lbl)
                for s, lbl, key in (("--", "train", "train"), ("-", "validation", "val")) if key in splits_seen]
    if len(handles) > 1:
        fig.legend(handles=handles, loc="lower center", ncol=len(handles), bbox_to_anchor=(0.5, -0.02))
    if title:
        fig.suptitle(title, fontweight="bold", color=INK)
    fig.tight_layout(rect=(0, 0.07 if len(handles) > 1 else 0, 1, 1))
    return _finish(fig, out_path)


def plot_attention(
    weights,
    src_tokens: Sequence[str],
    tgt_tokens: Sequence[str],
    out_path: str | Path | None = None,
    title: str | None = None,
    annotate: bool = False,
    ax=None,
):
    """Attention heatmap: rows = generated tokens, columns = source tokens.

    weights is [len(tgt_tokens), len(src_tokens)] (tensor or array); each row is one
    decoding step's attention over the source. Used by Q1 and Q3 (with attention_window).
    """
    _ensure_style()
    weights = np.asarray(weights.detach().cpu().float() if hasattr(weights, "detach") else weights, dtype=float)
    if weights.shape != (len(tgt_tokens), len(src_tokens)):
        raise ValueError(f"weights are {weights.shape} but there are {len(tgt_tokens)} target "
                         f"and {len(src_tokens)} source tokens")
    _warn_if_no_devanagari(list(src_tokens) + list(tgt_tokens))

    created = ax is None
    if created:
        width = min(16, max(4.5, 0.38 * len(src_tokens) + 2))
        height = min(14, max(3.0, 0.34 * len(tgt_tokens) + 1.6))
        fig, ax = plt.subplots(figsize=(width, height))
    else:
        fig = ax.figure
    image = ax.imshow(weights, cmap=SEQUENTIAL, aspect="auto", vmin=0, vmax=max(weights.max(), 1e-9))
    ax.set_xticks(range(len(src_tokens)), labels=src_tokens, rotation=60, ha="right", rotation_mode="anchor")
    ax.set_yticks(range(len(tgt_tokens)), labels=tgt_tokens)
    ax.tick_params(length=0)
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xlabel("source")
    ax.set_ylabel("generated")
    if annotate:
        for i in range(weights.shape[0]):
            for j in range(weights.shape[1]):
                ax.text(j, i, f"{weights[i, j]:.2f}", ha="center", va="center", fontsize=7,
                        color=_label_color(image.cmap(image.norm(weights[i, j]))))
    if title:
        ax.set_title(title)
    if created:
        fig.colorbar(image, ax=ax, fraction=0.035, pad=0.02).set_label("attention weight", color=INK_SECONDARY)
        fig.tight_layout()
    return _finish(fig, out_path)


def attention_window(weights, src_tokens: Sequence[str], width: int = 60):
    """Cut a long source down to the `width` consecutive tokens that get the most attention.

    A 512-token article is unreadable as a heatmap (Q3 asks for a 50-100 token window).
    Returns (weights_window, tokens_window, start) to pass to plot_attention.
    """
    weights = np.asarray(weights.detach().cpu().float() if hasattr(weights, "detach") else weights, dtype=float)
    n = weights.shape[1]
    if n <= width:
        return weights, list(src_tokens), 0
    column_mass = weights.sum(axis=0)
    window_mass = np.convolve(column_mass, np.ones(width), mode="valid")
    start = int(window_mass.argmax())
    return weights[:, start:start + width], list(src_tokens[start:start + width]), start


def plot_grouped_bars(
    data: Mapping[str, Mapping[str, float]],
    ylabel: str,
    out_path: str | Path | None = None,
    title: str | None = None,
    value_format: str | None = "{:.1f}",
    xlabel: str | None = None,
):
    """Grouped bars: {"Baseline": {"1-5": 21.3, "6-10": 15.2}, "Attention": {...}}.

    Category order follows the first model's keys. Made for the Q1 length buckets,
    but works for any per-category comparison (ablations, per-class F1).
    """
    _ensure_style()
    models = list(data)
    if len(models) > len(SERIES):
        raise ValueError(f"at most {len(SERIES)} models per figure")
    categories = list(data[models[0]])
    x = np.arange(len(categories))
    slot = 0.8 / len(models)

    fig, ax = plt.subplots(figsize=(max(4.5, 1.1 * len(categories) * max(1, len(models) / 2) + 1.5), 3.4))
    for i, model in enumerate(models):
        values = [data[model].get(c, np.nan) for c in categories]
        bars = ax.bar(x - 0.4 + slot * (i + 0.5), values, width=slot, color=SERIES[i],
                      edgecolor=SURFACE, linewidth=1.5, label=model)
        if value_format and len(models) <= 4:
            ax.bar_label(bars, labels=["" if np.isnan(v) else value_format.format(v) for v in values],
                         padding=2, fontsize=8, color=INK_SECONDARY)
    ax.set_xticks(x, labels=categories)
    ax.set_ylabel(ylabel)
    if xlabel:
        ax.set_xlabel(xlabel)
    ax.grid(axis="x", visible=False)
    ax.tick_params(axis="x", length=0)
    if len(models) > 1:
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.14 if xlabel else -0.08), ncol=len(models))
    if title:
        ax.set_title(title)
    fig.tight_layout()
    return _finish(fig, out_path)


def plot_confusion_matrix(
    cm,
    labels: Sequence[str],
    out_path: str | Path | None = None,
    title: str | None = None,
    normalize: bool = True,
    ax=None,
):
    """Confusion matrix with rows = true class, columns = predicted class.

    normalize=True divides each row by its total, so cells read as "share of true
    class X predicted as Y" (what Q2 asks for).
    """
    _ensure_style()
    cm = np.asarray(cm, dtype=float)
    if normalize:
        row_sums = cm.sum(axis=1, keepdims=True)
        cm = np.divide(cm, row_sums, out=np.zeros_like(cm), where=row_sums > 0)
    created = ax is None
    if created:
        size = max(3.6, 0.65 * len(labels) + 1.6)
        fig, ax = plt.subplots(figsize=(size + 0.6, size))
    else:
        fig = ax.figure
    image = ax.imshow(cm, cmap=SEQUENTIAL, vmin=0, vmax=1 if normalize else cm.max())
    ax.set_xticks(range(len(labels)), labels=labels)
    ax.set_yticks(range(len(labels)), labels=labels)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.grid(False)
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            text = f"{cm[i, j]:.2f}" if normalize else f"{int(cm[i, j])}"
            ax.text(j, i, text, ha="center", va="center", fontsize=8,
                    color=_label_color(image.cmap(image.norm(cm[i, j]))))
    if title:
        ax.set_title(title)
    if created:
        fig.colorbar(image, ax=ax, fraction=0.045, pad=0.03)
        fig.tight_layout()
    return _finish(fig, out_path)
