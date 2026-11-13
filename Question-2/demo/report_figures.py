"""The live demo's output for a few reviews, as figures and numbers for the report (appendix A).

    # from Question-2/
    ../.venv/bin/python demo/report_figures.py              # add --device cpu if the GPU is short of memory
    ../.venv/bin/python demo/report_figures.py --redraw     # figures only, from the saved numbers

Runs the same code as the app's Live tab (inference.Demo) on the reviews in REVIEWS and writes
outputs/figures/demo_example_<n>.png and outputs/results/demo_examples.json. Each figure shows what
the app shows for one review: both models' stars, confidence, expected rating and probabilities, the
BiGRU's attention weights, RoBERTa's integrated gradients, and the pooling ablation.
"""

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import inference  # noqa: E402  (sets offline mode and sys.path before anything imports transformers)

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from matplotlib.colors import to_rgb  # noqa: E402
from matplotlib.patches import FancyBboxPatch, Rectangle  # noqa: E402

import results  # noqa: E402
from common import figures_dir, results_dir  # noqa: E402
from common.plotting import AXIS, INK, INK_MUTED, INK_SECONDARY, SERIES, savefig, set_style  # noqa: E402

SEED = 42          # the app's default seed
IG_STEPS = 256     # the report's and the app's setting; doubled while the attributions do not add up (see explain)
IG_MAX_GAP = 0.05  # largest completeness gap accepted, as a share of the logit change
# ("example", review id) is one of the report's test examples; ("probe", name) is a typed probe from results.PROBES.
REVIEWS = [("example", "en_0600154"), ("example", "en_0239182"), ("probe", "Sarcasm")]

BLUE, ORANGE, GREEN, GOLD, RED = SERIES[0], SERIES[1], SERIES[2], SERIES[3], SERIES[7]   # as in app.py
SUBTITLE = {"bigru": "GloVe → 2-layer BiGRU → additive attention · 12.0 M parameters",
            "mean": "roberta-base → masked mean pooling · 124.1 M parameters"}
MINUS = "−"


def signed(x: float, digits: int = 2) -> str:
    return f"{x:+.{digits}f}".replace("-", MINUS)


def tint(colour: str, alpha: float) -> tuple:
    """`colour` at opacity `alpha` over white, as the app's rgba() backgrounds look on a light page."""
    return tuple(1 - alpha * (1 - c) for c in to_rgb(colour))


# ======================================================================= run the demo
def reviews() -> list[dict]:
    examples = {x["id"]: x for x in results.report_examples()}
    probes = {name: (title, body) for name, title, body in results.PROBES}
    out = []
    for kind, key in REVIEWS:
        if kind == "example":
            x = examples[key]
            out.append({"kind": kind, "id": key, "letter": x["letter"], "name": x["name"], "text": x["text"],
                        "true_stars": x["stars"]})
        else:
            title, body = probes[key]
            out.append({"kind": kind, "name": key, "title": title, "body": body,
                        "text": inference.Demo.text(title, body), "true_stars": None})
    return out


def prediction(pred: inference.Prediction) -> dict:
    return {"stars": pred.stars, "confidence": pred.confidence, "expected": pred.expected,
            "probs": [float(p) for p in pred.probs]}


def top(words, scores, k=3, sign=1) -> list[dict]:
    order = [i for i in np.argsort(-sign * np.asarray(scores)) if sign * scores[i] > 0][:k]
    return [{"word": words[i], "score": float(scores[i])} for i in order]


def run(demo: inference.Demo, review: dict, chunk: int) -> dict:
    """What the Live tab computes for one review with the main models: both predictions and the explanation."""
    text = review["text"]
    bigru = demo.predict_bigru(text, SEED)
    mean = demo.predict_roberta(text, "mean", SEED)
    exp, first_gap = explain(demo, text, mean.stars, chunk)
    return {
        **review, "seed": SEED,
        "bigru": {**prediction(bigru), "tokens": bigru.tokens, "attention": [float(a) for a in bigru.attention],
                  "unknown": [bool(u) for u in bigru.unknown],
                  "top_attention": top(bigru.tokens, bigru.attention)},
        "roberta_mean": {**prediction(mean), "ig_target_stars": exp.target_stars, "ig_words": exp.words,
                         "ig_scores": [float(s) for s in exp.scores], "ig_logit_change": exp.logit_change,
                         "ig_gap": exp.gap, "ig_steps": exp.steps, f"ig_gap_at_{IG_STEPS}_steps": first_gap,
                         "top_for": top(exp.words, exp.scores), "top_against": top(exp.words, exp.scores, sign=-1)},
    }


def explain(demo: inference.Demo, text: str, stars: int, chunk: int) -> tuple[inference.Explanation, float]:
    """Integrated gradients with IG_STEPS steps, doubled until the scores add up to the logit change.

    The midpoint rule does not converge evenly: on the sarcasm probe 256 steps leave a gap of 1.01 on a logit
    change of 4.38 (nearly all of it on one word), while 1024 steps leave 0.11. Returns the explanation and
    the gap at IG_STEPS.
    """
    steps = IG_STEPS
    exp = demo.explain_roberta(text, "mean", SEED, stars, steps, chunk)
    first_gap = exp.gap
    while abs(exp.gap) > IG_MAX_GAP * abs(exp.logit_change) and steps < 4096:
        steps *= 2
        exp = demo.explain_roberta(text, "mean", SEED, stars, steps, chunk)
    return exp, first_gap


def release(demo: inference.Demo) -> None:
    """Drop the loaded models, so only one RoBERTa checkpoint is in memory at a time (8 GB laptop)."""
    demo._models.clear()
    if demo.device.type == "mps":
        torch.mps.empty_cache()


def collect(device: str | None, chunk: int) -> list[dict]:
    demo = inference.Demo(device)
    outputs = [run(demo, review, chunk) for review in reviews()]
    release(demo)
    for out in outputs:                                   # the pooling ablation shown under the two cards
        out["roberta_first"] = prediction(demo.predict_roberta(out["text"], "first", SEED))
    other = next(s for s in inference.SEEDS if s != SEED)  # the other training seed: numbers for the report's table
    for out in outputs:
        out["other_seed"] = {"seed": other, "bigru": prediction(demo.predict_bigru(out["text"], other))}
    for pooling in ("mean", "first"):
        release(demo)
        for out in outputs:
            out["other_seed"][f"roberta_{pooling}"] = prediction(demo.predict_roberta(out["text"], pooling, other))
    return outputs


# ======================================================================= drawing
class Canvas:
    """Collects text and shapes at positions in inches (y grows downwards), then renders a figure as tall as needed."""

    def __init__(self, width: float):
        self.width, self.items = width, []
        self._measure = plt.figure(figsize=(width, 1), dpi=200)   # the dpi savefig uses, so hinted widths match
        self._renderer = self._measure.canvas.get_renderer()

    def text_width(self, text: str, size: float, weight: str = "normal") -> float:
        artist = self._measure.text(0, 0, text, fontsize=size, fontweight=weight)
        width = artist.get_window_extent(self._renderer).width / self._measure.dpi
        artist.remove()
        return width

    def text(self, x, y, text, size, colour=INK, weight="normal", ha="left", alpha=1.0) -> float:
        self.items.append(("text", x, y, text, dict(fontsize=size, color=colour, fontweight=weight, ha=ha, alpha=alpha)))
        return self.text_width(text, size, weight)

    def box(self, x, y, w, h, **style) -> None:
        self.items.append(("box", x, y, w, h, style))

    def flow(self, x0, x1, y, pieces, size, line, gap=None, pad=0.0) -> float:
        """Wrapped left-to-right text between x0 and x1. pieces: (text, style) with optional style keys
        colour, weight, fill (background colour) and underline (colour). Returns the y of the last line."""
        space = self.text_width("a a", size) - self.text_width("aa", size)
        gap = space if gap is None else gap
        x = x0
        for text, style in pieces:
            weight = style.get("weight", "normal")
            inset = pad if "fill" in style else 0.0
            w = self.text_width(text, size, weight) + 2 * inset
            if x + w > x1 and x > x0:
                x, y = x0, y + line
            height = size * 1.45 / 72
            if style.get("fill") is not None:
                self.box(x, y - height / 2, w, height, facecolor=style["fill"], edgecolor="none")
            if style.get("underline"):
                self.box(x + inset, y + height / 2 + 0.012, w - 2 * inset, 0.014, facecolor=style["underline"], edgecolor="none")
            self.text(x + inset, y, text, size, style.get("colour", INK), weight)
            x += w + gap
        return y

    def paragraph(self, x0, x1, y, text, size, line, colour=INK) -> float:
        """Plain wrapped text, one text object per line. Returns the y of the last line."""
        current = ""
        for word in text.split():
            longer = f"{current} {word}".strip()
            if current and self.text_width(longer, size) > x1 - x0:
                self.text(x0, y, current, size, colour)
                current, y = word, y + line
            else:
                current = longer
        self.text(x0, y, current, size, colour)
        return y

    def render(self, height: float, out_path: Path) -> None:
        plt.close(self._measure)
        fig = plt.figure(figsize=(self.width, height))
        ax = fig.add_axes((0, 0, 1, 1))
        ax.set_xlim(0, self.width)
        ax.set_ylim(height, 0)
        ax.axis("off")
        for kind, x, y, *rest in self.items:
            if kind == "text":
                ax.text(x, y, rest[0], va="center_baseline", **rest[1])
            else:
                w, h, style = rest
                rounding = style.pop("rounding", 0)
                if rounding:
                    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={rounding}", **style))
                else:
                    ax.add_patch(Rectangle((x, y), w, h, **style))
        savefig(fig, out_path)


def words(text: str, **style) -> list[tuple[str, dict]]:
    return [(w, style) for w in text.split()]


def draw_card(c: Canvas, x: float, y: float, w: float, h: float, name: str, subtitle: str, pred: dict,
              true_stars: int | None, colour: str) -> None:
    c.box(x, y, w, h, facecolor="none", edgecolor=AXIS, linewidth=0.8, rounding=0.06)
    left = x + 0.13
    c.text(left, y + 0.19, name, 9.5, weight="bold")
    c.text(left, y + 0.37, subtitle, 6.4, INK_MUTED)

    # stars, verdict, confidence, expected rating
    star_x = left
    for k in range(5):
        star_x += c.text(star_x, y + 0.78, "★", 17, GOLD, alpha=1.0 if k < pred["stars"] else 0.25) + 0.015
    verdict = f"{pred['stars']}★"
    if true_stars is not None:
        off = abs(pred["stars"] - true_stars)
        verdict += "  ✓ correct" if off == 0 else f"  ✗ off by {off}"
    c.text(left, y + 1.11, verdict, 8.5, weight="bold")
    c.text(left, y + 1.28, f"{pred['confidence']:.0%} confident", 8, INK_SECONDARY)
    c.text(left, y + 1.44, f"expected {pred['expected']:.2f}★", 8, INK_SECONDARY)

    # probability of each rating; the predicted one in full colour
    bars_x0, bars_x1, base, full = x + 0.46 * w, x + w - 0.13, y + 1.26, 0.72
    slot = (bars_x1 - bars_x0) / 5
    bar_w, radius = slot - 0.07, 0.03
    c.box(bars_x0, base, bars_x1 - bars_x0, 0.006, facecolor=AXIS, edgecolor="none")
    for k, p in enumerate(pred["probs"]):
        bx, bh = bars_x0 + k * slot + 0.035, full * p
        fill = colour if k + 1 == pred["stars"] else tint(colour, 0.35)
        if bh > 2 * radius:   # rounded top, square at the baseline
            c.box(bx, base - bh, bar_w, bh, facecolor=fill, edgecolor="none", rounding=radius)
            c.box(bx, base - bh + radius, bar_w, bh - radius, facecolor=fill, edgecolor="none")
        elif bh > 0.004:
            c.box(bx, base - bh, bar_w, bh, facecolor=fill, edgecolor="none")
        c.text(bx + bar_w / 2, base - bh - 0.085, f"{p:.0%}", 7, INK_SECONDARY, ha="center",
               weight="bold" if k + 1 == pred["stars"] else "normal")
        c.text(bx + bar_w / 2, base + 0.13, f"{k + 1}★", 7.5, INK_MUTED, ha="center")


def draw(out: dict, out_path: Path, width: float = 7.2) -> None:
    set_style()
    c = Canvas(width)
    m, size, line = 0.06, 8.6, 8.6 * 1.8 / 72
    small, small_line = 7.2, 7.2 * 1.55 / 72
    bigru, mean, first = out["bigru"], out["roberta_mean"], out["roberta_first"]
    target = mean["ig_target_stars"]

    # header
    if out["kind"] == "example":
        head = f"Test review {out['letter']} ({out['name'].split(' (')[0].lower()}) · true rating {out['true_stars']}★"
    else:
        head = f"Typed probe ({out['name'].lower()}) · no true rating"
    y = 0.14
    c.text(m, y, head, 9.5, weight="bold")
    c.text(width - m, y, f"live demo, seed {out['seed']}", 7.2, INK_MUTED, ha="right")

    # the two model cards
    y += 0.16
    gutter, card_h = 0.16, 1.62
    card_w = (width - 2 * m - gutter) / 2
    draw_card(c, m, y, card_w, card_h, "BiGRU + attention", SUBTITLE["bigru"], bigru, out["true_stars"], BLUE)
    draw_card(c, m + card_w + gutter, y, card_w, card_h, "RoBERTa (mean)", SUBTITLE["mean"], mean, out["true_stars"], ORANGE)
    y += card_h

    # BiGRU attention
    n_unknown = sum(bigru["unknown"])
    y += 0.24
    y = c.flow(m, width - m, y, words("What the BiGRU attends to", weight="bold")
               + words("· attention weights α, darker = more weight" + (
                   "; underlined = not in its vocabulary, read as <unk>" if n_unknown else ""), colour=INK_SECONDARY),
               size, line)
    scale = max(bigru["attention"]) or 1.0
    tokens = [(t, {"fill": tint(BLUE, 0.65 * a / scale), "underline": RED if u else None})
              for t, a, u in zip(bigru["tokens"], bigru["attention"], bigru["unknown"])]
    y = c.flow(m, width - m, y + line, tokens, size, line, gap=0.035, pad=0.03)
    note = "Top: " + ", ".join(f"“{t['word']}” {t['score']:.0%}" for t in bigru["top_attention"]) + "."
    y = c.paragraph(m, width - m, y + line * 0.95, note, small, small_line, INK_MUTED)

    # RoBERTa integrated gradients
    y += 0.3
    y = c.flow(m, width - m, y, words(f"Why RoBERTa says {target}★", weight="bold")
               + words(f"· integrated gradients for the {target}★ logit:", colour=INK_SECONDARY)
               + [(f"pushes towards {target}★", {"fill": tint(GREEN, 0.55)}), ("pushes away", {"fill": tint(RED, 0.55)})],
               size, line, pad=0.03)
    scores = np.asarray(mean["ig_scores"])
    scale = float(np.abs(scores).max()) or 1.0
    tokens = [(w, {"fill": tint(GREEN if s >= 0 else RED, 0.65 * min(abs(s) / scale, 1.0))})
              for w, s in zip(mean["ig_words"], scores)]
    y = c.flow(m, width - m, y + line, tokens, size, line, gap=0.035, pad=0.03)

    def listed(items):
        return ", ".join(f"“{t['word']}” ({signed(t['score'])})" for t in items) or "none"
    note = (f"For {target}★: {listed(mean['top_for'])}. Against: {listed(mean['top_against'])}. Scores add up to the "
            f"logit change {signed(mean['ig_logit_change'])} (gap {signed(mean['ig_gap'], 3)}, {mean['ig_steps']} steps).")
    y = c.paragraph(m, width - m, y + line * 0.95, note, small, small_line, INK_MUTED)

    # pooling ablation
    y += 0.27
    y = c.flow(m, width - m, y, words("Pooling ablation", weight="bold")
               + words(f"· mean pooling {mean['stars']}★ ({mean['confidence']:.0%}) · first-token <s> pooling "
                       f"{first['stars']}★ ({first['confidence']:.0%})", colour=INK_SECONDARY), size, line)
    c.render(y + 0.14, out_path)


# ======================================================================= main
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--device", default=None, help="cpu, mps or cuda (default: what the app uses)")
    parser.add_argument("--chunk", type=int, default=8, help="integrated-gradient steps per forward pass")
    parser.add_argument("--redraw", action="store_true", help="only redraw the figures from demo_examples.json")
    args = parser.parse_args()

    path = results_dir(2) / "demo_examples.json"
    if args.redraw:
        outputs = json.loads(path.read_text())
    else:
        outputs = collect(args.device, args.chunk)
        for n, out in enumerate(outputs, start=1):
            out["figure"] = f"demo_example_{n}.png"
        path.write_text(json.dumps(outputs, indent=1, ensure_ascii=False) + "\n")
        print("wrote", path)
    for out in outputs:
        draw(out, figures_dir(2) / out["figure"])
        print(f"{out['figure']}: {out.get('letter') or out['name']} · BiGRU {out['bigru']['stars']}★ "
              f"({out['bigru']['confidence']:.0%}) · RoBERTa {out['roberta_mean']['stars']}★ "
              f"({out['roberta_mean']['confidence']:.0%}) · first-token {out['roberta_first']['stars']}★")


if __name__ == "__main__":
    main()
