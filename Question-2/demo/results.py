"""Everything the demo shows from the committed results: metrics, saved test predictions, curves, examples.

Nothing here runs a model. The test predictions are the ones 04_evaluation.ipynb saved once.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from common import results_dir

RESULTS = results_dir(2, create=False)
MODELS = {   # display name -> (runs, colour); colours match common.plotting.SERIES used in the report
    "BiGRU + attention": (["bigru_seed42", "bigru_seed43"], "#2a78d6"),
    "RoBERTa (mean)": (["roberta_mean_seed42", "roberta_mean_seed43"], "#eb6834"),
    "RoBERTa (first token)": (["roberta_first_seed42", "roberta_first_seed43"], "#1baf7a"),
}
BIGRU, ROBERTA, FIRST = MODELS

# Why each report example is hard (report section 8, Table 6). Letters as in the report.
REPORT_EXAMPLES = {
    "en_0600154": ("A", "Negated negatives", "\"don't have cravings\", \"horrible\" … resolved by \"the cravings are gone!!\" "
                   "at the end. The BiGRU latches onto the negative words."),
    "en_0792507": ("B", "Typos + 'not to bad'", "Typos (\"isnt\", \"to bad\") and a negated negative; "
                   "the reviewer is mildly positive."),
    "en_0623344": ("C", "Rare word + comparison", "\"Senseo\" is <unk> for the BiGRU and the verdict hinges on "
                   "\"better than the original\"."),
    "en_0298106": ("D", "Irony, mixed, long", "Ironic \"well, let's just say … deus ex machina\" wrapped around praise "
                   "for the characters: a real 3★."),
    "en_0753534": ("E", "Problem → solution", "A long description of a problem (\"cooked kind of smell\") before the "
                   "product solves it."),
    "en_0501949": ("F", "Pros/cons list, '4.5/5'", "The reviewer even writes \"Overall: 4.5/5\"; the two BiGRU seeds "
                   "say 2★ and 5★."),
    "en_0133223": ("G", "Policy complaint (BiGRU wins)", "A complaint about the return policy, not the product. "
                   "RoBERTa softens it to 2★."),
    "en_0239182": ("H", "Concession (BiGRU wins)", "\"Probably very good … probably deserves 5\" yet the verdict "
                   "is 1★. RoBERTa weighs the hedge."),
    "en_0648962": ("I", "\"Decent camera but useless\" (BiGRU wins)", "A concession inside a harsh 1★ review."),
}

# Hand-written probes: no true label, just to poke at the models live.
PROBES = [
    ("Negation", "Not bad at all.", "I expected it to fall apart within a week, but it's still working perfectly."),
    ("Sarcasm", "Great.", "Oh great, another charger that works for exactly one day. Truly fantastic quality, thanks a lot."),
    ("Mixed", "Wanted to love it", "The design is beautiful and setup was easy, but it stopped charging after two days "
                                   "and support never replied."),
    ("Middle of the road", "", "It's okay. Does what it says, nothing more. Packaging was a bit damaged."),
    ("Typos and slang", "", "absolutly luv it!! wud defo buy agian, my kidz use it evry day"),
    ("Twist at the end", "", "Terrible packaging, the box was crushed and the manual was missing. "
                             "The blender itself, though, is the best I've ever owned."),
    ("Seller vs product", "", "Five stars for the seller's lightning fast shipping, zero stars for the product, "
                              "which broke on first use."),
]


def _json(path: Path):
    return json.loads(path.read_text())


def _jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def test_summary() -> dict:
    return _json(RESULTS / "test_summary.json")


def report_examples() -> list[dict]:
    examples = _json(RESULTS / "test_examples.json")
    rows = {x["id"]: x for key in ("roberta_right_bigru_wrong", "bigru_right_roberta_wrong") for x in examples[key]}
    out = []
    for review_id, (letter, name, why) in REPORT_EXAMPLES.items():
        x = rows[review_id]
        out.append({"id": review_id, "letter": letter, "name": name, "why": why, "text": x["text"], "stars": x["stars"],
                    "bigru_stars": x["bigru_stars"], "roberta_stars": x["roberta_stars"]})
    return out


def metrics_table() -> pd.DataFrame:
    summary = test_summary()["summary"]
    rows = []
    for model in MODELS:
        s = summary[model]
        rows.append({
            "Model": model,
            "Accuracy (%)": f"{100 * s['accuracy']['mean']:.1f} ± {100 * s['accuracy']['std']:.1f}",
            "Macro-F1 (%)": f"{100 * s['macro_f1']['mean']:.1f} ± {100 * s['macro_f1']['std']:.1f}",
            "MAE (stars)": f"{s['mae']['mean']:.3f} ± {s['mae']['std']:.3f}",
            "Off by ≥ 2 stars (%)": f"{100 * s['off_by_two_plus']['mean']:.1f}",
            "1★ ↔ 5★ swaps": f"{s['extreme_confusions']['mean']:.1f}",
        })
    return pd.DataFrame(rows)


def headline() -> dict:
    s = test_summary()["summary"]
    return {key: (s[ROBERTA][key]["mean"], s[BIGRU][key]["mean"])
            for key in ("accuracy", "macro_f1", "mae", "extreme_confusions", "off_by_two_plus")}


def per_class_f1() -> pd.DataFrame:
    per_run = test_summary()["per_run"]
    rows = []
    for model, (runs, _) in MODELS.items():
        f1 = np.mean([per_run[r]["per_class_f1"] for r in runs], axis=0)
        rows += [{"Model": model, "Stars": f"{k + 1}★", "F1": float(v)} for k, v in enumerate(f1)]
    return pd.DataFrame(rows)


def confusion(model: str) -> pd.DataFrame:
    """Row-normalised confusion matrix, both seeds pooled (as in the report)."""
    per_run = test_summary()["per_run"]
    cm = np.sum([per_run[r]["confusion_matrix"] for r in MODELS[model][0]], axis=0).astype(float)
    norm = cm / cm.sum(axis=1, keepdims=True)
    return pd.DataFrame([{"True": f"{i + 1}★", "Predicted": f"{j + 1}★", "Share": norm[i, j], "Count": int(cm[i, j])}
                         for i in range(5) for j in range(5)])


def error_distance() -> pd.DataFrame:
    """How far off the errors are, share of all test reviews, both seeds pooled."""
    preds = saved_predictions()
    rows = []
    for model, (runs, _) in MODELS.items():
        dist = np.concatenate([np.abs(preds[r + "_pred"] - preds["label"]) for r in runs])
        for d in (1, 2, 3, 4):
            rows.append({"Model": model, "Off by": f"{d} star" + ("s" if d > 1 else ""), "Share": float(np.mean(dist == d))})
    return pd.DataFrame(rows)


def curves() -> pd.DataFrame:
    rows = []
    for model, (runs, _) in MODELS.items():
        for run in runs:
            for h in _json(RESULTS / run / "history.json"):
                for split in ("train", "val"):
                    if f"{split}_macro_f1" in h:
                        rows.append({"Model": model, "Seed": run[-2:], "Epoch": h["epoch"], "Split": split,
                                     "Macro-F1": h[f"{split}_macro_f1"], "Run": f"{model} · seed {run[-2:]} · {split}"})
    return pd.DataFrame(rows)


def saved_predictions() -> pd.DataFrame:
    """One row per test review: id, label, and every run's saved prediction and probabilities."""
    frame = None
    for runs, _ in MODELS.values():
        for run in runs:
            rows = _jsonl(RESULTS / run / "test_predictions.jsonl")
            part = pd.DataFrame({"id": [r["id"] for r in rows], "label": [r["label"] for r in rows],
                                 run + "_pred": [r["pred"] for r in rows],
                                 run + "_probs": [r["probs"] for r in rows]})
            frame = part if frame is None else frame.merge(part.drop(columns="label"), on="id")
    return frame
