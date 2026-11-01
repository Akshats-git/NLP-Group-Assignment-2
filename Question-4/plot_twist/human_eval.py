"""Blinded human evaluation: building the rating sheet and analysing the ratings.

Ratings are 1-5 on four criteria. Statistics here are deliberately plain so they can be explained in
the report: means per model, a paired comparison on matched (prompt, decoding) pairs with a bootstrap
interval, and inter-rater agreement (quadratic-weighted kappa, Krippendorff's alpha).
"""

import csv
import itertools
import random
import statistics
from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path

import numpy as np

CRITERIA = ("coherence", "grammar", "relevance", "creativity")
CRITERIA_DESCRIPTIONS = {
    "coherence": "Does the story hang together? Events follow from each other, characters and details stay consistent.",
    "grammar": "Are the sentences grammatical and readable? Ignore the story's content; judge only the language.",
    "relevance": "Does the continuation stay connected to the opening it was given?",
    "creativity": "Is there something interesting, unexpected or surprising, beyond repeating the opening or generic filler?",
}


def select_samples(rows: Sequence[dict], models: Sequence[str], decoding_labels: Sequence[str], seed: int,
                   part: str = "plot_twist") -> tuple[list[dict], dict]:
    """Choose the stories raters will see, with a fixed rule instead of by hand.

    Rule: for every plot-twist prompt, every model and every listed decoding configuration, take
    sample 0 (the first sample generated, never a favourite). The pool is shuffled with `seed` and
    given neutral ids S01, S02, ... Returns (public samples without labels, private key id -> metadata).
    """
    pool = [r for r in rows if r["part"] == part and r["model"] in models and r["decoding_label"] in decoding_labels
            and r["sample_index"] == 0]
    pool.sort(key=lambda r: r["id"])
    random.Random(seed).shuffle(pool)
    public, key = [], {}
    for i, r in enumerate(pool, 1):
        sid = f"S{i:02d}"
        public.append({"id": sid, "prompt": r["prompt"], "text": r["text"]})
        key[sid] = {"model": r["model"], "decoding": r["decoding_label"], "prompt_key": r["prompt_key"],
                    "source_id": r["id"], "finished": r["finished"], "n_tokens": r["n_tokens"]}
    return public, key


def load_ratings(paths: Sequence[Path]) -> list[dict]:
    """Rows: rater, sample_id, coherence, grammar, relevance, creativity (ints 1-5)."""
    rows = []
    for path in paths:
        with Path(path).open(encoding="utf-8", newline="") as f:
            for r in csv.DictReader(f):
                row = {"rater": r["rater"].strip(), "sample_id": r["sample_id"].strip()}
                for c in CRITERIA:
                    v = int(r[c])
                    if not 1 <= v <= 5:
                        raise ValueError(f"{path}: {c} must be 1-5, got {v} for {row['sample_id']}")
                    row[c] = v
                rows.append(row)
    return rows


def _by_cell(ratings: Sequence[dict], criterion: str) -> dict:
    """{sample_id: {rater: value}}"""
    cells = defaultdict(dict)
    for r in ratings:
        cells[r["sample_id"]][r["rater"]] = r[criterion]
    return cells


def krippendorff_alpha(cells: dict, metric: str = "ordinal") -> float:
    """Krippendorff's alpha for {unit: {rater: value}}; units with fewer than 2 ratings are ignored.

    metric="interval" uses squared differences; "ordinal" uses squared cumulative-frequency distances.
    1 = perfect agreement, 0 = agreement no better than chance, negative = systematic disagreement.
    """
    units = [list(v.values()) for v in cells.values() if len(v) >= 2]
    values = sorted({x for u in units for x in u})
    if len(values) < 2:
        return float("nan")
    index = {v: i for i, v in enumerate(values)}
    k = len(values)
    o = np.zeros((k, k))
    for u in units:
        m = len(u)
        for a, b in itertools.permutations(range(m), 2):
            o[index[u[a]], index[u[b]]] += 1 / (m - 1)
    n_c = o.sum(axis=1)
    n = n_c.sum()
    if metric == "interval":
        delta = np.array([[(values[i] - values[j]) ** 2 for j in range(k)] for i in range(k)], dtype=float)
    else:
        cum = np.cumsum(n_c)
        delta = np.zeros((k, k))
        for i in range(k):
            for j in range(k):
                lo, hi = min(i, j), max(i, j)
                delta[i, j] = (cum[hi] - n_c[hi] / 2 - (cum[lo] - n_c[lo] / 2)) ** 2 if i != j else 0.0
    d_o = (o * delta).sum() / n
    d_e = (np.outer(n_c, n_c) * delta).sum() / (n * (n - 1))  # the diagonal has delta 0, so n_c * (n_c - 1) vs n_c ** 2 is moot
    return float(1 - d_o / d_e) if d_e > 0 else float("nan")


def quadratic_kappa(a: Sequence[int], b: Sequence[int], levels: int = 5) -> float:
    """Quadratic-weighted Cohen's kappa between two raters on the 1..levels scale."""
    a, b = np.asarray(a) - 1, np.asarray(b) - 1
    obs = np.zeros((levels, levels))
    for x, y in zip(a, b):
        obs[x, y] += 1
    obs /= obs.sum()
    exp = np.outer(obs.sum(axis=1), obs.sum(axis=0))
    w = np.array([[(i - j) ** 2 for j in range(levels)] for i in range(levels)], dtype=float) / (levels - 1) ** 2
    denom = (w * exp).sum()
    return float(1 - (w * obs).sum() / denom) if denom > 0 else float("nan")


def agreement(ratings: Sequence[dict]) -> dict:
    """Per criterion: alpha, mean pairwise quadratic kappa, exact and within-one agreement, mean |difference|."""
    out = {}
    raters = sorted({r["rater"] for r in ratings})
    for c in CRITERIA:
        cells = _by_cell(ratings, c)
        kappas, exact, within1, diffs = [], 0, 0, []
        for ra, rb in itertools.combinations(raters, 2):
            shared = [s for s, v in cells.items() if ra in v and rb in v]
            if len(shared) < 3:
                continue
            xa, xb = [cells[s][ra] for s in shared], [cells[s][rb] for s in shared]
            kappas.append(quadratic_kappa(xa, xb))
            d = [abs(x - y) for x, y in zip(xa, xb)]
            diffs += d
            exact += sum(v == 0 for v in d)
            within1 += sum(v <= 1 for v in d)
        out[c] = {
            "krippendorff_alpha_ordinal": krippendorff_alpha(cells),
            "mean_pairwise_quadratic_kappa": float(np.nanmean(kappas)) if kappas else float("nan"),
            "exact_agreement": exact / len(diffs) if diffs else float("nan"),
            "within_one_agreement": within1 / len(diffs) if diffs else float("nan"),
            "mean_abs_difference": statistics.fmean(diffs) if diffs else float("nan"),
        }
    return out


def rater_summary(ratings: Sequence[dict]) -> dict:
    """Each rater's mean per criterion: shows who is lenient or harsh."""
    out = {}
    for rater in sorted({r["rater"] for r in ratings}):
        mine = [r for r in ratings if r["rater"] == rater]
        out[rater] = {"n": len(mine), **{c: statistics.fmean(r[c] for r in mine) for c in CRITERIA}}
    return out


def group_means(ratings: Sequence[dict], key: dict, by: Sequence[str]) -> dict:
    """Mean rating per criterion for groups of stories, e.g. by=("model",) or by=("model", "decoding").

    Each story is first averaged over raters, then stories are averaged, so a story rated by 4 people does
    not count more than one rated by 3. The spread is the standard deviation of the story means.
    """
    per_story = defaultdict(lambda: defaultdict(list))
    for r in ratings:
        for c in CRITERIA:
            per_story[r["sample_id"]][c].append(r[c])
    groups = defaultdict(list)
    for sid, vals in per_story.items():
        meta = key[sid]
        groups[tuple(meta[b] for b in by)].append({c: statistics.fmean(v) for c, v in vals.items()})
    out = {}
    for g, stories in sorted(groups.items()):
        row = {"n_stories": len(stories)}
        for c in CRITERIA:
            xs = [s[c] for s in stories]
            row[c] = statistics.fmean(xs)
            row[f"{c}_sd"] = statistics.stdev(xs) if len(xs) > 1 else 0.0
        row["overall"] = statistics.fmean(row[c] for c in CRITERIA)
        out["/".join(g)] = row
    return out


def paired_difference(ratings: Sequence[dict], key: dict, a: str, b: str, n_boot: int = 10_000, seed: int = 0) -> dict:
    """Mean of (model a minus model b) over matched stories, with a bootstrap 95% interval and win/tie/loss.

    Stories are matched on (prompt, decoding): the same opening and the same decoding configuration, so only
    the model differs. Rater scores are averaged per story first.
    """
    per_story = defaultdict(lambda: defaultdict(list))
    for r in ratings:
        for c in CRITERIA:
            per_story[r["sample_id"]][c].append(r[c])
    pairs = defaultdict(dict)
    for sid, vals in per_story.items():
        meta = key[sid]
        pairs[(meta["prompt_key"], meta["decoding"])][meta["model"]] = {c: statistics.fmean(v) for c, v in vals.items()}
    matched = [p for p in pairs.values() if a in p and b in p]
    rng = random.Random(seed)
    out = {"n_pairs": len(matched)}
    for c in (*CRITERIA, "overall"):
        diffs = [(statistics.fmean(p[a][k] for k in CRITERIA) - statistics.fmean(p[b][k] for k in CRITERIA)) if c == "overall"
                 else p[a][c] - p[b][c] for p in matched]
        boots = sorted(statistics.fmean(rng.choices(diffs, k=len(diffs))) for _ in range(n_boot)) if diffs else [float("nan")]
        out[c] = {"mean_diff": statistics.fmean(diffs) if diffs else float("nan"),
                  "ci95": [boots[int(0.025 * n_boot)], boots[int(0.975 * n_boot) - 1]] if diffs else None,
                  "wins": sum(d > 0 for d in diffs), "ties": sum(d == 0 for d in diffs), "losses": sum(d < 0 for d in diffs)}
    return out
