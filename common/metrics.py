"""Metrics more than one question needs.

Question-specific ones live with the question: BLEU (Q1), classification metrics
and MAE (Q2), ROUGE (Q3), Distinct-n and repetition rate (Q4).
"""

import math
import statistics
from collections.abc import Sequence

import torch

# Sentence-length buckets from the Q1 brief: 1-5, 6-10, 11-15, 16+ tokens.
DEFAULT_LENGTH_EDGES = (5, 10, 15)


def masked_accuracy(logits: torch.Tensor, targets: torch.Tensor, pad_id: int) -> tuple[int, int]:
    """Token accuracy ignoring padding. Returns (correct, total) so batches can be summed exactly."""
    mask = targets != pad_id
    correct = ((logits.argmax(dim=-1) == targets) & mask).sum().item()
    return int(correct), int(mask.sum().item())


def perplexity(mean_nll: float) -> float:
    """exp of the mean per-token negative log-likelihood (natural log, padding excluded)."""
    try:
        return math.exp(mean_nll)
    except OverflowError:
        return float("inf")


def mean_std(values: Sequence[float]) -> tuple[float, float]:
    """Mean and sample standard deviation (n-1). With one value the std is 0."""
    values = [float(v) for v in values]
    mean = statistics.fmean(values)
    std = statistics.stdev(values) if len(values) > 1 else 0.0
    return mean, std


def format_mean_std(values: Sequence[float], digits: int = 2, scale: float = 1.0) -> str:
    """'84.21 ± 0.35'. Use scale=100 to show a 0-1 metric as a percentage."""
    mean, std = mean_std([v * scale for v in values])
    return f"{mean:.{digits}f} ± {std:.{digits}f}"


def summarize_runs(runs: Sequence[dict]) -> dict:
    """Combine the metric dicts of several seeds into {metric: {"mean", "std", "values"}}."""
    summary = {}
    for key in runs[0]:
        values = [run[key] for run in runs if isinstance(run.get(key), (int, float)) and not isinstance(run[key], bool)]
        if len(values) == len(runs):
            mean, std = mean_std(values)
            summary[key] = {"mean": mean, "std": std, "values": values}
    return summary


def bucket_labels(edges: Sequence[int] = DEFAULT_LENGTH_EDGES) -> list[str]:
    """(5, 10, 15) -> ['1-5', '6-10', '11-15', '16+']"""
    labels, low = [], 1
    for high in edges:
        labels.append(f"{low}-{high}")
        low = high + 1
    labels.append(f"{low}+")
    return labels


def bucket_label(length: int, edges: Sequence[int] = DEFAULT_LENGTH_EDGES) -> str:
    labels = bucket_labels(edges)
    for label, high in zip(labels, edges):
        if length <= high:
            return label
    return labels[-1]


def bucket_indices(lengths: Sequence[int], edges: Sequence[int] = DEFAULT_LENGTH_EDGES) -> dict[str, list[int]]:
    """Example indices per length bucket, in bucket order.

    Use this for corpus-level metrics: BLEU of a bucket is computed on that bucket's
    sentences together, not as an average of sentence BLEU.
    """
    groups = {label: [] for label in bucket_labels(edges)}
    for i, length in enumerate(lengths):
        groups[bucket_label(length, edges)].append(i)
    return groups


def group_by_bucket(lengths: Sequence[int], values: Sequence[float], edges: Sequence[int] = DEFAULT_LENGTH_EDGES) -> dict:
    """Mean of a per-example value in each length bucket: {label: {"mean", "count"}}."""
    out = {}
    for label, idx in bucket_indices(lengths, edges).items():
        out[label] = {"mean": statistics.fmean(values[i] for i in idx) if idx else float("nan"), "count": len(idx)}
    return out
