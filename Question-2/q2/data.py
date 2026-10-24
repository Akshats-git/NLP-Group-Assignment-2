"""Loading the Amazon reviews, with each review's title put back in front of its body.

The dataset the brief names, SetFit/amazon_reviews_multi_en, only kept the review body in
`text`. mteb/amazon_reviews_multi (config "en") is the same data with the same ids and
splits, but its `text` is title + "\n\n" + body. Splits and labels come from SetFit as the
brief asks; titles come from mteb, matched by id. A title is only used when the mteb body
equals the SetFit body, so a bad match can never attach the wrong title.

The test split is only loaded when asked for by name. Training and model selection use
train and validation; the final evaluation is the one place that loads "test".
"""

from collections.abc import Sequence

import numpy as np
from datasets import Dataset, DatasetDict, load_dataset, load_from_disk

from common import data_dir

SOURCE = "SetFit/amazon_reviews_multi_en"
TITLE_SOURCE = "mteb/amazon_reviews_multi"
NUM_CLASSES = 5
STAR_LABELS = ["1", "2", "3", "4", "5"]  # label i is i + 1 stars
TRAIN_SPLITS = ("train", "validation")


def split_title(titled_text: str) -> tuple[str, str]:
    """mteb's "title\n\nbody" -> (title, body). No separator means there is no title."""
    title, sep, body = titled_text.partition("\n\n")
    return (title.strip(), body) if sep else ("", titled_text)


def join_title_body(title: str, body: str) -> str:
    """One input text: the title as a first sentence, then the body."""
    title, body = title.strip(), body.strip()
    if not title:
        return body
    if title[-1] not in ".!?":
        title += "."
    return f"{title} {body}"


def _squash(text: str) -> str:
    return " ".join(text.split())


def merge_titles(reviews: Dataset, titled: Dataset) -> Dataset:
    """Add title / body / merged text columns to the SetFit split from the mteb split."""
    by_id = dict(zip(titled["id"], titled["text"]))

    def add_title(batch):
        out = {"title": [], "body": [], "text": [], "title_found": []}
        for review_id, body in zip(batch["id"], batch["text"]):
            title, titled_body = split_title(by_id.get(review_id, ""))
            found = bool(title) and _squash(titled_body) == _squash(body)
            title = title if found else ""
            out["title"].append(title)
            out["body"].append(body)
            out["text"].append(join_title_body(title, body))
            out["title_found"].append(found)
        return out

    merged = reviews.map(add_title, batched=True, desc="adding titles")
    return merged.remove_columns([c for c in ("label_text",) if c in merged.column_names])


def _load_titled(split: str) -> Dataset:
    # mteb's copy is script-based, which current `datasets` refuses to run, so read the
    # Parquet export the Hub keeps of it instead
    files = f"hf://datasets/{TITLE_SOURCE}@refs/convert/parquet/en/{split}/*.parquet"
    return load_dataset("parquet", data_files={split: files}, split=split)


def load_reviews(splits: Sequence[str] = TRAIN_SPLITS, limit: int | None = None, seed: int = 0) -> DatasetDict:
    """Columns: id, title, body, text (title + body), label (0-4), title_found.

    Merged splits are cached in Question-2/data/reviews/. limit takes a seeded random
    subset of each split for quick checks (the raw files are ordered by label, so the
    first N rows would all be 1-star).
    """
    cache = data_dir(2) / "reviews"
    out = {}
    for split in splits:
        path = cache / split
        if path.exists():
            ds = load_from_disk(str(path))
        else:
            ds = merge_titles(load_dataset(SOURCE, split=split), _load_titled(split))
            ds.save_to_disk(str(path))
            ds = load_from_disk(str(path))
        if limit is not None and limit < len(ds):
            ds = ds.shuffle(seed=seed).select(range(limit)).flatten_indices()
        out[split] = ds
    return DatasetDict(out)


def stars(labels) -> np.ndarray:
    """Class ids 0-4 -> star ratings 1-5, the scale MAE is reported on."""
    return np.asarray(labels) + 1


def label_counts(ds: Dataset) -> list[int]:
    return np.bincount(ds["label"], minlength=NUM_CLASSES).tolist()


def length_stats(lengths, cutoffs: Sequence[int] = ()) -> dict:
    """Percentiles of sequence length and the share of examples longer than each cutoff."""
    lengths = np.asarray(lengths)
    stats = {
        "mean": float(lengths.mean()),
        **{f"p{q}": float(np.percentile(lengths, q)) for q in (50, 90, 95, 99)},
        "max": int(lengths.max()),
    }
    for cutoff in cutoffs:
        stats[f"share_over_{cutoff}"] = float((lengths > cutoff).mean())
    return stats
