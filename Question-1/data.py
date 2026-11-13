"""IIT Bombay English-Hindi parallel corpus: download, clean, tokenise, build vocabs.

Corpus source: HuggingFace hub — cfilt/iitb-english-hindi
  https://huggingface.co/datasets/cfilt/iitb-english-hindi

Usage from the training script:

    from data import load_splits, build_vocabs, NMTDataset
    train, dev, test, stats = load_splits(max_train=30_000)
    vocab_en, vocab_hi = build_vocabs(train)
    train_ds = NMTDataset(train, vocab_en, vocab_hi)
"""

import random
import re
from pathlib import Path

import torch
from torch.utils.data import Dataset

from common import Vocab, normalize, data_dir, save_json, load_json

# ── cache paths ───────────────────────────────────────────────────────────────

_DATA_ROOT = data_dir(1)

# ── downloading via HuggingFace datasets ──────────────────────────────────────

def _load_hf_dataset():
    """Load the IITB English-Hindi corpus from the HuggingFace hub (cached by `datasets`)."""
    try:
        from datasets import load_dataset
    except ImportError as e:
        raise ImportError(
            "The 'datasets' package is required. Install it with: pip install datasets"
        ) from e
    print("  loading cfilt/iitb-english-hindi from HuggingFace hub ...")
    return load_dataset("cfilt/iitb-english-hindi")


# ── cleaning and tokenisation ─────────────────────────────────────────────────

# punctuation is split off as its own token so "hello," and "hello" share an entry
_PUNCT = re.compile(r"([.,;:!?()\[\]\"\u201c\u201d\u2018\u2019|\u0964])")
_DEVANAGARI = re.compile(r"[\u0900-\u097F]")
_LATIN = re.compile(r"[A-Za-z]")


def _tokenize(text: str, lower: bool) -> list[str]:
    return _PUNCT.sub(r" \1 ", normalize(text, lower=lower)).split()


def _tokenize_en(text: str) -> list[str]:
    """Lowercase, split punctuation, whitespace split."""
    return _tokenize(text, lower=True)


def _tokenize_hi(text: str) -> list[str]:
    """NFC-normalise, split punctuation (including the danda), whitespace split."""
    return _tokenize(text, lower=False)


def _is_valid_pair(en_raw: str, hi_raw: str, en_tokens: list[str], hi_tokens: list[str],
                   min_len: int = 1, max_len: int = 50, max_ratio: float | None = 3.0) -> bool:
    """Drop empty, too-short, too-long and misaligned pairs.

    Misaligned = the Hindi side has no Devanagari, the English side has no Latin letter,
    or (when max_ratio is set) the two lengths differ by more than max_ratio times.
    """
    if not (min_len <= len(en_tokens) <= max_len and min_len <= len(hi_tokens) <= max_len):
        return False
    if not _LATIN.search(en_raw) or not _DEVANAGARI.search(hi_raw):
        return False
    if max_ratio is not None:
        longer, shorter = max(len(en_tokens), len(hi_tokens)), min(len(en_tokens), len(hi_tokens))
        if longer > max_ratio * shorter:
            return False
    return True


def _clean_pair(en_raw: str, hi_raw: str, min_len: int, max_len: int,
                max_ratio: float | None) -> dict | None:
    en_raw, hi_raw = (en_raw or "").strip(), (hi_raw or "").strip()
    if not en_raw or not hi_raw:
        return None
    en_tok, hi_tok = _tokenize_en(en_raw), _tokenize_hi(hi_raw)
    if not _is_valid_pair(en_raw, hi_raw, en_tok, hi_tok, min_len, max_len, max_ratio):
        return None
    return {"en_tokens": en_tok, "hi_tokens": hi_tok,
            "en_text": " ".join(en_tok), "hi_text": " ".join(hi_tok)}


def _clean_split(split, min_len: int, max_len: int, max_ratio: float | None) -> list[dict]:
    out = []
    for item in split:
        pair = _clean_pair(item["translation"]["en"], item["translation"]["hi"],
                           min_len, max_len, max_ratio)
        if pair:
            out.append(pair)
    return out


# ── public API ────────────────────────────────────────────────────────────────

DEFAULT_MAX_TRAIN = 50_000   # top of the brief's 20k-50k range
DEFAULT_MIN_LEN = 5          # training pairs need >= this many English tokens


def load_splits(max_train: int = DEFAULT_MAX_TRAIN, max_len: int = 50,
                seed: int = 42, min_len: int = DEFAULT_MIN_LEN) -> tuple[list[dict], list[dict], list[dict], dict]:
    """Download (or load from cache), clean, tokenise and subsample the corpus.

    Training pairs are drawn in a seeded random order from the raw corpus and the first
    `max_train` pairs that survive cleaning are kept, so the subset has exactly that size
    without tokenising all 1.6M pairs. Training pairs also need at least `min_len` English
    tokens: the corpus is full of one-word UI strings, while the dev/test news sentences
    average more than 20 tokens. Dev and test are cleaned in full (no ratio filter,
    so the evaluation sets stay as close to the official ones as possible).

    Returns (train, dev, test, stats); each split is a list of dicts with keys
    en_tokens, hi_tokens, en_text, hi_text.
    """
    cache = _DATA_ROOT / f"splits_n{max_train}_min{min_len}_len{max_len}_seed{seed}.json"
    if cache.exists():
        blob = load_json(cache)
        print(f"  loaded cached splits from {cache.name}")
        return blob["train"], blob["dev"], blob["test"], blob["stats"]

    ds = _load_hf_dataset()
    train_raw = ds["train"]
    n_raw = len(train_raw)

    print("cleaning and tokenising ...")
    rng = random.Random(seed)
    order = list(range(n_raw))
    rng.shuffle(order)
    train, scanned = [], 0
    for i in order:
        scanned += 1
        item = train_raw[i]["translation"]
        pair = _clean_pair(item["en"], item["hi"], 1, max_len, 3.0)
        if pair and len(pair["en_tokens"]) >= min_len:
            train.append(pair)
            if max_train is not None and len(train) >= max_train:
                break

    dev  = _clean_split(ds["validation"], 1, max_len, None)
    test = _clean_split(ds["test"],       1, max_len, None)

    en_lens = [len(p["en_tokens"]) for p in train]
    hi_lens = [len(p["hi_tokens"]) for p in train]
    stats = {
        "train_pairs": len(train),
        "dev_pairs":   len(dev),
        "test_pairs":  len(test),
        "total_raw_train": n_raw,
        "raw_train_scanned": scanned,
        "max_en_len_train": max(en_lens),
        "max_hi_len_train": max(hi_lens),
        "avg_en_len_train": sum(en_lens) / len(en_lens),
        "avg_hi_len_train": sum(hi_lens) / len(hi_lens),
        "max_len_filter": max_len,
        "max_len_ratio_filter_train": 3.0,
        "min_en_len_filter_train": min_len,
        "max_train_subsample": max_train,
        "sampling_seed": seed,
    }
    print(
        f"  train: {stats['train_pairs']:,} pairs  |  "
        f"dev: {stats['dev_pairs']:,}  |  test: {stats['test_pairs']:,}"
    )
    save_json({"train": train, "dev": dev, "test": test, "stats": stats}, cache)
    return train, dev, test, stats


def build_vocabs(train_pairs: list[dict],
                 min_freq: int = 2,
                 max_size: int | None = None) -> tuple[Vocab, Vocab, dict]:
    """Build English and Hindi vocabularies from training data only.

    Returns (vocab_en, vocab_hi, vocab_stats).
    """
    vocab_en = Vocab.build(
        (p["en_tokens"] for p in train_pairs),
        min_freq=min_freq, max_size=max_size,
    )
    vocab_hi = Vocab.build(
        (p["hi_tokens"] for p in train_pairs),
        min_freq=min_freq, max_size=max_size,
    )
    vocab_stats = {
        "en_vocab_size": len(vocab_en),
        "hi_vocab_size": len(vocab_hi),
        "en_min_freq": min_freq,
        "hi_min_freq": min_freq,
    }
    print(f"  English vocab: {len(vocab_en):,}  |  Hindi vocab: {len(vocab_hi):,}")
    return vocab_en, vocab_hi, vocab_stats


class NMTDataset(Dataset):
    """PyTorch dataset for the NMT task.

    Each item returns a dict with:
        src:      list[int]  — English token IDs with <sos> and <eos>
        tgt:      list[int]  — Hindi token IDs with <sos> and <eos>
        src_text: str        — original English sentence (for error analysis)
        tgt_text: str        — original Hindi sentence (for error analysis)
        src_len:  int        — length of src in tokens (excluding padding)
    """

    def __init__(self, pairs: list[dict], vocab_en: Vocab, vocab_hi: Vocab,
                 max_len: int | None = None):
        self.pairs = pairs
        self.vocab_en = vocab_en
        self.vocab_hi = vocab_hi
        self.max_len = max_len

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> dict:
        pair = self.pairs[idx]
        src_ids = self.vocab_en.encode(
            pair["en_tokens"], add_sos=True, add_eos=True, max_len=self.max_len
        )
        tgt_ids = self.vocab_hi.encode(
            pair["hi_tokens"], add_sos=True, add_eos=True, max_len=self.max_len
        )
        return {
            "src": src_ids,
            "tgt": tgt_ids,
            "src_text": pair["en_text"],
            "tgt_text": pair["hi_text"],
            "src_len": len(src_ids),
        }


def save_data_report(stats: dict, vocab_stats: dict,
                     vocab_en: Vocab, vocab_hi: Vocab,
                     dev_pairs: list[dict], test_pairs: list[dict],
                     out_dir: Path) -> Path:
    """Write a JSON report with all the data preparation statistics."""
    report = {
        **stats,
        **vocab_stats,
        "en_oov_rate_dev":  vocab_en.oov_rate(p["en_tokens"] for p in dev_pairs),
        "hi_oov_rate_dev":  vocab_hi.oov_rate(p["hi_tokens"] for p in dev_pairs),
        "en_oov_rate_test": vocab_en.oov_rate(p["en_tokens"] for p in test_pairs),
        "hi_oov_rate_test": vocab_hi.oov_rate(p["hi_tokens"] for p in test_pairs),
    }
    path = save_json(report, out_dir / "data_stats.json")
    print(f"  data report saved to {path}")
    return path
