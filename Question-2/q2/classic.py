"""Classic pipeline for the BiGRU baseline.

    raw text -> strip HTML / URLs -> NFC + lowercase -> spaCy tokens -> Vocab ids (max 200)
             -> GloVe 6B 300-d embedding matrix

Lowercasing matches GloVe 6B, which is an uncased vocabulary. Only spaCy's rule-based
English tokenizer is used (spacy.blank("en")), not a full pipeline: no tagger or parser is
needed and it tokenises the 200k training reviews in about a minute.
"""

import os
import zipfile
from pathlib import Path
from urllib.request import urlretrieve

import numpy as np
import spacy
import torch
from datasets import DatasetDict
from torch.utils.data import DataLoader

from common import Vocab, data_dir, dataloader_kwargs, normalize, pad_collate, strip_html_and_urls
from common.vocab import SPECIALS

MAX_LEN = 200
MIN_FREQ = 2
GLOVE_DIM = 300
GLOVE_URL = "https://downloads.cs.stanford.edu/nlp/data/glove.6B.zip"  # nlp.stanford.edu/data/ often refuses connections

_tokenizer = None


def clean(text: str) -> str:
    return normalize(strip_html_and_urls(text), lower=True)


def tokenize(texts) -> list[list[str]]:
    global _tokenizer
    if _tokenizer is None:
        _tokenizer = spacy.blank("en").tokenizer
    docs = _tokenizer.pipe((clean(t) for t in texts), batch_size=1000)
    return [[token.text for token in doc if not token.is_space] for doc in docs]


def add_tokens(reviews: DatasetDict, num_proc: int | None = None) -> DatasetDict:
    """Adds a `tokens` column (full length, not yet truncated) to every split."""
    return reviews.map(lambda batch: {"tokens": tokenize(batch["text"])}, batched=True,
                       num_proc=num_proc, desc="spaCy tokenising")


def build_vocab(train_tokens, min_freq: int = MIN_FREQ) -> Vocab:
    """From the training split only."""
    return Vocab.build(train_tokens, min_freq=min_freq)


def encode(reviews: DatasetDict, vocab: Vocab, max_len: int = MAX_LEN) -> DatasetDict:
    """Adds `ids` (truncated to max_len, head kept so the title survives) and `n_tokens` (before truncation)."""
    def to_ids(batch):
        return {"ids": [vocab.encode(t, max_len=max_len) for t in batch["tokens"]],
                "n_tokens": [len(t) for t in batch["tokens"]]}
    return reviews.map(to_ids, batched=True, desc="encoding")


def find_glove(dim: int = GLOVE_DIM, path: str | Path | None = None, download: bool = True) -> Path:
    """Locate glove.6B.<dim>d.txt, downloading the Stanford zip (822 MB) only as a last resort.

    Looks at, in order: `path`, $Q2_GLOVE, Question-2/data/glove/, and any attached Kaggle
    dataset under /kaggle/input/. Attaching a Kaggle GloVe dataset is much faster than downloading.
    """
    name = f"glove.6B.{dim}d.txt"
    candidates = [path, os.environ.get("Q2_GLOVE"), data_dir(2) / "glove" / name]
    candidates += sorted(Path("/kaggle/input").glob(f"**/{name}")) if Path("/kaggle/input").exists() else []
    for candidate in candidates:
        if candidate and Path(candidate).is_file():
            return Path(candidate)
    if not download:
        raise FileNotFoundError(f"{name} not found; set Q2_GLOVE or attach a GloVe dataset")
    target = data_dir(2) / "glove"
    target.mkdir(parents=True, exist_ok=True)
    archive = target / "glove.6B.zip"
    if not archive.exists():
        print(f"downloading {GLOVE_URL} ...")
        urlretrieve(GLOVE_URL, archive)
    with zipfile.ZipFile(archive) as zf:
        zf.extract(name, target)
    return target / name


def load_glove(path: str | Path, words) -> dict[str, np.ndarray]:
    """Vectors for the given words only, so the 400k-word file never sits in memory."""
    words = set(words)
    vectors = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            word, _, rest = line.rstrip("\n").partition(" ")
            if word in words:
                vectors[word] = np.asarray(rest.split(" "), dtype=np.float32)
    return vectors


def embedding_matrix(vocab: Vocab, vectors: dict[str, np.ndarray], seed: int = 0) -> tuple[torch.Tensor, dict]:
    """[len(vocab), dim] matrix: GloVe rows where available, N(0, glove std) otherwise, zeros for <pad>.

    Also returns coverage: the share of vocabulary types with a GloVe vector, and the
    share of training tokens they account for (the more telling number).
    """
    found = np.stack(list(vectors.values()))
    rng = np.random.default_rng(seed)
    matrix = rng.normal(0.0, found.std(), (len(vocab), found.shape[1])).astype(np.float32)
    regular = vocab.itos[len(SPECIALS):]
    counts = getattr(vocab, "counts", {})
    hit_types = hit_tokens = 0
    for word in regular:
        if word in vectors:
            matrix[vocab.stoi[word]] = vectors[word]
            hit_types += 1
            hit_tokens += counts.get(word, 0)
    matrix[vocab.pad_id] = 0.0
    total_tokens = sum(counts.get(w, 0) for w in regular)
    coverage = {
        "vocab_size": len(vocab),
        "types_with_vector": hit_types,
        "type_coverage": hit_types / len(regular),
        "token_coverage": hit_tokens / total_tokens if total_tokens else None,
    }
    return torch.from_numpy(matrix), coverage


def make_loader(split, batch_size: int, shuffle: bool = False, seed: int = 0, num_workers: int = 2) -> DataLoader:
    """Batches of {"ids": [batch, time], "ids_lengths": [batch], "label": [batch]}, padded to the longest review."""
    split = split.select_columns(["ids", "label"])
    extra = dataloader_kwargs(seed) if shuffle else {}
    return DataLoader(split, batch_size=batch_size, shuffle=shuffle, collate_fn=pad_collate(0),  # <pad> is always id 0
                      num_workers=num_workers, pin_memory=torch.cuda.is_available(), **extra)
