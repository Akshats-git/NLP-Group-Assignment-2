"""TinyStories: subset selection, cleaning, word-level tokenisation, vocabulary and batching.

Every step that touches the corpus lives here so that both language models, the generation
code and the evaluation all see exactly the same tokens.

Splits are made at the story level before any sequence is built: a story belongs to
exactly one of train / validation / test, and every training sequence is one complete
story (<sos> ... <eos>), so no window of a story can leak into another split.
"""

import json
import random
import re
from collections import Counter
from collections.abc import Iterable, Iterator
from pathlib import Path

import torch
from torch.utils.data import Dataset, Sampler

from common import Vocab, normalize, pad_sequences

# Word-level tokens: words (keeping contractions like "didn't" whole), numbers, and each punctuation mark.
TOKEN_RE = re.compile(r"[a-z0-9]+(?:'[a-z]+)*|[^\sa-z0-9]")
# Some TinyStories files were saved with a double encoding: a curly quote (U+201C) shows up as the three characters
# "â€œ". About 6% of stories contain it, and a model trained on it prints "â € œ" in its stories, so it is repaired first.
_MOJIBAKE = [
    ("\u00e2\u20ac\u0153", '"'), ("\u00e2\u20ac\u009d", '"'), ("\u00e2\u20ac\u2122", "'"), ("\u00e2\u20ac\u02dc", "'"),
    ("\u00e2\u20ac\u201c", "-"), ("\u00e2\u20ac\u201d", " - "), ("\u00e2\u20ac\u00a6", "..."),
    ("\u00e2\u20ac", '"'),  # a closing quote whose last byte (0x9d) was lost
]
_CHAR_FIXES = {
    "\u2018": "'", "\u2019": "'", "\u201a": "'", "\u201b": "'",
    "\u201c": '"', "\u201d": '"', "\u201e": '"',
    "\u2013": "-", "\u2014": " - ", "\u2026": "...",
}
_CHAR_TABLE = str.maketrans({k: v for k, v in _CHAR_FIXES.items()})
_NON_ASCII = re.compile(r"[^\x00-\x7f]")


def clean(text: str) -> str:
    """Repair mojibake, straighten quotes and dashes, drop any other non-ASCII character, lowercase, collapse whitespace."""
    for bad, good in _MOJIBAKE:
        text = text.replace(bad, good)
    return normalize(_NON_ASCII.sub(" ", text.translate(_CHAR_TABLE)), lower=True)


def tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(clean(text))


_NO_SPACE_BEFORE = {".", ",", "!", "?", ";", ":", "...", ")"}


def detokenize(tokens: Iterable[str]) -> str:
    """Tokens back to readable text: no space before punctuation, quotes hug the quoted words.

    Used for display and for the human-evaluation sheets only. Metrics are computed on tokens.
    """
    out: list[str] = []
    in_quote = False
    glue_next = False
    for tok in tokens:
        if tok == '"':
            if not in_quote:  # opening quote: space before it, none after
                out.append(('' if not out else ' ') + tok)
                glue_next = True
            else:
                out.append(tok)
                glue_next = False
            in_quote = not in_quote
            continue
        if tok in _NO_SPACE_BEFORE or glue_next or not out:
            out.append(tok)
        else:
            out.append(" " + tok)
        glue_next = False
    return "".join(out)


# Capitalised in the corpus because they open a quotation, not because they are names.
_INTERJECTIONS = {"achoo", "ahoy", "ew", "eww", "excuse", "hey", "hi", "hmm", "mmm", "oh", "oops", "ouch", "uh", "woah",
                  "wow", "yay", "yeah", "yuck", "who's"}


def find_proper_nouns(raw_stories: Iterable[str], min_count: int = 5, min_share: float = 0.9) -> list[str]:
    """Words that are almost always capitalised in the middle of a sentence in the raw training text (names, places).

    The models work on lowercase tokens; this list only restores capitals when a story is shown to a reader.
    """
    upper, total = Counter(), Counter()
    for story in raw_stories:
        text = _NON_ASCII.sub(" ", story)
        for m in re.finditer(r"[A-Za-z]+(?:'[a-z]+)?", text):
            before = text[:m.start()].rstrip().rstrip('"\'')
            if not before or before[-1] in ".!?":
                continue  # sentence start: a capital tells nothing
            word = m.group(0)
            total[word.lower()] += 1
            upper[word.lower()] += word[0].isupper()
    return sorted(w for w, n in total.items() if n >= min_count and upper[w] / n >= min_share and w != "i" and w not in _INTERJECTIONS)


_DISPLAY_CASE_FILE = Path(__file__).resolve().parents[1] / "outputs" / "display_case.json"
_proper_nouns: frozenset | None = None


def proper_nouns() -> frozenset:
    """The names to capitalise on display, from outputs/display_case.json (written by prepare_data.py)."""
    global _proper_nouns
    if _proper_nouns is None:
        _proper_nouns = frozenset(json.loads(_DISPLAY_CASE_FILE.read_text())) if _DISPLAY_CASE_FILE.exists() else frozenset()
    return _proper_nouns


def _show(word: str) -> str:
    return "TV" if word == "tv" else word.capitalize()


def sentence_case(text: str) -> str:
    """Restore capitals for display: sentence starts (also after a closing quote that ends a sentence), 'I', names."""
    text = re.sub(r'(^|[.!?]\s+|\."\s+|[.!?]\s+")([a-z])', lambda m: m.group(1) + m.group(2).upper(), text)
    text = re.sub(r"(^\"|\s\")([a-z])", lambda m: m.group(1) + m.group(2).upper(), text)
    text = re.sub(r"\bi\b", "I", text)
    names = proper_nouns()
    if names:
        text = re.sub(r"(?<![A-Za-z'])[a-z]+(?:'[a-z]+)?", lambda m: _show(m.group(0)) if m.group(0) in names else m.group(0), text)
    return text


def load_story_subset(
    n_train: int = 20_000,
    n_val: int = 1_000,
    n_test: int = 1_000,
    max_tokens: int = 400,
    seed: int = 42,
    pool_factor: int = 3,
) -> dict[str, list[str]]:
    """Draw the documented subset of TinyStories.

    - train: a seeded sample from the first `pool_factor * n_train` stories of the official train split
      (streamed, so the 2 GB file is never downloaded)
    - validation and test: disjoint, seeded samples from the official validation split, minus any story that
      also appears in the training sample

    Stories longer than `max_tokens` tokens are dropped (not truncated) *before* sampling,
    so every kept story is complete and still has its real ending.
    Returns ({"train": [...], "validation": [...], "test": [...]} of raw story strings, filtering statistics).
    """
    from datasets import load_dataset

    rng = random.Random(seed)
    seen = dropped = 0

    def keep(story: str) -> bool:
        nonlocal seen, dropped
        story = story.strip()
        if not story:
            return False
        seen += 1
        if len(tokenize(story)) > max_tokens:
            dropped += 1
            return False
        return True

    train_pool = []
    stream = load_dataset("roneneldan/TinyStories", split="train", streaming=True)
    for row in stream:
        if keep(row["text"]):
            train_pool.append(row["text"].strip())
        if len(train_pool) >= pool_factor * n_train:
            break
    train_seen, train_dropped = seen, dropped
    seen = dropped = 0
    val_pool = [r["text"].strip() for r in load_dataset("roneneldan/TinyStories", split="validation") if keep(r["text"])]

    rng.shuffle(train_pool)
    rng.shuffle(val_pool)
    train = train_pool[:n_train]
    train_set = set(train)  # TinyStories contains a few exact duplicates; held-out stories must not repeat training ones
    held_out = [s for s in val_pool if s not in train_set]
    duplicates_removed = len(val_pool) - len(held_out)
    if len(train) < n_train or len(held_out) < n_val + n_test:
        raise RuntimeError("not enough stories after filtering; lower the subset sizes")
    splits = {"train": train, "validation": held_out[:n_val], "test": held_out[n_val:n_val + n_test]}
    stats = {"train_pool_stories_seen": train_seen, "train_pool_dropped_over_max_tokens": train_dropped,
             "validation_split_stories_seen": seen, "validation_split_dropped_over_max_tokens": dropped,
             "held_out_duplicates_of_train_removed": duplicates_removed}
    return splits, stats


def build_vocab(train_tokens: Iterable[list[str]], min_freq: int = 5, max_size: int | None = None) -> Vocab:
    """Vocabulary from the training split only. Rarer words become <unk>."""
    return Vocab.build(train_tokens, min_freq=min_freq, max_size=max_size)


class StoryDataset(Dataset):
    """One item per story: ids = <sos> w1 ... wn <eos>.

    The model input is ids[:-1] and the target ids[1:], built in `collate_lm`, so the model
    predicts each next token from all the earlier tokens of the same story.
    """

    def __init__(self, token_lists: list[list[str]], vocab: Vocab):
        self.ids = [vocab.encode(toks, add_sos=True, add_eos=True) for toks in token_lists]

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, i: int) -> list[int]:
        return self.ids[i]


def collate_lm(pad_id: int):
    """Pad a batch of stories on the right; return input, target and the number of real target tokens.

    Right padding is safe for a left-to-right RNN: the padded positions come after every real
    token, so they cannot change any real hidden state. Their targets are <pad>, which the
    loss ignores (ignore_index=pad_id).
    """
    def collate(batch: list[list[int]]) -> dict:
        ids, lengths = pad_sequences(batch, pad_id)
        return {"inputs": ids[:, :-1], "targets": ids[:, 1:], "lengths": lengths - 1}
    return collate


class BucketBatchSampler(Sampler):
    """Batches of similar-length stories, so little compute is wasted on padding.

    Each epoch: shuffle, cut into chunks of `chunk * batch_size`, sort each chunk by length, slice
    it into batches, then shuffle the batches. Fully determined by (seed, epoch).
    """

    def __init__(self, lengths: list[int], batch_size: int, seed: int, chunk: int = 50, shuffle: bool = True):
        self.lengths, self.batch_size, self.seed, self.chunk, self.shuffle = lengths, batch_size, seed, chunk, shuffle
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __iter__(self) -> Iterator[list[int]]:
        order = list(range(len(self.lengths)))
        if not self.shuffle:
            order.sort(key=lambda i: self.lengths[i])
            yield from (order[i:i + self.batch_size] for i in range(0, len(order), self.batch_size))
            return
        rng = random.Random(self.seed * 1000 + self.epoch)
        rng.shuffle(order)
        size = self.chunk * self.batch_size
        batches = []
        for start in range(0, len(order), size):
            block = sorted(order[start:start + size], key=lambda i: self.lengths[i])
            batches += [block[i:i + self.batch_size] for i in range(0, len(block), self.batch_size)]
        rng.shuffle(batches)
        yield from batches

    def __len__(self) -> int:
        return -(-len(self.lengths) // self.batch_size)


def make_loader(dataset: StoryDataset, pad_id: int, batch_size: int, seed: int, shuffle: bool):
    from torch.utils.data import DataLoader

    sampler = BucketBatchSampler([len(x) for x in dataset.ids], batch_size, seed, shuffle=shuffle)
    return DataLoader(dataset, batch_sampler=sampler, collate_fn=collate_lm(pad_id)), sampler


def encode_prompt(vocab: Vocab, text: str, device: str | torch.device = "cpu") -> torch.Tensor:
    """Prompt text -> [1, t] ids starting with <sos> (no <eos>, so the model keeps writing)."""
    return torch.tensor([vocab.encode(tokenize(text), add_sos=True)], device=device)
