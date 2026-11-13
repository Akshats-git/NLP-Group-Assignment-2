"""CNN/DailyMail loading, tokenisation and batching for Question 3.

Pipeline for one example:
    article    -> lowercase -> WordPiece ids [CLS] ... [SEP] -> first 400 tokens (always ends with [SEP])
    highlights -> lowercase, newline -> [NL] -> ids [CLS] ... [SEP] -> first 128 tokens

[CLS] is the start token and [SEP] the end token. The extra [NL] token marks the line
breaks between highlight sentences, so the model learns to produce sentence breaks and
its output can be split into lines for ROUGE-Lsum. Articles and summaries share one
vocabulary (bert-base-uncased, 30,522 WordPiece tokens + [NL]).
"""

from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm.auto import tqdm

DATASET_NAME = "abisee/cnn_dailymail"   # the current Hugging Face home of "cnn_dailymail"
DATASET_VERSION = "3.0.0"
TOKENIZER_NAME = "bert-base-uncased"    # uncased WordPiece, so lowercasing matches the tokenizer
NEWLINE_TOKEN = "[NL]"


def load_cnn_dailymail(cache_dir=None):
    from datasets import load_dataset
    return load_dataset(DATASET_NAME, DATASET_VERSION, cache_dir=cache_dir)


def load_tokenizer(name=TOKENIZER_NAME):
    """Pre-trained tokenizer (allowed by the brief) plus one new token for line breaks."""
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(name)
    tokenizer.add_special_tokens({"additional_special_tokens": [NEWLINE_TOKEN]})
    tokenizer.model_max_length = 10**6   # we truncate ourselves; this silences the 512-token warning
    return tokenizer


@dataclass(frozen=True)
class SpecialIds:
    pad: int
    unk: int
    bos: int   # [CLS], start of every summary
    eos: int   # [SEP], end of every summary
    nl: int    # [NL], sentence break inside a summary


def special_ids(tokenizer) -> SpecialIds:
    ids = SpecialIds(pad=tokenizer.pad_token_id, unk=tokenizer.unk_token_id, bos=tokenizer.cls_token_id,
                     eos=tokenizer.sep_token_id, nl=tokenizer.convert_tokens_to_ids(NEWLINE_TOKEN))
    assert ids.pad == 0, "the arrays below use 0 as padding"
    assert len(tokenizer) < 2**16, "ids are stored as uint16"
    return ids


def prepare_article(text: str) -> str:
    return text.lower()


def prepare_summary(text: str) -> str:
    """Lowercase and turn the newline between highlight sentences into the [NL] token."""
    lines = [line.strip() for line in text.lower().split("\n") if line.strip()]
    return f" {NEWLINE_TOKEN} ".join(lines)


def truncate(ids: list[int], max_len: int) -> list[int]:
    """Keep the first max_len tokens but always keep the final [SEP] (end token)."""
    return ids if len(ids) <= max_len else ids[: max_len - 1] + ids[-1:]


def encode_texts(split, field, prepare, tokenizer, max_len, chunk_size=2000, desc=None):
    """Tokenise one column of a dataset split.

    Returns (ids, full_lengths): ids is a [N, max_len] uint16 array padded with 0, and
    full_lengths the token counts before truncation (for the statistics in the report).
    """
    n = len(split)
    ids = np.zeros((n, max_len), dtype=np.uint16)
    full_lengths = np.zeros(n, dtype=np.int32)
    for start in tqdm(range(0, n, chunk_size), desc=desc, leave=False):
        texts = [prepare(t) for t in split[start:start + chunk_size][field]]
        for row, seq in enumerate(tokenizer(texts)["input_ids"], start=start):   # adds [CLS] ... [SEP]
            full_lengths[row] = len(seq)
            seq = truncate(seq, max_len)
            ids[row, : len(seq)] = seq
    return ids, full_lengths


def length_stats(full_lengths, limit) -> dict:
    full_lengths = np.asarray(full_lengths)
    return {
        "median": float(np.median(full_lengths)),
        "p90": float(np.percentile(full_lengths, 90)),
        "max": int(full_lengths.max()),
        "limit": int(limit),
        "pct_truncated": float(100 * (full_lengths > limit).mean()),
    }


class SummaryDataset(Dataset):
    """Rows of the padded id arrays. Index i gives (article ids, summary ids)."""

    def __init__(self, src: np.ndarray, tgt: np.ndarray):
        assert len(src) == len(tgt)
        self.src, self.tgt = src, tgt

    def __len__(self):
        return len(self.src)

    def __getitem__(self, i):
        return self.src[i], self.tgt[i]


def trim_padding(ids: torch.Tensor, pad_id: int = 0) -> torch.Tensor:
    """Drop the columns that are padding in every row (sequences are right-padded)."""
    longest = int((ids != pad_id).sum(dim=1).max())
    return ids[:, : max(longest, 1)]


def collate(batch):
    """Stack a list of (src, tgt) rows into two LongTensors, padded only to the longest row."""
    src = torch.from_numpy(np.stack([row[0] for row in batch]).astype(np.int64))
    tgt = torch.from_numpy(np.stack([row[1] for row in batch]).astype(np.int64))
    return trim_padding(src), trim_padding(tgt)


def ids_to_text(ids, tokenizer, special: SpecialIds) -> str:
    """Generated ids -> summary text with one sentence per line (needed for ROUGE-Lsum)."""
    ids = ids.tolist() if hasattr(ids, "tolist") else list(ids)
    if special.eos in ids:
        ids = ids[: ids.index(special.eos)]                    # stop at the end token
    sentences, current = [], []
    for i in ids:
        if i == special.nl:
            sentences.append(current)
            current = []
        elif i not in (special.pad, special.bos):
            current.append(i)
    sentences.append(current)
    lines = [tokenizer.decode(s, skip_special_tokens=True, clean_up_tokenization_spaces=True).strip()
             for s in sentences]
    return "\n".join(line for line in lines if line)


def tokens_for_plot(ids, tokenizer) -> list[str]:
    """WordPiece tokens as axis labels for the attention heatmap."""
    ids = ids.tolist() if hasattr(ids, "tolist") else list(ids)
    return tokenizer.convert_ids_to_tokens(ids)
