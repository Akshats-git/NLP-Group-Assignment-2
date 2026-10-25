"""RoBERTa pipeline: byte-level BPE with RobertaTokenizerFast, <s> ... </s>, truncation and dynamic padding.

Texts are tokenised once without padding. Each batch is then padded only to its longest
review by DataCollatorWithPadding, which is much cheaper than padding everything to 256
because most reviews are far shorter.
"""

import torch
from datasets import DatasetDict
from torch.utils.data import DataLoader
from transformers import DataCollatorWithPadding, RobertaTokenizerFast

from common import dataloader_kwargs

from .batching import LengthGroupedBatchSampler

MODEL_NAME = "roberta-base"
MAX_LEN = 256


def get_tokenizer(name: str = MODEL_NAME) -> RobertaTokenizerFast:
    return RobertaTokenizerFast.from_pretrained(name)


def tokenize(reviews: DatasetDict, tokenizer: RobertaTokenizerFast, max_length: int = MAX_LEN,
             num_proc: int | None = None) -> DatasetDict:
    """Adds input_ids / attention_mask (at most max_length, still ending in </s>) and n_tokens (before truncation).

    Truncation is done by hand rather than with truncation=True so the untruncated
    length comes for free, which is what the length statistics in the report need.
    """
    eos = tokenizer.eos_token_id

    def encode(batch):
        ids = tokenizer(batch["text"], verbose=False)["input_ids"]
        n_tokens = [len(x) for x in ids]
        ids = [x if len(x) <= max_length else x[: max_length - 1] + [eos] for x in ids]
        return {"input_ids": ids, "attention_mask": [[1] * len(x) for x in ids], "n_tokens": n_tokens}

    return reviews.map(encode, batched=True, num_proc=num_proc, desc="RoBERTa tokenising")


def collator(tokenizer: RobertaTokenizerFast) -> DataCollatorWithPadding:
    # multiples of 8 keep fp16 / bf16 matmuls on tensor cores
    return DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8)


def make_loader(split, tokenizer: RobertaTokenizerFast, batch_size: int, shuffle: bool = False,
                seed: int = 0, num_workers: int = 2) -> DataLoader:
    """Batches of {"input_ids", "attention_mask", "labels"}, padded per batch.

    shuffle=True draws seeded, length-grouped batches (see q2.batching); otherwise the
    split's own order is kept so predictions line up with review ids.
    """
    lengths = [len(ids) for ids in split["input_ids"]]
    split = split.select_columns(["input_ids", "attention_mask", "label"]).rename_column("label", "labels")
    kwargs = dict(collate_fn=collator(tokenizer), num_workers=num_workers, pin_memory=torch.cuda.is_available())
    if shuffle:
        sampler = LengthGroupedBatchSampler(lengths, batch_size, seed)
        return DataLoader(split, batch_sampler=sampler, worker_init_fn=dataloader_kwargs(seed)["worker_init_fn"], **kwargs)
    return DataLoader(split, batch_size=batch_size, **kwargs)
