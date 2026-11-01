"""Batches of similar-length reviews, so dynamic padding has little to pad.

Review lengths are very skewed (median 38 BPE tokens, 1% over 200), so a random batch of
32 is usually padded to one long review. Grouping by length removes most of that padding
while keeping training random: indices are shuffled, cut into pools of `pool_batches`
batches, each pool is sorted by length and cut into batches, and the batch order is
shuffled again. This is the same idea as `group_by_length` in Hugging Face's Trainer.
"""

import math
from collections.abc import Sequence

import torch
from torch.utils.data import Sampler


class LengthGroupedBatchSampler(Sampler[list[int]]):
    """batch_sampler for a DataLoader. Every pass over it draws a new, seeded shuffle.

    The batch holding the longest review comes first, so an out-of-memory error shows up
    in the first step instead of an hour into training.
    """

    def __init__(self, lengths: Sequence[int], batch_size: int, seed: int, pool_batches: int = 50):
        self.lengths = list(lengths)
        self.batch_size = batch_size
        self.seed = seed
        self.pool_batches = pool_batches
        self.epoch = 0

    def __len__(self) -> int:
        return math.ceil(len(self.lengths) / self.batch_size)

    def __iter__(self):
        generator = torch.Generator().manual_seed(self.seed + self.epoch)
        self.epoch += 1
        order = torch.randperm(len(self.lengths), generator=generator).tolist()
        pool = self.batch_size * self.pool_batches
        batches = []
        for start in range(0, len(order), pool):
            chunk = sorted(order[start:start + pool], key=lambda i: self.lengths[i], reverse=True)
            batches += [chunk[j:j + self.batch_size] for j in range(0, len(chunk), self.batch_size)]
        batches = [batches[i] for i in torch.randperm(len(batches), generator=generator).tolist()]
        longest = max(range(len(batches)), key=lambda b: self.lengths[batches[b][0]])
        batches.insert(0, batches.pop(longest))
        yield from batches
