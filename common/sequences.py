"""Padding, masks and batching for token-id sequences.

Mask convention used everywhere in this repo: True means "real token / allowed to
attend", False means "padding / blocked". To turn a mask into attention scores:

    scores = scores.masked_fill(~mask, torch.finfo(scores.dtype).min)

Use finfo.min rather than float("-inf"): a fully masked row of -inf gives NaN after
softmax, and -inf misbehaves under fp16.
"""

from collections.abc import Sequence

import torch


def pad_sequences(
    sequences: Sequence[Sequence[int]],
    pad_id: int,
    max_len: int | None = None,
    side: str = "right",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pad a list of id lists (or 1-D tensors) into a [batch, time] LongTensor.

    Returns (ids, lengths). Sequences longer than max_len are cut from the end.
    side="left" puts the padding first, which is handy for batched generation.
    """
    if side not in ("right", "left"):
        raise ValueError("side must be 'right' or 'left'")
    seqs = [list(s.tolist() if hasattr(s, "tolist") else s) for s in sequences]
    if max_len is not None:
        seqs = [s[:max_len] for s in seqs]
    lengths = torch.tensor([len(s) for s in seqs], dtype=torch.long)
    width = int(lengths.max()) if len(seqs) else 0
    ids = torch.full((len(seqs), width), pad_id, dtype=torch.long)
    for row, seq in enumerate(seqs):
        if not seq:
            continue
        if side == "right":
            ids[row, : len(seq)] = torch.tensor(seq, dtype=torch.long)
        else:
            ids[row, width - len(seq):] = torch.tensor(seq, dtype=torch.long)
    return ids, lengths


def lengths_to_mask(lengths: torch.Tensor, max_len: int | None = None) -> torch.Tensor:
    """[batch] lengths -> [batch, time] bool mask, True on real (right-padded) tokens."""
    max_len = max_len or int(lengths.max())
    positions = torch.arange(max_len, device=lengths.device)
    return positions.unsqueeze(0) < lengths.unsqueeze(1)


def padding_mask(ids: torch.Tensor, pad_id: int) -> torch.Tensor:
    """[batch, time] bool mask, True where the token is not padding."""
    return ids != pad_id


def causal_mask(size: int, device: str | torch.device | None = None) -> torch.Tensor:
    """[time, time] bool mask, True where position i may look at position j (j <= i)."""
    return torch.tril(torch.ones(size, size, dtype=torch.bool, device=device))


def teacher_forcing_pair(ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Split padded <sos> ... <eos> targets into decoder input and prediction target.

    input  = ids[:, :-1]  (starts with <sos>, i.e. the summary shifted right)
    target = ids[:, 1:]   (ends with <eos>,  i.e. the summary shifted left)
    """
    return ids[:, :-1], ids[:, 1:]


def pad_collate(pad_id: int, max_len: int | None = None, side: str = "right"):
    """collate_fn for a Dataset that returns dicts.

    Every value that is a list of ints (or a 1-D tensor) gets padded, and its lengths
    come back under "<key>_lengths". Numbers are stacked into a tensor, anything else
    (e.g. the raw text, kept for error analysis) is returned as a list.

        loader = DataLoader(ds, batch_size=64, collate_fn=pad_collate(vocab.pad_id))
        batch["src"], batch["src_lengths"], batch["tgt"], batch["text"]
    """
    def collate(examples: list[dict]) -> dict:
        batch = {}
        for key in examples[0]:
            values = [example[key] for example in examples]
            first = values[0]
            is_id_list = isinstance(first, (list, tuple)) and all(isinstance(v, int) for v in first)
            if is_id_list or (isinstance(first, torch.Tensor) and first.ndim == 1):
                batch[key], batch[f"{key}_lengths"] = pad_sequences(values, pad_id, max_len, side)
            elif isinstance(first, torch.Tensor):
                batch[key] = torch.stack(values)
            elif isinstance(first, bool) or not isinstance(first, (int, float)):
                batch[key] = values
            else:
                batch[key] = torch.tensor(values)
        return batch

    return collate
