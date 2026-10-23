"""Word-level vocabulary for the models that do not use a pre-trained tokenizer (Q1, Q2 baseline, Q4).

Special tokens always get the same ids so padding_idx=0 is correct in every embedding:

    <pad> = 0   <unk> = 1   <sos> = 2   <eos> = 3

Q4's BOS/EOS are the same tokens as <sos>/<eos>.
"""

import json
from collections import Counter
from collections.abc import Iterable, Sequence
from pathlib import Path

PAD, UNK, SOS, EOS = "<pad>", "<unk>", "<sos>", "<eos>"
SPECIALS = (PAD, UNK, SOS, EOS)


class Vocab:
    def __init__(self, tokens: Sequence[str]):
        if len(set(tokens)) != len(tokens):
            raise ValueError("vocabulary tokens must be unique")
        self.itos = list(tokens)
        self.stoi = {token: i for i, token in enumerate(self.itos)}

    @classmethod
    def build(
        cls,
        sequences: Iterable[Sequence[str]],
        min_freq: int = 1,
        max_size: int | None = None,
        specials: Sequence[str] = SPECIALS,
    ) -> "Vocab":
        """Build from tokenised training sentences only, never from validation or test.

        Tokens are ordered by frequency, ties broken alphabetically, so two people
        building from the same data get identical ids. max_size counts regular
        tokens only; the specials are always added on top.
        """
        counts = Counter(token for seq in sequences for token in seq)
        for special in specials:
            counts.pop(special, None)
        ranked = sorted((t for t, c in counts.items() if c >= min_freq), key=lambda t: (-counts[t], t))
        if max_size is not None:
            ranked = ranked[:max_size]
        vocab = cls(list(specials) + ranked)
        vocab.counts = counts
        return vocab

    def __len__(self) -> int:
        return len(self.itos)

    def __contains__(self, token: str) -> bool:
        return token in self.stoi

    def __getitem__(self, token: str) -> int:
        return self.stoi.get(token, self.unk_id)

    def _special_id(self, token: str) -> int:
        if token not in self.stoi:
            raise KeyError(f"{token} is not in this vocabulary")
        return self.stoi[token]

    @property
    def pad_id(self) -> int:
        return self._special_id(PAD)

    @property
    def unk_id(self) -> int:
        return self._special_id(UNK)

    @property
    def sos_id(self) -> int:
        return self._special_id(SOS)

    @property
    def eos_id(self) -> int:
        return self._special_id(EOS)

    def encode(
        self,
        tokens: Sequence[str],
        add_sos: bool = False,
        add_eos: bool = False,
        max_len: int | None = None,
    ) -> list[int]:
        """Map tokens to ids. max_len includes <sos>/<eos>; it truncates the content, never the <eos>."""
        ids = [self[token] for token in tokens]
        if max_len is not None:
            ids = ids[: max(0, max_len - add_sos - add_eos)]
        if add_sos:
            ids = [self.sos_id] + ids
        if add_eos:
            ids = ids + [self.eos_id]
        return ids

    def decode(self, ids, skip_special: bool = True, stop_at_eos: bool = True) -> list[str]:
        """Map ids back to tokens. Takes a list or a 1-D tensor."""
        if hasattr(ids, "tolist"):
            ids = ids.tolist()
        eos = self.stoi.get(EOS)
        specials = {self.stoi[s] for s in SPECIALS if s in self.stoi}
        tokens = []
        for i in ids:
            if stop_at_eos and i == eos:
                break
            if skip_special and i in specials:
                continue
            tokens.append(self.itos[i])
        return tokens

    def oov_rate(self, sequences: Iterable[Sequence[str]]) -> float:
        """Share of tokens that would map to <unk>. Worth reporting for the test split."""
        total = unknown = 0
        for seq in sequences:
            total += len(seq)
            unknown += sum(token not in self.stoi for token in seq)
        return unknown / total if total else 0.0

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"itos": self.itos}, ensure_ascii=False), encoding="utf-8")
        return path

    @classmethod
    def load(cls, path: str | Path) -> "Vocab":
        return cls(json.loads(Path(path).read_text(encoding="utf-8"))["itos"])
