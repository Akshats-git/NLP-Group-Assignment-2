"""Autoregressive generation shared by Q1 (translation), Q3 (summaries) and Q4 (stories).

The model is wrapped in a step function so one loop works for an RNN with a hidden
state and for a Transformer that re-reads the whole prefix:

    def step_fn(tokens, state):
        # tokens: [batch, t] everything generated so far (prompt included)
        # state:  whatever step_fn returned last time, None on the first call
        return logits, state             # logits: [batch, vocab] for the next token
        # or: return logits, state, extra  (e.g. this step's attention weights)

An LSTM language model (Q4):

    def step_fn(tokens, state):
        inputs = tokens if state is None else tokens[:, -1:]   # whole prompt once, then one token
        out, state = model.lstm(model.embed(inputs), state)
        return model.fc(out[:, -1]), state

Call model.eval() first; generate() already runs under torch.no_grad().
"""

from dataclasses import asdict, dataclass

import torch


@dataclass(frozen=True)
class Decoding:
    """One decoding configuration. Its label and to_dict() go into results and the report.

        Decoding()                                    greedy
        Decoding("sample", temperature=0.8)           temperature sampling
        Decoding("sample", top_k=40)                  top-k sampling
        Decoding("sample", temperature=0.8, top_k=40) both
    """

    strategy: str = "greedy"
    temperature: float = 1.0
    top_k: int | None = None
    top_p: float | None = None

    def __post_init__(self):
        if self.strategy not in ("greedy", "sample"):
            raise ValueError("strategy must be 'greedy' or 'sample'")
        if self.temperature <= 0:
            raise ValueError("temperature must be > 0 (use strategy='greedy' for argmax)")

    @property
    def label(self) -> str:
        if self.strategy == "greedy":
            return "greedy"
        parts = []
        if self.temperature != 1.0 or (self.top_k is None and self.top_p is None):
            parts.append(f"temp={self.temperature:g}")
        if self.top_k is not None:
            parts.append(f"top-k={self.top_k}")
        if self.top_p is not None:
            parts.append(f"top-p={self.top_p:g}")
        return ", ".join(parts)

    def to_dict(self) -> dict:
        return asdict(self)


def top_k_filter(logits: torch.Tensor, k: int | None) -> torch.Tensor:
    """Keep the k highest logits per row, set the rest to -inf."""
    if k is None or k <= 0 or k >= logits.size(-1):
        return logits
    kth = torch.topk(logits, k, dim=-1).values[..., -1:]
    return logits.masked_fill(logits < kth, float("-inf"))


def top_p_filter(logits: torch.Tensor, p: float | None) -> torch.Tensor:
    """Nucleus filtering: keep the smallest set of tokens whose probability adds up to p."""
    if p is None or p >= 1.0:
        return logits
    sorted_logits, order = torch.sort(logits, descending=True, dim=-1)
    probs = sorted_logits.softmax(dim=-1)
    drop_sorted = probs.cumsum(dim=-1) - probs >= p  # the tokens before this one already reach p
    drop = torch.zeros_like(drop_sorted).scatter(-1, order, drop_sorted)
    return logits.masked_fill(drop, float("-inf"))


def next_token(logits: torch.Tensor, decoding: Decoding = Decoding(), generator: torch.Generator | None = None) -> torch.Tensor:
    """Pick the next token for each row of [batch, vocab] logits. Returns [batch] ids.

    A seeded generator must live on the same device as the logits.
    """
    if decoding.strategy == "greedy":
        return logits.argmax(dim=-1)
    logits = logits.float() / decoding.temperature
    logits = top_k_filter(logits, decoding.top_k)
    logits = top_p_filter(logits, decoding.top_p)
    probs = logits.softmax(dim=-1)
    return torch.multinomial(probs, num_samples=1, generator=generator).squeeze(-1)


@dataclass
class GenerationOutput:
    tokens: torch.Tensor  # [batch, prompt + generated]
    new_tokens: torch.Tensor  # [batch, generated]; after <eos> rows are filled with pad_id
    finished: torch.Tensor  # [batch] bool, True if the row produced <eos>
    extras: list  # one entry per step if step_fn returned a third value, e.g. attention


@torch.no_grad()
def generate(
    step_fn,
    prompt: torch.Tensor,
    max_new_tokens: int,
    eos_id: int | None = None,
    pad_id: int = 0,
    decoding: Decoding = Decoding(),
    state=None,
    generator: torch.Generator | None = None,
) -> GenerationOutput:
    """Generate up to max_new_tokens after prompt ([batch, t] ids, e.g. just <sos>).

    Stops early once every row has produced eos_id. For Q1 / Q3, pass the encoder
    output in through `state` (or a closure) so the source is encoded only once.
    """
    if prompt.ndim != 2:
        raise ValueError("prompt must be [batch, time]")
    tokens = prompt
    finished = torch.zeros(prompt.size(0), dtype=torch.bool, device=prompt.device)
    extras = []
    for _ in range(max_new_tokens):
        out = step_fn(tokens, state)
        logits, state = out[0], out[1]
        if len(out) > 2:
            extras.append(out[2])
        nxt = next_token(logits, decoding, generator)
        if eos_id is not None:
            nxt = nxt.masked_fill(finished, pad_id)
        tokens = torch.cat([tokens, nxt.unsqueeze(1)], dim=1)
        if eos_id is not None:
            finished |= nxt == eos_id
            if finished.all():
                break
    return GenerationOutput(tokens=tokens, new_tokens=tokens[:, prompt.size(1):], finished=finished, extras=extras)
