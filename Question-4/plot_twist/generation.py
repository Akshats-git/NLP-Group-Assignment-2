"""Story generation on top of common.generate, with the decoding configurations of Part 3.

Every sampled run uses a generator seeded from (base seed, prompt key, decoding label), so a story
can be regenerated exactly from its record in generations.jsonl and both models see the same random
numbers for the same prompt and configuration.
"""

import zlib

import torch

from common import Decoding, Vocab, generate, make_generator
from .data import detokenize, encode_prompt, sentence_case, tokenize

# Part 3: greedy, temperature (0.5 / 0.8 / 1.2) and top-k (10, 40). Temperature is 1.0 for the top-k configs.
DECODING_CONFIGS = [
    Decoding("greedy"),
    Decoding("sample", temperature=0.5),
    Decoding("sample", temperature=0.8),
    Decoding("sample", temperature=1.2),
    Decoding("sample", top_k=10),
    Decoding("sample", top_k=40),
]


def config_by_label(label: str) -> Decoding:
    for cfg in DECODING_CONFIGS:
        if cfg.label == label:
            return cfg
    raise KeyError(f"unknown decoding label {label!r}; known: {[c.label for c in DECODING_CONFIGS]}")


def derive_seed(base_seed: int, *keys: str) -> int:
    """Stable 31-bit seed from a base seed and string keys (python's hash() is salted per process, crc32 is not)."""
    return (base_seed * 1_000_003 + zlib.crc32("|".join(keys).encode("utf-8"))) % (2**31)


def finish(vocab: Vocab, new_ids: torch.Tensor, finished: bool) -> dict:
    """One generated row -> tokens (up to <eos>), text, and whether it ended by itself."""
    tokens = vocab.decode(new_ids)  # stops at <eos>, drops specials
    return {"tokens": tokens, "n_tokens": len(tokens), "finished": bool(finished)}


@torch.no_grad()
def continue_story(model, vocab: Vocab, prompt_ids: torch.Tensor, decoding: Decoding, seed: int,
                   max_new_tokens: int, device) -> list[dict]:
    """Continue every row of prompt_ids [batch, t] (all the same length). Returns one dict per row."""
    prompt_ids = prompt_ids.to(device)
    generator = make_generator(seed, device) if decoding.strategy == "sample" else None
    out = generate(model.step_fn(), prompt_ids, max_new_tokens=max_new_tokens, eos_id=vocab.eos_id, pad_id=vocab.pad_id,
                   decoding=decoding, generator=generator)
    return [finish(vocab, out.new_tokens[i], out.finished[i].item()) for i in range(prompt_ids.size(0))]


def generate_from_text(model, vocab: Vocab, prompt: str, decoding: Decoding, n: int, seed: int,
                       max_new_tokens: int, device) -> list[dict]:
    """n continuations of a text prompt. Greedy is deterministic, so callers should ask for n=1."""
    ids = encode_prompt(vocab, prompt).repeat(n, 1)
    rows = continue_story(model, vocab, ids, decoding, seed, max_new_tokens, device)
    for row in rows:
        row["text"] = sentence_case(detokenize(row["tokens"]))
    return rows


def prompt_text_for_display(prompt: str) -> str:
    """The prompt as the model saw it (lowercased, re-spaced), so readers can tell what the <unk> check covers."""
    return sentence_case(detokenize(tokenize(prompt)))
