"""Perplexity of a language model on a split of stories."""

import math

import torch
import torch.nn.functional as F

from common import perplexity, to_device


@torch.no_grad()
def evaluate_loss(model, loader, device, pad_id: int = 0) -> dict:
    """Token-level negative log-likelihood over real target tokens (padding excluded).

    Returns mean NLL in nats, perplexity = exp(mean NLL), and the token count. Run in fp32.
    <eos> targets are counted, <unk> targets are counted too (the model is scored on predicting <unk>).
    """
    model.eval()
    total_nll, total_tokens = 0.0, 0
    for batch in loader:
        batch = to_device(batch, device)
        logits, _ = model(batch["inputs"])
        nll = F.cross_entropy(logits.reshape(-1, logits.size(-1)).float(), batch["targets"].reshape(-1),
                              ignore_index=pad_id, reduction="sum")
        total_nll += nll.item()
        total_tokens += int((batch["targets"] != pad_id).sum().item())
    mean_nll = total_nll / total_tokens
    return {"nll": mean_nll, "ppl": perplexity(mean_nll), "tokens": total_tokens, "bits_per_token": mean_nll / math.log(2)}
