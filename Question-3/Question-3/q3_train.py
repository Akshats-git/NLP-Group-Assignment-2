"""Loss, learning-rate schedule, validation loss, batched summary generation and resumable training state.

The training loop itself is in the notebook (section 5) so it can be read in one place.
"""

from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from common import autocast, teacher_forcing_pair

from q3_data import ids_to_text, trim_padding
from q3_model import greedy_summarize


def noam_lr(step: int, d_model: int, warmup_steps: int) -> float:
    """lr = d_model^-0.5 * min(step^-0.5, step * warmup^-1.5): linear warm-up, then 1/sqrt(step) decay."""
    step = max(step, 1)
    return d_model ** -0.5 * min(step ** -0.5, step * warmup_steps ** -1.5)


# ---------------------------------------------------------------- resuming after an interruption

STOPPER_FIELDS = ("best", "best_epoch", "bad_epochs", "epochs_seen")


def save_training_state(path, model, optimizer, scheduler, scaler, stopper, history, progress: dict):
    """Everything needed to continue training later: weights, optimiser, schedule, early stopping,
    history and progress (step, pairs seen, training seconds, stop reason).

    Written to a temporary file first and then renamed, so an interruption while saving never
    leaves a broken checkpoint behind.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    state = {
        "model": model.state_dict(), "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
        "scaler": scaler.state_dict(), "stopper": {k: getattr(stopper, k) for k in STOPPER_FIELDS},
        "history": history.records, "progress": progress,
    }
    tmp = path.with_suffix(".tmp")
    torch.save(state, tmp)
    tmp.replace(path)


def load_training_state(path, model, optimizer, scheduler, scaler, stopper, history) -> dict:
    """Load a state written by save_training_state into the given objects. Returns the progress dict."""
    state = torch.load(path, map_location="cpu", weights_only=False)   # our own file
    model.load_state_dict(state["model"])
    optimizer.load_state_dict(state["optimizer"])
    scheduler.load_state_dict(state["scheduler"])
    scaler.load_state_dict(state["scaler"])
    for key, value in state["stopper"].items():
        setattr(stopper, key, value)
    history.records = list(state["history"])
    return state["progress"]


def seq2seq_loss(outputs, targets, pad_id, smoothing=0.0, src=None):
    """Token-level cross-entropy that ignores padding.

    outputs: what the model returns, (logits [B, T, vocab], p_gen, copy_attention), or just logits.
    targets: [B, T] (the summary shifted left). src: [B, S] article ids, needed for copying.

    The loss is -log P(target word). With the copy mechanism
        P(word) = p_gen * P_vocab(word) + (1 - p_gen) * attention on article positions holding the word,
    computed for the target word only, so no extra [B, T, vocab] tensor is needed.
    With smoothing > 0, label smoothing is added on the vocabulary distribution:
        (1 - smoothing) * NLL(target) + smoothing * mean over the vocabulary of -log P_vocab
    (without copying this is exactly nn.CrossEntropyLoss(label_smoothing=...)).

    Returns (loss to optimise, sum of plain NLL over real tokens, number of real tokens).
    The plain NLL is what we report and use for early stopping.
    """
    logits, p_gen, copy_attention = outputs if isinstance(outputs, tuple) else (outputs, None, None)
    log_probs = F.log_softmax(logits.float(), dim=-1)
    nll = -log_probs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)   # [B, T]
    if p_gen is not None:
        in_article = src[:, None, :] == targets[:, :, None]           # [B, T, S]: article tokens equal to the target
        copy_prob = (copy_attention * in_article).sum(dim=-1)        # attention mass on those tokens
        prob = p_gen * torch.exp(-nll) + (1 - p_gen) * copy_prob
        nll = -prob.clamp_min(1e-12).log()
    real = targets != pad_id                                         # padding is never counted
    n_tokens = real.sum()
    if n_tokens == 0:
        raise ValueError("batch has no target tokens")
    per_token = nll if smoothing == 0 else (1 - smoothing) * nll - smoothing * log_probs.mean(dim=-1)
    loss = per_token[real].sum() / n_tokens
    return loss, nll[real].sum().detach(), n_tokens


@torch.no_grad()
def evaluate_loss(model, loader, pad_id, device):
    """Mean per-token cross-entropy (no label smoothing) over a whole loader, teacher forced."""
    model.eval()
    total_nll, total_tokens = 0.0, 0
    for src, tgt in loader:
        src, tgt = src.to(device), tgt.to(device)
        tgt_in, tgt_out = teacher_forcing_pair(tgt)
        with autocast(device):
            outputs = model(src, tgt_in)
        _, nll_sum, n_tokens = seq2seq_loss(outputs, tgt_out, pad_id, src=src)
        total_nll += nll_sum.item()
        total_tokens += n_tokens.item()
    return total_nll / total_tokens


@torch.no_grad()
def right_vs_wrong_article(model, src_ids: np.ndarray, tgt_ids: np.ndarray, pad_id, device, batch_size=64):
    """Validation loss of the reference summaries given their own article vs. another article.

    Each batch's articles are rotated by one, so every summary is scored against a different
    story. A model that reads the article does much better with the right one; if the two
    numbers are close, the decoder is only producing generic, summary-sounding text.
    Returns {"right": mean NLL, "wrong": mean NLL, "p_gen": mean generate-vs-copy gate on the
    right articles (None without the copy mechanism)}.
    """
    model.eval()
    sums = {"right": [0.0, 0], "wrong": [0.0, 0]}
    gate_sum, gate_count = 0.0, 0
    for start in range(0, len(src_ids), batch_size):
        src = trim_padding(torch.from_numpy(src_ids[start:start + batch_size].astype(np.int64))).to(device)
        tgt = trim_padding(torch.from_numpy(tgt_ids[start:start + batch_size].astype(np.int64))).to(device)
        if src.size(0) < 2:
            continue
        tgt_in, tgt_out = teacher_forcing_pair(tgt)
        for name, articles in (("right", src), ("wrong", src.roll(1, dims=0))):
            with autocast(device):
                outputs = model(articles, tgt_in)
            _, nll_sum, n_tokens = seq2seq_loss(outputs, tgt_out, pad_id, src=articles)
            sums[name][0] += nll_sum.item()
            sums[name][1] += n_tokens.item()
            p_gen = outputs[1] if isinstance(outputs, tuple) else None
            if name == "right" and p_gen is not None:
                gate_sum += p_gen[tgt_out != pad_id].sum().item()
                gate_count += int((tgt_out != pad_id).sum())
    result = {name: total / count for name, (total, count) in sums.items()}
    result["p_gen"] = gate_sum / gate_count if gate_count else None
    return result


@torch.no_grad()
def summarize_arrays(model, src_ids: np.ndarray, tokenizer, special, device, batch_size=64,
                     max_len=128, block_trigrams=False, desc=None):
    """Generate summaries (free running, no reference used) for rows of a padded id array.

    Returns a list of texts with one sentence per line.
    """
    from tqdm.auto import tqdm

    texts = []
    banned = [special.pad, special.bos, special.unk]
    for start in tqdm(range(0, len(src_ids), batch_size), desc=desc, leave=False, disable=desc is None):
        src = trim_padding(torch.from_numpy(src_ids[start:start + batch_size].astype(np.int64))).to(device)
        with autocast(device):
            new_tokens, _ = greedy_summarize(model, src, special.bos, special.eos, special.pad, max_len=max_len,
                                             banned_ids=banned, block_trigrams=block_trigrams,
                                             sentence_id=special.nl)
        texts.extend(ids_to_text(row, tokenizer, special) for row in new_tokens.cpu())
    return texts
