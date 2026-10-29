"""Final evaluation: trained runs on the test split, example picking and token importance.

The test split is predicted once per run and saved to outputs/results/<run>/test_predictions.jsonl.
Running the evaluation again reloads those files instead of predicting again, so the test set
is never used for anything but this one measurement.
"""

import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from common import Vocab, autocast, checkpoints_dir, data_dir, load_json, load_jsonl, results_dir, save_jsonl, to_device
from common.plotting import attention_window, plot_attention, savefig

from . import classic
from .data import load_reviews
from .models import BiGRUAttention, RobertaClassifier

KAGGLE_INPUT = Path("/kaggle/input")


def run_config(run_name: str) -> dict:
    return load_json(results_dir(2) / run_name / "run_info.json")["config"]


def _state(run_name: str) -> dict:
    return torch.load(checkpoints_dir(2) / run_name / "best.pt", map_location="cpu", weights_only=True)


def classic_vocab(limit: int | None = None, inputs: Path = KAGGLE_INPUT) -> Vocab:
    """The BiGRU vocabulary: Question-2/data/classic/, an attached Kaggle input, or rebuilt from train.

    Rebuilding gives the same ids, because Vocab.build breaks frequency ties alphabetically.
    A `limit` run built its vocabulary from its own subset, so that one is always rebuilt.
    """
    if limit is None:
        found = [data_dir(2) / "classic" / "vocab.json"]
        found += sorted(inputs.glob("**/Question-2/data/classic/vocab.json")) if inputs.exists() else []
        for path in found:
            if path.is_file():
                return Vocab.load(path)
    train = classic.add_tokens(load_reviews(("train",), limit=limit))["train"]
    return classic.build_vocab(train["tokens"])


def load_bigru(run_name: str, vocab: Vocab) -> BiGRUAttention:
    config, state = run_config(run_name), _state(run_name)
    if state["embedding.weight"].shape[0] != len(vocab):
        raise ValueError(f"{run_name} was trained with {state['embedding.weight'].shape[0]:,} words, "
                         f"the vocabulary has {len(vocab):,}")
    model = BiGRUAttention(torch.zeros_like(state["embedding.weight"]), config["hidden_size"], config["num_layers"],
                           config["attention_dim"], config["dropout"])
    model.load_state_dict(state)
    return model.eval()


def load_roberta(run_name: str) -> RobertaClassifier:
    config = run_config(run_name)
    model = RobertaClassifier(config["model_name"], config["pooling"], config["dropout"])
    model.load_state_dict(_state(run_name))
    return model.eval()


@torch.no_grad()
def predict_proba(model: torch.nn.Module, loader, device, inputs: tuple[str, str]) -> np.ndarray:
    """[reviews, 5] class probabilities in the loader's order. inputs names the two batch fields the model takes."""
    model.eval()
    probs = []
    for batch in loader:
        batch = to_device(batch, device)
        with autocast(device):
            logits = model(*(batch[key] for key in inputs))
        probs.append(torch.softmax(logits.float(), dim=-1).cpu())
    return torch.cat(probs).numpy()


def test_predictions(run_name: str, ids, labels, predict) -> list[dict]:
    """{"id", "label", "pred", "probs"} per test review. Calls predict() only if the run has no saved predictions."""
    path = results_dir(2) / run_name / "test_predictions.jsonl"
    if path.exists():
        rows = load_jsonl(path)
        if [r["id"] for r in rows] != list(ids):
            raise ValueError(f"{path} does not match the test split")
        return rows
    probs = predict()
    rows = [{"id": i, "label": y, "pred": int(p.argmax()), "probs": [round(float(x), 5) for x in p]}
            for i, y, p in zip(ids, labels, probs)]
    save_jsonl(rows, path)
    return rows


NEGATION = re.compile(r"\b(?:not|no|never|nothing|nobody|none|neither|nor|hardly|without)\b|n[’']t\b", re.I)
CONTRAST = re.compile(r"\b(?:but|however|although|though|except|otherwise|yet)\b", re.I)
LONG_REVIEW = 120   # spaCy tokens, about the 95th percentile of review length


def review_tags(text: str, n_tokens: int, n_unknown: int) -> list[str]:
    """Rough flags for the phenomena the brief names. Sarcasm can't be caught by a pattern, so check examples by eye."""
    tags = []
    if NEGATION.search(text):
        tags.append("negation")
    if CONTRAST.search(text):
        tags.append("mixed / contrast")
    if n_tokens > LONG_REVIEW:
        tags.append("long")
    if n_unknown >= 2:
        tags.append("rare words / typos")
    return tags


def pick_examples(candidates: list[dict], n: int, max_words: int = 150) -> list[dict]:
    """Up to n examples that cover as many different tags as possible, most confident first.

    candidates carry "tags" and "confidence". Reviews longer than max_words are only
    used for the "long" tag, so the rest stay readable in the report.
    """
    pool = sorted(candidates, key=lambda c: -c["confidence"])
    picked = []
    for tag in ("negation", "mixed / contrast", "long", "rare words / typos"):
        for c in pool:
            short = len(c["text"].split()) <= max_words
            if tag in c["tags"] and c not in picked and (short or tag == "long") and len(picked) < n:
                picked.append(c)
                break
    for c in pool:
        if len(picked) >= n:
            break
        if c not in picked and len(c["text"].split()) <= max_words:
            picked.append(c)
    return picked


def integrated_gradients(model: RobertaClassifier, input_ids: torch.Tensor, target: int, pad_id: int,
                         steps: int = 256, chunk: int = 32) -> tuple[torch.Tensor, float]:
    """Each token's contribution to the logit of `target` (Sundararajan et al., 2017).

    input_ids is one review, [1, time], with <s> and </s>. The baseline replaces every other
    token's embedding with the <pad> embedding and the path integral uses the midpoint rule.
    64 steps left a gap larger than the attributions on a test review; 256 closed it.
    Returns attributions [time] and the completeness gap: sum(attributions) minus
    (logit(review) - logit(baseline)), which should be small next to the logit difference.
    """
    model.eval()
    embed = model.encoder.embeddings.word_embeddings
    baseline_ids = input_ids.clone()
    baseline_ids[:, 1:-1] = pad_id
    x, x0 = embed(input_ids).detach(), embed(baseline_ids).detach()
    mask = torch.ones_like(input_ids)
    grads = torch.zeros_like(x)
    alphas = (torch.arange(steps, device=x.device, dtype=x.dtype) + 0.5) / steps
    for part in alphas.split(chunk):
        path = (x0 + part.view(-1, 1, 1) * (x - x0)).requires_grad_(True)
        logits = model(None, mask.expand(len(part), -1), inputs_embeds=path)
        grads += torch.autograd.grad(logits[:, target].sum(), path)[0].sum(dim=0, keepdim=True)
    attributions = ((x - x0) * grads / steps).sum(dim=-1)[0].detach()
    with torch.no_grad():
        gap = model(None, mask, inputs_embeds=x)[0, target] - model(None, mask, inputs_embeds=x0)[0, target]
    return attributions, float(attributions.sum() - gap)


def merge_subwords(tokens: list[str], scores, decode=None) -> tuple[list[str], np.ndarray]:
    """Byte-level BPE pieces ("Ġgre", "at") -> words, summing their scores. <s> and </s> are dropped.

    Pass decode=tokenizer.convert_tokens_to_string: pieces store bytes as characters, so
    without it a curly apostrophe stays "âĢĻ" and "I’m" comes out as "IâĢĻm".
    """
    pieces, merged = [], []
    for token, score in zip(tokens, np.asarray(scores, dtype=float)):
        if token in ("<s>", "</s>"):
            continue
        if token.startswith("Ġ") or not pieces:
            pieces.append([token])
            merged.append(score)
        else:
            pieces[-1].append(token)
            merged[-1] += score
    decode = decode or (lambda word: "".join(word).lstrip("Ġ"))
    return [decode(word).strip() for word in pieces], np.asarray(merged)


def plot_token_importance(rows: list[tuple[str, list[str], np.ndarray]], out_path=None, title: str | None = None,
                          width: int = 60):
    """One heatmap strip per (label, tokens, weights) row; long reviews are cut to the `width` busiest tokens."""
    longest = max(min(len(tokens), width) for _, tokens, _ in rows)
    fig, axes = plt.subplots(len(rows), 1, figsize=(min(16, 0.38 * longest + 2.5), 1.9 * len(rows)), squeeze=False)
    for ax, (label, tokens, weights) in zip(axes[:, 0], rows):
        window, words, _ = attention_window(np.asarray(weights, dtype=float)[None], tokens, width)
        plot_attention(window, words, [label], ax=ax)
        ax.set_xlabel("")
        ax.set_ylabel("")
    if title:
        fig.suptitle(title, fontweight="bold")
    fig.tight_layout()
    if out_path is not None:
        savefig(fig, out_path)
    return fig
