"""Tests for common/. Run from the repo root: pytest -q"""

import re
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn

from common import (
    Decoding,
    EarlyStopping,
    History,
    Vocab,
    bucket_indices,
    bucket_labels,
    causal_mask,
    format_mean_std,
    generate,
    lengths_to_mask,
    load_checkpoint,
    load_json,
    markdown_table,
    masked_accuracy,
    mean_std,
    normalize,
    pad_collate,
    pad_sequences,
    perplexity,
    save_checkpoint,
    save_json,
    set_seed,
    strip_html_and_urls,
    teacher_forcing_pair,
    to_device,
)
from common.decoding import next_token, top_k_filter, top_p_filter
from common.metrics import group_by_bucket, summarize_runs


def test_set_seed_repeats():
    set_seed(7)
    a = (torch.rand(3), np.random.rand(3))
    set_seed(7)
    b = (torch.rand(3), np.random.rand(3))
    assert torch.equal(a[0], b[0]) and np.array_equal(a[1], b[1])


def test_vocab_ids_and_roundtrip(tmp_path):
    vocab = Vocab.build([["b", "a", "b"], ["c", "a", "b"]], min_freq=2)
    assert vocab.itos[:4] == ["<pad>", "<unk>", "<sos>", "<eos>"]
    assert (vocab.pad_id, vocab.unk_id, vocab.sos_id, vocab.eos_id) == (0, 1, 2, 3)
    assert vocab.itos[4:] == ["b", "a"]  # by frequency, "c" dropped by min_freq
    ids = vocab.encode(["a", "c", "b"], add_sos=True, add_eos=True)
    assert ids == [2, 5, 1, 4, 3]
    assert vocab.decode(ids) == ["a", "b"]  # <unk> is a special, skipped
    assert vocab.decode(torch.tensor(ids), skip_special=False) == ["<sos>", "a", "<unk>", "b"]
    assert vocab.oov_rate([["a", "c"]]) == 0.5
    loaded = Vocab.load(vocab.save(tmp_path / "vocab.json"))
    assert loaded.itos == vocab.itos


def test_vocab_truncation_keeps_eos():
    vocab = Vocab.build([list("abcdef")])
    ids = vocab.encode(list("abcdef"), add_sos=True, add_eos=True, max_len=4)
    assert len(ids) == 4 and ids[0] == vocab.sos_id and ids[-1] == vocab.eos_id


def test_vocab_is_deterministic_on_ties():
    assert Vocab.build([["z", "y", "x"]]).itos == Vocab.build([["x", "z", "y"]]).itos


def test_text_helpers():
    assert normalize("  Hello\n\tWorld ", lower=True) == "hello world"
    assert normalize("क़") == normalize("क़")  # both spellings of qa end up identical
    assert strip_html_and_urls("Great<br />buy &amp; see http://x.com/a") == "Great buy & see"


def test_pad_sequences_and_masks():
    ids, lengths = pad_sequences([[5, 6, 7], [8]], pad_id=0)
    assert ids.tolist() == [[5, 6, 7], [8, 0, 0]] and lengths.tolist() == [3, 1]
    left, _ = pad_sequences([[5, 6, 7], [8]], pad_id=0, side="left")
    assert left.tolist() == [[5, 6, 7], [0, 0, 8]]
    cut, cut_lengths = pad_sequences([[1, 2, 3, 4]], pad_id=0, max_len=2)
    assert cut.tolist() == [[1, 2]] and cut_lengths.tolist() == [2]
    assert lengths_to_mask(lengths).tolist() == [[True, True, True], [True, False, False]]
    assert causal_mask(3).tolist() == [[True, False, False], [True, True, False], [True, True, True]]
    dec_in, target = teacher_forcing_pair(torch.tensor([[2, 9, 3]]))
    assert dec_in.tolist() == [[2, 9]] and target.tolist() == [[9, 3]]


def test_pad_collate_mixed_fields():
    collate = pad_collate(pad_id=0)
    batch = collate([
        {"src": [4, 5], "label": 3, "text": "hi"},
        {"src": [6], "label": 1, "text": "नमस्ते"},
    ])
    assert batch["src"].tolist() == [[4, 5], [6, 0]]
    assert batch["src_lengths"].tolist() == [2, 1]
    assert batch["label"].tolist() == [3, 1]
    assert batch["text"] == ["hi", "नमस्ते"]


def test_to_device_nested():
    batch = {"a": torch.ones(1), "b": [torch.zeros(1), "x"], "c": 3}
    moved = to_device(batch, "cpu")
    assert moved["b"][1] == "x" and moved["c"] == 3


def _tiny_model(value: float) -> nn.Module:
    model = nn.Linear(2, 1)
    nn.init.constant_(model.weight, value)
    return model


@pytest.mark.parametrize("use_disk", [False, True])
def test_early_stopping_restores_best(tmp_path, use_disk):
    stopper = EarlyStopping(mode="max", patience=2, checkpoint_path=tmp_path / "best.pt" if use_disk else None)
    model = _tiny_model(0.0)
    for epoch, score in enumerate([0.5, 0.7, 0.6, 0.65], start=1):
        nn.init.constant_(model.weight, float(epoch))
        stopper.step(score, model)
        if stopper.should_stop:
            break
    assert stopper.should_stop and stopper.best == 0.7 and stopper.best_epoch == 2
    stopper.restore(model)
    assert torch.all(model.weight == 2.0)


def test_early_stopping_min_delta():
    stopper = EarlyStopping(mode="min", patience=1, min_delta=0.1)
    stopper.step(1.0)
    assert not stopper.step(0.95)  # not better by more than min_delta
    assert stopper.should_stop


def test_checkpoint_roundtrip(tmp_path):
    model = _tiny_model(1.5)
    optimizer = torch.optim.Adam(model.parameters())
    path = save_checkpoint(tmp_path / "ckpt.pt", model, optimizer, epoch=3, config={"hidden": 8})
    fresh = _tiny_model(0.0)
    payload = load_checkpoint(path, fresh, torch.optim.Adam(fresh.parameters()))
    assert torch.all(fresh.weight == 1.5) and payload["epoch"] == 3 and payload["config"]["hidden"] == 8


def test_history(tmp_path):
    history = History()
    history.log(train_loss=2.0, val_loss=1.5)
    history.log(train_loss=1.0, val_loss=1.7)
    assert history["val_loss"] == [1.5, 1.7] and history["epoch"] == [1, 2]
    assert history.best("val_loss")["epoch"] == 1
    assert History.load(history.save(tmp_path / "h.json")).records == history.records


def test_metrics():
    logits = torch.tensor([[[0.1, 0.9], [0.8, 0.2], [0.5, 0.4]]])
    assert masked_accuracy(logits, torch.tensor([[1, 1, 0]]), pad_id=0) == (1, 2)
    assert perplexity(0.0) == 1.0 and perplexity(1e6) == float("inf")
    assert mean_std([1.0, 3.0]) == (2.0, pytest.approx(2**0.5))
    assert format_mean_std([0.8, 0.9], scale=100) == "85.00 ± 7.07"
    summary = summarize_runs([{"acc": 0.8, "name": "a"}, {"acc": 0.9, "name": "b"}])
    assert set(summary) == {"acc"} and summary["acc"]["mean"] == pytest.approx(0.85)


def test_length_buckets():
    assert bucket_labels() == ["1-5", "6-10", "11-15", "16+"]
    groups = bucket_indices([3, 5, 6, 15, 40])
    assert groups == {"1-5": [0, 1], "6-10": [2], "11-15": [3], "16+": [4]}
    stats = group_by_bucket([3, 5, 40], [1.0, 3.0, 5.0])
    assert stats["1-5"] == {"mean": 2.0, "count": 2} and stats["6-10"]["count"] == 0


def test_filters():
    logits = torch.tensor([[1.0, 3.0, 2.0, 0.0]])
    assert torch.isinf(top_k_filter(logits, 2)).tolist() == [[True, False, False, True]]
    probs = torch.tensor([[0.5, 0.3, 0.15, 0.05]]).log()
    kept = ~torch.isinf(top_p_filter(probs, 0.75))
    assert kept.tolist() == [[True, True, False, False]]
    kept = ~torch.isinf(top_p_filter(probs, 0.1))
    assert kept.tolist() == [[True, False, False, False]]  # the top token always survives


def test_next_token_strategies():
    logits = torch.tensor([[0.0, 5.0, 1.0]])
    assert next_token(logits).item() == 1
    generator = torch.Generator().manual_seed(0)
    draws = {next_token(logits, Decoding("sample", top_k=1), generator).item() for _ in range(20)}
    assert draws == {1}
    assert Decoding().label == "greedy"
    assert Decoding("sample", temperature=0.8).label == "temp=0.8"
    assert Decoding("sample", top_k=40).label == "top-k=40"
    with pytest.raises(ValueError):
        Decoding("sample", temperature=0)


def test_generate_stops_and_pads():
    eos, pad = 3, 0
    script = {0: [5, 6], 1: [3, 7], 2: [9, 3]}  # row 0 stops at step 1, row 1 at step 2

    def step_fn(tokens, state):
        step = 0 if state is None else state
        logits = torch.full((2, 10), -1e9)
        for row, token in enumerate(script[step]):
            logits[row, token] = 0.0
        return logits, step + 1, torch.tensor([step])

    out = generate(step_fn, torch.full((2, 1), 2), max_new_tokens=10, eos_id=eos, pad_id=pad)
    assert out.new_tokens.tolist() == [[5, 3, pad], [6, 7, 3]]
    assert out.finished.all() and len(out.extras) == 3


def test_files_and_tables(tmp_path):
    path = save_json({"x": np.float32(0.5), "t": torch.tensor(2), "arr": np.arange(2), "hi": "हिंदी"}, tmp_path / "r.json")
    assert load_json(path) == {"x": 0.5, "t": 2, "arr": [0, 1], "hi": "हिंदी"}
    assert "हिंदी" in path.read_text(encoding="utf-8")
    table = markdown_table([{"model": "A", "bleu": 12.345}])
    assert table.splitlines() == ["| model | bleu |", "|---|---|", "| A | 12.35 |"]


def test_plots_render(tmp_path):
    from common.plotting import attention_window, plot_attention, plot_confusion_matrix, plot_grouped_bars, plot_history

    history = History()
    for epoch in range(5):
        history.log(train_loss=2 - 0.3 * epoch, val_loss=2.1 - 0.25 * epoch, train_acc=0.3 + 0.1 * epoch)
    plot_history({"Baseline": history, "Attention": history}, metrics=("loss", "acc"), out_path=tmp_path / "h.png")
    plot_attention(np.random.rand(4, 5), ["I", "am", "going", "home", "."], ["मैं", "घर", "जा", "रहा"], out_path=tmp_path / "a.png")
    window, tokens, start = attention_window(np.random.rand(3, 200), [f"t{i}" for i in range(200)], width=50)
    assert window.shape == (3, 50) and len(tokens) == 50 and 0 <= start <= 150
    plot_grouped_bars({"A": {"1-5": 20.0, "16+": 8.0}, "B": {"1-5": 22.0, "16+": 13.0}}, "BLEU", out_path=tmp_path / "b.png")
    plot_confusion_matrix(np.eye(5) * 10 + 1, [str(i) for i in range(1, 6)], out_path=tmp_path / "c.png")
    assert all((tmp_path / name).stat().st_size > 0 for name in ("h.png", "a.png", "b.png", "c.png"))
    with pytest.raises(ValueError):
        plot_attention(np.random.rand(2, 2), ["a"], ["b", "c"])


BANNED_FOR_Q3 = [
    r"nn\.Transformer", r"TransformerEncoder", r"TransformerDecoder", r"MultiheadAttention",
    r"scaled_dot_product_attention", r"MultiHeadAttention",
]


def test_common_has_nothing_q3_is_banned_from():
    """Q3 imports common too, so common must never use the layers Q3 has to write itself."""
    for path in (Path(__file__).resolve().parents[1] / "common").glob("*.py"):
        source = path.read_text(encoding="utf-8")
        for pattern in BANNED_FOR_Q3:
            assert not re.search(pattern, source), f"{path.name} uses {pattern}"
