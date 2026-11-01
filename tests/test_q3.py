"""Tests for the Q3 Transformer in Question-3/. Run from the repo root: pytest -q

These repeat the pitfall checks the notebook runs (causal mask, padding masks, loss ignoring padding,
no banned layers), so they can be checked in seconds on a CPU without the dataset or a GPU.
"""

import ast
import math
import sys
from pathlib import Path

import pytest
import torch
from torch import nn

Q3 = Path(__file__).resolve().parents[1] / "Question-3"
sys.path.insert(0, str(Q3))

import q3_model  # noqa: E402
import q3_train  # noqa: E402

VOCAB, PAD = 50, 0


def small_model(copy_mechanism=False):
    torch.manual_seed(0)
    return q3_model.TransformerSummarizer(VOCAB, pad_id=PAD, d_model=32, n_heads=4, n_enc_layers=2, n_dec_layers=2,
                                          d_ff=64, dropout=0.0, copy_mechanism=copy_mechanism).eval()


def prefix(outputs, n):
    return tuple(None if x is None else x[:, :n] for x in outputs)


def same(a, b):
    return all(x is None or torch.allclose(x, y, atol=1e-5) for x, y in zip(a, b))


def test_no_banned_building_blocks():
    banned = {"Transformer", "TransformerEncoder", "TransformerDecoder", "TransformerEncoderLayer",
              "TransformerDecoderLayer", "MultiheadAttention", "scaled_dot_product_attention"}
    tree = ast.parse((Q3 / "q3_model.py").read_text())
    used = {n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)} | {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}
    assert not used & banned
    assert not any(isinstance(m, (nn.Transformer, nn.MultiheadAttention)) for m in small_model().modules())


def test_positional_encoding_matches_the_formula():
    pe = q3_model.PositionalEncoding(16, max_len=10)(torch.zeros(1, 10, 16))[0]
    pos, i = 3, 2
    assert pe[pos, 2 * i].item() == pytest.approx(math.sin(pos / 10000 ** (2 * i / 16)), abs=1e-6)
    assert pe[pos, 2 * i + 1].item() == pytest.approx(math.cos(pos / 10000 ** (2 * i / 16)), abs=1e-6)


def test_attention_is_scaled_dot_product_softmax():
    torch.manual_seed(0)
    attn = q3_model.MultiHeadAttention(16, 4).eval()
    x = torch.randn(2, 5, 16)
    mask = torch.ones(2, 1, 5, 5, dtype=torch.bool)
    out, weights = attn(x, x, x, mask)
    q, k, v = (attn.split_heads(w(x)) for w in (attn.w_q, attn.w_k, attn.w_v))
    expected = torch.softmax(q @ k.transpose(-2, -1) / math.sqrt(attn.d_k), dim=-1)
    assert torch.allclose(weights, expected, atol=1e-5)
    assert torch.allclose(weights.sum(-1), torch.ones(2, 4, 5), atol=1e-5)
    assert out.shape == (2, 5, 16)


def test_causal_mask_hides_future_tokens():
    model = small_model(copy_mechanism=True)
    src, tgt = torch.randint(5, VOCAB, (2, 12)), torch.randint(5, VOCAB, (2, 8))
    changed = tgt.clone()
    changed[:, 4:] = 7
    with torch.no_grad():
        before, after = model(src, tgt), model(src, changed)
    assert same(prefix(before, 4), prefix(after, 4))                              # earlier positions unchanged
    assert not torch.allclose(before[0][:, 4:], after[0][:, 4:], atol=1e-5)      # later ones do change


def test_source_padding_is_ignored_and_gets_no_attention():
    model = small_model()
    src, tgt = torch.randint(5, VOCAB, (2, 12)), torch.randint(5, VOCAB, (2, 6))
    padded_src = torch.cat([src, torch.full((2, 7), PAD)], dim=1)
    with torch.no_grad():
        base, padded = model(src, tgt), model(padded_src, tgt)
        memory, src_mask = model.encode(padded_src)
        _, cross = model.decode(tgt, memory, src_mask)
    assert same(base[:1], padded[:1])
    assert cross[..., -7:].max().item() < 1e-6


def test_target_padding_does_not_change_real_positions():
    model = small_model()
    src, tgt = torch.randint(5, VOCAB, (2, 12)), torch.randint(5, VOCAB, (2, 6))
    padded_tgt = torch.cat([tgt, torch.full((2, 4), PAD)], dim=1)
    with torch.no_grad():
        assert same(model(src, tgt), prefix(model(src, padded_tgt), 6))


def test_loss_ignores_padding():
    logits = torch.randn(2, 6, VOCAB, requires_grad=True)
    targets = torch.tensor([[5, 6, 7, 0, 0, 0], [8, 9, 10, 11, 12, 0]])
    loss, _, n_tokens = q3_train.seq2seq_loss(logits, targets, PAD)
    reference = nn.functional.cross_entropy(logits.reshape(-1, VOCAB), targets.reshape(-1), ignore_index=PAD)
    loss.backward()
    assert n_tokens.item() == 8
    assert torch.allclose(loss, reference)
    assert logits.grad[targets == PAD].abs().max().item() == 0


def test_noam_schedule_warms_up_then_decays():
    d_model, warmup = 512, 4000
    assert q3_train.noam_lr(warmup, d_model, warmup) > q3_train.noam_lr(100, d_model, warmup)
    assert q3_train.noam_lr(warmup, d_model, warmup) > q3_train.noam_lr(40000, d_model, warmup)
    assert q3_train.noam_lr(warmup, d_model, warmup) == pytest.approx(d_model ** -0.5 * warmup ** -0.5)


def test_greedy_decoding_is_free_running_and_stops():
    model = small_model()
    src = torch.randint(5, VOCAB, (3, 10))
    tokens, _ = q3_model.greedy_summarize(model, src, bos_id=1, eos_id=2, pad_id=PAD, max_len=12, banned_ids=(PAD, 1))
    assert tokens.shape[0] == 3 and tokens.shape[1] <= 12
    assert (tokens != 1).all()                                              # banned token is never generated


def test_trigram_blocking_forbids_repeats():
    logits = torch.zeros(1, 10)
    tokens = torch.tensor([[3, 4, 5, 3, 4]])                                # next token 5 would repeat the trigram 3 4 5
    blocked = q3_model.block_repeated_trigrams(logits.clone(), tokens)
    assert blocked[0, 5].item() == float("-inf")
    assert blocked[0, 6].item() == 0
