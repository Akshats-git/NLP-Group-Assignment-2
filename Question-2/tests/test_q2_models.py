"""Offline tests for batching, metrics and the RoBERTa classifier. Run from the repo root: pytest -q Question-2/tests"""

import sys
from pathlib import Path

import pytest
import torch
from transformers import RobertaConfig, RobertaModel

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from q2.batching import LengthGroupedBatchSampler
from q2.metrics import classification_metrics
from q2.models import AdditiveAttentionPooling, BiGRUAttention, RobertaClassifier, masked_mean


def test_length_grouped_batches():
    lengths = [(i * 37) % 101 for i in range(1000)]
    sampler = LengthGroupedBatchSampler(lengths, batch_size=16, seed=1, pool_batches=10)
    first, second = list(sampler), list(sampler)
    assert len(first) == len(sampler) == 63
    assert sorted(i for b in first for i in b) == list(range(1000))   # every review exactly once
    assert first != second                                             # new shuffle each epoch
    assert first == list(LengthGroupedBatchSampler(lengths, 16, seed=1, pool_batches=10))  # seeded
    assert lengths[first[0][0]] == max(lengths)                        # longest batch first
    spread = sum(max(lengths[i] for i in b) - min(lengths[i] for i in b) for b in first) / len(first)
    assert spread < 15                                                 # random batches spread ~95


def test_classification_metrics():
    m = classification_metrics([0, 1, 2, 3, 4, 4], [0, 2, 2, 3, 0, 4])
    assert m["accuracy"] == pytest.approx(4 / 6)
    assert m["mae"] == pytest.approx(5 / 6)
    assert m["off_by_one"] == pytest.approx(1 / 6) and m["off_by_two_plus"] == pytest.approx(1 / 6)
    assert m["adjacent_share_of_errors"] == 0.5 and m["extreme_confusions"] == 1
    assert m["confusion_matrix"][4] == [1, 0, 0, 0, 1]
    assert len(m["per_class_f1"]) == 5


def test_import_finished_runs(tmp_path, monkeypatch):
    from q2 import kaggle
    outputs = tmp_path / "repo" / "outputs"
    monkeypatch.setattr(kaggle, "output_dir", lambda q: outputs)
    old = tmp_path / "input" / "my-notebook" / "NLP-Group-Assignment-2" / "Question-2" / "outputs"
    for run, finished in (("roberta_mean_seed42", True), ("roberta_mean_seed43", False)):
        (old / "results" / run).mkdir(parents=True)
        (old / "checkpoints" / run).mkdir(parents=True)
        (old / "checkpoints" / run / "best.pt").write_bytes(b"w")
        if finished:
            (old / "results" / run / "summary.json").write_text("{}")
    assert kaggle.import_finished_runs("roberta_*", inputs=tmp_path / "input") == ["roberta_mean_seed42"]
    assert (outputs / "checkpoints" / "roberta_mean_seed42" / "best.pt").exists()
    assert not (outputs / "results" / "roberta_mean_seed43").exists()
    assert kaggle.import_finished_runs("roberta_*", inputs=tmp_path / "input") == []   # already there


def test_masked_mean_ignores_padding():
    hidden = torch.tensor([[[1.0], [3.0], [100.0]]])
    assert masked_mean(hidden, torch.tensor([[1, 1, 0]])).item() == 2.0


def test_attention_pooling_skips_padding():
    pool = AdditiveAttentionPooling(input_dim=4, attention_dim=3)
    hidden = torch.randn(2, 5, 4)
    mask = torch.tensor([[1, 1, 1, 0, 0], [1, 1, 1, 1, 1]], dtype=torch.bool)
    pooled, alpha = pool(hidden, mask)
    assert pooled.shape == (2, 4)
    assert torch.allclose(alpha.sum(-1), torch.ones(2))
    assert alpha[0, 3:].abs().max() == 0
    assert torch.allclose(pooled[0], alpha[0, :3] @ hidden[0, :3])


def test_bigru_padding_invariant_and_freezing():
    torch.manual_seed(0)
    embeddings = torch.randn(20, 6)
    embeddings[0] = 0
    model = BiGRUAttention(embeddings, hidden_size=8, num_layers=2, attention_dim=5).eval()
    ids = torch.tensor([[4, 5, 6, 0, 0, 0], [7, 8, 9, 10, 11, 12]])
    lengths = torch.tensor([3, 6])
    logits, alpha = model(ids, lengths, return_attention=True)
    assert logits.shape == (2, 5) and alpha.shape == (2, 6)
    alone = model(ids[:1, :3], lengths[:1])            # same review without padding or batch mates
    assert torch.allclose(logits[:1], alone, atol=1e-5)
    assert not model.embedding.weight.requires_grad
    model.set_embedding_trainable(True)
    assert model.embedding.weight.requires_grad
    assert torch.equal(model.embedding.weight, embeddings)


@pytest.fixture(scope="module")
def tiny_roberta(tmp_path_factory):
    path = tmp_path_factory.mktemp("tiny_roberta")
    config = RobertaConfig(vocab_size=50, hidden_size=16, num_hidden_layers=2, num_attention_heads=2,
                           intermediate_size=32, max_position_embeddings=40)
    RobertaModel(config, add_pooling_layer=False).save_pretrained(path)
    return str(path)


@pytest.mark.parametrize("pooling", ["mean", "first"])
def test_classifier_padding_invariant(tiny_roberta, pooling):
    torch.manual_seed(0)
    model = RobertaClassifier(tiny_roberta, pooling=pooling).eval()
    ids = torch.tensor([[0, 7, 8, 9, 2]])
    padded = torch.tensor([[0, 7, 8, 9, 2, 1, 1, 1]])   # <pad> is id 1 in RoBERTa
    mask = torch.ones_like(ids)
    padded_mask = torch.tensor([[1, 1, 1, 1, 1, 0, 0, 0]])
    logits = model(ids, mask)
    assert logits.shape == (1, 5)
    assert torch.allclose(logits, model(padded, padded_mask), atol=1e-5)


def test_mean_and_first_pooling_differ(tiny_roberta):
    torch.manual_seed(0)
    mean = RobertaClassifier(tiny_roberta, pooling="mean").eval()
    first = RobertaClassifier(tiny_roberta, pooling="first").eval()
    first.load_state_dict(mean.state_dict())
    ids, mask = torch.tensor([[0, 7, 8, 9, 2]]), torch.ones(1, 5, dtype=torch.long)
    assert not torch.allclose(mean(ids, mask), first(ids, mask))
