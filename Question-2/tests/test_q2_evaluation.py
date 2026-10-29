"""Offline tests for the evaluation helpers. Run from the repo root: pytest -q Question-2/tests"""

import sys
from pathlib import Path

import pytest
import torch
from transformers import RobertaConfig, RobertaModel

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from q2.evaluation import integrated_gradients, merge_subwords, pick_examples, plot_token_highlights, review_tags
from q2.metrics import confusions_between, error_distances
from q2.models import RobertaClassifier


def test_error_distances():
    assert error_distances([0, 1, 2, 4], [0, 2, 0, 0]) == [0.25, 0.25, 0.25, 0.0, 0.25]
    confusion = [[5, 0, 0, 0, 2], [0] * 5, [0] * 5, [0, 0, 0, 7, 1], [3, 0, 0, 4, 9]]
    assert confusions_between(confusion, 0, 4) == 5 and confusions_between(confusion, 3, 4) == 5


def test_review_tags():
    assert review_tags("It wasn't great, but it works", 6, 0) == ["negation", "mixed / contrast"]
    assert review_tags("Not bad", 2, 0) == ["negation"]
    assert review_tags("Nothingness is nice", 3, 0) == []
    assert review_tags("x " * 200, 200, 3) == ["long", "rare words / typos"]


def test_pick_examples_covers_tags():
    candidates = [{"text": "a", "tags": [], "confidence": 0.99},
                  {"text": "b", "tags": ["negation"], "confidence": 0.6},
                  {"text": "c", "tags": ["negation"], "confidence": 0.9},
                  {"text": "d " * 300, "tags": ["negation"], "confidence": 0.95},
                  {"text": "e", "tags": ["rare words / typos"], "confidence": 0.5}]
    assert [c["text"] for c in pick_examples(candidates, 3)] == ["c", "e", "a"]
    assert len(pick_examples(candidates, 10)) == 4                    # the long one only fills the "long" tag


def test_merge_subwords():
    words, scores = merge_subwords(["<s>", "Gre", "at", "Ġprod", "uct", "!", "</s>"], [9, 1, 2, 3, 4, 5, 9])
    assert words == ["Great", "product!"] and scores.tolist() == [3, 12]


@pytest.fixture(scope="module")
def tiny_classifier(tmp_path_factory):
    path = tmp_path_factory.mktemp("tiny_roberta")
    config = RobertaConfig(vocab_size=50, hidden_size=16, num_hidden_layers=2, num_attention_heads=2,
                           intermediate_size=32, max_position_embeddings=40)
    RobertaModel(config, add_pooling_layer=False).save_pretrained(path)
    torch.manual_seed(0)
    return RobertaClassifier(str(path)).eval()


def test_inputs_embeds_match_input_ids(tiny_classifier):
    ids, mask = torch.tensor([[0, 7, 8, 9, 2]]), torch.ones(1, 5, dtype=torch.long)
    embeds = tiny_classifier.encoder.embeddings.word_embeddings(ids)
    assert torch.allclose(tiny_classifier(ids, mask), tiny_classifier(None, mask, inputs_embeds=embeds), atol=1e-6)


def test_integrated_gradients_completeness(tiny_classifier):
    ids = torch.tensor([[0, 7, 8, 9, 11, 2]])
    attributions, gap = integrated_gradients(tiny_classifier, ids, target=3, pad_id=1, steps=128)
    assert attributions.shape == (6,)
    assert abs(gap) < 0.02 * max(attributions.abs().sum().item(), 1e-3)


def test_merge_subwords_decodes_bytes():
    from transformers import RobertaTokenizerFast
    try:
        tokenizer = RobertaTokenizerFast.from_pretrained("roberta-base", local_files_only=True)
    except OSError:
        pytest.skip("roberta-base tokenizer not cached")
    ids = tokenizer("I don’t mind")["input_ids"]
    words, _ = merge_subwords(tokenizer.convert_ids_to_tokens(ids), [0.0] * len(ids), tokenizer.convert_tokens_to_string)
    assert words == ["I", "don’t", "mind"]


def test_token_highlights_wrap_long_text():
    short = plot_token_highlights([("row", ["good"] * 5, [1.0] * 5)], width=4)
    long = plot_token_highlights([("row", ["good"] * 200, [1.0] * 200)], width=4)
    assert long.get_size_inches()[1] > 3 * short.get_size_inches()[1]      # wrapped onto more lines, not wider
    assert long.get_size_inches()[0] == short.get_size_inches()[0] == 4
