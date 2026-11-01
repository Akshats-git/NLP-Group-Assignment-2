"""Offline tests for the Q2 data pipelines. Run from the repo root: pytest -q Question-2/tests"""

import sys
from pathlib import Path

import numpy as np
import pytest
from datasets import Dataset, DatasetDict

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from q2 import classic
from q2.data import join_title_body, length_stats, merge_titles, split_title


def test_split_and_join_title():
    assert split_title("Great\n\nWorks well.\n\nWould buy again.") == ("Great", "Works well.\n\nWould buy again.")
    assert split_title("no title here") == ("", "no title here")
    assert join_title_body("Great", "Works well.") == "Great. Works well."
    assert join_title_body("Great!", "Works well.") == "Great! Works well."
    assert join_title_body("", " Works well. ") == "Works well."


def test_merge_titles_only_when_bodies_match():
    reviews = Dataset.from_dict({"id": ["a", "b", "c"], "text": ["Body  one.", "Body two.", "Body three."],
                                 "label": [0, 1, 2], "label_text": ["0", "1", "2"]})
    titled = Dataset.from_dict({"id": ["a", "b"], "text": ["Title A\n\nBody one.", "Title B\n\nSomething else."]})
    merged = merge_titles(reviews, titled)
    assert merged["title"] == ["Title A", "", ""]
    assert merged["text"] == ["Title A. Body  one.", "Body two.", "Body three."]
    assert merged["title_found"] == [True, False, False]
    assert "label_text" not in merged.column_names


def test_clean_and_tokenize():
    text = "GREAT <br />product!! See https://amazon.com/x &amp; don't buy"
    assert classic.clean(text) == "great product!! see & don't buy"
    assert classic.tokenize([text]) == [["great", "product", "!", "!", "see", "&", "do", "n't", "buy"]]


def test_encode_truncates_and_keeps_length():
    reviews = DatasetDict({"train": Dataset.from_dict({"text": ["a b c a b a", "a z"], "label": [0, 4]})})
    reviews = classic.add_tokens(reviews)
    vocab = classic.build_vocab(reviews["train"]["tokens"], min_freq=2)
    encoded = classic.encode(reviews, vocab, max_len=3)["train"]
    assert encoded["ids"] == [[vocab["a"], vocab["b"], vocab.unk_id], [vocab["a"], vocab.unk_id]]
    assert encoded["n_tokens"] == [6, 2]
    batch = next(iter(classic.make_loader(encoded, batch_size=2, num_workers=0)))
    assert batch["ids"].shape == (2, 3) and batch["ids_lengths"].tolist() == [3, 2]
    assert batch["label"].tolist() == [0, 4]


def test_glove_matrix(tmp_path):
    glove = tmp_path / "glove.txt"
    glove.write_text("good 1 2 3\nbad -1 -2 -3\nunused 9 9 9\n", encoding="utf-8")
    vocab = classic.build_vocab([["good", "good", "bad", "meh"]], min_freq=1)
    vectors = classic.load_glove(glove, vocab.itos)
    assert set(vectors) == {"good", "bad"}
    matrix, coverage = classic.embedding_matrix(vocab, vectors)
    assert matrix.shape == (len(vocab), 3)
    assert matrix[vocab["good"]].tolist() == [1, 2, 3]
    assert matrix[vocab.pad_id].abs().sum() == 0
    assert coverage["type_coverage"] == pytest.approx(2 / 3)
    assert coverage["token_coverage"] == pytest.approx(3 / 4)


def test_length_stats():
    stats = length_stats(np.arange(1, 101), cutoffs=(50, 90))
    assert stats["max"] == 100 and stats["share_over_50"] == 0.5 and stats["share_over_90"] == 0.1
