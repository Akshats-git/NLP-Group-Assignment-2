"""Tests for the Q4 Plot-Twist Generator code in Question-4/. Run from the repo root: pytest -q"""

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "Question-4"))

from common import Decoding, Vocab, generate  # noqa: E402
from plot_twist.data import (BucketBatchSampler, StoryDataset, collate_lm, detokenize, sentence_case, tokenize)  # noqa: E402
from plot_twist.generation import derive_seed, generate_from_text  # noqa: E402
from plot_twist.human_eval import (agreement, group_means, krippendorff_alpha, paired_difference,  # noqa: E402
                                   quadratic_kappa, select_samples)
from plot_twist.lm_eval import evaluate_loss  # noqa: E402
from plot_twist.metrics import distinct_n, repetition_rate, summarize_group  # noqa: E402
from plot_twist.models import StoryLM  # noqa: E402

STORIES = ["Once upon a time, a cat didn't like rain.", "The dog said \"hello\" to the cat.", "A bird sang. A bird sang!"]


@pytest.fixture
def vocab():
    return Vocab.build([tokenize(s) for s in STORIES], min_freq=1)


def test_tokenize_keeps_contractions_and_punctuation():
    assert tokenize("Don’t go, it’s late!") == ["don't", "go", ",", "it's", "late", "!"]


def test_mojibake_is_repaired_before_tokenising():
    broken = "He said \u00e2\u20ac\u0153Hi\u00e2\u20ac\u009d and it\u00e2\u20ac\u2122s fine \u00e2\u20ac\u201c ok"
    assert tokenize(broken) == ["he", "said", '"', "hi", '"', "and", "it's", "fine", "-", "ok"]
    assert all(ord(ch) < 128 for tok in tokenize("caf\u00e9 \u00e2\u20ac\u0153x") for ch in tok)


def test_display_casing_restores_sentence_starts_after_quotes():
    assert sentence_case('she said, "no." the dog ran. i saw it.') == 'She said, "No." The dog ran. I saw it.'


def test_detokenize_round_trip_reads_naturally():
    text = detokenize(tokenize('The dog said "hello" to the cat.'))
    assert text == 'the dog said "hello" to the cat.'
    assert sentence_case("i saw a cat. it ran.") == "I saw a cat. It ran."


def test_next_token_pairs_stay_inside_one_story(vocab):
    ds = StoryDataset([tokenize(s) for s in STORIES], vocab)
    batch = collate_lm(vocab.pad_id)([ds[0], ds[1]])
    for row in range(2):
        n = int(batch["lengths"][row])
        ids = ds[row]
        assert batch["inputs"][row, :n].tolist() == ids[:-1]
        assert batch["targets"][row, :n].tolist() == ids[1:]
        assert batch["targets"][row, n - 1] == vocab.eos_id  # every story ends with <eos>
        assert batch["inputs"][row, 0] == vocab.sos_id
    assert (batch["targets"][batch["lengths"] < batch["targets"].size(1), -1] == vocab.pad_id).all()


def test_bucket_sampler_covers_every_story_once_and_is_seeded():
    lengths = [5, 50, 7, 49, 6, 51, 8, 48, 9, 47]
    a = list(BucketBatchSampler(lengths, 3, seed=1, chunk=2))
    b = list(BucketBatchSampler(lengths, 3, seed=1, chunk=2))
    assert a == b
    assert sorted(i for batch in a for i in batch) == list(range(10))
    sampler = BucketBatchSampler(lengths, 3, seed=1, chunk=2)
    sampler.set_epoch(1)
    assert list(sampler) != a


@pytest.mark.parametrize("cell", ["rnn", "lstm"])
def test_right_padding_does_not_change_real_positions(cell):
    torch.manual_seed(0)
    model = StoryLM(30, cell=cell, emb_dim=8, hid_dim=16).eval()
    ids = torch.randint(4, 30, (1, 6))
    padded = torch.cat([ids, torch.zeros(1, 4, dtype=torch.long)], dim=1)
    a, _ = model(ids)
    b, _ = model(padded)
    assert torch.allclose(a, b[:, :6], atol=1e-6)


def test_models_differ_only_in_the_recurrent_cell():
    rnn, lstm = StoryLM(100, "rnn"), StoryLM(100, "lstm")
    shapes = lambda m: {n: p.shape for n, p in m.named_parameters() if not n.startswith("rnn")}
    assert shapes(rnn) == shapes(lstm)
    assert sum(p.numel() for p in lstm.rnn.parameters()) > 3 * sum(p.numel() for p in rnn.rnn.parameters())


def test_perplexity_of_uniform_model_equals_vocab_size(vocab):
    model = StoryLM(len(vocab), "lstm", emb_dim=4, hid_dim=4).eval()
    for p in model.parameters():
        torch.nn.init.zeros_(p)  # zero logits -> uniform distribution
    ds = StoryDataset([tokenize(s) for s in STORIES], vocab)
    loader = torch.utils.data.DataLoader(ds, batch_size=3, collate_fn=collate_lm(vocab.pad_id))
    result = evaluate_loss(model, loader, "cpu", vocab.pad_id)
    assert result["ppl"] == pytest.approx(len(vocab), rel=1e-4)


def test_generation_is_reproducible_and_never_emits_banned_tokens(vocab):
    torch.manual_seed(0)
    model = StoryLM(len(vocab), "lstm", emb_dim=8, hid_dim=16).eval()
    kw = dict(prompt="the cat", decoding=Decoding("sample", temperature=1.5), n=4, max_new_tokens=30, device="cpu")
    a = generate_from_text(model, vocab, seed=7, **kw)
    b = generate_from_text(model, vocab, seed=7, **kw)
    c = generate_from_text(model, vocab, seed=8, **kw)
    assert [r["tokens"] for r in a] == [r["tokens"] for r in b]
    assert [r["tokens"] for r in a] != [r["tokens"] for r in c]
    ids = generate(model.step_fn(), torch.tensor([[vocab.sos_id]] * 8), 40, eos_id=vocab.eos_id,
                   decoding=Decoding("sample", temperature=3.0), generator=torch.Generator().manual_seed(0)).new_tokens
    assert not ((ids == vocab.unk_id) | (ids == vocab.sos_id)).any()


def test_greedy_decoding_ignores_the_seed(vocab):
    torch.manual_seed(0)
    model = StoryLM(len(vocab), "rnn", emb_dim=8, hid_dim=16).eval()
    kw = dict(prompt="the cat", decoding=Decoding(), n=1, max_new_tokens=20, device="cpu")
    assert generate_from_text(model, vocab, seed=1, **kw)[0]["tokens"] == generate_from_text(model, vocab, seed=2, **kw)[0]["tokens"]


def test_derive_seed_is_stable_and_key_sensitive():
    assert derive_seed(42, "a", "greedy") == derive_seed(42, "a", "greedy")
    assert derive_seed(42, "a", "greedy") != derive_seed(42, "b", "greedy")
    assert derive_seed(42, "a", "greedy") != derive_seed(43, "a", "greedy")


def test_distinct_and_repetition_definitions():
    assert distinct_n([["a", "b", "a", "b"]], 1) == 0.5
    assert distinct_n([["a", "b", "a", "b"]], 2) == pytest.approx(2 / 3)
    assert repetition_rate(["a", "b", "c", "d"]) == 0.0
    assert repetition_rate(["a", "b", "c"] * 4) == pytest.approx(1 - 3 / 10)
    assert repetition_rate(["a", "b"]) == 0.0


def test_summarize_group_length_flags():
    stories = [["a"] * 10, ["b"] * 100, ["c"] * 100]
    s = summarize_group(stories, [True, True, False], max_new_tokens=100, short_below=30)
    assert s["too_short_share"] == pytest.approx(1 / 3)
    assert s["too_long_share"] == pytest.approx(1 / 3)
    assert s["avg_length"] == pytest.approx(70)


# ---- human evaluation ----

def _rows():
    rows = []
    for model in ("rnn", "lstm"):
        for pk in ("p1", "p2", "p3"):
            for dec in ("greedy", "temp=0.8"):
                for i in (0, 1):
                    rows.append({"id": f"{model}|{pk}|{dec}|{i}", "part": "plot_twist", "model": model, "prompt_key": pk,
                                 "decoding_label": dec, "sample_index": i, "prompt": "open", "text": f"{model}{pk}{dec}{i}",
                                 "finished": True, "n_tokens": 10})
    return rows


def test_select_samples_is_blind_balanced_and_deterministic():
    pub, key = select_samples(_rows(), ["rnn", "lstm"], ["greedy", "temp=0.8"], seed=1)
    assert len(pub) == 12
    assert {s["id"] for s in pub} == set(key)
    assert all(set(s) == {"id", "prompt", "text"} for s in pub)  # nothing that names the model
    assert all(k["source_id"].endswith("|0") for k in key.values())  # fixed rule: sample 0
    assert sorted(k["model"] for k in key.values()).count("rnn") == 6
    assert pub == select_samples(_rows(), ["rnn", "lstm"], ["greedy", "temp=0.8"], seed=1)[0]
    assert [s["id"] for s in pub] == [f"S{i:02d}" for i in range(1, 13)]


def test_agreement_statistics():
    perfect = {f"s{i}": {"a": v, "b": v, "c": v} for i, v in enumerate([1, 2, 3, 4, 5, 3, 2])}
    assert krippendorff_alpha(perfect) == pytest.approx(1.0)
    assert quadratic_kappa([1, 2, 3, 4, 5], [1, 2, 3, 4, 5]) == pytest.approx(1.0)
    opposite = quadratic_kappa([1, 2, 3, 4, 5], [5, 4, 3, 2, 1])
    assert opposite < -0.9
    noisy = {f"s{i}": {"a": v, "b": w} for i, (v, w) in enumerate([(1, 5), (5, 1), (2, 4), (4, 2), (3, 3), (1, 4), (5, 2)])}
    assert krippendorff_alpha(noisy) < 0


def test_group_means_and_paired_difference():
    key = {}
    ratings = []
    for i, (model, pk) in enumerate([(m, p) for p in ("p1", "p2", "p3", "p4") for m in ("rnn", "lstm")]):
        sid = f"S{i:02d}"
        key[sid] = {"model": model, "decoding": "greedy", "prompt_key": pk}
        base = 4 if model == "lstm" else 2
        for rater, bump in (("a", 0), ("b", 1)):
            ratings.append({"rater": rater, "sample_id": sid, "coherence": base + bump - (model == "lstm" and bump),
                            "grammar": base, "relevance": base, "creativity": base})
    means = group_means(ratings, key, ("model",))
    assert means["lstm"]["grammar"] == 4 and means["rnn"]["grammar"] == 2
    diff = paired_difference(ratings, key, "lstm", "rnn", n_boot=200)
    assert diff["n_pairs"] == 4
    assert diff["grammar"]["mean_diff"] == 2 and diff["grammar"]["wins"] == 4 and diff["grammar"]["losses"] == 0
    assert agreement(ratings)["grammar"]["exact_agreement"] == 1.0
