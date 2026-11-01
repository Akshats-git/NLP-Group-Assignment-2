"""Part 1: draw the TinyStories subset, tokenise it, build the vocabulary and report data statistics.

    python Question-4/prepare_data.py

Writes (stories are cached under data/, which is git-ignored and never shipped):
    Question-4/data/{train,validation,test}.jsonl      raw stories of the subset
    Question-4/outputs/vocab.json                      the vocabulary (shipped, needed to reproduce results)
    Question-4/outputs/display_case.json               names to capitalise when a story is shown to a reader (display only)
    Question-4/outputs/results/data_summary.json       sizes, vocab size, OOV rate, length stats, split hashes
    Question-4/outputs/results/next_token_example.md   one story turned into input/target pairs
"""

import hashlib
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from plot_twist import config as cfg
from plot_twist.data import StoryDataset, build_vocab, detokenize, find_proper_nouns, load_story_subset, tokenize
from common import load_json, load_jsonl, save_json, save_jsonl, set_seed


def split_hash(stories: list[str]) -> str:
    return hashlib.sha256("\n".join(stories).encode("utf-8")).hexdigest()[:16]


def length_stats(lengths: list[int]) -> dict:
    s = sorted(lengths)
    q = lambda p: s[min(len(s) - 1, int(p * len(s)))]
    return {"min": s[0], "median": q(0.5), "mean": round(statistics.fmean(s), 1), "p90": q(0.9), "p95": q(0.95), "max": s[-1]}


def main():
    p = cfg.paths()
    set_seed(cfg.DATA["subset_seed"])

    cache = {split: p["data"] / f"{split}.jsonl" for split in ("train", "validation", "test")}
    if all(path.exists() for path in cache.values()):
        splits = {split: [r["text"] for r in load_jsonl(path)] for split, path in cache.items()}
        filter_stats = load_json(p["results"] / "data_summary.json")["filtering"] if (p["results"] / "data_summary.json").exists() else None
        print("reusing cached stories in", p["data"])
    else:
        d = cfg.DATA
        splits, filter_stats = load_story_subset(d["n_train"], d["n_val"], d["n_test"], d["max_tokens"], d["subset_seed"], d["pool_factor"])
        for split, stories in splits.items():
            save_jsonl(({"text": s} for s in stories), cache[split])

    tokens = {split: [tokenize(s) for s in stories] for split, stories in splits.items()}
    vocab = build_vocab(tokens["train"], min_freq=cfg.DATA["min_freq"])
    vocab.save(p["results"].parent / "vocab.json")
    names = [w for w in find_proper_nouns(splits["train"]) if w in vocab]
    save_json(names, p["results"].parent / "display_case.json")  # only used to restore capitals when showing stories

    # Story-level split check: no story text appears in two splits.
    sets = {k: set(v) for k, v in splits.items()}
    overlaps = {f"{a}&{b}": len(sets[a] & sets[b]) for a, b in (("train", "validation"), ("train", "test"), ("validation", "test"))}

    summary = {
        "dataset": "roneneldan/TinyStories",
        "config": cfg.DATA,
        "filtering": filter_stats,
        "stories": {k: len(v) for k, v in splits.items()},
        "tokens": {k: sum(map(len, v)) for k, v in tokens.items()},
        "story_length_tokens": {k: length_stats([len(t) for t in v]) for k, v in tokens.items()},
        "vocab_size": len(vocab),
        "vocab_regular_words": len(vocab) - 4,
        "mojibake_stories_in_train_repaired": sum("\u00e2\u20ac" in s for s in splits["train"]),
        "train_tokens_covered_by_vocab": 1 - vocab.oov_rate(tokens["train"]),
        "oov_rate": {k: vocab.oov_rate(v) for k, v in tokens.items()},
        "exact_duplicate_stories_across_splits": overlaps,
        "split_sha256_16": {k: split_hash(v) for k, v in splits.items()},
    }
    save_json(summary, p["results"] / "data_summary.json")

    # One story as next-token prediction pairs.
    ds = StoryDataset(tokens["train"][:1], vocab)
    ids = ds.ids[0]
    inputs, targets = ids[:-1], ids[1:]
    lines = ["# Next-token prediction example", "",
             "Story (first 25 tokens shown):", "", "> " + detokenize(tokens["train"][0][:25]) + " ...", "",
             "| step | input token | target (next) token |", "|---|---|---|"]
    for i in range(12):
        lines.append(f"| {i} | `{vocab.itos[inputs[i]]}` | `{vocab.itos[targets[i]]}` |")
    lines += ["| ... | ... | ... |", f"| {len(inputs) - 1} | `{vocab.itos[inputs[-1]]}` | `{vocab.itos[targets[-1]]}` |", "",
              f"Full id sequence length: {len(ids)} (= `<sos>` + {len(ids) - 2} words/punctuation + `<eos>`).", ""]
    (p["results"] / "next_token_example.md").write_text("\n".join(lines), encoding="utf-8")

    for k, v in summary.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()
