"""Work out which training run a downloaded best.pt belongs to.

Kaggle keeps every notebook version's output, so the same file name (best.pt) turns up many
times, from smoke tests, the faulty transformers-5.16.1 RoBERTa runs, and the real runs. This
script tells them apart by behaviour: it runs each checkpoint on a seeded sample of the
*validation* split and compares its predictions with the committed
outputs/results/<run>/val_predictions.jsonl of all six runs. The right checkpoint agrees with its
run on about 99-100% of reviews; two different runs agree on at most 94%. The test split is never loaded.

    # from Question-2/
    ../.venv/bin/python demo/find_checkpoint.py ~/Downloads/kaggle            # check every *.pt under a folder
    ../.venv/bin/python demo/find_checkpoint.py a.pt b.pt --install           # also copy matches into outputs/checkpoints/
"""

import argparse
import hashlib
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE.parents[1]), str(HERE.parent)]

import datasets  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import transformers  # noqa: E402
from datasets import DatasetDict  # noqa: E402

datasets.disable_progress_bars()
transformers.logging.set_verbosity_error()
transformers.logging.disable_progress_bar()

from common import checkpoints_dir, get_device, load_jsonl, results_dir  # noqa: E402

RUNS = {"bigru": ["bigru_seed42", "bigru_seed43"],
        "roberta": ["roberta_mean_seed42", "roberta_mean_seed43", "roberta_first_seed42", "roberta_first_seed43"]}
MATCH = 0.98          # agreement needed to call it a match
VOCAB_SIZE = 32_878   # the full-data BiGRU vocabulary; smoke-test runs built a smaller one


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 22), b""):
            h.update(block)
    return h.hexdigest()


def load_state(path: Path) -> dict:
    state = torch.load(path, map_location="cpu", weights_only=True)
    if isinstance(state, dict) and isinstance(state.get("model"), dict):
        state = state["model"]
    return {k.removeprefix("module."): v for k, v in state.items()}


def kind_of(state: dict) -> str | None:
    if "embedding.weight" in state and any(k.startswith("gru.") for k in state):
        return "bigru"
    if any(k.startswith("encoder.embeddings.") for k in state) and "head.weight" in state:
        return "roberta"
    return None


class Validation:
    """A seeded validation sample, tokenised lazily for each pipeline."""

    def __init__(self, n: int, device: torch.device):
        from q2.data import load_reviews
        self.split = load_reviews(("validation",), limit=n)["validation"]
        self.ids, self.labels = list(self.split["id"]), np.array(self.split["label"])
        self.device = device
        self._bigru = self._roberta = None

    def bigru_loader(self):
        if self._bigru is None:
            from q2 import classic
            from q2.evaluation import classic_vocab
            vocab = classic_vocab()
            split = classic.encode(classic.add_tokens(DatasetDict({"v": self.split}))["v"], vocab)
            self._bigru = classic.make_loader(split, batch_size=256, num_workers=0)
        return self._bigru

    def roberta_loader(self):
        if self._roberta is None:
            from q2 import roberta_data
            tokenizer = roberta_data.get_tokenizer()
            split = roberta_data.tokenize(DatasetDict({"v": self.split}), tokenizer)["v"]
            self._roberta = roberta_data.make_loader(split, tokenizer, batch_size=32, num_workers=0)
        return self._roberta


@torch.no_grad()
def predict_bigru(state: dict, data: Validation) -> dict[str, np.ndarray]:
    from q2.models import BiGRUAttention
    model = BiGRUAttention(torch.zeros_like(state["embedding.weight"]))
    model.load_state_dict(state)
    model.to(data.device).eval()
    preds = [model(b["ids"].to(data.device), b["ids_lengths"].to(data.device)).argmax(-1).cpu()
             for b in data.bigru_loader()]
    return {"bigru": torch.cat(preds).numpy()}


@torch.no_grad()
def predict_roberta(state: dict, data: Validation) -> dict[str, np.ndarray]:
    """Pooling is not stored in the weights (both variants have the same parameters), so both are tried
    from one encoder pass."""
    from q2.models import RobertaClassifier, masked_mean
    model = RobertaClassifier(pooling="mean")
    model.load_state_dict(state)
    model.to(data.device).eval()
    preds = {"mean": [], "first": []}
    for b in data.roberta_loader():
        ids, mask = b["input_ids"].to(data.device), b["attention_mask"].to(data.device)
        hidden = model.encoder(input_ids=ids, attention_mask=mask).last_hidden_state
        preds["mean"].append(model.head(masked_mean(hidden.float(), mask)).argmax(-1).cpu())
        preds["first"].append(model.head(hidden[:, 0]).argmax(-1).cpu())
    return {k: torch.cat(v).numpy() for k, v in preds.items()}


def saved_predictions(run: str, ids: list[str]) -> np.ndarray:
    by_id = {r["id"]: r["pred"] for r in load_jsonl(results_dir(2) / run / "val_predictions.jsonl")}
    return np.array([by_id[i] for i in ids])


def check(path: Path, data: Validation) -> dict:
    state = load_state(path)
    kind = kind_of(state)
    out = {"path": path, "kind": kind, "size_mb": path.stat().st_size / 2**20, "run": None, "agreement": {}}
    if kind is None:
        out["note"] = "not a Q2 BiGRU or RoBERTa checkpoint"
        return out
    if kind == "bigru" and state["embedding.weight"].shape[0] != VOCAB_SIZE:
        out["note"] = (f"vocabulary of {state['embedding.weight'].shape[0]:,} words instead of {VOCAB_SIZE:,}: "
                       "a smoke-test (--limit) run")
        return out
    preds = predict_bigru(state, data) if kind == "bigru" else predict_roberta(state, data)
    for run in RUNS[kind]:
        variant = "bigru" if kind == "bigru" else run.split("_")[1]
        out["agreement"][run] = float(np.mean(preds[variant] == saved_predictions(run, data.ids)))
    best = max(out["agreement"], key=out["agreement"].get)
    variant = "bigru" if kind == "bigru" else best.split("_")[1]
    out["val_acc"] = float(np.mean(preds[variant] == data.labels))
    if out["agreement"][best] >= MATCH:
        out["run"] = best
    else:
        out["note"] = "matches none of the six committed runs (an older or faulty version, or a smoke test)"
    return out


def install(result: dict) -> str:
    target = checkpoints_dir(2) / result["run"] / "best.pt"
    if target.exists():
        return f"already in place: {target}" if sha256(target) == result["sha"] else f"NOT copied, {target} exists and differs"
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(result["path"], target)
    return f"copied to {target}"


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("paths", nargs="+", type=Path, help="best.pt files, or folders searched for *.pt")
    parser.add_argument("--n", type=int, default=500, help="validation reviews to compare on (default 500)")
    parser.add_argument("--device", default="", help="cpu, mps or cuda (default: best available)")
    parser.add_argument("--install", action="store_true", help="copy each match to outputs/checkpoints/<run>/best.pt")
    args = parser.parse_args(argv)

    files = []
    for p in args.paths:
        p = p.expanduser()
        files += sorted(p.rglob("*.pt")) if p.is_dir() else [p]
    files = [f for f in files if f.is_file() and checkpoints_dir(2) not in f.resolve().parents]
    if not files:
        sys.exit("no .pt files found (unzip Kaggle downloads first)")

    by_hash: dict[str, list[Path]] = {}
    for f in files:
        print(f"hashing {f}", flush=True)
        by_hash.setdefault(sha256(f), []).append(f)
    print(f"\n{len(files)} file(s), {len(by_hash)} distinct\n")

    data = Validation(args.n, get_device(args.device or None))
    results = []
    for digest, paths in by_hash.items():
        print(f"checking {paths[0]} on {args.n} validation reviews ...", flush=True)
        result = check(paths[0], data) | {"sha": digest, "copies": paths}
        results.append(result)

    print("\n" + "=" * 100)
    for r in results:
        print(f"\n{r['path']}  ({r['size_mb']:.0f} MB, sha {r['sha'][:12]})")
        for dup in r["copies"][1:]:
            print(f"  identical copy: {dup}")
        for run, a in sorted(r["agreement"].items(), key=lambda x: -x[1]):
            print(f"  agrees with {run:22} {a:6.1%}")
        if "val_acc" in r:
            print(f"  accuracy on this sample: {r['val_acc']:.1%}")
        print(f"  ==> {r['run'].upper() if r['run'] else 'NO MATCH'}" + (f"  ({r['note']})" if r.get("note") else ""))
        if args.install and r["run"]:
            print(f"  {install(r)}")

    found = {r["run"] for r in results if r["run"]}
    print("\nFound:  " + (", ".join(sorted(found)) or "none"))
    print("Missing: " + (", ".join(run for runs in RUNS.values() for run in runs if run not in found) or "none"))


if __name__ == "__main__":
    main()
