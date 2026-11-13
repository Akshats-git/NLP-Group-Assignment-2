"""Live inference for the demo: both trained models on one review, with their explanations.

Everything goes through the q2 package, so a typed review sees exactly the pipelines of the report:
"<title>. <body>" -> (BiGRU) clean + lowercase + spaCy tokens + vocab ids, max 200
                  -> (RoBERTa) byte-level BPE, <s> ... </s>, max 256 keeping </s>
Checkpoints come from Question-2/outputs/checkpoints/<run>/best.pt (see find_checkpoint.py).
"""

import os

os.environ.setdefault("HF_HUB_OFFLINE", "1")        # roberta-base comes from the local Hugging Face cache,
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")  # so the demo works without internet

import sys  # noqa: E402
import threading  # noqa: E402
import time  # noqa: E402
from dataclasses import dataclass, field  # noqa: E402
from pathlib import Path  # noqa: E402

HERE = Path(__file__).resolve().parent
for path in (HERE.parents[1], HERE.parent):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import numpy as np  # noqa: E402
import torch  # noqa: E402
import transformers  # noqa: E402

from common import checkpoints_dir, get_device  # noqa: E402
from q2 import classic, roberta_data  # noqa: E402
from q2.data import join_title_body  # noqa: E402
from q2.evaluation import classic_vocab, integrated_gradients, load_bigru, load_roberta, merge_subwords, run_config  # noqa: E402

transformers.logging.set_verbosity_error()
transformers.logging.disable_progress_bar()

SEEDS = (42, 43)
STARS = np.arange(1, 6)


def run_name(model: str, seed: int) -> str:
    """model: "bigru", "mean" or "first"."""
    return f"bigru_seed{seed}" if model == "bigru" else f"roberta_{model}_seed{seed}"


def available_runs() -> list[str]:
    return sorted(p.parent.name for p in checkpoints_dir(2, create=False).glob("*/best.pt"))


@dataclass
class Prediction:
    probs: np.ndarray                     # [5], stars 1..5
    seconds: float
    n_tokens: int                         # tokens before truncation
    max_len: int
    tokens: list[str] = field(default_factory=list)          # BiGRU: the tokens it read
    attention: np.ndarray | None = None                      # BiGRU: alpha per token
    unknown: list[bool] = field(default_factory=list)        # BiGRU: token is <unk>

    @property
    def stars(self) -> int:
        return int(self.probs.argmax()) + 1

    @property
    def confidence(self) -> float:
        return float(self.probs.max())

    @property
    def expected(self) -> float:
        return float((self.probs * STARS).sum())

    @property
    def truncated(self) -> bool:
        return self.n_tokens > self.max_len


@dataclass
class Explanation:
    words: list[str]
    scores: np.ndarray        # contribution of each word to the target logit
    target_stars: int
    logit_change: float       # logit(review) - logit(baseline); the scores add up to this
    gap: float                # completeness error: sum(scores) - logit_change
    steps: int
    seconds: float


class Demo:
    """Loads models lazily and keeps them. One lock: Streamlit reruns in threads."""

    def __init__(self, device: str | None = None):
        self.device = get_device(device)           # RoBERTa: mps on a Mac, cuda, or cpu
        self.vocab = classic_vocab()
        self.tokenizer = roberta_data.get_tokenizer(run_config("roberta_mean_seed42")["model_name"])
        self.bigru_max_len = run_config("bigru_seed42")["max_len"]
        self.roberta_max_len = run_config("roberta_mean_seed42")["max_len"]
        self._models: dict[str, torch.nn.Module] = {}
        self._lock = threading.Lock()

    # ---------------------------------------------------------------- models
    def model(self, name: str) -> torch.nn.Module:
        if name not in self._models:
            if name.startswith("bigru"):
                self._models[name] = load_bigru(name, self.vocab)              # small, CPU is fastest
            else:
                self._models[name] = load_roberta(name).to(self.device)
        return self._models[name]

    def warm_up(self, names) -> None:
        for name in names:
            self.model(name)
        self.predict_bigru("warm up", SEEDS[0])
        self.predict_roberta("warm up", "mean", SEEDS[0])

    # ---------------------------------------------------------------- inputs
    @staticmethod
    def text(title: str, body: str) -> str:
        return join_title_body(title or "", body or "")

    def roberta_ids(self, text: str) -> tuple[torch.Tensor, int]:
        ids = self.tokenizer(text, verbose=False)["input_ids"]
        n = len(ids)
        if n > self.roberta_max_len:                                   # same rule as q2.roberta_data.tokenize
            ids = ids[: self.roberta_max_len - 1] + [self.tokenizer.eos_token_id]
        return torch.tensor([ids], device=self.device), n

    # ---------------------------------------------------------------- predictions
    @torch.no_grad()
    def predict_bigru(self, text: str, seed: int) -> Prediction:
        start = time.perf_counter()
        tokens = classic.tokenize([text])[0]
        ids = self.vocab.encode(tokens, max_len=self.bigru_max_len) or [self.vocab.unk_id]
        kept = tokens[: len(ids)] or ["<unk>"]
        with self._lock:
            logits, alpha = self.model(run_name("bigru", seed))(
                torch.tensor([ids]), torch.tensor([len(ids)]), return_attention=True)
        return Prediction(probs=torch.softmax(logits[0].float(), -1).numpy(), seconds=time.perf_counter() - start,
                          n_tokens=len(tokens), max_len=self.bigru_max_len, tokens=kept,
                          attention=alpha[0, : len(ids)].numpy(), unknown=[i == self.vocab.unk_id for i in ids])

    @torch.no_grad()
    def predict_roberta(self, text: str, pooling: str, seed: int) -> Prediction:
        start = time.perf_counter()
        ids, n = self.roberta_ids(text)
        with self._lock:
            logits = self.model(run_name(pooling, seed))(ids, torch.ones_like(ids))
        return Prediction(probs=torch.softmax(logits[0].float(), -1).cpu().numpy(),
                          seconds=time.perf_counter() - start, n_tokens=n, max_len=self.roberta_max_len)

    def explain_roberta(self, text: str, pooling: str, seed: int, target_stars: int, steps: int = 64,
                        chunk: int = 32) -> Explanation:
        """Integrated gradients for the logit of `target_stars`, sub-word pieces summed into words.

        `chunk` is how many of the steps go through the model at once: it changes memory and speed, not the result.
        """
        start = time.perf_counter()
        ids, _ = self.roberta_ids(text)
        with self._lock:
            scores, gap = integrated_gradients(self.model(run_name(pooling, seed)), ids, target_stars - 1,
                                               self.tokenizer.pad_token_id, steps=steps, chunk=chunk)
        scores = scores.float().cpu().numpy()
        pieces = self.tokenizer.convert_ids_to_tokens(ids[0].tolist())
        words, merged = merge_subwords(pieces, scores, decode=self.tokenizer.convert_tokens_to_string)
        return Explanation(words=words, scores=merged, target_stars=target_stars, logit_change=float(scores.sum() - gap),
                           gap=gap, steps=steps, seconds=time.perf_counter() - start)
