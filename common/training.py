"""Training-loop building blocks. The loops themselves stay in each question.

Metric names in History, so plots and tables line up across questions:

    epoch, train_loss, val_loss, train_acc, val_acc, val_macro_f1, val_ppl, lr, time_s
"""

import copy
import time
from pathlib import Path

import torch
from torch import nn

from .files import load_json, save_json


class AverageMeter:
    """Running mean, weighted by batch size: meter.update(loss.item(), n=batch_size)."""

    def __init__(self):
        self.total = 0.0
        self.count = 0

    def update(self, value: float, n: int = 1) -> None:
        self.total += value * n
        self.count += n

    @property
    def avg(self) -> float:
        return self.total / self.count if self.count else 0.0


class Timer:
    """with Timer() as t: ...  then t.elapsed seconds. Training duration is a Q4 deliverable."""

    def __enter__(self):
        self.start = time.perf_counter()
        self.elapsed = 0.0
        return self

    def __exit__(self, *exc):
        self.elapsed = time.perf_counter() - self.start
        return False


def format_seconds(seconds: float) -> str:
    seconds = int(round(seconds))
    hours, rest = divmod(seconds, 3600)
    minutes, secs = divmod(rest, 60)
    return f"{hours}h {minutes:02d}m {secs:02d}s" if hours else f"{minutes}m {secs:02d}s"


def count_parameters(model: nn.Module) -> dict:
    """{"total": ..., "trainable": ...}. Q3 and Q4 must report parameter counts."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total": total, "trainable": trainable}


def format_count(n: int) -> str:
    for unit, size in (("B", 1e9), ("M", 1e6), ("K", 1e3)):
        if n >= size:
            return f"{n / size:.2f}{unit}"
    return str(n)


class History:
    """Per-epoch metrics: history.log(epoch=1, train_loss=..., val_loss=...)."""

    def __init__(self, records: list[dict] | None = None):
        self.records = list(records or [])

    def log(self, **metrics) -> dict:
        metrics.setdefault("epoch", len(self.records) + 1)
        self.records.append(metrics)
        return metrics

    def __getitem__(self, key: str) -> list:
        return [record.get(key) for record in self.records]

    def __len__(self) -> int:
        return len(self.records)

    def keys(self) -> list[str]:
        seen = {}
        for record in self.records:
            seen.update(dict.fromkeys(record))
        return list(seen)

    def best(self, key: str, mode: str = "min") -> dict:
        candidates = [r for r in self.records if r.get(key) is not None]
        if not candidates:
            raise KeyError(f"no epoch has {key!r}")
        pick = min if mode == "min" else max
        return pick(candidates, key=lambda r: r[key])

    def save(self, path: str | Path) -> Path:
        return save_json(self.records, path)

    @classmethod
    def load(cls, path: str | Path) -> "History":
        return cls(load_json(path))


class EarlyStopping:
    """Stop when a validation metric stops improving, and bring back the best weights.

        stopper = EarlyStopping(mode="max", patience=3)    # Q2: val macro-F1
        stopper = EarlyStopping(mode="min", patience=3,    # Q3: val loss
                                checkpoint_path=ckpt_dir / "best.pt")
        for epoch in ...:
            ...
            stopper.step(val_metric, model)
            if stopper.should_stop:
                break
        stopper.restore(model)   # test the best checkpoint, not the last one

    Without checkpoint_path the best weights are kept as a CPU copy in memory. For
    big models (RoBERTa, the Q3 Transformer) pass a path so they go to disk instead.
    """

    def __init__(
        self,
        mode: str = "min",
        patience: int = 3,
        min_delta: float = 0.0,
        checkpoint_path: str | Path | None = None,
    ):
        if mode not in ("min", "max"):
            raise ValueError("mode must be 'min' or 'max'")
        self.mode = mode
        self.patience = patience
        self.min_delta = min_delta
        self.checkpoint_path = Path(checkpoint_path) if checkpoint_path else None
        self.best = None
        self.best_epoch = None
        self.bad_epochs = 0
        self.epochs_seen = 0
        self._best_state = None

    def _improved(self, value: float) -> bool:
        if self.best is None:
            return True
        if self.mode == "min":
            return value < self.best - self.min_delta
        return value > self.best + self.min_delta

    def step(self, value: float, model: nn.Module | None = None) -> bool:
        """Record one epoch's value. Returns True if it is a new best."""
        self.epochs_seen += 1
        improved = self._improved(value)
        if improved:
            self.best = value
            self.best_epoch = self.epochs_seen
            self.bad_epochs = 0
            if model is not None:
                self._save_best(model)
        else:
            self.bad_epochs += 1
        return improved

    def _save_best(self, model: nn.Module) -> None:
        if self.checkpoint_path is not None:
            self.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), self.checkpoint_path)
        else:
            self._best_state = {k: v.detach().to("cpu", copy=True) for k, v in model.state_dict().items()}

    @property
    def should_stop(self) -> bool:
        return self.bad_epochs >= self.patience

    def restore(self, model: nn.Module) -> nn.Module:
        if self.checkpoint_path is not None and self.checkpoint_path.exists():
            state = torch.load(self.checkpoint_path, map_location="cpu", weights_only=True)
        elif self._best_state is not None:
            state = copy.deepcopy(self._best_state)
        else:
            raise RuntimeError("nothing to restore: step() was never called with a model")
        model.load_state_dict(state)
        return model


def save_checkpoint(path: str | Path, model: nn.Module, optimizer=None, scheduler=None, scaler=None, **extra) -> Path:
    """Save weights plus whatever is needed to resume or reproduce (epoch, config, vocab path...)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"model": model.state_dict(), **extra}
    for name, obj in (("optimizer", optimizer), ("scheduler", scheduler), ("scaler", scaler)):
        if obj is not None:
            payload[name] = obj.state_dict()
    torch.save(payload, path)
    return path


def load_checkpoint(
    path: str | Path,
    model: nn.Module | None = None,
    optimizer=None,
    scheduler=None,
    scaler=None,
    map_location="cpu",
    strict: bool = True,
) -> dict:
    """Load a checkpoint written by save_checkpoint into whichever objects are passed.

    Returns the whole payload, so extras like payload["epoch"] or payload["config"]
    are available. weights_only=False because the extras can hold plain Python
    objects; only load checkpoints we made ourselves.
    """
    payload = torch.load(path, map_location=map_location, weights_only=False)
    if model is not None:
        model.load_state_dict(payload["model"], strict=strict)
    for name, obj in (("optimizer", optimizer), ("scheduler", scheduler), ("scaler", scaler)):
        if obj is not None and name in payload:
            obj.load_state_dict(payload[name])
    return payload
