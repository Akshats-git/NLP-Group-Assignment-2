"""Train the BiGRU + additive attention baseline.

    python -m q2.train_bigru --seed 42                                  # from Question-2/
    python -m q2.train_bigru --limit 2000 --epochs 2 --device cpu       # quick check

Adam (lr 1e-3), gradient clipping at 1.0, GloVe embeddings frozen for the first epoch and
fine-tuned after that, validation after every epoch and early stopping on validation
macro-F1 (patience 3). The best epoch's weights are kept. The test split is never loaded here.

Writes to outputs/results/<run_name>/: run_info.json, history.json, summary.json and
val_predictions.jsonl. The best weights go to outputs/checkpoints/<run_name>/best.pt.
"""

import argparse
import os
import time
from dataclasses import asdict, dataclass, fields

import torch
from torch import nn

from common import (REPO_ROOT, EarlyStopping, History, autocast, checkpoints_dir, count_parameters, get_device,
                    grad_scaler, results_dir, save_json, save_jsonl, save_run_info, set_seed, to_device)
from common.device import amp_dtype

from . import classic
from .metrics import classification_metrics
from .models import BiGRUAttention
from .training import Segment


@dataclass
class BiGRUConfig:
    seed: int = 42
    min_freq: int = classic.MIN_FREQ
    max_len: int = classic.MAX_LEN
    hidden_size: int = 256
    num_layers: int = 2
    attention_dim: int = 256
    dropout: float = 0.3
    lr: float = 1e-3
    max_grad_norm: float = 1.0
    freeze_epochs: int = 1
    epochs: int = 15
    patience: int = 3
    batch_size: int = 128
    eval_batch_size: int = 512
    limit: int | None = None
    num_workers: int = 2
    device: str = ""   # empty picks cuda, then mps, then cpu

    @property
    def run_name(self) -> str:
        name = f"bigru_seed{self.seed}"
        return f"{name}_limit{self.limit}" if self.limit else name


@torch.no_grad()
def predict(model: nn.Module, loader, device) -> tuple[float, list[int], list[int]]:
    """Mean cross-entropy, predicted labels and true labels over a loader, in the loader's order."""
    model.eval()
    loss_fn = nn.CrossEntropyLoss(reduction="sum")
    total_loss, preds, labels = 0.0, [], []
    for batch in loader:
        batch = to_device(batch, device)
        with autocast(device):
            logits = model(batch["ids"], batch["ids_lengths"])
        total_loss += loss_fn(logits.float(), batch["label"]).item()
        preds += logits.argmax(dim=-1).tolist()
        labels += batch["label"].tolist()
    return total_loss / len(labels), preds, labels


def train(config: BiGRUConfig) -> dict:
    set_seed(config.seed)
    device = get_device(config.device or None)
    run_dir = results_dir(2) / config.run_name
    checkpoint = checkpoints_dir(2) / config.run_name / "best.pt"

    data, vocab, embeddings = classic.prepare(config.limit, config.min_freq, config.max_len, num_proc=os.cpu_count())
    set_seed(config.seed)   # prepare() may have drawn random vectors for words without GloVe
    train_loader = classic.make_loader(data["train"], config.batch_size, shuffle=True, seed=config.seed,
                                       num_workers=config.num_workers)
    val_loader = classic.make_loader(data["validation"], config.eval_batch_size, num_workers=config.num_workers)

    model = BiGRUAttention(embeddings, config.hidden_size, config.num_layers, config.attention_dim,
                           config.dropout).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)   # frozen embeddings get no grad, so Adam skips them
    scaler = grad_scaler(device)
    loss_fn = nn.CrossEntropyLoss()

    model.set_embedding_trainable(config.freeze_epochs == 0)
    parameters = {"total": count_parameters(model)["total"], "embedding": model.embedding.weight.numel()}
    save_run_info(run_dir, asdict(config), seed=config.seed, run_name=config.run_name, amp_dtype=amp_dtype(device),
                  parameters=parameters, vocab_size=len(vocab), train_reviews=len(data["train"]),
                  val_reviews=len(data["validation"]))
    print(f"{config.run_name}: {len(data['train']):,} train reviews, vocab {len(vocab):,}, "
          f"{len(train_loader):,} batches/epoch, {device}, amp {amp_dtype(device)}")

    history = History()
    stopper = EarlyStopping(mode="max", patience=config.patience, checkpoint_path=checkpoint)
    best_preds, start = None, time.perf_counter()
    for epoch in range(1, config.epochs + 1):
        frozen = epoch <= config.freeze_epochs
        model.set_embedding_trainable(not frozen)
        model.train()
        segment = Segment()
        for batch in train_loader:
            batch = to_device(batch, device)
            with autocast(device):
                logits = model(batch["ids"], batch["ids_lengths"])
            loss = loss_fn(logits.float(), batch["label"])
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
            segment.update(loss, logits, batch["label"])

        val_loss, val_preds, val_labels = predict(model, val_loader, device)
        val = classification_metrics(val_labels, val_preds)
        seen = classification_metrics(segment.labels, segment.preds)
        record = history.log(epoch=epoch, train_loss=segment.loss.avg, val_loss=val_loss,
                             train_acc=seen["accuracy"], val_acc=val["accuracy"],
                             train_macro_f1=seen["macro_f1"], val_macro_f1=val["macro_f1"], val_mae=val["mae"],
                             embeddings_frozen=frozen, time_s=time.perf_counter() - start)
        if stopper.step(val["macro_f1"], model):
            best_preds = val_preds
        history.save(run_dir / "history.json")
        print(f"  epoch {epoch:>2}{' (frozen emb)' if frozen else ''} | train loss {record['train_loss']:.4f} | "
              f"val loss {val_loss:.4f} acc {val['accuracy']:.4f} macro-F1 {val['macro_f1']:.4f} "
              f"MAE {val['mae']:.4f} | {record['time_s'] / 60:.1f} min")
        if stopper.should_stop:
            print(f"  early stop: no gain for {config.patience} epochs")
            break

    best = history.best("val_macro_f1", mode="max")
    val_split = data["validation"]
    save_jsonl(({"id": i, "label": y, "pred": p} for i, y, p in zip(val_split["id"], val_split["label"], best_preds)),
               run_dir / "val_predictions.jsonl")
    summary = {"run_name": config.run_name, "best_epoch": best["epoch"], "train_time_s": time.perf_counter() - start,
               "checkpoint": str(checkpoint.relative_to(REPO_ROOT)),
               "val": classification_metrics(val_split["label"], best_preds)}
    save_json(summary, run_dir / "summary.json")
    print(f"best: epoch {best['epoch']}, val macro-F1 {best['val_macro_f1']:.4f}, "
          f"total {summary['train_time_s'] / 60:.1f} min")
    return summary


def parse_args(argv=None) -> BiGRUConfig:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    for field in fields(BiGRUConfig):
        kind = int if field.name == "limit" else type(field.default)
        parser.add_argument(f"--{field.name.replace('_', '-')}", type=kind, default=field.default)
    return BiGRUConfig(**vars(parser.parse_args(argv)))


if __name__ == "__main__":
    train(parse_args())
