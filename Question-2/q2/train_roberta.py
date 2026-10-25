"""Fine-tune RoBERTa for 5-class rating prediction.

    python -m q2.train_roberta --pooling mean --seed 42          # from Question-2/
    python -m q2.train_roberta --pooling first --seed 43 --limit 2000 --epochs 1   # quick check

AdamW (lr 2e-5, weight decay 0.01, none on biases and LayerNorm), 6% linear warmup then
linear decay, fp16 / bf16 autocast, gradient clipping at 1.0. Validation runs
`evals_per_epoch` times per epoch; the checkpoint with the best validation macro-F1 is
kept. The test split is never loaded here.

Writes to outputs/results/<run_name>/: run_info.json, history.json, summary.json and
val_predictions.jsonl. The best weights go to outputs/checkpoints/<run_name>/best.pt.
"""

import argparse
import math
import os
import time
from dataclasses import asdict, dataclass, fields

import torch
from torch import nn
from transformers import get_linear_schedule_with_warmup

from common import (REPO_ROOT, AverageMeter, EarlyStopping, History, autocast, checkpoints_dir, count_parameters,
                    get_device, grad_scaler, results_dir, save_json, save_jsonl, save_run_info, set_seed, to_device)
from common.device import amp_dtype

from .data import load_reviews
from .metrics import classification_metrics
from .models import RobertaClassifier
from .roberta_data import MAX_LEN, MODEL_NAME, get_tokenizer, make_loader, tokenize


@dataclass
class RobertaConfig:
    seed: int = 42
    pooling: str = "mean"
    model_name: str = MODEL_NAME
    max_len: int = MAX_LEN
    dropout: float = 0.1
    lr: float = 2e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.06
    epochs: int = 2
    batch_size: int = 32
    grad_accum: int = 1
    max_grad_norm: float = 1.0
    evals_per_epoch: int = 2
    patience: int = 3
    eval_batch_size: int = 128
    limit: int | None = None
    num_workers: int = 2
    device: str = ""   # empty picks cuda, then mps, then cpu

    @property
    def run_name(self) -> str:
        name = f"roberta_{self.pooling}_seed{self.seed}"
        return f"{name}_limit{self.limit}" if self.limit else name


def param_groups(model: nn.Module, weight_decay: float) -> list[dict]:
    """No weight decay on biases and LayerNorm weights, the usual BERT-style setting."""
    decay, no_decay = [], []
    for name, param in model.named_parameters():
        (no_decay if param.ndim < 2 or "LayerNorm" in name else decay).append(param)
    return [{"params": decay, "weight_decay": weight_decay}, {"params": no_decay, "weight_decay": 0.0}]


@torch.no_grad()
def predict(model: nn.Module, loader, device) -> tuple[float, list[int], list[int]]:
    """Mean cross-entropy, predicted labels and true labels over a loader, in the loader's order."""
    model.eval()
    loss_fn = nn.CrossEntropyLoss(reduction="sum")
    total_loss, preds, labels = 0.0, [], []
    for batch in loader:
        batch = to_device(batch, device)
        with autocast(device):
            logits = model(batch["input_ids"], batch["attention_mask"])
        total_loss += loss_fn(logits.float(), batch["labels"]).item()
        preds += logits.argmax(dim=-1).tolist()
        labels += batch["labels"].tolist()
    return total_loss / len(labels), preds, labels


class Segment:
    """Training loss and predictions since the last evaluation (dropout on, so a rough train signal)."""

    def __init__(self):
        self.loss, self.preds, self.labels = AverageMeter(), [], []

    def update(self, loss: torch.Tensor, logits: torch.Tensor, labels: torch.Tensor) -> None:
        self.loss.update(loss.item(), n=len(labels))
        self.preds += logits.argmax(dim=-1).tolist()
        self.labels += labels.tolist()


def train(config: RobertaConfig) -> dict:
    set_seed(config.seed)
    device = get_device(config.device or None)
    run_dir = results_dir(2) / config.run_name
    checkpoint = checkpoints_dir(2) / config.run_name / "best.pt"

    tokenizer = get_tokenizer(config.model_name)
    data = tokenize(load_reviews(limit=config.limit), tokenizer, config.max_len, num_proc=os.cpu_count())
    train_loader = make_loader(data["train"], tokenizer, config.batch_size, shuffle=True, seed=config.seed,
                               num_workers=config.num_workers)
    val_loader = make_loader(data["validation"], tokenizer, config.eval_batch_size, num_workers=config.num_workers)

    model = RobertaClassifier(config.model_name, config.pooling, config.dropout).to(device)
    gpus = torch.cuda.device_count()
    net = nn.DataParallel(model) if gpus > 1 else model  # Kaggle's T4 x2: each batch is split over both GPUs

    optimizer = torch.optim.AdamW(param_groups(model, config.weight_decay), lr=config.lr)
    total_steps = math.ceil(len(train_loader) / config.grad_accum) * config.epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, int(config.warmup_ratio * total_steps), total_steps)
    scaler = grad_scaler(device)
    loss_fn = nn.CrossEntropyLoss()

    save_run_info(run_dir, asdict(config), seed=config.seed, run_name=config.run_name, gpus=gpus,
                  amp_dtype=amp_dtype(device), parameters=count_parameters(model), total_steps=total_steps,
                  train_reviews=len(data["train"]), val_reviews=len(data["validation"]))
    print(f"{config.run_name}: {len(data['train']):,} train reviews, {len(train_loader):,} batches/epoch, "
          f"{total_steps:,} optimiser steps, {device} x{max(gpus, 1)}, amp {amp_dtype(device)}")

    history = History()
    stopper = EarlyStopping(mode="max", patience=config.patience, checkpoint_path=checkpoint)
    eval_at = {math.ceil(len(train_loader) * k / config.evals_per_epoch) for k in range(1, config.evals_per_epoch + 1)}
    step, best_preds, start = 0, None, time.perf_counter()

    for epoch in range(1, config.epochs + 1):
        net.train()
        segment = Segment()
        for i, batch in enumerate(train_loader, start=1):
            batch = to_device(batch, device)
            with autocast(device):
                logits = net(batch["input_ids"], batch["attention_mask"])
            loss = loss_fn(logits.float(), batch["labels"])
            scaler.scale(loss / config.grad_accum).backward()
            segment.update(loss, logits, batch["labels"])

            if i % config.grad_accum == 0 or i == len(train_loader):
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()
                step += 1

            if i not in eval_at:
                continue
            val_loss, val_preds, val_labels = predict(net, val_loader, device)
            val = classification_metrics(val_labels, val_preds)
            seen = classification_metrics(segment.labels, segment.preds)
            record = history.log(epoch=round(epoch - 1 + i / len(train_loader), 3), step=step,
                                 train_loss=segment.loss.avg, val_loss=val_loss,
                                 train_acc=seen["accuracy"], val_acc=val["accuracy"],
                                 train_macro_f1=seen["macro_f1"], val_macro_f1=val["macro_f1"],
                                 val_mae=val["mae"], lr=scheduler.get_last_lr()[0], time_s=time.perf_counter() - start)
            if stopper.step(val["macro_f1"], model):
                best_preds = val_preds
            history.save(run_dir / "history.json")
            print(f"  epoch {record['epoch']:.2f} | train loss {record['train_loss']:.4f} | val loss {val_loss:.4f} "
                  f"acc {val['accuracy']:.4f} macro-F1 {val['macro_f1']:.4f} MAE {val['mae']:.4f} | "
                  f"{record['time_s'] / 60:.1f} min")
            segment = Segment()
            net.train()
            if stopper.should_stop:
                break
        if stopper.should_stop:
            print(f"  early stop: no gain for {config.patience} evaluations")
            break

    best = history.best("val_macro_f1", mode="max")
    val_split = data["validation"]
    save_jsonl(({"id": i, "label": y, "pred": p} for i, y, p in zip(val_split["id"], val_split["label"], best_preds)),
               run_dir / "val_predictions.jsonl")
    summary = {"run_name": config.run_name, "best_epoch": best["epoch"], "best_step": best["step"],
               "train_time_s": time.perf_counter() - start, "checkpoint": str(checkpoint.relative_to(REPO_ROOT)),
               "val": classification_metrics(val_split["label"], best_preds)}
    save_json(summary, run_dir / "summary.json")
    print(f"best: epoch {best['epoch']:.2f}, val macro-F1 {best['val_macro_f1']:.4f}, "
          f"total {summary['train_time_s'] / 60:.1f} min")
    return summary


def parse_args(argv=None) -> RobertaConfig:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    for field in fields(RobertaConfig):
        kind = int if field.name == "limit" else type(field.default)
        parser.add_argument(f"--{field.name.replace('_', '-')}", type=kind, default=field.default)
    return RobertaConfig(**vars(parser.parse_args(argv)))


if __name__ == "__main__":
    train(parse_args())
