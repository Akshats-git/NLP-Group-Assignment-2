"""Train baseline and attention NMT models on the IIT Bombay English-Hindi corpus.

Run from repo root:
    python -m Question-1.train              # train both models
    python -m Question-1.train --model baseline --epochs 3   # quick smoke test

Or from the Question-1 folder:
    python train.py --model attention --epochs 25
"""

import argparse
import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

# add repo root to path so `import common` works when running from Question-1/
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common import (
    set_seed, get_device, autocast, grad_scaler, to_device,
    History, EarlyStopping, AverageMeter, Timer,
    save_run_info, save_json, results_dir, checkpoints_dir, data_dir,
    pad_collate, padding_mask, teacher_forcing_pair, masked_accuracy,
    dataloader_kwargs, count_parameters, format_count,
)
from data import DEFAULT_MAX_TRAIN, DEFAULT_MIN_LEN, load_splits, build_vocabs, NMTDataset, save_data_report
from models import build_model, DEFAULT_CONFIG


# ── configuration ────────────────────────────────────────────────────────────

def get_config(args) -> dict:
    return {
        "model_type": args.model,
        "max_train": args.max_train,
        "max_len": args.max_len,
        "min_len": args.min_len,
        "min_freq": args.min_freq,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "lr": args.lr,
        "clip_grad": args.clip_grad,
        "patience": args.patience,
        "seed": args.seed,
        **DEFAULT_CONFIG,
    }


# ── training loop ────────────────────────────────────────────────────────────

def train_one_epoch(model, loader, optimizer, criterion, scaler, device,
                    pad_id: int, is_attention: bool, clip_grad: float = 1.0):
    """One training epoch. Returns average loss and accuracy."""
    model.train()
    loss_meter = AverageMeter()
    correct_total, count_total = 0, 0

    for batch in tqdm(loader, desc="  train", leave=False):
        batch = to_device(batch, device)
        src = batch["src"]                                     # [B, T_src]
        tgt_in, tgt_out = teacher_forcing_pair(batch["tgt"])   # [B, T-1] each

        with autocast(device):
            if is_attention:
                src_mask = padding_mask(src, pad_id)
                logits = model(src, tgt_in, src_mask)
            else:
                logits = model(src, tgt_in)
            # logits: [B, T, V], tgt_out: [B, T]
            loss = criterion(logits.reshape(-1, logits.size(-1)), tgt_out.reshape(-1))

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
        scaler.step(optimizer)
        scaler.update()

        bs = src.size(0)
        loss_meter.update(loss.item(), n=bs)
        c, t = masked_accuracy(logits, tgt_out, pad_id)
        correct_total += c
        count_total += t

    acc = correct_total / count_total if count_total else 0.0
    return loss_meter.avg, acc


@torch.no_grad()
def evaluate(model, loader, criterion, device, pad_id: int, is_attention: bool):
    """Validation pass. Returns average loss and accuracy."""
    model.eval()
    loss_meter = AverageMeter()
    correct_total, count_total = 0, 0

    for batch in tqdm(loader, desc="  val  ", leave=False):
        batch = to_device(batch, device)
        src = batch["src"]
        tgt_in, tgt_out = teacher_forcing_pair(batch["tgt"])

        with autocast(device):
            if is_attention:
                src_mask = padding_mask(src, pad_id)
                logits = model(src, tgt_in, src_mask)
            else:
                logits = model(src, tgt_in)
            loss = criterion(logits.reshape(-1, logits.size(-1)), tgt_out.reshape(-1))

        bs = src.size(0)
        loss_meter.update(loss.item(), n=bs)
        c, t = masked_accuracy(logits, tgt_out, pad_id)
        correct_total += c
        count_total += t

    acc = correct_total / count_total if count_total else 0.0
    return loss_meter.avg, acc


def train_model(model_type: str, config: dict):
    """Full training pipeline for one model."""
    seed = config["seed"]
    set_seed(seed)
    device = get_device()
    print(f"\n{'='*60}")
    print(f"Training {model_type.upper()} model  (device={device}, seed={seed})")
    print(f"{'='*60}")

    # ── data ─────────────────────────────────────────────────────────────
    train_pairs, dev_pairs, test_pairs, data_stats = load_splits(
        max_train=config["max_train"], max_len=config["max_len"], seed=seed,
        min_len=config["min_len"],
    )
    vocab_en, vocab_hi, vocab_stats = build_vocabs(train_pairs, min_freq=config["min_freq"])
    pad_id = vocab_en.pad_id

    # save vocabs for later inference
    vocab_dir = data_dir(1) / "vocabs"
    vocab_en.save(vocab_dir / "vocab_en.json")
    vocab_hi.save(vocab_dir / "vocab_hi.json")

    # save data report (once)
    run_name = f"{model_type}_seed{seed}"
    run_dir = results_dir(1) / run_name
    save_data_report(data_stats, vocab_stats, vocab_en, vocab_hi, dev_pairs, test_pairs, run_dir)

    # datasets and loaders
    train_ds = NMTDataset(train_pairs, vocab_en, vocab_hi)
    dev_ds   = NMTDataset(dev_pairs,   vocab_en, vocab_hi)

    collate = pad_collate(pad_id)
    train_loader = DataLoader(
        train_ds, batch_size=config["batch_size"], shuffle=True,
        collate_fn=collate, **dataloader_kwargs(seed),
    )
    dev_loader = DataLoader(
        dev_ds, batch_size=config["batch_size"] * 2, shuffle=False,
        collate_fn=collate,
    )

    # ── model ────────────────────────────────────────────────────────────
    model = build_model(model_type, len(vocab_en), len(vocab_hi), pad_idx=pad_id)
    model = model.to(device)
    params = count_parameters(model)
    print(f"  parameters: {format_count(params['trainable'])} trainable "
          f"/ {format_count(params['total'])} total")

    # save run info
    save_run_info(run_dir, config, seed=seed, parameters=params,
                  data_stats=data_stats, vocab_stats=vocab_stats)

    # ── optimiser, scheduler, criterion ──────────────────────────────────
    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])
    criterion = nn.CrossEntropyLoss(ignore_index=pad_id)
    scaler = grad_scaler(device)
    is_attention = model_type == "attention"

    stopper = EarlyStopping(
        mode="min", patience=config["patience"],
        checkpoint_path=checkpoints_dir(1) / f"{model_type}_best.pt",
    )
    history = History()

    # ── training ─────────────────────────────────────────────────────────
    print(f"\nStarting training for up to {config['epochs']} epochs ...")
    for epoch in range(1, config["epochs"] + 1):
        with Timer() as t:
            train_loss, train_acc = train_one_epoch(
                model, train_loader, optimizer, criterion, scaler, device,
                pad_id, is_attention, config["clip_grad"],
            )
            val_loss, val_acc = evaluate(
                model, dev_loader, criterion, device, pad_id, is_attention,
            )

        improved = stopper.step(val_loss, model)
        record = history.log(
            epoch=epoch,
            train_loss=train_loss, val_loss=val_loss,
            train_acc=train_acc, val_acc=val_acc,
            time_s=t.elapsed,
        )
        marker = " *" if improved else ""
        print(f"  epoch {epoch:3d}  |  train loss {train_loss:.4f}  acc {train_acc:.4f}"
              f"  |  val loss {val_loss:.4f}  acc {val_acc:.4f}"
              f"  |  {t.elapsed:.0f}s{marker}")

        if stopper.should_stop:
            print(f"  early stopping at epoch {epoch} (best was epoch {stopper.best_epoch})")
            break

    stopper.restore(model)
    history.save(run_dir / "history.json")
    print(f"\n  best val loss: {stopper.best:.4f} at epoch {stopper.best_epoch}")
    print(f"  history saved to {run_dir / 'history.json'}")

    return model, history, vocab_en, vocab_hi, train_pairs, dev_pairs, test_pairs


# ── CLI ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Train Q1 NMT models")
    p.add_argument("--model", choices=["baseline", "attention", "both"], default="both",
                   help="which model to train (default: both)")
    p.add_argument("--max-train", type=int, default=DEFAULT_MAX_TRAIN,
                   help="training pairs to use (default: 50000)")
    p.add_argument("--max-len", type=int, default=50,
                   help="max sentence length in tokens (default: 50)")
    p.add_argument("--min-len", type=int, default=DEFAULT_MIN_LEN,
                   help="min English tokens for a training pair (default: 5)")
    p.add_argument("--min-freq", type=int, default=2,
                   help="min token frequency for vocabulary (default: 2)")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--epochs", type=int, default=25)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--clip-grad", type=float, default=1.0)
    p.add_argument("--patience", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()
    config = get_config(args)

    models_to_train = ["baseline", "attention"] if args.model == "both" else [args.model]
    results = {}

    for model_type in models_to_train:
        config["model_type"] = model_type
        model, history, vocab_en, vocab_hi, train_p, dev_p, test_p = train_model(model_type, config)
        results[model_type] = {
            "model": model,
            "history": history,
            "vocab_en": vocab_en,
            "vocab_hi": vocab_hi,
            "test_pairs": test_p,
        }

    print("\n" + "="*60)
    print("Training complete.")
    if len(results) == 2:
        for name, r in results.items():
            best = r["history"].best("val_loss")
            print(f"  {name:12s}  best val_loss={best['val_loss']:.4f}  "
                  f"val_acc={best.get('val_acc', 0):.4f}  (epoch {best['epoch']})")
    print("="*60)


if __name__ == "__main__":
    main()
