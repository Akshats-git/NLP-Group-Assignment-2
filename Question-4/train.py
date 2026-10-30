"""Part 2: train one language model (Vanilla RNN or LSTM) with next-token cross-entropy.

    python Question-4/train.py --cell rnn  --seed 42
    python Question-4/train.py --cell lstm --seed 42

Run both cells with the same seed to get the comparable pair. Options only override the shared
settings in plot_twist/config.py for quick experiments; leave them out for the reported runs.

Writes, for a run named <cell>_seed<seed>:
    outputs/checkpoints/<run>.pt          best-validation weights + model config (Q4 ships these)
    outputs/results/<run>/history.json    per-epoch train/val loss, perplexity, lr, time
    outputs/results/<run>/metrics.json    parameters, training time, best epoch, val and test perplexity
    outputs/results/<run>/run_info.json   config, seed, environment
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import torch
import torch.nn.functional as F

from common import (AverageMeter, EarlyStopping, History, Timer, Vocab, autocast, count_parameters, format_count,
                    get_device, grad_scaler, load_jsonl, perplexity, save_checkpoint, save_json, save_run_info,
                    set_seed, to_device)
from plot_twist import config as cfg
from plot_twist.data import StoryDataset, make_loader, tokenize
from plot_twist.lm_eval import evaluate_loss
from plot_twist.models import build_model


def load_splits(vocab: Vocab, data_path: Path) -> dict[str, StoryDataset]:
    out = {}
    for split in ("train", "validation", "test"):
        stories = [r["text"] for r in load_jsonl(data_path / f"{split}.jsonl")]
        out[split] = StoryDataset([tokenize(s) for s in stories], vocab)
    return out


def main():
    sys.stdout.reconfigure(line_buffering=True)  # progress shows up in redirected logs
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cell", choices=cfg.CELLS, required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--hid-dim", type=int, help="override hidden size (e.g. for the parameter-matched RNN control)")
    ap.add_argument("--tag", default="", help="suffix for the run name, e.g. 'matched'")
    ap.add_argument("--epochs", type=int, help="override max epochs (smoke tests)")
    ap.add_argument("--amp", action="store_true",
                    help="mixed precision (off by default: on a GTX 1650 fp16 autocast was ~5x slower than fp32, and the models are small)")
    args = ap.parse_args()

    p = cfg.paths()
    model_cfg = dict(cfg.MODEL, **({"hid_dim": args.hid_dim} if args.hid_dim else {}))
    train_cfg = dict(cfg.TRAIN, **({"max_epochs": args.epochs} if args.epochs else {}))
    run = f"{args.cell}_seed{args.seed}" + (f"_{args.tag}" if args.tag else "")
    run_dir = p["results"] / run

    set_seed(args.seed)  # before the model and the loaders are built
    device = get_device()
    vocab = Vocab.load(p["results"].parent / "vocab.json")
    data = load_splits(vocab, p["data"])
    pad = vocab.pad_id

    model = build_model(args.cell, len(vocab), model_cfg).to(device)
    params = count_parameters(model)
    config = {"cell": args.cell, "model": model_cfg, "train": train_cfg, "data": cfg.DATA, "vocab_size": len(vocab),
              "amp": args.amp, "device": str(device)}
    save_run_info(run_dir, config, seed=args.seed)
    print(f"{run}: {format_count(params['total'])} parameters ({params['total']:,}), vocab {len(vocab)}, device {device}")

    train_loader, sampler = make_loader(data["train"], pad, train_cfg["batch_size"], args.seed, shuffle=True)
    val_loader, _ = make_loader(data["validation"], pad, 32, args.seed, shuffle=False)
    test_loader, _ = make_loader(data["test"], pad, 32, args.seed, shuffle=False)

    optimizer = torch.optim.Adam(model.parameters(), lr=train_cfg["lr"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=train_cfg["max_epochs"], eta_min=train_cfg["lr"] * train_cfg["lr_min_factor"])
    scaler = grad_scaler(device) if args.amp else torch.amp.GradScaler(device.type, enabled=False)
    stopper = EarlyStopping(mode="min", patience=train_cfg["patience"])
    history = History()

    with Timer() as total_time:
        for epoch in range(1, train_cfg["max_epochs"] + 1):
            sampler.set_epoch(epoch)
            model.train()
            meter = AverageMeter()
            with Timer() as epoch_time:
                for batch in train_loader:
                    batch = to_device(batch, device)
                    n_tokens = int((batch["targets"] != pad).sum())
                    optimizer.zero_grad(set_to_none=True)
                    batch_loss = 0.0
                    # One optimiser step per batch of stories. The batch is processed in row chunks (gradient
                    # accumulation) so the [rows, time, vocab] logits fit on a 4 GB GPU; the loss of every chunk is
                    # divided by the whole batch's token count, so the result equals a single full-batch step.
                    for lo in range(0, batch["inputs"].size(0), train_cfg["rows_per_pass"]):
                        sl = slice(lo, lo + train_cfg["rows_per_pass"])
                        width = int(batch["lengths"][sl].max())
                        inputs, targets = batch["inputs"][sl, :width], batch["targets"][sl, :width]
                        with autocast(device, enabled=args.amp):
                            logits, _ = model(inputs)
                        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)).float(), targets.reshape(-1),
                                               ignore_index=pad, reduction="sum") / n_tokens
                        scaler.scale(loss).backward()
                        batch_loss += loss.item()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), train_cfg["grad_clip"])
                    scaler.step(optimizer)
                    scaler.update()
                    meter.update(batch_loss, n=n_tokens)  # token-weighted, so it matches how perplexity is defined
            val = evaluate_loss(model, val_loader, device, pad)
            lr = optimizer.param_groups[0]["lr"]
            history.log(epoch=epoch, train_loss=meter.avg, val_loss=val["nll"], train_ppl=perplexity(meter.avg),
                        val_ppl=val["ppl"], lr=lr, time_s=epoch_time.elapsed)
            improved = stopper.step(val["nll"], model)
            scheduler.step()
            print(f"epoch {epoch:2d}  train {meter.avg:.4f}  val {val['nll']:.4f} (ppl {val['ppl']:.2f})  lr {lr:.1e}  "
                  f"{epoch_time.elapsed:.0f}s{'  *' if improved else ''}")
            if stopper.should_stop:
                print(f"early stop: no improvement for {train_cfg['patience']} epochs")
                break

    stopper.restore(model)  # evaluate and ship the best epoch, not the last
    val = evaluate_loss(model, val_loader, device, pad)
    test = evaluate_loss(model, test_loader, device, pad)  # touched once, after model selection
    metrics = {
        "run": run, "cell": args.cell, "seed": args.seed, "parameters": params, "best_epoch": stopper.best_epoch,
        "epochs_run": len(history), "train_time_s": total_time.elapsed,
        "train_time_per_epoch_s": total_time.elapsed / len(history),
        "val_nll": val["nll"], "val_ppl": val["ppl"], "test_nll": test["nll"], "test_ppl": test["ppl"],
        "val_tokens": val["tokens"], "test_tokens": test["tokens"],
    }
    history.save(run_dir / "history.json")
    save_json(metrics, run_dir / "metrics.json")
    save_checkpoint(p["checkpoints"] / f"{run}.pt", model, model_config=model.config, seed=args.seed,
                    best_epoch=stopper.best_epoch, vocab_path="Question-4/outputs/vocab.json")
    print(f"best epoch {stopper.best_epoch}: val ppl {val['ppl']:.2f}, test ppl {test['ppl']:.2f}, "
          f"{total_time.elapsed / 60:.1f} min total")


if __name__ == "__main__":
    main()
