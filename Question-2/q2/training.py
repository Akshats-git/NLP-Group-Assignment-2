"""Pieces both training loops (BiGRU and RoBERTa) share."""

import torch

from common import AverageMeter


class Segment:
    """Training loss and predictions since the last evaluation (dropout on, so a rough train signal)."""

    def __init__(self):
        self.loss, self.preds, self.labels = AverageMeter(), [], []

    def update(self, loss: torch.Tensor, logits: torch.Tensor, labels: torch.Tensor) -> None:
        self.loss.update(loss.item(), n=len(labels))
        self.preds += logits.argmax(dim=-1).tolist()
        self.labels += labels.tolist()
