"""Q2 classifiers."""

import torch
from torch import nn
from transformers import RobertaModel

from .data import NUM_CLASSES


def masked_mean(hidden: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """[batch, time, dim] -> [batch, dim], averaging only positions where mask is 1 (real tokens)."""
    mask = mask.unsqueeze(-1).to(hidden.dtype)
    return (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)


class RobertaClassifier(nn.Module):
    """roberta-base -> pooling over the last hidden layer -> dropout -> linear (5).

    pooling="mean" is the model the brief asks for: the masked mean over every non-padding
    token, <s> and </s> included. pooling="first" uses only the <s> vector; it is the
    ablation. The whole encoder is fine-tuned.
    """

    def __init__(self, model_name: str = "roberta-base", pooling: str = "mean", dropout: float = 0.1,
                 num_classes: int = NUM_CLASSES):
        super().__init__()
        if pooling not in ("mean", "first"):
            raise ValueError("pooling must be 'mean' or 'first'")
        self.pooling = pooling
        self.encoder = RobertaModel.from_pretrained(model_name, add_pooling_layer=False)
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(self.encoder.config.hidden_size, num_classes)

    def pool(self, hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        if self.pooling == "first":
            return hidden[:, 0]
        return masked_mean(hidden.float(), attention_mask)  # sum in fp32, fp16 can overflow on long reviews

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        hidden = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        pooled = self.pool(hidden, attention_mask)
        return self.head(self.dropout(pooled.to(self.head.weight.dtype)))
