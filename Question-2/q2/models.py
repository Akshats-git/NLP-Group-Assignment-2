"""Q2 classifiers."""

import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from transformers import RobertaModel

from common import lengths_to_mask

from .data import NUM_CLASSES


class AdditiveAttentionPooling(nn.Module):
    """Bahdanau-style attention pooling: score_t = v . tanh(W h_t + b), alpha = softmax over real tokens.

    Returns the weighted sum of the hidden states and the weights alpha, which are kept for
    the attention visualisation.
    """

    def __init__(self, input_dim: int, attention_dim: int):
        super().__init__()
        self.proj = nn.Linear(input_dim, attention_dim)
        self.score = nn.Linear(attention_dim, 1, bias=False)

    def forward(self, hidden: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        scores = self.score(torch.tanh(self.proj(hidden))).squeeze(-1)           # [batch, time]
        scores = scores.masked_fill(~mask, torch.finfo(scores.dtype).min)
        alpha = torch.softmax(scores, dim=-1)
        return torch.bmm(alpha.unsqueeze(1), hidden).squeeze(1), alpha           # [batch, dim], [batch, time]


class BiGRUAttention(nn.Module):
    """GloVe embeddings -> 2-layer bidirectional GRU -> additive attention pooling -> dropout -> linear (5).

    Sequences are packed, so the GRU never reads padding in either direction. The
    embedding starts from the GloVe matrix and is frozen until `set_embedding_trainable(True)`.
    `dropout` is used between the two GRU layers and before the linear layer.
    """

    def __init__(self, embeddings: torch.Tensor, hidden_size: int = 256, num_layers: int = 2,
                 attention_dim: int = 256, dropout: float = 0.3, num_classes: int = NUM_CLASSES):
        super().__init__()
        self.embedding = nn.Embedding.from_pretrained(embeddings, freeze=True, padding_idx=0)
        self.gru = nn.GRU(embeddings.shape[1], hidden_size, num_layers=num_layers, batch_first=True,
                          bidirectional=True, dropout=dropout if num_layers > 1 else 0.0)
        self.attention = AdditiveAttentionPooling(2 * hidden_size, attention_dim)
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(2 * hidden_size, num_classes)

    def set_embedding_trainable(self, trainable: bool) -> None:
        self.embedding.weight.requires_grad_(trainable)

    def forward(self, ids: torch.Tensor, lengths: torch.Tensor, return_attention: bool = False):
        packed = pack_padded_sequence(self.embedding(ids), lengths.clamp(min=1).cpu(), batch_first=True,
                                      enforce_sorted=False)
        hidden, _ = pad_packed_sequence(self.gru(packed)[0], batch_first=True, total_length=ids.shape[1])
        pooled, alpha = self.attention(hidden, lengths_to_mask(lengths, ids.shape[1]))
        logits = self.head(self.dropout(pooled))
        return (logits, alpha) if return_attention else logits


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
                 num_classes: int = NUM_CLASSES, attn_implementation: str | None = None):
        super().__init__()
        if pooling not in ("mean", "first"):
            raise ValueError("pooling must be 'mean' or 'first'")
        self.pooling = pooling
        extra = {"attn_implementation": attn_implementation} if attn_implementation else {}   # "eager" or "sdpa"
        self.encoder = RobertaModel.from_pretrained(model_name, add_pooling_layer=False, **extra)
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
