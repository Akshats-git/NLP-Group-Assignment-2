"""LSTM encoder-decoder models for English-to-Hindi NMT.

Two architectures sharing the same encoder:
    BaselineSeq2Seq  — fixed-size context from the encoder's final hidden state
    AttentionSeq2Seq — Bahdanau (additive) attention over all encoder states

Both expose .encode() and .decode_step() so they work with common.generate.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


# ── Encoder (shared) ─────────────────────────────────────────────────────────

class Encoder(nn.Module):
    """Bidirectional LSTM encoder.

    Returns:
        enc_out:  [batch, src_len, 2*hidden]  — all time-step outputs (for attention)
        hidden:   tuple(h, c) each [n_layers, batch, hidden] — final state for decoder init
    """

    def __init__(self, vocab_size: int, emb_dim: int, hidden_dim: int,
                 n_layers: int = 2, dropout: float = 0.3, pad_idx: int = 0):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=pad_idx)
        self.rnn = nn.LSTM(
            emb_dim, hidden_dim, num_layers=n_layers,
            dropout=dropout if n_layers > 1 else 0,
            batch_first=True, bidirectional=True,
        )
        self.dropout = nn.Dropout(dropout)
        # project bidirectional hidden to decoder-sized hidden
        self.fc_h = nn.Linear(hidden_dim * 2, hidden_dim)
        self.fc_c = nn.Linear(hidden_dim * 2, hidden_dim)
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.pad_idx = pad_idx

    def forward(self, src: torch.Tensor) -> tuple[torch.Tensor, tuple]:
        """src: [batch, src_len] token IDs."""
        embedded = self.dropout(self.embedding(src))        # [B, T, emb]
        # pack so the LSTM never reads <pad>: the final states (and the backward
        # direction, which starts at the last real token) are then the true ones
        lengths = (src != self.pad_idx).sum(1).clamp(min=1).cpu()
        packed = pack_padded_sequence(embedded, lengths, batch_first=True, enforce_sorted=False)
        packed_out, (h, c) = self.rnn(packed)
        enc_out, _ = pad_packed_sequence(packed_out, batch_first=True, total_length=src.size(1))

        # h, c: [2*layers, B, H] -> merge directions for each layer
        # reshape to [layers, 2, B, H], concat directions -> [layers, B, 2H], project
        h = h.view(self.n_layers, 2, -1, self.hidden_dim)   # [layers, 2, B, H]
        c = c.view(self.n_layers, 2, -1, self.hidden_dim)
        h = torch.cat([h[:, 0], h[:, 1]], dim=-1)           # [layers, B, 2H]
        c = torch.cat([c[:, 0], c[:, 1]], dim=-1)
        h = torch.tanh(self.fc_h(h))                        # [layers, B, H]
        c = torch.tanh(self.fc_c(c))

        return enc_out, (h, c)


# ── Bahdanau (additive) attention ─────────────────────────────────────────────

class BahdanauAttention(nn.Module):
    """Additive attention: score(s, h) = v^T tanh(W_s · s + W_h · h).

    This is our own implementation; no pre-built attention modules are used.
    Returns the context vector and the attention weights (for heatmaps).
    """

    def __init__(self, decoder_dim: int, encoder_dim: int, attention_dim: int = 256):
        super().__init__()
        self.W_s = nn.Linear(decoder_dim, attention_dim, bias=False)
        self.W_h = nn.Linear(encoder_dim, attention_dim, bias=False)
        self.v = nn.Linear(attention_dim, 1, bias=False)

    def project_keys(self, encoder_outputs: torch.Tensor) -> torch.Tensor:
        """W_h h_i for all source positions; compute once per sentence and reuse every step."""
        return self.W_h(encoder_outputs)

    def forward(self, decoder_hidden: torch.Tensor,
                encoder_outputs: torch.Tensor,
                mask: torch.Tensor | None = None,
                keys: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            decoder_hidden: [batch, decoder_dim]  — s_{t-1}
            encoder_outputs: [batch, src_len, encoder_dim]  — h_1..h_n
            mask: [batch, src_len] bool, True = real token

        Returns:
            context: [batch, encoder_dim]  — weighted sum
            alpha:   [batch, src_len]      — attention weights (for visualisation)
        """
        # decoder_hidden: [B, D] -> [B, 1, attn_dim]
        query = self.W_s(decoder_hidden).unsqueeze(1)
        # encoder_outputs: [B, T, E] -> [B, T, attn_dim]
        if keys is None:
            keys = self.W_h(encoder_outputs)

        # energy: [B, T, 1] -> [B, T]
        energy = self.v(torch.tanh(query + keys)).squeeze(-1)

        # mask out padding positions
        if mask is not None:
            energy = energy.masked_fill(~mask, torch.finfo(energy.dtype).min)

        alpha = F.softmax(energy, dim=-1)                    # [B, T]
        context = torch.bmm(alpha.unsqueeze(1), encoder_outputs).squeeze(1)  # [B, E]
        return context, alpha


# ── Baseline decoder (no attention) ──────────────────────────────────────────

class BaselineDecoder(nn.Module):
    """LSTM decoder that relies solely on the encoder's final hidden state.

    At each time step it takes the previous token embedding and produces logits
    over the target vocabulary.
    """

    def __init__(self, vocab_size: int, emb_dim: int, hidden_dim: int,
                 n_layers: int = 2, dropout: float = 0.3, pad_idx: int = 0):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=pad_idx)
        self.rnn = nn.LSTM(
            emb_dim, hidden_dim, num_layers=n_layers,
            dropout=dropout if n_layers > 1 else 0,
            batch_first=True,
        )
        self.fc_out = nn.Linear(hidden_dim, vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, tgt: torch.Tensor, hidden: tuple) -> tuple[torch.Tensor, tuple]:
        """Full sequence forward for teacher forcing.
        tgt: [batch, tgt_len], hidden: from encoder.
        Returns logits [batch, tgt_len, vocab].
        """
        embedded = self.dropout(self.embedding(tgt))
        output, hidden = self.rnn(embedded, hidden)
        logits = self.fc_out(output)
        return logits, hidden

    def step(self, token: torch.Tensor, hidden: tuple) -> tuple[torch.Tensor, tuple]:
        """Single-step decode for inference.
        token: [batch] single token ID.
        Returns logits [batch, vocab], updated hidden.
        """
        embedded = self.dropout(self.embedding(token.unsqueeze(1)))  # [B, 1, emb]
        output, hidden = self.rnn(embedded, hidden)                  # [B, 1, H]
        logits = self.fc_out(output.squeeze(1))                      # [B, vocab]
        return logits, hidden


# ── Attention decoder ────────────────────────────────────────────────────────

class AttentionDecoder(nn.Module):
    """LSTM decoder with Bahdanau attention over encoder outputs.

    At each step the decoder computes attention, and the context vector is
    concatenated with the token embedding before being fed to the LSTM.
    decode_step returns the attention weights so they can be collected for heatmaps.
    """

    def __init__(self, vocab_size: int, emb_dim: int, hidden_dim: int,
                 encoder_dim: int, attention_dim: int = 256,
                 n_layers: int = 2, dropout: float = 0.3, pad_idx: int = 0):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=pad_idx)
        self.attention = BahdanauAttention(hidden_dim, encoder_dim, attention_dim)
        # input to LSTM is embedding + context
        self.rnn = nn.LSTM(
            emb_dim + encoder_dim, hidden_dim, num_layers=n_layers,
            dropout=dropout if n_layers > 1 else 0,
            batch_first=True,
        )
        self.fc_out = nn.Linear(hidden_dim + encoder_dim + emb_dim, vocab_size)
        self.dropout = nn.Dropout(dropout)

    def _one_step(self, token_emb: torch.Tensor, hidden: tuple,
                  enc_out: torch.Tensor,
                  mask: torch.Tensor | None,
                  keys: torch.Tensor | None = None) -> tuple[torch.Tensor, tuple, torch.Tensor]:
        """One decoder step. Returns (logits, hidden, alpha)."""
        # use the top-layer hidden state for attention
        s = hidden[0][-1]                                          # [B, H]
        context, alpha = self.attention(s, enc_out, mask, keys)    # [B, E], [B, T_src]

        rnn_input = torch.cat([token_emb, context.unsqueeze(1)], dim=-1)  # [B, 1, emb+E]
        output, hidden = self.rnn(rnn_input, hidden)                       # [B, 1, H]

        # prediction combines LSTM output, context and embedding
        combined = torch.cat([output.squeeze(1), context, token_emb.squeeze(1)], dim=-1)
        logits = self.fc_out(self.dropout(combined))               # [B, vocab]
        return logits, hidden, alpha

    def forward(self, tgt: torch.Tensor, hidden: tuple,
                enc_out: torch.Tensor,
                mask: torch.Tensor | None = None) -> tuple[torch.Tensor, tuple]:
        """Full sequence forward for teacher forcing.
        Returns logits [batch, tgt_len, vocab].
        """
        embedded = self.dropout(self.embedding(tgt))               # [B, T, emb]
        keys = self.attention.project_keys(enc_out)
        all_logits = []
        for t in range(tgt.size(1)):
            token_emb = embedded[:, t:t+1, :]                     # [B, 1, emb]
            logits_t, hidden, _ = self._one_step(token_emb, hidden, enc_out, mask, keys)
            all_logits.append(logits_t)
        return torch.stack(all_logits, dim=1), hidden              # [B, T, vocab]

    def step(self, token: torch.Tensor, hidden: tuple,
             enc_out: torch.Tensor,
             mask: torch.Tensor | None = None) -> tuple[torch.Tensor, tuple, torch.Tensor]:
        """Single-step decode for inference. Returns (logits, hidden, alpha)."""
        embedded = self.dropout(self.embedding(token.unsqueeze(1)))  # [B, 1, emb]
        return self._one_step(embedded, hidden, enc_out, mask)


# ── Full seq2seq wrappers ────────────────────────────────────────────────────

class BaselineSeq2Seq(nn.Module):
    """Encoder-decoder without attention: the encoder's final hidden state is
    the only information the decoder receives from the source.
    """

    def __init__(self, enc_vocab: int, dec_vocab: int, emb_dim: int = 256,
                 hidden_dim: int = 512, n_layers: int = 2, dropout: float = 0.3,
                 pad_idx: int = 0):
        super().__init__()
        self.encoder = Encoder(enc_vocab, emb_dim, hidden_dim, n_layers, dropout, pad_idx)
        self.decoder = BaselineDecoder(dec_vocab, emb_dim, hidden_dim, n_layers, dropout, pad_idx)

    def forward(self, src: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
        """Teacher-forced forward pass. Returns logits [batch, tgt_len, vocab]."""
        _, hidden = self.encoder(src)
        logits, _ = self.decoder(tgt, hidden)
        return logits

    def encode(self, src: torch.Tensor):
        """Encode source for inference. Returns (enc_out, hidden)."""
        return self.encoder(src)

    def decode_step(self, token: torch.Tensor, hidden: tuple,
                    enc_out: torch.Tensor = None):
        """Single decode step for common.generate."""
        logits, hidden = self.decoder.step(token, hidden)
        return logits, hidden


class AttentionSeq2Seq(nn.Module):
    """Encoder-decoder with Bahdanau attention over all encoder hidden states."""

    def __init__(self, enc_vocab: int, dec_vocab: int, emb_dim: int = 256,
                 hidden_dim: int = 512, n_layers: int = 2, dropout: float = 0.3,
                 attention_dim: int = 256, pad_idx: int = 0):
        super().__init__()
        self.encoder = Encoder(enc_vocab, emb_dim, hidden_dim, n_layers, dropout, pad_idx)
        self.decoder = AttentionDecoder(
            dec_vocab, emb_dim, hidden_dim,
            encoder_dim=hidden_dim * 2,  # bidirectional encoder
            attention_dim=attention_dim,
            n_layers=n_layers, dropout=dropout, pad_idx=pad_idx,
        )

    def forward(self, src: torch.Tensor, tgt: torch.Tensor,
                src_mask: torch.Tensor | None = None) -> torch.Tensor:
        """Teacher-forced forward. Returns logits [batch, tgt_len, vocab]."""
        enc_out, hidden = self.encoder(src)
        logits, _ = self.decoder(tgt, hidden, enc_out, src_mask)
        return logits

    def encode(self, src: torch.Tensor):
        """Encode source for inference. Returns (enc_out, hidden)."""
        return self.encoder(src)

    def decode_step(self, token: torch.Tensor, hidden: tuple,
                    enc_out: torch.Tensor,
                    src_mask: torch.Tensor | None = None):
        """Single decode step for common.generate. Returns (logits, hidden, alpha)."""
        logits, hidden, alpha = self.decoder.step(token, hidden, enc_out, src_mask)
        return logits, hidden, alpha


# ── model factory ────────────────────────────────────────────────────────────

DEFAULT_CONFIG = {
    "emb_dim": 256,
    "hidden_dim": 512,
    "n_layers": 2,
    "dropout": 0.3,
    "attention_dim": 256,
}


def build_model(model_type: str, enc_vocab_size: int, dec_vocab_size: int,
                config: dict | None = None, pad_idx: int = 0) -> nn.Module:
    """Create a model by name ('baseline' or 'attention')."""
    cfg = {**DEFAULT_CONFIG, **(config or {})}
    if model_type == "baseline":
        return BaselineSeq2Seq(
            enc_vocab_size, dec_vocab_size,
            emb_dim=cfg["emb_dim"], hidden_dim=cfg["hidden_dim"],
            n_layers=cfg["n_layers"], dropout=cfg["dropout"], pad_idx=pad_idx,
        )
    elif model_type == "attention":
        return AttentionSeq2Seq(
            enc_vocab_size, dec_vocab_size,
            emb_dim=cfg["emb_dim"], hidden_dim=cfg["hidden_dim"],
            n_layers=cfg["n_layers"], dropout=cfg["dropout"],
            attention_dim=cfg["attention_dim"], pad_idx=pad_idx,
        )
    else:
        raise ValueError(f"unknown model type: {model_type!r}")
