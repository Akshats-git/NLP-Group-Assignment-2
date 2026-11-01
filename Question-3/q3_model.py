"""Transformer encoder-decoder for summarisation, written from scratch.

Everything here is built from basic tensor operations and the simple layers the brief
allows (nn.Linear, nn.Embedding, nn.LayerNorm, nn.Dropout). No nn.Transformer*,
nn.MultiheadAttention or F.scaled_dot_product_attention.

Shapes used in the comments:
    B = batch size, S = source (article) length, T = target (summary) length,
    d = d_model, h = number of heads, d_k = d / h

Mask convention (same as common/): True = real token / allowed to attend.
"""

import math

import torch
from torch import nn

from common import Decoding, causal_mask, generate, padding_mask


class PositionalEncoding(nn.Module):
    """Fixed sinusoidal position encodings from "Attention Is All You Need".

        PE(pos, 2i)   = sin(pos / 10000^(2i / d_model))
        PE(pos, 2i+1) = cos(pos / 10000^(2i / d_model))
    """

    def __init__(self, d_model: int, max_len: int = 1024):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)             # [max_len, 1]
        two_i = torch.arange(0, d_model, 2)                       # 0, 2, 4, ... (the "2i" in the formula)
        angle = position / (10000 ** (two_i / d_model))           # [max_len, d/2]
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(angle)                            # even dimensions
        pe[:, 1::2] = torch.cos(angle)                            # odd dimensions
        self.register_buffer("pe", pe, persistent=False)          # fixed, not learned, not saved

    def forward(self, x: torch.Tensor) -> torch.Tensor:          # x: [B, time, d]
        if x.size(1) > self.pe.size(0):
            raise ValueError(f"sequence of length {x.size(1)} is longer than max_len={self.pe.size(0)}")
        return x + self.pe[: x.size(1)]


class MultiHeadAttention(nn.Module):
    """Scaled dot-product attention with h heads:

        Attention(Q, K, V) = softmax(Q K^T / sqrt(d_k) + mask) V

    The "+ mask" is done with masked_fill: blocked positions get the most negative
    number the dtype can hold, so their softmax weight is (practically) zero.
    """

    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)    # mixes the heads back together

    def split_heads(self, x: torch.Tensor) -> torch.Tensor:
        """[B, time, d] -> [B, h, time, d_k]"""
        batch, time, _ = x.shape
        return x.view(batch, time, self.n_heads, self.d_k).transpose(1, 2)

    def forward(self, query, key, value, mask):
        """query: [B, Tq, d]; key, value: [B, Tk, d]; mask: bool, broadcastable to [B, h, Tq, Tk].

        Returns (output [B, Tq, d], attention weights [B, h, Tq, Tk]).
        """
        q = self.split_heads(self.w_q(query))
        k = self.split_heads(self.w_k(key))
        v = self.split_heads(self.w_v(value))

        # Scale q before the product (same maths as dividing the scores) so fp16 cannot overflow.
        scores = (q / math.sqrt(self.d_k)) @ k.transpose(-2, -1)          # [B, h, Tq, Tk]
        scores = scores.masked_fill(~mask, torch.finfo(scores.dtype).min)
        weights = scores.softmax(dim=-1)                                  # each row sums to 1
        context = weights.to(v.dtype) @ v                                 # [B, h, Tq, d_k]

        batch, _, q_len, _ = context.shape
        context = context.transpose(1, 2).reshape(batch, q_len, self.n_heads * self.d_k)
        return self.w_o(context), weights


class FeedForward(nn.Module):
    """Position-wise feed-forward network: FFN(x) = max(0, x W1 + b1) W2 + b2."""

    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        return self.linear2(torch.relu(self.linear1(x)))


class EncoderLayer(nn.Module):
    """Self-attention, then feed-forward. Each sub-layer: LayerNorm(x + Dropout(Sublayer(x)))."""

    def __init__(self, d_model, n_heads, d_ff, dropout):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, n_heads)
        self.ffn = FeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, src_mask):
        attn_out, _ = self.self_attn(x, x, x, src_mask)
        x = self.norm1(x + self.dropout(attn_out))
        x = self.norm2(x + self.dropout(self.ffn(x)))
        return x


class DecoderLayer(nn.Module):
    """Masked self-attention, encoder-decoder (cross) attention, then feed-forward."""

    def __init__(self, d_model, n_heads, d_ff, dropout):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, n_heads)
        self.cross_attn = MultiHeadAttention(d_model, n_heads)
        self.ffn = FeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, y, memory, tgt_mask, src_mask):
        self_out, _ = self.self_attn(y, y, y, tgt_mask)                    # summary looks at its own past
        y = self.norm1(y + self.dropout(self_out))
        cross_out, cross_weights = self.cross_attn(y, memory, memory, src_mask)  # summary looks at the article
        y = self.norm2(y + self.dropout(cross_out))
        y = self.norm3(y + self.dropout(self.ffn(y)))
        return y, cross_weights


class TransformerSummarizer(nn.Module):
    """Encoder-decoder Transformer that maps article token ids to summary logits.

    One embedding matrix is shared by the encoder input, the decoder input and the final
    linear layer (as in the paper, section 3.4). Embeddings are multiplied by sqrt(d_model)
    before the positional encodings are added.

    copy_mechanism=True adds a pointer / copy switch on top of the Transformer (See et al.,
    2017, "Get To The Point"). At every summary position a gate p_gen in (0, 1) decides
    between generating a word from the vocabulary softmax and copying an article token,
    using the last decoder layer's cross-attention (averaged over heads) as the copy
    distribution:

        P(word) = p_gen * P_vocab(word) + (1 - p_gen) * sum of attention on article positions holding word

    Names and numbers can then be taken straight from the article instead of being guessed
    from 30k vocabulary entries. With copy_mechanism=False this is the paper's Transformer.
    """

    def __init__(self, vocab_size, pad_id=0, d_model=512, n_heads=8, n_enc_layers=4,
                 n_dec_layers=4, d_ff=2048, dropout=0.1, max_len=1024, copy_mechanism=True):
        super().__init__()
        self.pad_id = pad_id
        self.d_model = d_model
        self.copy_mechanism = copy_mechanism
        self.embed = nn.Embedding(vocab_size, d_model, padding_idx=pad_id)
        self.positions = PositionalEncoding(d_model, max_len)
        self.dropout = nn.Dropout(dropout)
        self.encoder_layers = nn.ModuleList(
            [EncoderLayer(d_model, n_heads, d_ff, dropout) for _ in range(n_enc_layers)])
        self.decoder_layers = nn.ModuleList(
            [DecoderLayer(d_model, n_heads, d_ff, dropout) for _ in range(n_dec_layers)])
        self.output = nn.Linear(d_model, vocab_size, bias=False)   # decoder state -> vocabulary logits
        self.output.weight = self.embed.weight                      # weight tying
        # p_gen from the decoder state and the article context it attends to
        self.copy_gate = nn.Linear(2 * d_model, 1) if copy_mechanism else None
        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear) and module is not self.output:
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        # Small embeddings: after the sqrt(d) scale they have unit size, and the tied
        # output layer starts with logits of a sensible size.
        nn.init.normal_(self.embed.weight, mean=0.0, std=self.d_model ** -0.5)
        with torch.no_grad():
            self.embed.weight[self.pad_id].zero_()

    def embed_tokens(self, ids):
        return self.dropout(self.positions(self.embed(ids) * math.sqrt(self.d_model)))

    def encode(self, src):
        """src: [B, S] ids -> (memory [B, S, d], src_mask [B, 1, 1, S])."""
        src_mask = padding_mask(src, self.pad_id)[:, None, None, :]   # hide padded article tokens
        x = self.embed_tokens(src)
        for layer in self.encoder_layers:
            x = layer(x, src_mask)
        return x, src_mask

    def decode(self, tgt_in, memory, src_mask):
        """tgt_in: [B, T] ids (starts with the start token).

        Returns (decoder states [B, T, d], last layer's cross-attention [B, h, T, S]).
        """
        length = tgt_in.size(1)
        future_blocked = causal_mask(length, device=tgt_in.device)[None, None]   # [1, 1, T, T], j <= i
        tgt_pad = padding_mask(tgt_in, self.pad_id)[:, None, None, :]           # [B, 1, 1, T]
        tgt_mask = future_blocked & tgt_pad                                      # [B, 1, T, T]
        y = self.embed_tokens(tgt_in)
        cross_weights = None
        for layer in self.decoder_layers:
            y, cross_weights = layer(y, memory, tgt_mask, src_mask)
        return y, cross_weights

    def word_distribution(self, states, cross_weights, memory):
        """Turn decoder states into (vocabulary logits [B, T, vocab], p_gen [B, T], copy attention [B, T, S]).

        Without the copy mechanism p_gen and the copy attention are None.
        """
        logits = self.output(states)
        if not self.copy_mechanism:
            return logits, None, None
        copy_attention = cross_weights.float().mean(dim=1)          # last layer, averaged over heads
        context = copy_attention @ memory                           # [B, T, d]: what is read from the article
        p_gen = torch.sigmoid(self.copy_gate(torch.cat([states, context], dim=-1))).squeeze(-1).float()
        return logits, p_gen, copy_attention

    def forward(self, src, tgt_in):
        """Teacher-forced pass used in training: the word distribution for every summary position."""
        memory, src_mask = self.encode(src)
        states, cross_weights = self.decode(tgt_in, memory, src_mask)
        return self.word_distribution(states, cross_weights, memory)


def next_word_log_probs(logits, p_gen, copy_attention, src):
    """log P(next word) over the whole vocabulary for one decoding step.

    logits: [B, vocab]; p_gen: [B] or None; copy_attention: [B, S]; src: [B, S] article ids.
    """
    probs = logits.float().softmax(dim=-1)
    if p_gen is not None:
        probs = p_gen[:, None] * probs
        probs = probs.scatter_add(1, src, (1 - p_gen[:, None]) * copy_attention)   # add copy mass per article token
    return probs.clamp_min(1e-12).log()


def block_repeated_trigrams(logits, tokens, ignore_id=None):
    """Forbid any next token that would repeat a trigram already in the summary.

    tokens: [B, t] generated so far. Optional decoding trick (Paulus et al., 2018); the
    model is still decoded greedily, just never allowed to say the same trigram twice.
    Trigrams that contain ignore_id (the sentence-break token) are not blocked, so
    sentences may still start with the same word.
    """
    if tokens.size(1) < 3:
        return logits
    first, second, third = tokens[:, :-2], tokens[:, 1:-1], tokens[:, 2:]     # every earlier trigram
    repeats = (first == tokens[:, -2:-1]) & (second == tokens[:, -1:])       # starts with the last two tokens
    if ignore_id is not None:
        repeats &= (first != ignore_id) & (second != ignore_id) & (third != ignore_id)
    rows, cols = repeats.nonzero(as_tuple=True)
    logits[rows, third[rows, cols]] = float("-inf")
    return logits


@torch.no_grad()
def greedy_summarize(model, src, bos_id, eos_id, pad_id, max_len=128, banned_ids=(),
                     block_trigrams=False, sentence_id=None, record_attention=False):
    """Greedy decoding without teacher forcing.

    The article is encoded once. Starting from the start token, the decoder is run on
    everything generated so far, the most likely next token is appended, and this
    repeats until every summary has produced the end token or max_len tokens.

    Returns (new_tokens [B, steps], attention or None). After a summary ends its row is
    filled with pad_id. With record_attention=True, attention is the last decoder layer's
    cross-attention for each generated token, recorded during generation: [B, h, steps, S].
    """
    model.eval()
    memory, src_mask = model.encode(src)                      # encode each article once
    banned = list(banned_ids)

    def step_fn(tokens, state):
        states, cross = model.decode(tokens, memory, src_mask)
        newest = model.word_distribution(states[:, -1:], cross[:, :, -1:], memory)   # only the newest position
        logits, p_gen, copy_attention = (x[:, 0] if x is not None else None for x in newest)
        logits = next_word_log_probs(logits, p_gen, copy_attention, src)
        if banned:
            logits[:, banned] = float("-inf")                 # e.g. never generate padding or [UNK]
        if block_trigrams:
            logits = block_repeated_trigrams(logits, tokens, ignore_id=sentence_id)
        if record_attention:
            return logits, state, cross[:, :, -1, :].float()  # [B, h, S] for the newest token
        return logits, state

    start = torch.full((src.size(0), 1), bos_id, dtype=torch.long, device=src.device)
    out = generate(step_fn, start, max_new_tokens=max_len, eos_id=eos_id, pad_id=pad_id, decoding=Decoding())
    attention = torch.stack(out.extras, dim=2) if record_attention else None
    return out.new_tokens, attention
