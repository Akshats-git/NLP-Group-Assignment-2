"""Model A (Vanilla RNN) and Model B (LSTM) language models.

    Embedding -> {RNN | LSTM} -> Linear -> vocabulary logits

One class serves both so that everything except the recurrent cell is identical by
construction: same embedding size, hidden size, depth, dropout, initialisation scheme and output layer.
"""

import torch
from torch import nn

PAD_ID, UNK_ID, SOS_ID, EOS_ID = 0, 1, 2, 3  # fixed by common.Vocab


class StoryLM(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        cell: str = "lstm",
        emb_dim: int = 256,
        hid_dim: int = 512,
        num_layers: int = 1,
        dropout: float = 0.3,
        pad_id: int = PAD_ID,
    ):
        super().__init__()
        if cell not in ("rnn", "lstm"):
            raise ValueError("cell must be 'rnn' or 'lstm'")
        self.cell = cell
        self.config = dict(vocab_size=vocab_size, cell=cell, emb_dim=emb_dim, hid_dim=hid_dim,
                           num_layers=num_layers, dropout=dropout, pad_id=pad_id)
        self.embed = nn.Embedding(vocab_size, emb_dim, padding_idx=pad_id)
        rnn_cls = nn.RNN if cell == "rnn" else nn.LSTM
        kwargs = {"nonlinearity": "tanh"} if cell == "rnn" else {}
        self.rnn = rnn_cls(emb_dim, hid_dim, num_layers=num_layers, batch_first=True,
                           dropout=dropout if num_layers > 1 else 0.0, **kwargs)
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(hid_dim, vocab_size)
        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.normal_(self.embed.weight, std=0.1)
        with torch.no_grad():
            self.embed.weight[self.embed.padding_idx].zero_()
        for name, p in self.rnn.named_parameters():
            if "weight_hh" in name:
                for chunk in p.data.chunk(4 if self.cell == "lstm" else 1):
                    nn.init.orthogonal_(chunk)  # keeps the recurrent map close to norm-preserving at the start
            elif "weight_ih" in name:
                nn.init.xavier_uniform_(p)
            elif "bias" in name:
                nn.init.zeros_(p)
        if self.cell == "lstm":  # forget-gate bias 1: remember by default
            for name, p in self.rnn.named_parameters():
                if "bias_hh" in name:
                    p.data[self.config["hid_dim"]:2 * self.config["hid_dim"]].fill_(1.0)
        nn.init.xavier_uniform_(self.fc.weight)
        nn.init.zeros_(self.fc.bias)

    def forward(self, ids: torch.Tensor, state=None):
        """ids [batch, time] -> (logits [batch, time, vocab], new state)."""
        out, state = self.rnn(self.drop(self.embed(ids)), state)
        return self.fc(self.drop(out)), state

    def step_fn(self, banned_ids: tuple[int, ...] = (PAD_ID, UNK_ID, SOS_ID)):
        """Adapter for common.generate: reads the whole prompt once, then one token per step.

        banned_ids get -inf logits, so they are never generated: <pad> and <sos> are structural,
        and <unk> carries no information (it would print as '<unk>' in the story).
        """
        banned = list(banned_ids)

        def step(tokens: torch.Tensor, state):
            inputs = tokens if state is None else tokens[:, -1:]
            logits, state = self(inputs, state)
            logits = logits[:, -1].float()
            if banned:
                logits[:, banned] = float("-inf")
            return logits, state

        return step


def build_model(cell: str, vocab_size: int, config: dict) -> StoryLM:
    return StoryLM(vocab_size, cell=cell, emb_dim=config["emb_dim"], hid_dim=config["hid_dim"],
                   num_layers=config["num_layers"], dropout=config["dropout"])


def load_model(checkpoint_path, device="cpu") -> tuple[StoryLM, dict]:
    """Rebuild a model from a checkpoint written by train.py. Returns (model in eval mode, payload)."""
    from common import load_checkpoint

    payload = load_checkpoint(checkpoint_path, map_location=device)
    model = StoryLM(**payload["model_config"]).to(device)
    model.load_state_dict(payload["model"])
    return model.eval(), payload
