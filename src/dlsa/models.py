"""Released network ordering, with one optional lagged-stock-holdings input.

Architecture implementation derives from Guijarro-Ordonez and Zanotti's released
code; see THIRD_PARTY_NOTICES.md for the retained license.
"""
import os

import torch
from torch import nn

from .config import EXPERIMENT


class CNNBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv1d(1, 8, 2)
        self.conv2 = nn.Conv1d(8, 8, 2)
        self.relu = nn.ReLU(inplace=True)
        self.left_zero_padding = nn.ConstantPad1d((1, 0), 0)
        self.normalization1 = nn.InstanceNorm1d(1)
        self.normalization2 = nn.InstanceNorm1d(8)

    def forward(self, inputs):
        inputs = self.normalization1(inputs)
        hidden = self.relu(self.conv1(self.left_zero_padding(inputs)))
        hidden = self.normalization2(hidden)
        hidden = self.relu(self.conv2(self.left_zero_padding(hidden)))
        return hidden + inputs.repeat(1, 8, 1)


class Policy(nn.Module):
    def __init__(self, architecture, holdings=False, seed=0):
        super().__init__()
        torch.manual_seed(seed)
        self.architecture = architecture
        self.holdings = holdings
        if architecture == "cnn":
            self.convBlocks = nn.ModuleList([CNNBlock()])
            self.encoder = nn.TransformerEncoderLayer(
                d_model=8, nhead=4, dim_feedforward=16, dropout=EXPERIMENT.dropout
            )
            self.linear = nn.Linear(8, 1)
            self.signal_size = 8
        else:
            self.hiddenLayers = nn.ModuleList([
                nn.Sequential(nn.Linear(left, right), nn.ReLU(True), nn.Dropout(EXPERIMENT.dropout))
                for left, right in ((30, 16), (16, 8), (8, 4))
            ])
            self.finalLayer = nn.Linear(4, 1)
            self.signal_size = 4
        if holdings:
            original = self.head
            # The extension starts at zero and does not consume additional RNG.
            with torch.random.fork_rng(devices=[]):
                extended = nn.Linear(self.signal_size + 1, 1)
            with torch.no_grad():
                extended.weight[:, :-1].copy_(original.weight)
                extended.weight[:, -1].zero_()
                extended.bias.copy_(original.bias)
            if architecture == "cnn":
                self.linear = extended
            else:
                self.finalLayer = extended

    @property
    def head(self):
        return self.linear if self.architecture == "cnn" else self.finalLayer

    def encode(self, inputs):
        shape = inputs.shape[:-1]
        hidden = inputs.reshape(-1, EXPERIMENT.lookback)
        if self.architecture == "cnn":
            hidden = self.convBlocks[0](hidden[:, None, :]).permute(2, 0, 1)
            hidden = self.encoder(hidden)[-1]
        else:
            for layer in self.hiddenLayers:
                hidden = layer(hidden)
        return hidden.reshape(*shape, self.signal_size)

    def allocate(self, signals, previous=None):
        if self.holdings:
            signals = torch.cat([signals, previous.to(signals.dtype).unsqueeze(-1)], dim=-1)
        return self.head(signals).squeeze(-1)

    def forward(self, inputs, previous=None):
        return self.allocate(self.encode(inputs), previous)


def configure(architecture, device="auto", arm="signal_only"):
    """Match the study's precision policy; CNN uses CUDA by default."""
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    chosen = ("cuda" if architecture == "cnn" else "cpu") if device == "auto" else device
    torch.set_num_threads(4 if architecture == "cnn" or arm == "signal_only" else 1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    return torch.device(chosen)


def batches(length):
    for start in range(0, length, EXPERIMENT.batch_size):
        yield slice(start, min(start + EXPERIMENT.batch_size, length))


def stock_weights(q, b, c, batched=False):
    """Net residual positions into stocks, then normalize total stock exposure."""
    q = q.to(b.dtype)
    if batched:
        raw = q - torch.bmm(torch.bmm(q.unsqueeze(1), b), c.transpose(1, 2)).squeeze(1)
    else:
        raw = q - torch.einsum("...k,...nk->...n", torch.einsum("...n,...nk->...k", q, b), c)
    gross = raw.abs().sum(-1, keepdim=True)
    return raw / torch.where(gross == 0, torch.ones_like(gross), gross)
