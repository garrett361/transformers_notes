import math

import torch
import torch.nn as nn
from defaults import B, D, F, S

# Apologies for the lack of type-hinting, but it makes the latex less readable.


class MLP(nn.Module):
    def __init__(
        self,
        hidden_dim=D,
        expansion_factor=F,
        dropout=0.1,
    ):
        super().__init__()
        self.linear_1 = nn.Linear(hidden_dim, expansion_factor * hidden_dim)
        self.linear_2 = nn.Linear(expansion_factor * hidden_dim, hidden_dim)
        self.gelu = nn.GELU()
        self.drop = nn.Dropout(dropout)

    def forward(self, inputs):
        outputs = self.linear_1(inputs)
        outputs = self.gelu(outputs)
        outputs = self.linear_2(outputs)
        outputs = self.drop(outputs)
        return outputs


def test_mlp():
    inputs = torch.randn(B, S, D)
    m = MLP()
    outputs = m(inputs)
    assert outputs.shape == inputs.shape
