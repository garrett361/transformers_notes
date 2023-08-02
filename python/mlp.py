import torch
import torch.nn as nn
from defaults import B, D, E, K


class MLP(nn.Module):
    def __init__(
        self,
        hidden_dim=D,
        expansion_factor=E,
        dropout=0.1,
    ):
        super().__init__()
        linear_1 = nn.Linear(hidden_dim, expansion_factor * hidden_dim)
        linear_2 = nn.Linear(expansion_factor * hidden_dim, hidden_dim)
        gelu = nn.GELU()
        self.layers = nn.Sequential(linear_1, gelu, linear_2)
        self.dropout = nn.Dropout(dropout)

    def forward(self, inputs):
        z = self.layers(inputs)
        z = self.dropout(z)
        return z


def test_mlp():
    inputs = torch.randn(B, K, D)
    m = MLP()
    outputs = m(inputs)
    assert outputs.shape == inputs.shape


if __name__ == "__main__":
    test_mlp()
