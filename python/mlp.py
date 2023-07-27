import torch
import torch.nn as nn
from defaults import B, D, E, S

# Apologies for the lack of type-hinting, but it makes the latex less readable.


class MLP(nn.Module):
    def __init__(
        self,
        hidden_dim=D,
        expansion_factor=E,
    ):
        super().__init__()
        linear_1 = nn.Linear(hidden_dim, expansion_factor * hidden_dim)
        linear_2 = nn.Linear(expansion_factor * hidden_dim, hidden_dim)
        gelu = nn.GELU()
        self.layers = nn.Sequential(linear_1, gelu, linear_2)

    def forward(self, inputs):
        z = self.layers(inputs)
        return z


def test_mlp():
    inputs = torch.randn(B, S, D)
    m = MLP()
    outputs = m(inputs)
    assert outputs.shape == inputs.shape


if __name__ == "__main__":
    test_mlp()
