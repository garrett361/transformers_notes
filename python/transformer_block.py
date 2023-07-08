import torch
import torch.nn as nn
from causal_attention import CausalAttention
from defaults import A, B, D, F, K, L, S, V
from mlp import MLP

# Apologies for the lack of type-hinting, but it makes the latex less readable.


class TransformerBlock(nn.Module):
    def __init__(
        self,
        attn_heads=A,
        block_size=K,
        dropout=0.1,
        expansion_factor=F,
        hidden_dim=D,
        layers=L,
        vocab_size=V,
    ):
        super().__init__()
        self.attn_ln = nn.LayerNorm(hidden_dim)
        self.mlp_ln = nn.LayerNorm(hidden_dim)
        self.attn = CausalAttention(attn_heads, hidden_dim, block_size, dropout)
        self.mlp = MLP(hidden_dim, expansion_factor, dropout)

    def forward(self, inputs):
        z = self.attn(self.attn_ln(inputs)) + inputs
        z = self.mlp(self.mlp_ln(z)) + z
        return z


def test_transformer_block():
    inputs = torch.randn(B, S, D)
    t = TransformerBlock()
    outputs = t(inputs)
    assert outputs.shape == inputs.shape
