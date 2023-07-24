import torch
import torch.nn as nn
from causal_attention import CausalAttention
from defaults import B, D, E, H, K, L, S, V
from mlp import MLP

# Apologies for the lack of type-hinting, but it makes the latex less readable.


class TransformerBlock(nn.Module):
    def __init__(
        self,
        attn_heads=H,
        block_size=K,
        dropout=0.1,
        expansion_factor=E,
        hidden_dim=D,
        layers=L,
        vocab_size=V,
    ):
        super().__init__()
        self.attn_ln = nn.LayerNorm(hidden_dim)
        self.attn = CausalAttention(attn_heads, hidden_dim, block_size, dropout)
        self.attn_drop = nn.Dropout(dropout)

        self.mlp_ln = nn.LayerNorm(hidden_dim)
        self.mlp = MLP(hidden_dim, expansion_factor)
        self.mlp_drop = nn.Dropout(dropout)

    def forward(self, inputs):
        z_attn = self.attn_ln(inputs)
        z_attn = self.attn(z_attn)
        z_attn = self.attn_drop(z_attn) + inputs

        z_mlp = self.mlp_ln(z_attn)
        z_mlp = self.mlp(z_mlp)
        z = self.mlp_drop(z_mlp) + z_attn
        return z


def test_transformer_block():
    inputs = torch.randn(B, S, D)
    t = TransformerBlock()
    outputs = t(inputs)
    assert outputs.shape == inputs.shape
