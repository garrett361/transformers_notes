import torch
import torch.nn as nn
from causal_attention import CausalAttention
from defaults import A, B, D, E, K, L, V
from mlp import MLP


class TransformerBlock(nn.Module):
    def __init__(
        self,
        block_size=K,
        dropout=0.1,
        expansion_factor=E,
        hidden_dim=D,
        num_attn_heads=A,
        num_layers=L,
        vocab_size=V,
    ):
        super().__init__()
        self.block_size = block_size
        self.dropout = dropout
        self.expansion_factor = expansion_factor
        self.hidden_dim = hidden_dim
        self.num_attn_heads = num_attn_heads
        self.num_layers = num_layers
        self.vocab_size = vocab_size

        self.attn_ln = nn.LayerNorm(hidden_dim)
        self.attn = CausalAttention(
            block_size=block_size,
            dropout=dropout,
            hidden_dim=hidden_dim,
            num_attn_heads=num_attn_heads,
        )

        self.mlp_ln = nn.LayerNorm(hidden_dim)
        self.mlp = MLP(hidden_dim, expansion_factor, dropout)

    def forward(self, inputs):
        z_attn = self.attn_ln(inputs)
        z_attn = self.attn(z_attn) + inputs

        z_mlp = self.mlp_ln(z_attn)
        z_mlp = self.mlp(z_mlp) + z_attn
        return z_mlp


def test_transformer_block():
    inputs = torch.randn(B, K, D)
    t = TransformerBlock()
    outputs = t(inputs)
    assert outputs.shape == inputs.shape


if __name__ == "__main__":
    test_transformer_block()
