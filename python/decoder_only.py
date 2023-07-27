import torch
import torch.nn as nn
from defaults import A, B, D, E, K, L, S, V
from transformer_block import TransformerBlock

# Apologies for the lack of type-hinting, but it makes the latex less readable.


class DecoderOnly(nn.Module):
    def __init__(
        self,
        attn_heads=A,
        block_size=K,
        dropout=0.1,
        expansion_factor=E,
        hidden_dim=D,
        layers=L,
        vocab_size=V,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        self.pos_encoding = nn.Parameter(torch.randn(1, block_size, hidden_dim))
        self.drop = nn.Dropout(dropout)
        self.trans_blocks = nn.ModuleList(
            [
                TransformerBlock(
                    attn_heads,
                    block_size,
                    dropout,
                    expansion_factor,
                    hidden_dim,
                    layers,
                    vocab_size,
                )
                for _ in range(layers)
            ]
        )
        self.final_ln = nn.LayerNorm(hidden_dim)
        self.lm_head = nn.Linear(hidden_dim, vocab_size, bias=False)
        self.lm_head.weight = self.embedding.weight  # Weight tying.

    def forward(self, inputs):
        S = inputs.shape[1]
        z = self.embedding(inputs) + self.pos_encoding[:, :S]
        z = self.drop(z)
        for block in self.trans_blocks:
            z = block(z)
        z = self.final_ln(z)
        z = self.lm_head(z)
        return z


def test_decoder():
    inputs = torch.randint(high=V, size=(B, S))
    d = DecoderOnly()
    outputs = d(inputs)
    assert outputs.shape == torch.Size([B, S, V])


if __name__ == "__main__":
    test_decoder()
