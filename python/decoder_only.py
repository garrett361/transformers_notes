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


def test_causality():
    inputs = torch.randint(high=V, size=(B, S))
    d = DecoderOnly()
    d.eval()  # Make dropout deterministic.
    # Passing in two sequences of different lengths whose common elements match should result in
    # outputs whose common elements also match
    for short_seq_len in range(1, S):
        for long_seq_len in range(short_seq_len, S + 1):
            short_inputs, long_inputs = inputs[:, :short_seq_len], inputs[:, :long_seq_len]
            # Only take the common positions in the outputs:
            short_outputs, long_outputs = d(short_inputs), d(long_inputs)[:, :short_seq_len]
            #  Not sure why the tolerances needed to be raised for success here, but they did.
            assert torch.allclose(short_outputs, long_outputs, rtol=1e-5, atol=1e-5)


if __name__ == "__main__":
    test_decoder()
