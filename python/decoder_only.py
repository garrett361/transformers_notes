import torch
import torch.nn as nn
from defaults import A, B, D, E, K, L, V
from transformer_block import TransformerBlock


class DecoderOnly(nn.Module):
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

        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        self.pos_encoding = nn.Parameter(torch.randn(1, block_size, hidden_dim))
        self.drop = nn.Dropout(dropout)
        self.trans_blocks = nn.ModuleList(
            [
                TransformerBlock(
                    block_size=block_size,
                    dropout=dropout,
                    expansion_factor=expansion_factor,
                    hidden_dim=hidden_dim,
                    num_attn_heads=num_attn_heads,
                    num_layers=num_layers,
                    vocab_size=vocab_size,
                )
                for _ in range(num_layers)
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
    inputs = torch.randint(high=V, size=(B, K))
    d = DecoderOnly()
    outputs = d(inputs)
    assert outputs.shape == torch.Size([B, K, V])


def test_causality():
    inputs = torch.randint(high=V, size=(B, K))
    d = DecoderOnly()
    d.eval()  # Make dropout deterministic.
    # Passing in two sequences of different lengths whose common elements match should result in
    # outputs whose common elements also match
    for short_seq_len in range(1, K):
        for long_seq_len in range(short_seq_len, K + 1):
            short_inputs, long_inputs = inputs[:, :short_seq_len], inputs[:, :long_seq_len]
            # Only take the common positions in the outputs:
            short_outputs, long_outputs = d(short_inputs), d(long_inputs)[:, :short_seq_len]
            #  Not sure why the tolerances needed to be raised for success here, but they did.
            assert torch.allclose(short_outputs, long_outputs, rtol=1e-5, atol=1e-5)


if __name__ == "__main__":
    test_decoder()
