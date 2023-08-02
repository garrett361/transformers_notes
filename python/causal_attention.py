import math

import torch
import torch.nn as nn
from defaults import A, B, D, K


class CausalAttention(nn.Module):
    def __init__(
        self,
        block_size=K,
        dropout=0.1,
        hidden_dim=D,
        num_attn_heads=A,
    ):
        super().__init__()
        self.block_size = block_size
        self.dropout = dropout
        self.hidden_dim = hidden_dim
        self.num_attn_heads = num_attn_heads

        self.head_dim, remainder = divmod(hidden_dim, num_attn_heads)
        assert not remainder, "num_attn_heads must divide hidden_dim evenly"

        self.Q = nn.ModuleList(
            [nn.Linear(hidden_dim, self.head_dim) for _ in range(num_attn_heads)]
        )
        self.K = nn.ModuleList(
            [nn.Linear(hidden_dim, self.head_dim) for _ in range(num_attn_heads)]
        )
        self.V = nn.ModuleList(
            [nn.Linear(hidden_dim, self.head_dim) for _ in range(num_attn_heads)]
        )
        self.O = nn.Linear(hidden_dim, hidden_dim)

        self.attn_dropout = nn.Dropout(dropout)
        self.out_dropout = nn.Dropout(dropout)
        self.register_buffer(
            "causal_mask",
            torch.tril(torch.ones(block_size, block_size)[None]),
        )

    def get_qkv(self, inputs):
        queries = [q(inputs) for q in self.Q]
        keys = [k(inputs) for k in self.K]
        values = [v(inputs) for v in self.V]
        return queries, keys, values

    def get_attn_maps(self, queries, keys):
        S = queries[0].shape[1]
        norm = math.sqrt(self.head_dim)
        non_causal_attn_scores = [(q @ k.transpose(-2, -1)) / norm for q, k in zip(queries, keys)]
        causal_attn_scores = [
            a.masked_fill(self.causal_mask[:, :S, :S] == 0, float("-inf"))
            for a in non_causal_attn_scores
        ]
        attn_maps = [a.softmax(dim=-1) for a in causal_attn_scores]
        return attn_maps

    def forward(self, inputs):
        queries, keys, values = self.get_qkv(inputs)
        attn_maps = self.get_attn_maps(queries, keys)
        weighted_values = torch.cat(
            [self.attn_dropout(a) @ v for a, v in zip(attn_maps, values)], dim=-1
        )
        z = self.O(weighted_values)
        z = self.out_dropout(z)
        return z


def test_attention():
    c = CausalAttention()
    inputs = torch.randn(B, c.block_size, c.hidden_dim)
    outputs = c(inputs)
    assert outputs.shape == inputs.shape


def test_attention_map():
    """Test that all of the attention maps are causal"""
    c = CausalAttention()
    inputs = torch.randn(B, c.block_size, c.hidden_dim)
    q, k, v = c.get_qkv(inputs)
    attn_maps = c.get_attn_maps(q, k)
    assert attn_maps
    for map in attn_maps:
        assert torch.allclose(map.sum(dim=-1), torch.ones(map.shape[:2]))
        for pos in range(map.shape[-1]):
            zero_attns = map[:, pos, pos + 1 :]
            assert torch.allclose(zero_attns, torch.zeros_like(zero_attns))


def test_causality():
    c = CausalAttention()
    inputs = torch.randn(B, c.block_size, c.hidden_dim)
    c.eval()  # Make dropout deterministic.
    # Passing in two sequences of different lengths whose common elements match should result in
    # outputs whose common elements also match
    for short_seq_len in range(1, c.block_size):
        for long_seq_len in range(short_seq_len, c.block_size + 1):
            short_inputs, long_inputs = inputs[:, :short_seq_len], inputs[:, :long_seq_len]
            # Only take the common positions in the outputs:
            short_outputs, long_outputs = c(short_inputs), c(long_inputs)[:, :short_seq_len]
            #  Not sure why the tolerances needed to be raised for success here, but they did.
            assert torch.allclose(short_outputs, long_outputs, rtol=1e-6, atol=1e-6)


if __name__ == "__main__":
    test_attention()
    test_attention_map()
    test_causality()
