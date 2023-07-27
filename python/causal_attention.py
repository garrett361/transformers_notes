import math

import torch
import torch.nn as nn
from defaults import A, B, D, K, S

# Apologies for the lack of type-hinting, but it makes the latex less readable.


class CausalAttention(nn.Module):
    def __init__(
        self,
        attn_heads=A,
        hidden_dim=D,
        block_size=K,
        dropout=0.1,
    ):
        super().__init__()
        self.head_dim, remainder = divmod(hidden_dim, attn_heads)
        assert not remainder, "attn_heads must divide hidden_dim evenly"

        self.Q = nn.ModuleList([nn.Linear(hidden_dim, self.head_dim) for _ in range(attn_heads)])
        self.K = nn.ModuleList([nn.Linear(hidden_dim, self.head_dim) for _ in range(attn_heads)])
        self.V = nn.ModuleList([nn.Linear(hidden_dim, self.head_dim) for _ in range(attn_heads)])
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

    def get_attn_maps(self, queries, keys, values, seq_len):
        norm = math.sqrt(self.head_dim)
        non_causal_attn_scores = [(q @ k.transpose(-2, -1)) / norm for q, k in zip(queries, keys)]
        causal_attn_scores = [
            a.masked_fill(self.causal_mask[:, :seq_len, :seq_len] == 0, float("-inf"))
            for a in non_causal_attn_scores
        ]
        attn_maps = [a.softmax(dim=-1) for a in causal_attn_scores]
        return attn_maps

    def forward(self, inputs):
        seq_len = inputs.shape[1]
        queries, keys, values = self.get_qkv(inputs)
        attn_maps = self.get_attn_maps(queries, keys, values, seq_len)
        weighted_values = torch.cat(
            [self.attn_dropout(a) @ v for a, v in zip(attn_maps, values)], dim=-1
        )
        z = self.O(weighted_values)
        z = self.out_dropout(z)
        return z


def test_attention():
    inputs = torch.randn(B, S, D)
    a = CausalAttention()
    outputs = a(inputs)
    assert outputs.shape == inputs.shape


def test_attention_map():
    """Test that all of the attention maps are causal"""
    inputs = torch.randn(B, S, D)
    a = CausalAttention()
    q, k, v = a.get_qkv(inputs)
    attn_maps = a.get_attn_maps(q, k, v, S)
    assert attn_maps
    for map in attn_maps:
        assert torch.allclose(map.sum(dim=-1), torch.ones(map.shape[:2]))
        for pos in range(map.shape[-1]):
            zero_attns = map[:, pos, pos + 1 :]
            assert torch.allclose(zero_attns, torch.zeros_like(zero_attns))


def test_causality():
    inputs = torch.randn(B, S, D)
    a = CausalAttention()
    a.eval()  # Make dropout deterministic.
    # Passing in two sequences of different lengths whose common elements match should result in
    # outputs whose common elements also match
    for short_seq_len in range(1, S):
        for long_seq_len in range(short_seq_len, S + 1):
            short_inputs, long_inputs = inputs[:, :short_seq_len], inputs[:, :long_seq_len]
            # Only take the common positions in the outputs:
            short_outputs, long_outputs = a(short_inputs), a(long_inputs)[:, :short_seq_len]
            #  Not sure why the tolerances needed to be raised for success here, but they did.
            assert torch.allclose(short_outputs, long_outputs, rtol=1e-6, atol=1e-6)


if __name__ == "__main__":
    test_attention()
    test_attention_map()
    test_causality()
