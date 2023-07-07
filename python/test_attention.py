import math

import torch
import torch.nn as nn
from defaults import A, B, D, S

# Apologies for the lack of type-hinting, but it makes the latex less readable.


class Attention(nn.Module):
    def __init__(
        self,
        attn_heads=A,
        hidden_dim=D,
        block_size=S,
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
        self.final_dropout = nn.Dropout(dropout)
        self.register_buffer(
            "causal_mask",
            torch.tril(torch.ones(block_size, block_size)[None]),
        )

    def get_qkv(self, inputs):
        queries = [q(inputs) for q in self.Q]
        keys = [k(inputs) for k in self.K]
        values = [v(inputs) for v in self.V]
        return queries, keys, values

    def get_attn_maps(self, queries, keys, values):
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
        attn_maps = self.get_attn_maps(queries, keys, values)
        weighted_values = torch.concat(
            [self.attn_dropout(a) @ v for a, v in zip(attn_maps, values)], dim=-1
        )
        outputs = self.final_dropout(self.O(weighted_values))
        return outputs


def test_attention():
    inputs = torch.randn(B, S, D)
    a = Attention()
    outputs = a(inputs)
    assert outputs.shape == inputs.shape


def test_attention_map():
    """Test that all of the attention maps are causal"""
    inputs = torch.randn(B, S, D)
    a = Attention()
    q, k, v = a.get_qkv(inputs)
    attn_maps = a.get_attn_maps(q, k, v)
    assert attn_maps
    for map in attn_maps:
        assert torch.allclose(map.sum(dim=-1), torch.ones(map.shape[:2]))
        for pos in range(map.shape[-1]):
            zero_attns = map[:, pos, pos + 1 :]
            assert torch.allclose(zero_attns, torch.zeros_like(zero_attns))
