import math

import torch
import torch.nn as nn
from causal_attention import CausalAttention
from defaults import A, B, D, K


class CausalAttentionWithCache(CausalAttention):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.cached_keys = self.cached_values = None

    def get_qkv(self, inputs):
        queries = [q(inputs) for q in self.Q]
        keys = [k(inputs) for k in self.K]
        values = [v(inputs) for v in self.V]
        return queries, keys, values

    def forward(self, inputs, use_cache=True):
        """Forward method with optional cache. When use_cache == True, the output will have a
        sequence length of one."""
        if not use_cache:
            return super().forward(inputs)
        # Get all q, k, v values if the cache is not initialized.
        if self.cached_keys is None:
            assert (
                self.cached_values is None
            ), "If cached_keys is None, cached_values should be None, too"
            queries, keys, values = self.get_qkv(inputs)
        else:
            # Otherwise, we only need q, k, v values for the last sequence position.
            queries, new_keys, new_values = self.get_qkv(inputs[:, [-1]])
            keys = [torch.cat([ck, nk], dim=1) for ck, nk in zip(self.cached_keys, new_keys)]
            values = [torch.cat([cv, nv], dim=1) for cv, nv in zip(self.cached_values, new_values)]
        # Update/initialize the cache.
        self.cached_keys = [k[:, -self.block_size + 1 :] for k in keys]
        self.cached_values = [v[:, -self.block_size + 1 :] for v in values]
        last_queries = [q[:, [-1]] for q in queries]
        attn_maps = self.get_attn_maps(last_queries, keys)
        weighted_values = torch.cat(
            [self.attn_dropout(a) @ v for a, v in zip(attn_maps, values)], dim=-1
        )
        z = self.O(weighted_values)
        z = self.out_dropout(z)
        return z

    def clear_cache(self):
        self.cached_keys = self.cached_values = None


def test_attention():
    c = CausalAttentionWithCache()
    inputs = torch.randn(B, c.block_size // 2, c.hidden_dim)
    while inputs.shape[1] <= c.block_size:
        outputs = c(inputs, use_cache=True)
        assert outputs.shape[1] == 1
        inputs = torch.cat([inputs, outputs], dim=1)


def test_consistency():
    """Verify that the cached forward method gives the same results as the model without a cache."""
    c = CausalAttentionWithCache()
    c_no_cache = CausalAttention()
    # Ensure they have the same parameters
    for (n1, p1), (n2, p2) in zip(c.named_parameters(), c_no_cache.named_parameters()):
        assert n1 == n2
        p1.data = p2.data
    c.eval()
    c_no_cache.eval()

    inputs = torch.randn(B, c.block_size // 2, c.hidden_dim)
    while inputs.shape[1] <= c.block_size:
        with torch.inference_mode():
            outputs = c(inputs, use_cache=True)
            outputs_no_cache = c_no_cache(inputs)[:, [-1]]
            assert torch.allclose(outputs, outputs_no_cache, rtol=1e-6, atol=1e-6)
            inputs = torch.cat([inputs, outputs], dim=1)


if __name__ == "__main__":
    test_attention()
    test_consistency()
