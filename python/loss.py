import torch
import torch.nn as nn
from decoder_only import DecoderOnly
from defaults import B, D, F, H, K, L, V

# Apologies for the lack of type-hinting, but it makes the latex less readable.


def test_loss():
    model = DecoderOnly(
        attn_heads=H,
        block_size=K,
        dropout=0.1,
        expansion_factor=F,
        hidden_dim=D,
        layers=L,
        vocab_size=V,
    )
    tokens = torch.randint(V, size=(B, K + 1))
    inputs, targets = tokens[:, :-1], tokens[:, 1:]
    outputs = model(inputs)
    outputs_flat, targets_flat = outputs.reshape(-1, outputs.shape[-1]), targets.reshape(-1)
    criterion = nn.CrossEntropyLoss()
    loss = criterion(outputs_flat, targets_flat)
    assert loss
