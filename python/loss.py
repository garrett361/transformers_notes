import torch
import torch.nn.functional as F
from decoder_only import DecoderOnly
from defaults import A, B, D, E, K, L, V


def test_loss():
    model = DecoderOnly(
        num_attn_heads=A,
        block_size=K,
        dropout=0.1,
        expansion_factor=E,
        hidden_dim=D,
        num_layers=L,
        vocab_size=V,
    )
    tokens = torch.randint(model.vocab_size, size=(B, model.block_size + 1))
    inputs, targets = tokens[:, :-1], tokens[:, 1:]
    outputs = model(inputs)
    outputs_flat, targets_flat = outputs.reshape(-1, outputs.shape[-1]), targets.reshape(-1)
    loss = F.cross_entropy(outputs_flat, targets_flat)
    assert loss


if __name__ == "__main__":
    test_loss()
