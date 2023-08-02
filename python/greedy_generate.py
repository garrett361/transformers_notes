import torch
from decoder_only import DecoderOnly
from defaults import B


class DecoderOnlyGreedy(DecoderOnly):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def generate(self, inputs, max_length):
        """
        Naive, minimal generation method. Assumes inputs are already tokenized. max_length can be
        longer than the block_size, but only up to block_size tokens can ever be included in the
        context.
        """
        self.eval()
        outputs = inputs.clone()
        while outputs.shape[1] < max_length:
            context = outputs[:, -self.block_size :]
            last_token_pred_logits = self(context)[:, -1]
            most_probable_token = last_token_pred_logits.argmax(dim=-1)[:, None]
            outputs = torch.cat([outputs, most_probable_token], dim=-1)
        return outputs


def test_generation():
    model = DecoderOnlyGreedy()
    inputs = torch.randint(high=model.vocab_size, size=(B, model.block_size // 2))
    outputs = model.generate(inputs, model.block_size)
    assert outputs.shape == torch.Size([B, model.block_size])
    assert outputs.max() <= model.vocab_size


def test_long_generation():
    model = DecoderOnlyGreedy()
    inputs = torch.randint(high=model.vocab_size, size=(B, model.block_size // 2))
    outputs = model.generate(inputs, 2 * model.block_size)
    assert outputs.shape == torch.Size([B, 2 * model.block_size])
    assert outputs.max() <= model.vocab_size


if __name__ == "__main__":
    test_generation()
    test_long_generation()
