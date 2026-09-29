"""Character encoder with tied masked-character decoder and learned gap head."""

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint


class CharacterEncoder(nn.Module):
    def __init__(self, config, vocab_size):
        super().__init__()
        h = config["hidden_size"]
        self.config = config
        self.embedding = nn.Embedding(vocab_size, h, padding_idx=0)
        self.position = nn.Embedding(config["max_length"], h)
        # The token matrix is also the MLM decoder: unit-variance embeddings
        # would create overconfident random logits after the final LayerNorm.
        nn.init.normal_(self.embedding.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.position.weight, mean=0.0, std=0.02)
        with torch.no_grad():
            self.embedding.weight[0].zero_()
        self.blocks = nn.ModuleList(
            [
                nn.TransformerEncoderLayer(
                    h,
                    config["num_heads"],
                    config["intermediate_size"],
                    config.get("dropout", 0.1),
                    activation="gelu",
                    batch_first=True,
                    norm_first=True,
                )
                for _ in range(config["num_layers"])
            ]
        )
        self.norm = nn.LayerNorm(h)
        self.mlm_bias = nn.Parameter(torch.zeros(vocab_size))
        self.boundary = nn.Linear(2 * h, 1)
        self.activation_checkpointing = False

    def forward(self, ids, task="pretrain"):
        x = self.embedding(ids) + self.position(
            torch.arange(ids.shape[1], device=ids.device)
        )
        padding = ids.eq(0)
        for block in self.blocks:
            if self.training and self.activation_checkpointing:
                x = checkpoint(
                    lambda a, b=block: b(a, src_key_padding_mask=padding),
                    x,
                    use_reentrant=False,
                )
            else:
                x = block(x, src_key_padding_mask=padding)
        x = self.norm(x)
        if task == "pretrain":
            return torch.nn.functional.linear(x, self.embedding.weight, self.mlm_bias)
        return self.boundary(torch.cat((x[:, :-1], x[:, 1:]), dim=-1)).squeeze(-1)
