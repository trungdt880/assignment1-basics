import torch
import torch.nn as nn
from jaxtyping import Bool, Float, Int

from cs336_basics.models.embedding import Embedding
from cs336_basics.models.linear import Linear
from cs336_basics.models.rmsnorm import RMSNorm
from cs336_basics.models.rope import RotaryPositionalEmbedding
from cs336_basics.models.transformer_block import TransformerBlock


class TransformerLM(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        context_length: int,
        d_model: int,
        num_layers: int,
        num_heads: int,
        d_ff: int,
        rope_theta: float,
    ):
        super().__init__()

        self.vocab_size = vocab_size
        self.context_length = context_length
        self.num_layers = num_layers
        self.d_model = d_model
        self.d_ff = d_ff
        self.num_heads = num_heads
        self.rope_theta = rope_theta
        self.head_dim = d_model // num_heads

        self.token_embeddings = Embedding(vocab_size, d_model)
        self.rope = RotaryPositionalEmbedding(rope_theta, self.head_dim, context_length)
        self.layers = nn.ModuleList(
            [TransformerBlock(d_model, num_heads, d_ff) for _ in range(num_layers)]
        )
        self.ln_final = RMSNorm(d_model)
        self.lm_head = Linear(d_model, vocab_size)

    def forward(
        self,
        x: Int[torch.Tensor, "batch seq_len"],
        mask: Bool[torch.Tensor, "... seq_len seq_len"] | None = None,
    ) -> Float[torch.Tensor, "batch seq_len vocab_size"]:
        token_positions = torch.arange(0, x.shape[1], device=x.device)
        cos = self.rope.cosines[token_positions]
        sin = self.rope.sines[token_positions]
        position_embeddings = (cos, sin)

        x = self.token_embeddings(x)

        for _, layer in enumerate(self.layers):
            x = layer(x, position_embeddings=position_embeddings)
        x = self.ln_final(x)
        x = self.lm_head(x)
        return x
