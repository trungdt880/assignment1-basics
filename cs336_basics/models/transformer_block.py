import torch
import torch.nn as nn
from jaxtyping import Bool, Float

from cs336_basics.models.attention import MultiHeadAttention
from cs336_basics.models.rmsnorm import RMSNorm
from cs336_basics.models.swiglu import SwiGLU


class TransformerBlock(nn.Module):
    def __init__(
        self, d_model: int, num_heads: int, d_ff: int, device=None, dtype=None
    ):
        super().__init__()

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff

        self.attn = MultiHeadAttention(d_model, num_heads)
        self.ffn = SwiGLU(d_model, d_ff)
        self.ln1 = RMSNorm(d_model)
        self.ln2 = RMSNorm(d_model)

    def forward(
        self,
        x: Float[torch.Tensor, "... seq_len d_model"],
        mask: Bool[torch.Tensor, "... seq_len seq_len"] | None = None,
        position_embeddings: (
            tuple[
                Float[torch.Tensor, "max_seq_len half_head_dim"],
                Float[torch.Tensor, "max_seq_len half_head_dim"],
            ]
            | None
        ) = None,
    ) -> Float[torch.Tensor, "... seq_len d_model"]:
        attn = self.attn(self.ln1(x), position_embeddings=position_embeddings)
        x = x + attn
        ffn = self.ffn(self.ln2(x))
        return x + ffn
        # Total: 2*((model_dim)**2*seq_len)*4 + 4*seq_len*d_model + 6*d_model*seq_len*d_ff
