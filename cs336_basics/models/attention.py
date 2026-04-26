import einx
import torch
import torch.nn as nn
from jaxtyping import Bool, Float

from cs336_basics.models.linear import Linear
from cs336_basics.models.rope import RotaryPositionalEmbedding
from cs336_basics.models.sdpa import scaled_dot_product_attention


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, device=None, dtype=None):
        super().__init__()

        assert (
            d_model % num_heads == 0
        ), f"Model dim ({d_model=}) not divisible with {num_heads=}"

        self.model_dim = d_model
        self.num_heads = num_heads

        self.head_dim = self.model_dim // self.num_heads

        self.q_proj = Linear(
            self.model_dim, self.num_heads * self.head_dim, device=device, dtype=dtype
        )
        self.k_proj = Linear(
            self.model_dim, self.num_heads * self.head_dim, device=device, dtype=dtype
        )
        self.v_proj = Linear(
            self.model_dim, self.num_heads * self.head_dim, device=device, dtype=dtype
        )
        self.output_proj = Linear(
            self.num_heads * self.head_dim, self.model_dim, device=device, dtype=dtype
        )

    def forward(
        self,
        x: Float[torch.Tensor, "... seq_len model_dim"],
        mask: Bool[torch.Tensor, "... seq_len seq_len"] | None = None,
        position_embeddings: (
            tuple[
                Float[torch.Tensor, "max_seq_len half_head_dim"],
                Float[torch.Tensor, "max_seq_len half_head_dim"],
            ]
            | None
        ) = None,
    ) -> Float[torch.Tensor, "... seq_len model_dim"]:
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        q = einx.id(
            "... seq_len (num_heads head_dim) -> ... num_heads seq_len head_dim",
            q,
            num_heads=self.num_heads,
        )
        k = einx.id(
            "... seq_len (num_heads head_dim) -> ... num_heads seq_len head_dim",
            k,
            num_heads=self.num_heads,
        )
        if position_embeddings is not None:
            cos, sin = position_embeddings
            q = RotaryPositionalEmbedding.apply_rotary(q, sin=sin, cos=cos)
            k = RotaryPositionalEmbedding.apply_rotary(k, sin=sin, cos=cos)

        v = einx.id(
            "... seq_len (num_heads head_dim) -> ... num_heads seq_len head_dim",
            v,
            num_heads=self.num_heads,
        )

        if mask is None:
            num_token = x.shape[-2]
            mask = torch.ones(
                (1, 1, num_token, num_token), dtype=torch.bool, device=x.device
            ).tril()

            mask = mask.repeat((q.shape[0], q.shape[1], 1, 1))
        output = scaled_dot_product_attention(q, k, v, mask)
        output = einx.id(
            "... num_heads seq_len head_dim -> ... seq_len (num_heads head_dim)",
            output,
            num_heads=self.num_heads,
        )
        output = self.output_proj(output)
        return output
