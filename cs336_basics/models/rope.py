import torch
import torch.nn as nn
from jaxtyping import Float, Int


class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None):
        super().__init__()

        self.theta = theta
        self.d_k = d_k
        self.max_seq_len = max_seq_len

        token_ids = torch.arange(0, max_seq_len, device=device)
        steps = torch.arange(0, d_k // 2, device=device)
        steps = (2 * steps) / d_k
        denum = self.theta ** (steps)
        angles = token_ids[..., None] / denum
        cosines = torch.cos(angles)
        sines = torch.sin(angles)
        self.register_buffer("cosines", cosines, persistent=False)
        self.register_buffer("sines", sines, persistent=False)

    @staticmethod
    def apply_rotary(
        x: Float[torch.Tensor, "... seq_len d_k"],
        sin: Float[torch.Tensor, "seq_len half_d_k"],
        cos: Float[torch.Tensor, "seq_len half_d_k"],
    ) -> Float[torch.Tensor, "... seq_len d_k"]:
        x_even = x[..., ::2]
        x_odd = x[..., 1::2]
        new_even = cos * x_even - sin * x_odd
        new_odd = sin * x_even + cos * x_odd
        x = torch.stack([new_even, new_odd], dim=-1).flatten(-2)
        return x

    def forward(
        self,
        x: Float[torch.Tensor, "... seq_len d_k"],
        token_positions: Int[torch.Tensor, "... seq_len"],
    ) -> Float[torch.Tensor, "... seq_len d_k"]:
        token_sine = self.sines[token_positions]
        token_cosine = self.cosines[token_positions]
        return self.apply_rotary(x, sin=token_sine, cos=token_cosine)
