import einx
import torch
import torch.nn as nn
from jaxtyping import Float

from cs336_basics.models.common import round_up_to_multiple
from cs336_basics.models.linear import Linear
from cs336_basics.models.silu import SiLU


class SwiGLU(nn.Module):
    def __init__(self, d_model: int, d_ff: int | None, device=None, dtype=None):
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        if d_ff is None:
            self.d_ff = round_up_to_multiple(d_model, 64)

        self.silu = SiLU()
        self.w1 = Linear(d_model, d_ff, device, dtype)
        self.w2 = Linear(d_ff, d_model, device, dtype)
        self.w3 = Linear(d_model, d_ff, device, dtype)

    def forward(self, x):
        return self.w2(self.silu(self.w1(x)) * self.w3(x))
