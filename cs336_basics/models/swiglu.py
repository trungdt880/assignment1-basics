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
        self.linear1 = Linear(d_model, d_ff, device, dtype)
        self.linear2 = Linear(d_ff, d_model, device, dtype)
        self.linear3 = Linear(d_model, d_ff, device, dtype)

    def forward(self, x):
        return self.linear2(self.silu(self.linear1(x)) * self.linear3(x))
