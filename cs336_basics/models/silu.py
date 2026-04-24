import torch
import torch.nn as nn
from jaxtyping import Float


class SiLU(nn.Module):
    def forward(
        self, x: Float[torch.Tensor, "... dim"]
    ) -> Float[torch.Tensor, "... dim"]:
        return torch.sigmoid(x) * x
