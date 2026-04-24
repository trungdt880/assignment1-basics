import einx
import torch
import torch.nn as nn
from jaxtyping import Float


class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None):
        super().__init__()
        self.d_model = d_model
        self.eps = eps
        tensor = torch.ones(d_model, device=device, dtype=dtype)
        self.weights = nn.Parameter(tensor)

    def forward(
        self, x: Float[torch.Tensor, "... d_model"]
    ) -> Float[torch.Tensor, "... d_model"]:
        in_dtype = x.dtype
        x = x.to(torch.float32)
        rms = torch.sqrt(einx.mean("... ([d_model])", x**2) + self.eps)
        result = einx.multiply("... d_model, d_model -> ... d_model", x, self.weights)
        result /= rms

        return result.to(in_dtype)
