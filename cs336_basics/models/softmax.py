import einx
import torch
import torch.nn as nn
from jaxtyping import Float


def softmax(
    x: Float[torch.Tensor, "..."], dim: int, temperature: float | None = None
) -> Float[torch.Tensor, "..."]:
    if temperature is not None:
        x = x / temperature
    max_x = torch.max(x, dim=dim, keepdim=True)[0]
    exp = torch.exp(x - max_x)
    sum_exp = torch.sum(exp, dim=dim, keepdim=True)
    return exp / sum_exp
