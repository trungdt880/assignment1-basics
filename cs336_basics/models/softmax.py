import einx
import torch
import torch.nn as nn
from jaxtyping import Float


def softmax(x: Float[torch.Tensor, "..."], dim):
    max = torch.max(x, dim=dim)[0][..., None]
    exp = torch.exp(x - max)
    sum_exp = torch.sum(exp, dim=dim, keepdim=True)
    return exp / sum_exp
