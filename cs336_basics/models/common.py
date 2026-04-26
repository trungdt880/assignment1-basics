import math

import torch
from jaxtyping import Float, Int


def round_up_to_multiple(x: int, m: int) -> int:
    return m * math.ceil(x / m)


def cross_entropy_loss(
    inputs: Float[torch.Tensor, " batch_size vocab_size"],
    targets: Int[torch.Tensor, " batch_size"],
) -> Float[torch.Tensor, ""]:
    max = torch.max(inputs, dim=-1, keepdim=True)[0]
    normed_inputs = inputs - max

    inputs_exp = torch.exp(normed_inputs)
    sum_exp = torch.sum(inputs_exp, dim=-1, keepdim=True)
    loss = torch.log(sum_exp) - normed_inputs.gather(-1, targets[..., None])
    return loss.mean()
