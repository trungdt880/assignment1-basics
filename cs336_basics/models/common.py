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


def lr_cosine_schedule(
    cur_step: int, lr_min: float, lr_max: float, warmup_step: int, cosine_cycle_step
) -> float:
    if cur_step < warmup_step:
        return cur_step / warmup_step * lr_max
    elif cur_step <= cosine_cycle_step:
        step_length = cosine_cycle_step - warmup_step
        real_cur_step = cur_step - warmup_step

        return lr_min + 1 / 2 * (
            1 + math.cos(real_cur_step / step_length * math.pi)
        ) * (lr_max - lr_min)
    elif cur_step > cosine_cycle_step:
        return lr_min
    else:
        raise ValueError(f"Invalid step {cur_step}")


def gradient_clipping(
    params: Float[torch.Tensor, "..."], max_l2_norm: float, eps: float = 1e-6
):
    grads = [p.grad.data for p in params if p.grad is not None]

    total_grads = torch.sqrt(torch.sum([(g**2).sum() for g in grads]))

    if total_grads >= max_l2_norm:
        scale = max_l2_norm / (total_grads + eps)
        for g in grads:
            g.mul_(scale)
