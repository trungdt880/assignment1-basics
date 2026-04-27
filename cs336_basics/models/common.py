import math
import os
from collections.abc import Iterable
from typing import IO, BinaryIO

import numpy as np
import numpy.typing as npt
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
    params: Iterable[Float[torch.Tensor, "..."]], max_l2_norm: float, eps: float = 1e-6
):
    grads = [p.grad.data for p in params if p.grad is not None]

    total_grads = torch.sqrt(sum([(g**2).sum() for g in grads]))

    if total_grads >= max_l2_norm:
        scale = max_l2_norm / (total_grads + eps)
        for g in grads:
            g.mul_(scale)


def get_batch(
    dataset: npt.NDArray, batch_size: int, context_length: int, device: str
) -> tuple[torch.Tensor, torch.Tensor]:
    ids = np.random.randint(0, len(dataset) - context_length, size=batch_size)
    ids = ids[..., None] + np.arange(context_length)
    inputs = dataset[ids]
    targets = dataset[ids + 1]
    inputs = torch.tensor(inputs, dtype=torch.long, device=device)
    targets = torch.tensor(targets, dtype=torch.long, device=device)
    return inputs, targets


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    iteration: int,
    out: str | os.PathLike | BinaryIO | IO[bytes],
):
    model_state_dict = model.state_dict()
    opt_state_dict = optimizer.state_dict()
    final_state_dict = dict(
        model=model_state_dict, optimizer=opt_state_dict, iteration=iteration
    )
    torch.save(final_state_dict, out)


def load_checkpoint(
    src: str | os.PathLike | BinaryIO | IO[bytes],
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
) -> int:
    state_dict = torch.load(src)
    model.load_state_dict(state_dict["model"])
    optimizer.load_state_dict(state_dict["optimizer"])
    return state_dict["iteration"]


def set_seed(seed: int = 42):
    import random

    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
