import math
from collections.abc import Callable, Iterable

import torch
import torch.nn as nn


class SGDDecay(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr=}")
        defaults = {"lr": lr}
        super().__init__(params, defaults)

    def step(self, closure: Callable | None = None) -> float | None:
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"]
            for p in group["params"]:
                if p.grad is None:
                    continue

                state = self.state[p]
                t = state.get("t", 0)
                grad = p.grad.data
                p.data -= lr / math.sqrt(t + 1) * grad
                state["t"] = t + 1

        return loss


class AdamW(torch.optim.Optimizer):
    def __init__(
        self,
        params,
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 1e-2,
    ):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr=}")
        defaults = {"lr": lr, "betas": betas, "eps": eps, "weight_decay": weight_decay}
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure: Callable | None = None) -> float | None:
        loss = None if closure is None else closure()
        for group in self.param_groups:
            alpha = group["lr"]
            betas = group["betas"]
            beta1, beta2 = betas
            eps = group["eps"]
            weight_decay = group["weight_decay"]
            for p in group["params"]:
                if p.grad is None:
                    continue

                state = self.state[p]
                t = state.get("t", 1)
                m = state.get("m", torch.zeros_like(p))
                v = state.get("v", torch.zeros_like(p))

                beta1_t = beta1**t
                beta2_t = beta2**t
                alpha_t = alpha * math.sqrt(1 - beta2_t) / (1 - beta1_t)
                grad = p.grad

                # weight decay
                p -= alpha * weight_decay * p

                m = beta1 * m + (1 - beta1) * grad
                v = beta2 * v + (1 - beta2) * (grad**2)

                # moment-based weight update
                p.data -= alpha_t * (m / (torch.sqrt(v) + eps))

                state["t"] = t + 1
                state["m"] = m
                state["v"] = v
        return loss


if __name__ == "__main__":
    w = 5 * torch.rand((10, 10))
    for lr in [1]:
        # for lr in [1, 1e2, 1e3]:
        print("=" * 80)
        print(f"{lr=}")
        weights = nn.Parameter(w.clone())
        # opt = SGDDecay([weights], lr=lr)
        opt = AdamW([weights], lr=lr)
        for t in range(10):
            opt.zero_grad()
            loss = (weights**2).mean()
            print(loss.cpu().item())
            loss.backward()
            opt.step()
