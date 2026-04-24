import einx
import torch
import torch.nn as nn


class Linear(nn.Module):
    def __init__(self, in_features, out_features, device=None, dtype=None):
        super().__init__()
        tensor = torch.empty((out_features, in_features), device=device, dtype=dtype)
        std = 2 / (in_features + out_features)
        weights = nn.init.trunc_normal_(tensor, std=std, a=-3 * std, b=3 * std)
        self.weights = nn.Parameter(weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return einx.dot(
            "... [in_dim], out_dim [in_dim] -> ... out_dim", x, self.weights
        )
