import einx
import torch
import torch.nn as nn


class Linear(nn.Module):
    def __init__(self, in_features, out_features, device=None, dtype=None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        tensor = torch.empty((out_features, in_features), device=device, dtype=dtype)
        self.weights = nn.Parameter(tensor)
        self.init_parameters()

    def init_parameters(self):
        std = 2 / (self.in_features + self.out_features)
        nn.init.trunc_normal_(self.weights, std=std, a=-3 * std, b=3 * std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return einx.dot(
            "... [in_dim], out_dim [in_dim] -> ... out_dim", x, self.weights
        )
