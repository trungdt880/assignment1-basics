import einx
import torch
import torch.nn as nn
from jaxtyping import Float, Int


class Embedding(nn.Module):
    def __init__(
        self, num_embeddings: int, embedding_dim: int, device=None, dtype=None
    ):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        tensor = torch.empty(
            (num_embeddings, embedding_dim), device=device, dtype=dtype
        )
        self.weights = nn.Parameter(tensor)
        self.init_parameters()

    def init_parameters(self):
        std = 2 / (self.num_embeddings + self.embedding_dim)
        torch.nn.init.trunc_normal_(self.weights, std=std, a=-3 * std, b=-3 * std)

    def forward(
        self, token_ids: Int[torch.Tensor, "..."]
    ) -> Float[torch.Tensor, "... embedding_dim"]:
        return einx.get_at(
            "[num_embeddings] embedding_dim, b idx  -> b idx embedding_dim",
            self.weights,
            token_ids,
        )
